"""Diagnose prefill-logit nondeterminism through the adapter (full model): same prompt, repeated, same / different slots,
fresh blocks each time, with and without the large scratch region."""
import os
import sys

import numpy as np
import torch

import ttnn

sys.path.insert(0, "/home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal")
from transformers import AutoConfig

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import open_mesh
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm import TTLfm2MoeForCausalLM
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import resolve_model_path

N_BLOCKS = 4096


def row(blocks):
    r = torch.zeros(2048, dtype=torch.int32)
    r[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return r


def main():
    mesh = open_mesh(768 << 20)
    try:
        cfg = AutoConfig.from_pretrained(str(resolve_model_path()))
        ad = TTLfm2MoeForCausalLM.initialize_vllm_model(cfg, mesh, 32, max_seq_len=131072)
        kv = ad.allocate_kv_cache((N_BLOCKS, 2, 64, 64), torch.bfloat16, 10)
        for t in (False, True):
            ad.warmup_model_prefill(kv_cache=kv, enable_trace=t)
            ad.warmup_model_decode(kv_cache=kv, max_batch_size=32, num_blocks=2048, enable_trace=t)
        g = torch.Generator().manual_seed(4242)
        PLEN = int(os.environ.get("DIAG_LEN", "300"))
        p = torch.randint(100, 60000, (PLEN,), generator=g)
        nxt = [1]

        def take(n):
            b = list(range(nxt[0], nxt[0] + n))
            nxt[0] += n
            return b

        def pf(slot, blocks):
            toks = p.reshape(1, -1)
            return ad.prefill_forward(
                tokens=toks,
                start_pos=np.array([0]),
                prompt_lens=np.array([PLEN]),
                page_table=row(blocks).reshape(1, -1),
                kv_cache=kv,
                empty_slots=[slot],
            )

        from vllm_tt_plugin.model_input import TTSamplingParams

        def sp(n):
            return TTSamplingParams(
                temperature=[0.0] * n,
                top_k=[1] * n,
                top_p=[1.0] * n,
                presence_penalty=[0.0] * n,
                frequency_penalty=[0.0] * n,
                repetition_penalty=[1.0] * n,
                seed=[None] * n,
                num_logprobs=[-2] * n,
                enable_log_probs=[False] * n,
            )

        # three live decode requests in slots 0..2
        live = [torch.randint(100, 60000, (n,), generator=g) for n in (90, 140, 200)]
        lblk = [take(12) for _ in live]
        L = max(len(x) for x in live)
        toks = torch.zeros(3, L, dtype=torch.int64)
        for i, x in enumerate(live):
            toks[i, : len(x)] = x
        first = ad.prefill_forward(
            tokens=toks,
            start_pos=np.array([0, 0, 0]),
            prompt_lens=np.array([len(x) for x in live]),
            page_table=torch.stack([row(b) for b in lblk]),
            kv_cache=kv,
            empty_slots=[0, 1, 2],
            sampling_params=sp(3),
        )
        cur = [int(t) for t in first[:, 0]]
        pos = [len(x) for x in live]

        def dec(reload):
            tk = torch.zeros(4, 1, dtype=torch.int64)
            st = torch.full((4,), -1, dtype=torch.int64)
            pt = torch.zeros(4, 2048, dtype=torch.int32)
            for r in range(3):
                tk[r, 0], st[r], pt[r] = cur[r], pos[r], row(lblk[r])
            out = ad.decode_forward(
                tokens=tk,
                start_pos=st,
                page_table=pt,
                kv_cache=kv,
                read_from_device=True,
                sampling_params=sp(4),
                reload_inputs=reload,
                reload_page_table=False,
                reload_sampling_params=reload,
                reset_sampling_state=reload,
            )
            for r in range(3):
                cur[r] = int(out[r, 0])
                pos[r] += 1

        ref, bad = None, []
        slots = [10, 13, 5, 20, 31, 3]
        for it in range(int(os.environ.get("DIAG_ITERS", "150"))):
            s = slots[it % len(slots)]
            if os.environ.get("DIAG_SYNC") == "1":
                ttnn.synchronize_device(mesh)
            o = pf(s, take((PLEN + 63) // 64))
            if ref is None:
                ref = o
            elif not torch.equal(o, ref):
                d = (o - ref).abs()
                bad.append(it)
                print(
                    f"MISMATCH it={it} slot={s} maxdiff={d.max().item():.4g} ndiff={(d>0).sum().item()} argmax {int(o[0,0].argmax())} vs {int(ref[0,0].argmax())}",
                    flush=True,
                )
            if os.environ.get("DIAG_NODECODE") != "1":
                dec(reload=True)
                for _ in range(2):
                    dec(reload=False)
            if nxt[0] > N_BLOCKS - 80:
                nxt[0] = 100  # recycle blocks (fresh content is not required)
        print("mismatches", len(bad), "of", it, "len", PLEN)
        ad.release_persistent_capture()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

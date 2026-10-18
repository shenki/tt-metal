"""Diagnose prefill-logit nondeterminism through the adapter (full model): same prompt, repeated, same / different slots,
fresh blocks each time, with and without the large scratch region."""
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
        p = torch.randint(100, 60000, (300,), generator=g)
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
                prompt_lens=np.array([300]),
                page_table=row(blocks).reshape(1, -1),
                kv_cache=kv,
                empty_slots=[slot],
            )

        outs = []
        for i, slot in enumerate([0, 0, 0, 13, 13, 0, 5, 0]):
            blk = take(5)
            outs.append((slot, pf(slot, blk)))
        ref = outs[0][1]
        for slot, o in outs:
            d = (o - ref).abs().max().item()
            print(
                f"slot {slot:2d} equal={torch.equal(o, ref)} maxdiff={d:.4g} argmax={int(o[0,0].argmax())} vs {int(ref[0,0].argmax())}"
            )
        # same blocks reused (no fresh blocks)
        blk = take(5)
        a = pf(0, blk)
        b = pf(0, blk)
        print("same blocks, same slot equal:", torch.equal(a, b), (a - b).abs().max().item())
        ad.release_persistent_capture()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

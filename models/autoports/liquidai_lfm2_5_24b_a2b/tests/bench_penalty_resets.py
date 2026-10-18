# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cost of the TTPenalties request-boundary calls on the 1x4 mesh (vocab 129024, 32 rows): prompt-mask reset for one
slot (prefill of a penalized request) and the full history rebuild (decode transition with reset_sampling_state).

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.bench_penalty_resets
"""

import time
from types import SimpleNamespace

import torch

import ttnn


def main():
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import open_mesh
    from models.common.sampling.tt_penalties import TTPenalties

    mesh = open_mesh(64 << 20)
    try:
        args = SimpleNamespace(vocab_size=128000, padded_vocab_size=129024, cluster_shape=(1, 4), max_batch_size=32)
        pen = TTPenalties(mesh, args)
        prompt = torch.randint(0, 128000, (32, 128), dtype=torch.int32)
        out = torch.randint(0, 128000, (32, 64), dtype=torch.int32)

        def t(name, fn, n=10):
            fn()
            ttnn.synchronize_device(mesh)
            s = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(mesh)
            print(f"{name}: {(time.perf_counter() - s) / n * 1000:.2f} ms")

        t("reset_prompt_tokens(one slot)", lambda: pen.reset_prompt_tokens(prompt, slots=[3]))
        t("reset_output_tokens(slots=[3])", lambda: pen.reset_output_tokens(slots=[3]))
        t("reset_prompt_tokens(all rows)", lambda: pen.reset_prompt_tokens(prompt))
        t("reset_output_tokens(all rows)", lambda: pen.reset_output_tokens(out))
        t("host histogram [32, V]", lambda: pen._token_counts_host(prompt))
        mask = torch.zeros(32, 129024, dtype=torch.int32)
        mapper = ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, 4))
        t(
            "from_torch int32 tile [32, V] sharded (host)",
            lambda: ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper),
        )
        h = ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        t("copy_host_to_device_tensor [32, V/4] x4", lambda: ttnn.copy_host_to_device_tensor(h, pen.prompt_mask))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

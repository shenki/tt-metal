# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-sampling logits readback variants on the 1x4 mesh (vocab-sharded [1, 1, 32, 32256] bf16 tile tensor per chip, the
decode / prefill logits layout): what the adapter's host-sampling compat path pays per step before the CPU sampler.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.bench_logits_readback
"""

import time

import torch

import ttnn


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    return (time.perf_counter() - t) / n * 1000, out


def main():
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import open_mesh

    mesh = open_mesh(64 << 20)
    try:
        V, B = 129024, 32
        ref = torch.randn(1, 1, B, V).bfloat16()
        lg = ttnn.from_torch(
            ref,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)

        def a_current():  # adapter today: cpu() of the tile tensor, host untilize + concat, float
            h = lg.cpu(blocking=True)
            return ttnn.to_torch(h, mesh_composer=comp)[0, 0, :1, :128000].float()

        def b_per_device():
            h = lg.cpu(blocking=True)
            return torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(h)], -1)[0, 0, :1, :128000].float()

        def c_device_rm(rows):
            def f():
                rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
                sl = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, rows, V // 4]) if rows < B else rm
                h = sl.cpu(blocking=True)
                ttnn.deallocate(rm)
                if sl is not rm:
                    ttnn.deallocate(sl)
                return torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(h)], -1)[0, 0, :1, :128000].float()

            return f

        res = {}
        for name, fn in [
            ("A tile cpu + to_torch(ConcatMeshToTensor)", a_current),
            ("B tile cpu + per-device to_torch + cat", b_per_device),
            ("C device untilize, all 32 rows", c_device_rm(B)),
            ("C device untilize + slice 1 row", c_device_rm(1)),
            ("C device untilize + slice 8 rows", c_device_rm(8)),
        ]:
            ms, out = timeit(fn)
            ok = torch.equal(out, ref[0, 0, :1, :128000].float())
            res[name] = ms
            print(f"{name}: {ms:.2f} ms  exact={ok}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

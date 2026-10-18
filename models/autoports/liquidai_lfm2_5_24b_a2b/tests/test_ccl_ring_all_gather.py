# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the multicast all-gather completion race on the 1x4 Blackhole ring.

The prefill all-reduce of this model (``ttnn.all_reduce`` -> reduce_scatter + ``ttnn.all_gather``) ran the multicast
all-gather factory for S = 64..512 rows; with route alternation its completion credit travelled only the primary routes,
so the op could finish while packets from the chip two hops away were still in flight and the next op read stale tiles
(intermittent prefill-logit nondeterminism, doc/vllm_integration/README.md).  This traces ``all_gather`` immediately
followed by ``add`` and replays it with fresh random inputs each time (stale tiles from an earlier replay then differ),
checking every replay against torch.

    pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_ccl_ring_all_gather.py
"""

import os

import pytest
import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import mesh_device_params


@pytest.mark.parametrize("device_params", [mesh_device_params(64 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("rows", [128, 512])
def test_traced_all_gather_then_add(mesh_device, rows):
    replays = int(os.environ.get("AG_REPLAYS", "1000"))
    width, tp = 512, 4
    dram = ttnn.DRAM_MEMORY_CONFIG

    def shards(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )

    x0 = torch.randn(1, 1, rows, width * tp)
    x_dev = ttnn.from_torch(
        x0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    bias = torch.randn(1, 1, rows, width * tp)
    b_dev = ttnn.from_torch(
        bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def step():
        g = ttnn.all_gather(x_dev, dim=3, cluster_axis=1, topology=ttnn.Topology.Ring, memory_config=dram)
        out = ttnn.add(g, b_dev, memory_config=dram)
        ttnn.deallocate(g)
        return out

    ttnn.deallocate(step())  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    bad = []
    try:
        for r in range(replays):
            x = torch.randn(1, 1, rows, width * tp)
            ttnn.copy_host_to_device_tensor(shards(x), x_dev)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ref = (x.bfloat16() + bias.bfloat16()).float()
            for d, t in enumerate(ttnn.get_device_tensors(out)):
                got = ttnn.to_torch(t).float()
                if not torch.allclose(got, ref, atol=0.07, rtol=0.02):
                    bad.append((r, d, (got - ref).abs().max().item()))
    finally:
        ttnn.release_trace(mesh_device, tid)
    assert not bad, f"{len(bad)} stale all_gather outputs of {replays * tp}: {bad[:8]}"

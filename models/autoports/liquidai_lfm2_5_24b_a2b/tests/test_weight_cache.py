# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-weight cache (``tt/weights.py::WeightCache``) is bitwise transparent: the reduced probe (layers 0, 2, 3 +
embedding / final norm / LM head / RoPE, max_seq_len 131072 as the full model) built without the cache, with a cold cache (writing) and with the warm cache
(reading) holds identical device tensors: every ``ttnn.Tensor`` reachable from the model, on every chip, with the same
shape / dtype / layout / memory config."""

import gc
import hashlib
import time

from loguru import logger

import ttnn

from ..tt.generator import open_mesh
from ..tt.model import Lfm2Model

PROBE = (0, 2, 3)


def fingerprint(model):
    seen, out = set(), {}

    def walk(obj, path, depth):
        if id(obj) in seen or depth > 8:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            if not obj.is_allocated():
                return
            t = ttnn.to_torch(obj, mesh_composer=ttnn.ConcatMeshToTensor(obj.device(), dim=0))
            out[path] = (
                tuple(obj.shape),
                str(obj.dtype),
                str(obj.layout),
                str(obj.memory_config()),
                hashlib.sha256(t.contiguous().view(-1).float().numpy().tobytes()).hexdigest(),
            )
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", depth + 1)
        elif hasattr(obj, "__dict__") and "lfm2_5_24b_a2b" in type(obj).__module__:
            for k, v in vars(obj).items():
                walk(v, f"{path}.{k}", depth + 1)

    walk(model, "model", 0)
    return out


def test_weight_cache_bitwise(tmp_path):
    mesh = open_mesh(64 << 20)
    try:
        prints, times, caches = {}, {}, {}
        for name, root in (("off", False), ("cold", str(tmp_path)), ("warm", str(tmp_path))):
            t0 = time.perf_counter()
            m = Lfm2Model(mesh, layer_indices=PROBE, max_seq_len=131072, weight_cache=root)
            times[name] = time.perf_counter() - t0
            caches[name] = m.weight_cache
            prints[name] = fingerprint(m)
            del m
            gc.collect()
        logger.info(f"weight cache build times {times}, caches {caches}, tensors {len(prints['off'])}")
        assert len(prints["off"]) > 200, f"walker found only {len(prints['off'])} tensors"
        assert caches["cold"]["misses"] > 0 and caches["cold"]["hits"] == 0
        assert caches["warm"]["hits"] == caches["cold"]["misses"] and caches["warm"]["misses"] == 0
        for name in ("cold", "warm"):
            diff = [k for k in prints["off"] if prints["off"][k] != prints[name].get(k)]
            assert set(prints[name]) == set(prints["off"]) and not diff, (name, diff[:10])
    finally:
        ttnn.close_mesh_device(mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for LFM2.5-24B-A2B.

The checkpoint stores experts per expert (``experts.{e}.w1/w3/w2``: gate/up/down, torch Linear
layout ``[out, in]``). ``load_layer_state_dict`` returns one decoder layer's tensors with the
``model.layers.{i}.`` prefix stripped; only the safetensors shards that hold the layer are opened.
"""

import json
import re
from pathlib import Path

import torch


def _weight_map(model_path):
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def load_tensors(model_path, keys):
    from safetensors import safe_open

    weight_map = _weight_map(model_path)
    by_file = {}
    for k in keys:
        by_file.setdefault(weight_map[k], []).append(k)
    out = {}
    for fname, ks in by_file.items():
        with safe_open(str(Path(model_path) / fname), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def load_layer_state_dict(model_path, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in _weight_map(model_path) if k.startswith(prefix)]
    if not keys:
        raise KeyError(f"no tensors for layer {layer_idx}")
    return {k[len(prefix) :]: v for k, v in load_tensors(model_path, keys).items()}


def load_global_state_dict(model_path):
    keys = ["model.embed_tokens.weight", "model.embedding_norm.weight"]
    return {k[len("model.") :]: v for k, v in load_tensors(model_path, keys).items()}


def stacked_expert_weights(layer_sd, num_experts):
    """Per-expert checkpoint tensors -> (gate [E,I,H], up [E,I,H], down [E,H,I])."""
    gate = torch.stack([layer_sd[f"feed_forward.experts.{e}.w1.weight"] for e in range(num_experts)])
    up = torch.stack([layer_sd[f"feed_forward.experts.{e}.w3.weight"] for e in range(num_experts)])
    down = torch.stack([layer_sd[f"feed_forward.experts.{e}.w2.weight"] for e in range(num_experts)])
    return gate, up, down


def to_hf_module_state_dict(layer_sd, num_experts):
    """Checkpoint-layout layer dict -> ``Lfm2MoeDecoderLayer.state_dict()`` layout (fused experts)."""
    out = {}
    for k, v in layer_sd.items():
        if ".experts." not in k:
            out[k] = v
    if "feed_forward.gate.weight" in layer_sd:
        gate, up, down = stacked_expert_weights(layer_sd, num_experts)
        out["feed_forward.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
        out["feed_forward.experts.down_proj"] = down
        out["feed_forward.gate.weight"] = layer_sd["feed_forward.gate.weight"]
    return out


# ------------------------------------------------------------------------------------------------
# device-weight cache (build time only; numerics unchanged)
# ------------------------------------------------------------------------------------------------
# modules whose code decides device weight contents / layouts (the generator does not; model.py builds the embedding,
# final norm and LM head, and model.WEIGHT_LAYOUT_VERSION is in the key as well)
WEIGHT_CODE_FILES = (
    "functional_decoder.py",
    "fused_decoder.py",
    "optimized_decoder.py",
    "multichip_decoder.py",
    "weights.py",
    "model_config.py",
    "model.py",
)


class WeightCache:
    """Routes the device-bound ``ttnn.from_torch`` calls of a model build through ``ttnn.as_tensor(cache_file_name=...)``.

    Active only inside ``with cache:`` (the model constructor); every weight-upload site of the decoder modules goes through
    ``ttnn.from_torch``, so one hook covers them all without touching the decoder code.  File name = scope (e.g. the layer)
    / call sequence in that scope / shape / dtype / layout; the directory key hashes the weight-defining code (``WEIGHT_CODE_FILES``), the HF
    revision and the construction configuration, so any code or policy change starts a fresh cache.  On a hit
    ``as_tensor`` loads the tensorized (tilized / block-float) host tensor and places it with its recorded memory config,
    skipping the host conversion.  Tensors below ``min_numel`` elements, all-zero tensors (mutable state such as the
    ShortConv state, zero constants), host-only tensors, calls with uncommon arguments and everything after ``stop()``
    (decode state / inputs) bypass the cache.  The key also covers the checkpoint files (``checkpoint_fingerprint``)."""

    def __init__(self, root, key, min_numel=1 << 16):
        self.dir = Path(root) / key
        self.min_numel = min_numel
        self.scope_name, self.seq = "globals", 0
        self.hits = self.misses = self.bypassed = 0
        self._orig = None
        self._inside = False

    @staticmethod
    def key_for(**config):
        import hashlib

        h = hashlib.sha256()
        for f in sorted((Path(__file__).parent).glob("*.py")):
            if f.name in WEIGHT_CODE_FILES:
                h.update(f.name.encode())
                h.update(f.read_bytes())
        h.update(repr(sorted(config.items())).encode())
        return h.hexdigest()[:20]

    @staticmethod
    def checkpoint_fingerprint(model_path):
        """Content identity of every safetensors shard and the index, independent of where the checkpoint is mounted:
        in a Hugging Face cache snapshot each file is a symlink to a blob named by its content hash, so the host and a
        container (or another machine) that see the same revision select the same cache key. Files that are not HF
        cache symlinks to hash-named blobs fall back to (name, size, mtime) under the resolved directory."""
        p = Path(model_path).resolve()
        files = sorted(p.glob("*.safetensors")) + [p / "model.safetensors.index.json"]
        targets = [(f.name, f.resolve().name) for f in files if f.exists() and f.is_symlink()]
        if targets and len(targets) == sum(f.exists() for f in files):
            if all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", t) for _, t in targets):
                return tuple(targets)
        return (str(p), tuple((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in files if f.exists()))

    def scope(self, name):
        self.scope_name, self.seq = name, 0

    def stop(self):
        """End of the weight-construction region: later device tensors (state, inputs) bypass the cache."""
        self.scope_name, self.stopped = None, True

    def __enter__(self):
        import ttnn

        self._orig = ttnn.from_torch
        orig, cache = self._orig, self

        def from_torch(tensor, dtype=None, **kw):
            device = kw.get("device")
            extra = set(kw) - {"layout", "device", "memory_config", "mesh_mapper"}
            if (
                cache._inside
                or getattr(cache, "stopped", False)
                or device is None
                or extra
                or not isinstance(tensor, torch.Tensor)
                or tensor.numel() < cache.min_numel
                or not bool(tensor.any())  # zero-initialised state / constants are created fresh, never cached
            ):
                cache.bypassed += 0 if cache._inside else 1
                return orig(tensor, dtype, **kw)
            layout = kw.get("layout", ttnn.ROW_MAJOR_LAYOUT)
            mem = kw.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG
            shape = "x".join(str(s) for s in tensor.shape)
            name = cache.dir / cache.scope_name / f"{cache.seq:04d}_{shape}_{type(kw.get('mesh_mapper')).__name__}"
            cache.seq += 1
            path = Path(f"{name}_dtype_{dtype.name if dtype is not None else 'None'}_layout_{layout.name}.tensorbin")
            if path.exists():
                cache.hits += 1
            else:
                cache.misses += 1
            cache._inside = True
            try:
                return ttnn.as_tensor(
                    tensor,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    memory_config=mem,
                    mesh_mapper=kw.get("mesh_mapper"),
                    cache_file_name=str(name),
                )
            finally:
                cache._inside = False

        ttnn.from_torch = from_torch
        return self

    def __exit__(self, *exc):
        import ttnn

        ttnn.from_torch = self._orig
        return False

    def summary(self):
        return {"dir": str(self.dir), "hits": self.hits, "misses": self.misses, "bypassed": self.bypassed}

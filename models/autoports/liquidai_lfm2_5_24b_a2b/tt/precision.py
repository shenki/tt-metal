# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision configuration of the full model (stage 8, datatype sweep).

A ``PrecisionConfig`` is the complete dtype / compute-fidelity choice the model is built with: the decoder ``Policy``
(weight dtypes per group, fidelities, KV-cache dtype), the ``MeshPlan`` (incl. the decode all-reduce payload dtype), per-
layer exceptions and the LM head (dtype, DS config, fidelity).  JSON form = overrides relative to the stage-7 baseline
(``MC_POLICY`` / ``DEFAULT_MESH_PLAN`` / bfp8-HiFi2 head), so a file stays valid when unrelated policy fields gain defaults.

Resolution used by ``Lfm2Model`` when the caller passes no explicit precision (``resolve``):
``LFM2_PRECISION_CONFIG`` (a JSON path, or ``baseline`` = the stage-7 policy) -> ``SELECTED_PATH`` (the datatype-sweep
winner, ``doc/datatype_sweep/selected_precision_config.json``) if it exists -> baseline.
"""

import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .multichip_decoder import DEFAULT_MESH_PLAN, MC_POLICY, MeshPlan
from .optimized_decoder import Policy

SELECTED_PATH = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"
BASELINE_HEAD = {"dtype": "bfp8", "cfg": (64, 1, 3), "fidelity": "hifi2"}


def _tuplify(v):
    return tuple(_tuplify(x) for x in v) if isinstance(v, list) else v


def _overrides(cls, kv):
    names = {f.name for f in fields(cls)}
    bad = set(kv) - names
    if bad:
        raise KeyError(f"unknown {cls.__name__} fields {sorted(bad)}")
    return {k: _tuplify(v) for k, v in kv.items()}


@dataclass
class PrecisionConfig:
    config_id: str = "baseline"
    policy_overrides: dict = field(default_factory=dict)
    plan_overrides: dict = field(default_factory=dict)
    layer_exceptions: dict = field(default_factory=dict)  # {layer index: {Policy field: value}}
    head: dict = field(default_factory=lambda: dict(BASELINE_HEAD))
    source: str = "baseline"
    extra: dict = field(default_factory=dict)  # descriptive fields of the JSON (weight_groups, accuracy, ...)

    def __post_init__(self):
        self.policy = replace(MC_POLICY, **_overrides(Policy, self.policy_overrides))
        self.plan = replace(DEFAULT_MESH_PLAN, **_overrides(MeshPlan, self.plan_overrides))
        self.layer_exceptions = {int(k): dict(v) for k, v in self.layer_exceptions.items()}
        for ov in self.layer_exceptions.values():
            _overrides(Policy, ov)
        h = dict(BASELINE_HEAD)
        h.update(self.head or {})
        h["cfg"] = tuple(h["cfg"])
        self.head = h

    def policy_for_layer(self, i):
        ov = self.layer_exceptions.get(int(i))
        return replace(self.policy, **_overrides(Policy, ov)) if ov else self.policy

    def key(self):
        """Everything that decides the model build (weight-cache key input)."""
        return {
            "policy": repr(self.policy),
            "plan": repr(self.plan),
            "layer_exceptions": {k: sorted(v.items()) for k, v in sorted(self.layer_exceptions.items())},
            "head": (self.head["dtype"], self.head["cfg"], self.head["fidelity"]),
        }

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        known = {"config_id", "policy_overrides", "plan_overrides", "layer_exceptions", "head"}
        return cls(
            config_id=d.get("config_id", Path(path).stem),
            policy_overrides=d.get("policy_overrides", {}),
            plan_overrides=d.get("plan_overrides", {}),
            layer_exceptions=d.get("layer_exceptions", {}),
            head=d.get("head", {}),
            source=str(Path(path).resolve()),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def to_json(self, path=None, **extra):
        d = {
            "config_id": self.config_id,
            "policy_overrides": self.policy_overrides,
            "plan_overrides": self.plan_overrides,
            "layer_exceptions": {str(k): v for k, v in sorted(self.layer_exceptions.items())},
            "head": {"dtype": self.head["dtype"], "cfg": list(self.head["cfg"]), "fidelity": self.head["fidelity"]},
            "policy": self.policy.summary(),
            "plan": self.plan.summary(),
        }
        d.update(self.extra)
        d.update(extra)
        text = json.dumps(d, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o)) + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text


def baseline():
    return PrecisionConfig()


def resolve(precision=None):
    """``precision``: a PrecisionConfig, a JSON path, "baseline", or None (env / selected file / baseline)."""
    if isinstance(precision, PrecisionConfig):
        return precision
    if precision is None:
        precision = os.environ.get("LFM2_PRECISION_CONFIG") or (
            str(SELECTED_PATH) if SELECTED_PATH.exists() else "baseline"
        )
    if precision == "baseline":
        return baseline()
    return PrecisionConfig.from_json(precision)

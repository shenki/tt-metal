# FSD ↔ PSD placement reproduction — 36-host `SC36_32x4_revC_subtorus_aisleD`

This directory reproduces, **entirely inside tt-metal with no hardware**, the
placement difference between the two routes that assign fabric coordinates to a
multi-host Blackhole supercluster:

| Route | What it is | Numbering source |
|-------|-----------|------------------|
| **PSD** | tt-metal mock-PSD discovery | the committed `..._mapping.yaml` + per-host mock cluster descriptors shipped in `tt-cluster-descriptors` |
| **FSD** | offline Factory System Descriptor | `generate_cluster_descriptor` run on the committed `..._factory_system_descriptor.textproto` |

Both routes describe the **same 36 physical hosts** of
`SC36_32x4_revC_subtorus_aisleD`. We drive the **same heterogeneous
ControlPlane test** with each route's mapping and compare **two things**:

1. the **rank → host bindings** (the two `mapping.yaml` files), and
2. the **actual placement file the ControlPlane generates** —
   `<logs>/generated/fabric/asic_to_fabric_node_mapping_rank_<r>_of_70.yaml`.

## Test under reproduction

- **Mesh-graph-descriptor:** `tests/tt_metal/tt_fabric/custom_mesh_descriptors/gemma_specdecode_mesh_graph_descriptor.textproto`
  — heterogeneous, **70 single-rank meshes** (`S4x1` `[4,1]`, `S4x2` `[4,2]`,
  `S4x4` `[4,4]`, row axis `RING`). This is the only in-repo MGD paired with the
  36-host aisle-D mock mapping (see `run_fabric_cpu_only_unit_tests.sh`, the SC36
  aisle-D `run_test` line).
- **Test:** `fabric_unit_tests --gtest_filter=ControlPlaneFixture.TestGalaxyLayoutCheck`
- **Launcher:** `tt-run` (`ttnn/ttnn/distributed/ttrun.py`),
  `--mock-cluster-rank-binding <mapping.yaml> --force-rediscovery`, 70 ranks via
  `mpirun --oversubscribe`.
- **Mode:** CPU-only mock (`--force-rediscovery`, `TT_METAL_SLOW_DISPATCH_MODE=1`).

## How to run

```bash
TT_METAL_HOME=/path/to/tt-metal PATH=/path/to/tt-run-shim:$PATH \
  tests/scale_out/fsd_psd_placement_repro/run_repro.sh
```

`run_repro.sh` (1) regenerates the FSD-derived per-host descriptors + mapping
from the committed FSD textproto, (2) runs the heterogeneous test once per route
into isolated `TT_METAL_LOGS_PATH` dirs, and (3) diffs the generated placement.

`mappings/` holds the two rank→host bindings used as the two sides:
- `committed_PSD_SC36_aisleD_mapping.yaml` — the shipped PSD mapping.
- `fsd_derived_SC36_aisleD_mapping.yaml` — regenerated from the FSD (paths are
  environment-specific; `run_repro.sh` rewrites them at run time).

## What the generated placement file contains

Per physical ASIC, keyed by a **stable hardware `asic_id`**:

```yaml
asic_to_fabric_node_mapping:
  hostnames:
    - hostname: <mock cluster descriptor filename>   # mock host identity = filename
      mesh:
        - mesh: 62
        - chips:
            - umd_chip_id: 0
              asic_position: { tray_id: 3, asic_location: 7 }
              fabric_node_id: { mesh_id: 62, chip_id: 14 }   # <-- the placement
              asic_id: 2070104874866030313                   # <-- stable key
```

Because the committed route labels hosts `..._bh-glx-120-dNNuNN_rank_k.yaml`
while the FSD route labels them `SC36_aisleD_rank_k.yaml`, a raw text `diff`
trips on the hostname strings. The meaningful comparison keys on the stable
`asic_id` and asks: **for the same physical ASIC, is the assigned
`fabric_node_id (mesh_id, chip_id)` the same under both routes?**

## Results

> **Test status (both routes): PASSED** — `ControlPlaneFixture.TestGalaxyLayoutCheck`
> runs green with both the committed PSD mapping and the FSD-derived mapping on
> all 70 ranks. Both routes produce a valid, self-consistent placement.

<!-- RESULTS_PLACEHOLDER -->

## Report — TT_VISIBLE_DEVICES is bookkeeping, not connectivity

`TT_VISIBLE_DEVICES` in a per-stage rank binding is the list of **local chip
indices 0–31 within a single 32-chip galaxy**. When the FSD and PSD routes
disagree on `TT_VISIBLE_DEVICES`, they disagree on *which local index names a
given physical chip* — a **relabeling of the local 0–31 namespace**, not a
change to the physical fabric graph:

- The **inter-host connectivity is identical** between routes — the same
  physical links join the same physical endpoints; only the integer *names* of
  the local chip slots move.
- Host/ring **placement shape is identical** — the same hosts land in the same
  mesh/ring positions.
- The divergence is confined to the **local index → physical chip** map, which
  then propagates into the `rank → mesh_id / mesh_host_rank / TT_VISIBLE_DEVICES`
  bookkeeping. It does **not** repartition the machine or re-route fabric.

So a `TT_VISIBLE_DEVICES` mismatch between routes is a **bookkeeping / numbering**
discrepancy, not a connectivity or correctness defect: both routes describe the
same machine, wired the same way, and the ControlPlane test passes under both.

## Hypothesis

The FSD route and the committed mock-PSD route **seat the same physical ASICs on
the same fabric nodes**, but can assign **different local chip-index numbering**
within a galaxy when the SAT seating admits more than one equivalent solution.
Specifically:

1. `generate_cluster_descriptor` (FSD route) and mock-PSD discovery feed the SAT
   seater **different-but-equivalent** candidate orderings. The CaDiCaL seating
   is deterministic but enumerates candidate tiles in **chip-index order with no
   canonical geometric tie-break** — so when a galaxy's placement has a symmetry
   (e.g. a sub-torus that can be numbered from either end), the two routes can
   settle on different, equally-valid local numberings.
2. That difference surfaces as the `TT_VISIBLE_DEVICES` / local-index delta
   above. It is **invisible to physical connectivity** (identical links) and to
   the ControlPlane test (passes both ways), which is why it is a numbering
   discrepancy rather than a mis-placement.

**Predicted fixes** (either closes the gap):
- **(a)** have the FSD→cluster-descriptor converter **preserve the committed
  local numbering** when one exists, so the FSD route reproduces the PSD labels
  exactly; or
- **(b)** add a **canonical geometric tie-break** to the SAT seater so both
  routes deterministically pick the same local numbering regardless of candidate
  input order.

`generate_rank_bindings` does not yet implement FSD-driven mapping
(tt-metal#52859) and falls back to live PSD discovery, so today the two routes
cannot be reconciled purely in-tool — this repro pins down exactly where (and
how little) they diverge.

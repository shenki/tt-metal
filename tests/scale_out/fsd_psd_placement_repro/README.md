# FSD ↔ PSD placement reproduction — 36-host `SC36_32x4_revC_subtorus_aisleD`

This directory reproduces, **entirely inside tt-metal with no hardware**, a real
placement divergence between the two routes that assign fabric coordinates to a
multi-host Blackhole supercluster:

| Route | What it is | Numbering source |
|-------|-----------|------------------|
| **PSD** | tt-metal mock-PSD discovery | the committed `..._mapping.yaml` + per-host mock cluster descriptors shipped in `tt-cluster-descriptors` |
| **FSD** | offline Factory System Descriptor | `generate_cluster_descriptor` run on the committed `..._factory_system_descriptor.textproto` |

Both routes describe the **same 36 physical hosts** of
`SC36_32x4_revC_subtorus_aisleD`. We drive the **same heterogeneous ControlPlane
test** with each route's mapping and compare **two things**:

1. the **rank → host bindings** (the two `mapping.yaml` files), and
2. the **actual placement file the ControlPlane generates** —
   `<logs>/generated/fabric/asic_to_fabric_node_mapping_rank_<r>_of_70.yaml`.

## TL;DR result

| Route | `ControlPlaneFixture.TestGalaxyLayoutCheck` |
|-------|---------------------------------------------|
| committed PSD mapping | **PASSED** — 0 failures |
| FSD-derived mapping   | **FAILED** — 490 assertion failures across the 4×1 meshes (1, 2, 57, 58) |

Same hosts, same MGD, same test — **only the descriptor route differs**, and the
FSD route produces a physically-invalid galaxy layout. This is a genuine
placement bug reproduction, not a cosmetic numbering difference.

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

## The exact placement difference

`TestGalaxyLayoutCheck` validates that every 4×1 RING rank group seats on a
physically-valid galaxy tray geometry: the 4 chips must sit on **one halftray**
(`asic_locations {1,2,5,6}` or `{3,4,7,8}`) or span a **valid tray pair**
(`{1,2}`/`{3,4}` for rev-C), with each tray contributing **exactly 2 chips** in a
single column pair (`{1,5}`,`{2,6}`,`{3,7}`,`{4,8}`). The tray/asic-location for
each ASIC is read from the mock PSD (`psd.get_tray_id(asic_id)` /
`psd.get_asic_location(asic_id)`).

Pulling **mesh 57's** 4×1 group straight out of the generated
`asic_to_fabric_node_mapping` file for each route:

| | committed PSD | FSD-derived |
|---|---|---|
| trays occupied | `{3, 4}` ✅ valid pair | `{1, 2, 3, 4}` ❌ all four trays |
| asic_locations | `{4, 8}` ✅ valid column | `{5}` ❌ single location |
| chips per tray | 2 + 2 ✅ | 1 + 1 + 1 + 1 ❌ |

```
mesh 57, committed/PSD           mesh 57, FSD-derived
  chip 0  tray 3  loc 4            chip 0  tray 1  loc 5
  chip 1  tray 3  loc 8            chip 1  tray 2  loc 5
  chip 2  tray 4  loc 8            chip 2  tray 4  loc 5
  chip 3  tray 4  loc 4            chip 3  tray 3  loc 5
```

The PSD route folds the 4×1 ring onto the `{3,4}` tray pair (2 chips each,
column `{4,8}`) — a physical galaxy layout. The FSD route smears the same 4×1
ring **one chip per tray down a single column (loc 5)** — geometrically a 4-cycle
in the descriptor's abstract space, but not a shape the physical galaxy admits,
so the check rejects it:

```
mesh 57 host_rank 0 4x1 rank group tray pair must be {1,3} or {2,4}
mesh 57 host_rank 0 4x1 rank group tray 1 must contain exactly 2 chips  (got 1)
mesh 57 host_rank 0 4x1 rank group must use asic locations {1,5}, {2,6}, {3,7}, or {4,8}, got a different set
```

Note the error cites the **rev-A/B** tray pairs `{1,3}/{2,4}`, not rev-C
`{1,2}/{3,4}` — because the synthetic descriptor also loses the board identity
that makes `psd.is_bh_galaxy_rev_c()` return true, so the check falls back to the
rev-A/B geometry. That is a second, independent symptom of the same lost physical
metadata.

Failure tally over the whole FSD run: tray-pair-invalid ×210, tray-not-2-chips
×280, halftray-invalid ×70, column-invalid ×70.

## Why the descriptors differ (root cause)

The committed per-host descriptor is a **physically-discovered** PSD; the
FSD-derived one, produced by `generate_cluster_descriptor`, is **idealized**.
For the same host (`d01u02` / rank 0):

| field | committed PSD descriptor | FSD-derived descriptor |
|-------|--------------------------|------------------------|
| `cluster_id` | `bh-glx-120-d01u02` | *(absent)* |
| `chip_unique_ids` | real 64-bit ASIC ids (`17824262955932436622`, …) | sequential placeholders (`2, 0, 5, 7, …`) |
| `boards` | **1** `ubb_blackhole` board, 32 chips in true physical order (`4, 14, 25, 21, …`) | **4** boards of 8 chips each, sequential (`0–7`, `8–15`, `16–23`, `24–31`) |
| `ethernet_connections` | 280 links (physical) | 280 links, **relabeled** to the sequential numbering |
| `asic_locations` | `chip→1..8` | identical |

The 4-sequential-boards split and the placeholder ids mean the ControlPlane
derives a **different chip→tray partition** than the physical hardware has — which
is exactly what the per-mesh table above shows.

Crucially, **the FSD textproto itself is not missing this information**: it
carries 11,664 `tray_id` entries, 11,520 `asic_location` entries, per-ASIC
`host_id`, `board_types` (tray 1–4 `UBB_BLACKHOLE` per host), and `board_locations`
(rack / shelf_u / aisle / hall). The physical placement is fully specified in the
FSD — the converter just doesn't carry it through to the cluster descriptor.

## Report — TT_VISIBLE_DEVICES: when a relabel is benign vs. when it isn't

`TT_VISIBLE_DEVICES` in a per-stage rank binding is the list of **local chip
indices 0–31 within a 32-chip galaxy**. A difference there is, by itself, a
**relabeling of the local 0–31 namespace** — *which* local index names a given
physical chip — and in many prior cases that relabel is benign: inter-host
connectivity is preserved and host/ring placement shape is identical, so it reads
as bookkeeping.

This reproduction is the case where **the relabel is not benign**. Because the
FSD-derived descriptor also re-partitions the galaxy's boards/trays (4 synthetic
trays vs the real physical tray wiring), the local renumbering carries through
into the **physical tray assignment** the ControlPlane computes. The galaxy
layout check — which enforces the physical tray-pair geometry — then fails. So
the lesson is: a `TT_VISIBLE_DEVICES`/local-index delta is only safe when the
**board/tray composition is preserved**; when the route also changes board/tray
grouping, the same delta becomes a real mis-placement.

## Hypothesis

`generate_cluster_descriptor` (the FSD route) does **not** reconstruct the real
physical board composition, chip ordering, and ASIC identity of a discovered
galaxy. It emits an idealized descriptor — 4 sequential 8-chip boards, placeholder
`chip_unique_ids`, connectivity relabeled to match — even though the source FSD
carries the full per-ASIC `tray_id` / `asic_location` / `host_id` / board data.

Consequently the chip→tray partition derived from the FSD route differs from the
physically-discovered PSD, and 4×1 rank groups seat across trays in a way that
violates the galaxy tray-pair invariant `TestGalaxyLayoutCheck` enforces — while
the committed PSD mapping satisfies it.

**Predicted fix:** the FSD→cluster-descriptor conversion must **preserve (or
faithfully reconstruct) the physical board grouping, chip ordering, and ASIC
unique ids from the FSD's `tray_id` / `asic_location` / board fields** — rather
than synthesizing sequential boards — so the FSD route reproduces the committed
physical placement and passes the same layout check. This is consistent with
`generate_rank_bindings` not yet implementing FSD-driven mapping
(tt-metal#52859); today the FSD route cannot reproduce discovered placement
in-tool, and this repro pins down precisely where and how it diverges.

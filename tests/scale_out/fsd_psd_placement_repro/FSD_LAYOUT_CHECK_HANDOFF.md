# Handoff — why the FSD route fails `TestGalaxyLayoutCheck`, and how to fix it

**Audience:** the agent/engineer picking up the fix.
**Scope:** `SC36_32x4_revC_subtorus_aisleD` (36-host Blackhole galaxy, rev-C), driven
by the heterogeneous `gemma_specdecode` MGD (70 single-rank meshes). Everything
below is CPU-only / mock — no hardware needed to reproduce.

**One-line summary:** `generate_cluster_descriptor` must *synthesize* three pieces
of physical identity that the FSD does not carry — silicon `chip_unique_ids`, the
board **revision**, and the real **board grouping** — and the way it synthesizes
them makes the ControlPlane seat 4×1 meshes on a physically-invalid galaxy
geometry. The committed (physically-discovered) descriptors pass the same check.

---

## 0. Reproduce (start here)

```bash
TT_METAL_HOME=$PWD PATH=/workspace/group/bin:$PATH \
  tests/scale_out/fsd_psd_placement_repro/run_repro.sh
```

Result today: committed-PSD mapping **PASSES** (0 failures); FSD-derived mapping
**FAILS** — 490 assertions across the four 4×1 meshes (1, 2, 57, 58). Saved
evidence is in `evidence/` (both generated `asic_to_fabric_node_mapping` files,
the deduped failure lines, and the PASS/FAIL summary). The FSD-derived descriptor
is deterministic and reproducible: a fresh `generate_cluster_descriptor` run
byte-matches the one under test, so this is the tool's real output, not a
generation artifact.

---

## 1. The observed misalignment (confirmed from the generated placement)

Pulling **mesh 57**'s 4×1 group straight out of each route's generated
`asic_to_fabric_node_mapping` (`evidence/`):

| | committed / PSD (passes) | FSD-derived (fails) |
|---|---|---|
| trays occupied | `{3, 4}` — valid rev-C pair | `{1, 2, 3, 4}` — all four trays |
| asic_locations | `{4, 8}` — a valid column pair | `{5}` — one single column |
| chips per tray | 2 + 2 | 1 + 1 + 1 + 1 |

```
mesh 57 committed/PSD            mesh 57 FSD-derived
  umd 0  tray 3  loc 8             umd 0  tray 4  loc 5
  umd 1  tray 3  loc 4             umd 1  tray 3  loc 5
  umd 2  tray 4  loc 8             umd 2  tray 2  loc 5
  umd 3  tray 4  loc 4             umd 3  tray 1  loc 5
```

The physical galaxy folds a 4×1 ring onto **one half-tray / one tray pair, two
chips per tray, in a single column pair**. The FSD route smears the same ring
**one chip per tray straight down a single asic column** — a clean 4-cycle in the
descriptor's abstract graph, but not a shape the physical galaxy admits. The
check rejects it; failure classes: tray-pair-invalid ×210, tray-not-2-chips ×280,
half-tray-invalid ×70, column-invalid ×70.

This is **not** a benign `TT_VISIBLE_DEVICES`/local-index relabel. The per-chip
`(tray, asic_location)` values are preserved (see §3); what diverges is **which
physical ASICs each mesh lands on**.

---

## 2. Root cause at the schema level (why the tool is forced to guess)

`tools/scaleout/factory_system_descriptor/schemas/factory_system_descriptor.proto`
carries only:

- `Host`: hostname, hall, aisle, rack, shelf_u, motherboard, instance_path
- `BoardLocationType`: `host_id`, `tray_id`, `board_type` *(string, e.g. `"UBB_BLACKHOLE"`)*
- `EndPoint`: `host_id`, `tray_id`, `asic_location`, `chan_id`

It does **not** carry: silicon `chip_unique_id`, **board revision** (rev-A/B vs
rev-C), or real `board_id` / UBB serial. A physically-discovered PSD has all
three. So `generate_cluster_descriptor_from_fsd`
(`tools/scaleout/factory_system_descriptor/utils.cpp:676`) synthesizes them — and
each synthesis is where the ControlPlane's physical-geometry assumptions break.

---

## 3. The three code-level causes (confirmed from source)

### Cause A — board **revision** bits are 0 ⇒ rev-C detection fails *(airtight)*

- Converter: `encoded_board_id = (upi << 36) | unique_board_id`, with
  `unique_board_id` a small counter (`utils.cpp:974-975`).
- Detector: `is_bh_galaxy_rev_c()` reads `revision_bits = (board_id >> 32) & 0xF`
  and returns true only if `>= 3` (`tt_metal/fabric/physical_system_discovery.cpp:698-705`).
- Arithmetic: synthetic `board_id` → `(>>32)&0xF == 0` → **rev-C = false**.
  Committed real `board_id 4892798423057` → `(>>32)&0xF == 3` → **rev-C = true**.

With rev-C false, `TestGalaxyLayoutCheck` validates against the **rev-A/B** tray
pairs `{1,3}/{2,4}` instead of rev-C `{1,2}/{3,4}` — exactly the
`"tray pair must be {1,3} or {2,4}"` lines in `evidence/…_failures.txt`. **The FSD
has no revision field, so this cannot be fixed by passing existing data through —
it needs a new input** (schema field or CLI flag). The galaxy *is* rev-C (see the
directory name and the committed board_id).

### Cause B — one board **per tray** instead of one UBB board *(confirmed)*

`utils.cpp:953-1004` emits a separate board per FSD `board_location` (i.e. per
`tray_id`): 4 synthetic 8-chip boards with sequential ids (`4879082848256…+3`),
versus the physically-discovered **1 `UBB_BLACKHOLE` board of 32 chips**
(`board_id 4892798423057`). `is_bh_galaxy_rev_c` and any `get_board_id_for_chip`
co-location logic read chip 0's board; the per-tray split changes board identity
and is what forces the revision bits in Cause A to come from a counter.

### Cause C — synthetic chip ordering + placeholder `chip_unique_ids` *(confirmed)*

- Local chip ids are assigned by iterating a `std::map` keyed on
  `(host, tray_id, asic_location)` — i.e. **sorted** order (`utils.cpp:810-817`),
  not the physically-discovered fused-chip order the real PSD records.
- `chip_unique_ids` are the synthetic sequential counter
  (`utils.cpp:744-753`, emitted at `:827-829`), not real 64-bit silicon ids.

The ControlPlane keys fabric nodes on `asic_id` and walks connectivity; a
synthetic order + synthetic ids means the mesh→ASIC assignment is driven by the
tool's numbering, not the hardware's.

### What is NOT the cause (ruled out)

Per-chip `(tray, asic_location)` round-trips **correctly**: the converter encodes
`bus_id = tray_bus_id | (asic_location & 0x0F)` (`utils.cpp:769-770`), and
`get_ubb_id` decodes `tray = index(bus_id & 0xF0)`, `asic = bus_id & 0x0F`
(`tt_metal/fabric/control_plane.cpp:144-154`). So individual tray/asic values are
faithful — the divergence is in **mesh-to-ASIC assignment**, not per-chip
attributes.

---

## 4. Hypothesis for the seating divergence (the part to confirm while fixing)

Both descriptors should describe the *same* physical galaxy, so a topologically
faithful FSD-derived descriptor would seat mesh 57 on the same ASICs as discovery.
It doesn't — the FSD route rings chips at a single `asic_location` (5) across
trays 1-4, discovery rings chips within a half-tray at `{4,8}`. Since per-chip
`(tray, asic_location)` is preserved, the ring **edges** must bind to different
physical chips. The likely driver is one or both of:

1. **Synthetic chip ordering / ids (Cause C)** steering the topology mapper's
   ring walk and tie-breaks onto a different physical set, and/or
2. an **`asic_location` convention mismatch**: the converter's asic-location *set*
   comes from the board's canonical layout (`board.get_asic_locations()`,
   `utils.cpp:741,748`) while the connectivity endpoints carry the FSD's own
   `asic_location` numbering — if those conventions differ from the physical UBB
   `asic_id` convention that discovery uses, the golden ring maps to the wrong
   column. (Also check the silent `operator[]` default at `utils.cpp:784-785`: an
   endpoint key absent from `asic_to_unique_chip_id` inserts chip id 0 rather than
   erroring — verify every FSD endpoint's `asic_location` is in the canonical set.)

**To confirm:** instrument `TopologyMapper`'s mesh→ASIC assignment (or dump it)
for both descriptors and diff; that isolates whether the flip is ordering-driven
(Cause C) or convention-driven (hypothesis 2).

---

## 5. Suggested fix plan (prioritized)

1. **Instrument & isolate** — run the repro, dump the topology-mapper assignment
   for both routes, diff against `evidence/`. Decide whether the seating flip is
   driven by ordering (Cause C) or asic_location convention (§4.2).
2. **Rev-C (Cause A)** — add board revision to the FSD (`BoardLocationType` field,
   or a `--board-revision` / `--rev-c` CLI flag), and encode revision bits into
   the synthetic `board_id` so `(board_id>>32)&0xF >= 3` for rev-C. Smallest change
   that clears the `{1,3}/{2,4}` failure class. Confirm `board_upi_map` + `upi<<36`
   leaves bits [35:32] free for the revision.
3. **Board grouping (Cause B)** — emit **one `UBB_BLACKHOLE` board of 32 chips**
   per galaxy (matching discovery), not one board per tray.
4. **Ordering / ids (Cause C)** — reproduce the physically-discovered chip order,
   or make the mapper order-invariant, so the FSD route seats meshes identically.
   Offline, real silicon ids are unknowable; the ordering must still reproduce the
   discovered seating.
5. **Verify** — FSD route passes `ControlPlaneFixture.TestGalaxyLayoutCheck` with
   0 failures **and** its generated `asic_to_fabric_node_mapping` matches the
   committed one (the `run_repro.sh` step-4 diff goes empty).

Consistent with `generate_rank_bindings` not yet implementing FSD-driven mapping
(tt-metal#52859): today the FSD route cannot reproduce discovered placement
in-tool, and this repro pins down exactly where (schema gaps → synthesized board
revision / grouping / chip order) and how (mesh seating off a valid rev-C tray
geometry) it diverges.

---

## 6. File / line index

| What | Where |
|---|---|
| FSD→descriptor converter | `tools/scaleout/factory_system_descriptor/utils.cpp:676-1047` |
| synthetic `board_id` (Cause A/B) | `utils.cpp:953-1004` (id at `:974`) |
| local chip-id order + unique ids (Cause C) | `utils.cpp:744-753, 810-817, 827-829` |
| bus_id encode (per-chip tray/asic) | `utils.cpp:758-772` |
| silent `operator[]` on endpoint key | `utils.cpp:780-785` |
| FSD proto schema (missing fields) | `tools/scaleout/factory_system_descriptor/schemas/factory_system_descriptor.proto` |
| rev-C detector | `tt_metal/fabric/physical_system_discovery.cpp:698-705` |
| UBB tray/asic decode | `tt_metal/fabric/control_plane.cpp:144-154` |
| galaxy geometry check | `tests/tt_metal/tt_fabric/common/utils.cpp` (`TestGalaxyLayoutCheck`) |
| generated placement (evidence) | `tests/scale_out/fsd_psd_placement_repro/evidence/` |

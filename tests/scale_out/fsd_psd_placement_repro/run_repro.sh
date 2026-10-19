#!/usr/bin/env bash
# =============================================================================
# FSD <-> PSD placement reproduction for the 36-host SC36_32x4_revC_subtorus_aisleD
# supercluster, using the heterogeneous gemma_specdecode mesh-graph-descriptor.
#
# Goal: run the SAME heterogeneous ControlPlane test twice against the SAME 36
# physical hosts, differing ONLY in which rank->host mock-cluster mapping is fed:
#
#   (PSD side) committed mapping  -> SC36_32x4_revC_subtorus_aisleD_mapping.yaml
#                                    (the mock-PSD discovery numbering shipped in
#                                     tt-cluster-descriptors)
#   (FSD side) FSD-derived mapping -> regenerated here from the committed
#                                    factory_system_descriptor.textproto via
#                                    generate_cluster_descriptor
#
# Then diff BOTH:
#   1. the rank->host bindings (the two mapping.yaml files), and
#   2. the ACTUAL generated placement file the ControlPlane emits:
#        <logs>/generated/fabric/asic_to_fabric_node_mapping_rank_*_of_70.yaml
#
# Run it as (CPU-only mock, no hardware required):
#   TT_METAL_HOME=/path/to/tt-metal PATH=/path/to/tt-run/shim:$PATH \
#     tests/scale_out/fsd_psd_placement_repro/run_repro.sh
# =============================================================================
set -uo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export TT_METAL_HOME
cd "$TT_METAL_HOME"

SUBMOD="tt_metal/third_party/tt-cluster-descriptors"
AISLED="$SUBMOD/superclusters/blackhole/SC36_32x4_revC_subtorus_aisleD"
COMMITTED_MAP="$AISLED/SC36_32x4_revC_subtorus_aisleD_mapping.yaml"
FSD_TEXTPROTO="$AISLED/SC36_32x4_revC_subtorus_aisleD_factory_system_descriptor.textproto"
MGD="tests/tt_metal/tt_fabric/custom_mesh_descriptors/gemma_specdecode_mesh_graph_descriptor.textproto"
GTEST_FILTER="ControlPlaneFixture.TestGalaxyLayoutCheck"
FUT="./build/test/tt_metal/tt_fabric/fabric_unit_tests"

GEN_CLUSTER_DESC="${GEN_CLUSTER_DESC:-./build/tools/scaleout/generate_cluster_descriptor}"

OUT="${OUT:-/tmp/fsd_psd_placement_repro}"
FSD_DIR="$OUT/fsd_derived"
RUN_PSD="$OUT/run_committed_psd"
RUN_FSD="$OUT/run_fsd_derived"
rm -rf "$OUT"; mkdir -p "$FSD_DIR" "$RUN_PSD" "$RUN_FSD"

echo "### [1/4] Regenerate FSD-derived per-host descriptors + mapping from the committed FSD"
"$GEN_CLUSTER_DESC" -f "$FSD_TEXTPROTO" -o "$FSD_DIR" -b SC36_aisleD
# Build a rank->descriptor mapping (same schema the committed file uses).
FSD_MAP="$FSD_DIR/SC36_aisleD_mapping.yaml"
if [[ ! -f "$FSD_MAP" ]]; then
  echo "rank_to_cluster_mock_cluster_desc:" > "$FSD_MAP"
  i=0
  for f in $(ls "$FSD_DIR"/SC36_aisleD_rank_*.yaml | sort -V); do
    echo "  $i: $f" >> "$FSD_MAP"; i=$((i+1))
  done
fi

run_case() {
  local label="$1" mapping="$2" rundir="$3"
  echo "### running $label  (mapping=$mapping)"
  TT_METAL_LOGS_PATH="$rundir" \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=600 \
  tt-run \
    --mesh-graph-descriptor "$MGD" \
    --mock-cluster-rank-binding "$mapping" \
    --mpi-args "--allow-run-as-root --oversubscribe" \
    --force-rediscovery \
    "$FUT" --gtest_filter="$GTEST_FILTER" > "$rundir/stdout.log" 2>&1
  local rc=$?
  grep -E '\[  PASSED  \]|\[  FAILED  \]' "$rundir/stdout.log" | tail -1
  return $rc
}

echo "### [2/4] Run heterogeneous test with committed (PSD) mapping"
run_case "PSD/committed" "$COMMITTED_MAP" "$RUN_PSD"
echo "### [3/4] Run heterogeneous test with FSD-derived mapping"
run_case "FSD/derived"   "$FSD_MAP"       "$RUN_FSD"

echo "### [4/4] Compare the generated placement files"
# Every rank writes the same global view; compare one representative file.
PSD_FILE=$(find "$RUN_PSD" -name 'asic_to_fabric_node_mapping_rank_1_of_*.yaml' | head -1)
FSD_FILE=$(find "$RUN_FSD" -name 'asic_to_fabric_node_mapping_rank_1_of_*.yaml' | head -1)
echo "PSD placement: $PSD_FILE"
echo "FSD placement: $FSD_FILE"
if diff -q "$PSD_FILE" "$FSD_FILE" >/dev/null; then
  echo "RESULT: generated placement files are IDENTICAL"
else
  echo "RESULT: generated placement files DIFFER"
  diff "$PSD_FILE" "$FSD_FILE" | head -60
fi

#!/bin/zsh
# Isolated ANE specialize for one tower. Parent survives SIGABRT.
set -u
TOWER="${1:?tower}"
export ANEMLL_EMBEDDINGS_ARTIFACTS="${ANEMLL_EMBEDDINGS_ARTIFACTS:-/Volumes/Models/anemll-embeddings/artifacts}"
export ANEMLL_COREAI_PYTHON="${ANEMLL_COREAI_PYTHON:-/Users/anemll/anemll-forge/coreai/.venv/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/coreai-ane-${TOWER}.log"
: > "$LOG"
echo "=== ANE specialize ${TOWER} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"
/Volumes/Models/anemll-embeddings/.venv/bin/python "$ROOT/scripts/smoke_coreai_towers.py" \
  --compute ane --tower "$TOWER" >>"$LOG" 2>&1
rc=$?
echo "exit=${rc}" | tee -a "$LOG"
echo "--- unique errors ---"
rg -o "mps_spi[^\n]{0,160}|memref<[^>]+>|grouping for the value[^\n]{0,80}|ANE I/O[^\n]{0,120}|Incompatible element type[^\n]{0,120}|PASS |FAIL |Assertion[^\n]{0,80}|aborted" "$LOG" | sort -u | head -40
exit $rc

#!/bin/zsh
# Isolated ANE specialize for one tower. Parent survives SIGABRT.
#
#   model/ane_specialize_one.sh vision|audio|text|text_embeds
#
# Env (all optional):
#   ANEMLL_EMBEDDINGS_ARTIFACTS  default ~/.anemll-embeddings/artifacts
#   ANEMLL_COREAI_PYTHON         default: documented locations (api/runtime_paths.py)
#   ANEMLL_PYTHON                host venv python (default: <repo>/.venv/bin/python, else python3)
set -u
TOWER="${1:?usage: $0 vision|audio|text|text_embeds}"
case "$TOWER" in
  vision|audio|text|text_embeds) ;;
  *) echo "unknown tower: $TOWER (expected vision, audio, text, or text_embeds)" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ANEMLL_EMBEDDINGS_ARTIFACTS="${ANEMLL_EMBEDDINGS_ARTIFACTS:-$HOME/.anemll-embeddings/artifacts}"
PY="${ANEMLL_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
# Private, unpredictable log file (not a fixed /tmp name another user could pre-create).
LOG="$(mktemp "${TMPDIR:-/tmp}/coreai-ane-${TOWER}.XXXXXX")" || exit 1
echo "=== ANE specialize ${TOWER} $(date -u +%Y-%m-%dT%H:%M:%SZ) === log: $LOG" | tee "$LOG"
"$PY" "$ROOT/model/smoke_coreai_towers.py" --compute ane --tower "$TOWER" >>"$LOG" 2>&1
rc=$?
echo "exit=${rc}" | tee -a "$LOG"
echo "--- unique errors ---"
grep -oE "mps_spi.{0,160}|memref<[^>]+>|grouping for the value.{0,80}|ANE I/O.{0,120}|Incompatible element type.{0,120}|PASS |FAIL |Assertion.{0,80}|aborted" "$LOG" | sort -u | head -40
exit $rc

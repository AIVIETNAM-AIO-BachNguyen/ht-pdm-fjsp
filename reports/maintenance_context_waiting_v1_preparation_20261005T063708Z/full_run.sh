cd ~/ht-pdm-fjsp && \
export PATH="$HOME/.local/bin:$PATH" && \
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi && \
git fetch origin codex/maintenance-context-waiting-v1 && \
RUN_REVISION="e2ec53e3239881aaf2d7db57aa7b72c64342165f" && \
test "$(git rev-parse FETCH_HEAD)" = "$RUN_REVISION" && \
RUN_ID="maintenance_context_waiting_v1_full_cpu_$(date -u +%Y%m%dT%H%M%SZ)" && \
RUN_WORKTREE="$HOME/ht-pdm-fjsp-runs/$RUN_ID" && \
RUN_OUTPUT="$HOME/ht-pdm-fjsp/artifacts/$RUN_ID" && \
mkdir -p "$HOME/ht-pdm-fjsp-runs" && \
git worktree add --detach "$RUN_WORKTREE" "$RUN_REVISION" && \
cd "$RUN_WORKTREE" && \
export UV_PROJECT_ENVIRONMENT="$RUN_WORKTREE/.venv" && \
uv sync --frozen && \
set -o pipefail && \
test ! -e "$RUN_OUTPUT" && \
mkdir -p "$RUN_OUTPUT" && \
uv run --frozen --no-sync python -m ht_pdm_fjsp.maintenance_context_waiting \
  --profile full --device cpu --output-dir "$RUN_OUTPUT" \
  2>&1 | tee "$RUN_OUTPUT.log"

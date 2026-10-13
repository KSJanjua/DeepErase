# DeepErase -- source this at the start of every terminal on the GPU box:
#
#   source /workspace/Capstone_Project/DeepErase/gpu/env.sh
#
# It activates the project's own Python environment (made once by
# gpu/setup_env.sh), keeps downloads in the project's folder, and moves into the
# clone. It affects only the current terminal.

_de_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_de_home="${DEEPERASE_HOME:-$(dirname "$_de_repo")}"

if [ ! -x "$_de_home/.venv/bin/python" ]; then
  echo "No DeepErase environment at $_de_home/.venv. Create it once with:"
  echo "    bash $_de_repo/gpu/setup_env.sh"
  unset _de_repo _de_home
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$_de_home/.venv/bin/activate"
export DEEPERASE_HOME="$_de_home"
export HF_HOME="$_de_home/hf_cache"
export DEEPERASE_CACHE_DIR="$HF_HOME"
export HF_HUB_DISABLE_TELEMETRY=1
# setup_env.sh cached everything the runs read, so runs stay off the network: a
# Hub outage or rate limit cannot fail a job mid-matrix (the 13 Oct scan log
# shows one Hub lookup failing). To download something new in this terminal:
#     export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
cd "$_de_repo" || return 1

echo "DeepErase ready: $(python --version 2>&1) from $VIRTUAL_ENV"
echo "  repo  $PWD   (branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null), commit $(git rev-parse --short HEAD 2>/dev/null))"
echo "  cache $HF_HOME   (offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE)"
unset _de_repo _de_home

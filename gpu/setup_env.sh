#!/usr/bin/env bash
# DeepErase -- one-time setup of an isolated Python environment on the GPU box.
#
#   cd /workspace/Capstone_Project/DeepErase
#   bash gpu/setup_env.sh
#
# Everything is created under the folder that holds this clone (DEEPERASE_HOME,
# normally /workspace/Capstone_Project):
#
#   .venv/      Python 3.11 with the exact package versions the test suite is
#               verified against (requirements.txt); torch is the CUDA build
#   .tools/     uv (the installer), the Python it downloads, and its cache
#   hf_cache/   the two 1B TOFU checkpoints (and, later, the TOFU dataset)
#
# plus data/tofu/ inside the clone (TOFU's JSON files, which the tests read).
#
# Nothing is installed into the system Python, ~/.local or ~/.cache, so another
# project on this machine is unaffected, and deleting DEEPERASE_HOME removes
# every trace. Safe to re-run: steps that are already done are skipped.
#
# Options, as environment variables in front of the command:
#   DEEPERASE_HOME=/path   create the environment somewhere else
#   TORCH_CUDA=cu118       force torch's CUDA build (default: from the driver,
#                          cu124 for driver >= 525, cu118 below that)
#   SKIP_MODELS=1          do not pre-download the two model checkpoints

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${DEEPERASE_HOME:-$(dirname "$REPO")}"

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    OK    %s\n' "$*"; }
warn() { printf '    WARN  %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

case "$PROJECT" in
  / | /workspace | /root | "${HOME:-/root}")
    die "this clone sits directly in $PROJECT, so the environment would be created
there, next to whatever else lives in that folder. Clone into a folder of its own:
    mkdir -p /workspace/Capstone_Project && cd /workspace/Capstone_Project
    git clone -b rishabh-development https://github.com/KSJanjua/DeepErase.git" ;;
esac

VENV="$PROJECT/.venv"
TOOLS="$PROJECT/.tools"
CACHE="$PROJECT/hf_cache"
PY="$VENV/bin/python"
UV="$TOOLS/bin/uv"

# Keep uv, its downloads and its Python inside the project folder.
export UV_INSTALL_DIR="$TOOLS/bin"
export UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR="$TOOLS/cache"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME="$CACHE"
export HF_HUB_DISABLE_TELEMETRY=1

echo "DeepErase environment setup"
echo "  clone:        $REPO"
echo "  environment:  $VENV"
echo "  model cache:  $CACHE"

# ---------------------------------------------------------------------------
say "1/7  Machine and GPU"
[ "$(uname -m)" = "x86_64" ] || die "expected an x86_64 machine, found $(uname -m)."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no NVIDIA driver is visible here."
nvidia-smi -L | sed 's/^/    /'
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
DRIVER_MAJOR="${DRIVER%%.*}"
if [ -z "${TORCH_CUDA:-}" ]; then
  if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && [ "$DRIVER_MAJOR" -ge 525 ]; then
    TORCH_CUDA=cu124
  elif [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && [ "$DRIVER_MAJOR" -ge 450 ]; then
    TORCH_CUDA=cu118
  else
    TORCH_CUDA=cu124
    warn "could not read the driver version; assuming a recent driver"
  fi
fi
ok "driver ${DRIVER:-unknown} -> torch CUDA build $TORCH_CUDA"
mkdir -p "$PROJECT"
FREE_GB="$(df -Pk "$PROJECT" | awk 'NR==2 {print int($4/1048576)}')"
[ "$FREE_GB" -ge 20 ] || die "only ${FREE_GB} GB free under $PROJECT; about 20 GB is needed."
ok "${FREE_GB} GB free under $PROJECT"

# ---------------------------------------------------------------------------
say "2/7  uv, the package installer (into $TOOLS/bin)"
if [ ! -x "$UV" ]; then
  mkdir -p "$TOOLS/bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || die "the uv installer failed (no internet?)"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh || die "the uv installer failed (no internet?)"
  else
    python3 -m pip install --quiet --target "$TOOLS/uv-pip" uv || die "could not install uv"
    ln -sf "$TOOLS/uv-pip/bin/uv" "$UV"
  fi
fi
[ -x "$UV" ] || die "uv is not at $UV after installing it"
ok "$("$UV" --version)"

# ---------------------------------------------------------------------------
say "3/7  Python 3.11 environment at $VENV"
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
  ok "already there: $("$PY" --version)"
else
  if [ -e "$VENV" ]; then
    "$UV" venv --clear "$VENV" --python 3.11 --seed
  else
    "$UV" venv "$VENV" --python 3.11 --seed
  fi
  ok "created: $("$PY" --version)"
fi

# ---------------------------------------------------------------------------
say "4/7  PyTorch 2.6.0, CUDA build $TORCH_CUDA (about 3 GB; a few minutes)"
# The exact build label (2.6.0+cu124), so a re-run with another TORCH_CUDA swaps it.
"$UV" pip install --python "$PY" "torch==2.6.0+$TORCH_CUDA" \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
torch_version() { "$PY" -c 'import importlib.metadata as m; print(m.version("torch"))'; }
TORCH_BUILD="$(torch_version)"
ok "torch $TORCH_BUILD"

# ---------------------------------------------------------------------------
say "5/7  Project packages (requirements.txt, exact versions)"
# Everything except torch, which step 4 chose to match the driver.
grep -v -E '^torch([=<>!~ ]|$)' "$REPO/requirements.txt" > "$TOOLS/requirements-without-torch.txt"
"$UV" pip install --python "$PY" -r "$TOOLS/requirements-without-torch.txt"
AFTER="$(torch_version)"
[ "$AFTER" = "$TORCH_BUILD" ] || die "torch changed from $TORCH_BUILD to $AFTER while installing the other packages"
ok "torch still $AFTER"

# ---------------------------------------------------------------------------
say "6/7  Checking that PyTorch can use the GPU"
"$PY" - <<'PYEOF'
import os, sys
# Same precision settings as run_study: float32 must not be run as TF32.
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import torch
print(f"    torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("    FAIL  torch cannot see the GPU. If nvidia-smi works, the driver may be "
             "too old for this build: re-run with TORCH_CUDA=cu118 in front.")
dev = torch.device("cuda:0")
free, total = torch.cuda.mem_get_info(dev)
print(f"    {torch.cuda.get_device_name(dev)}: {total / 2**30:.1f} GiB, "
      f"{free / 2**30:.1f} GiB free")
torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(512, 512, dtype=torch.float64)
b = torch.randn(512, 512, dtype=torch.float64)
want = a @ b
got = (a.float().to(dev) @ b.float().to(dev)).double().cpu()
err = float((got - want).abs().max() / want.abs().max())
print(f"    float32 matmul on the GPU matches float64 on the CPU to {err:.1e}")
if err > 1e-4:
    sys.exit("    FAIL  float32 matmul is too inaccurate: something forces TF32 on")
import datasets, numpy, scipy, transformers
print(f"    transformers {transformers.__version__}, datasets {datasets.__version__}, "
      f"numpy {numpy.__version__}, scipy {scipy.__version__}")
PYEOF
ok "GPU works from the project environment"

# ---------------------------------------------------------------------------
say "7/7  TOFU files for the tests, and the two 1B checkpoints"
mkdir -p "$REPO/data/tofu"
for f in full forget01 forget05 forget10 retain90 retain95 retain99 \
         real_authors world_facts forget10_perturbed retain_perturbed \
         forget01_perturbed forget05_perturbed real_authors_perturbed \
         world_facts_perturbed; do
  if [ ! -s "$REPO/data/tofu/$f.json" ]; then
    curl -sSfL -o "$REPO/data/tofu/$f.json" \
      "https://huggingface.co/datasets/locuslab/TOFU/resolve/main/$f.json" \
      || die "could not download TOFU's $f.json"
  fi
done
"$PY" - "$REPO" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from deeperase.data.tofu import read_tofu_rows
want = {"full": 4000, "forget10": 400, "retain90": 3600, "forget10_perturbed": 400,
        "retain_perturbed": 400, "real_authors_perturbed": 100,
        "world_facts_perturbed": 117}
bad = {k: len(read_tofu_rows(k, f"{sys.argv[1]}/data/tofu")) for k in want}
bad = {k: n for k, n in bad.items() if n != want[k]}
if bad:
    sys.exit(f"    FAIL  truncated TOFU files (rows found): {bad}. Delete them and re-run.")
print("    TOFU row counts as expected")
PYEOF
if [ "${SKIP_MODELS:-0}" != "1" ]; then
  "$PY" - "$CACHE" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for split in ("full", "retain90"):
    repo = f"open-unlearning/tofu_Llama-3.2-1B-Instruct_{split}"
    snapshot_download(repo, cache_dir=sys.argv[1],
                      allow_patterns=["*.json", "*.safetensors"])
    print(f"    {repo}: cached")
PYEOF
fi

cat <<EOF

==========================================================================
Setup complete.

In every new terminal, start with:
    source $REPO/gpu/env.sh

Then check everything (about a minute):
    python gpu/bootstrap.py --run-tests
==========================================================================
EOF

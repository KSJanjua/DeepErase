#!/usr/bin/env bash
# Matched-damage T1 controls, one per method.
#
#   bash gpu/matched_controls.sh ga:1e-8 graddiff:5e-8 npo:1e-8
#
# For each method:rate it first runs scripts/control_probe.py, which finds the
# random-direction scale whose damage matches the real run's (about 25 min),
# then runs the full control study at that scale through gpu/replicate.py
# (about 50 min). SEED=1 in front picks another seed (default 0). A method whose
# probe finds no matching scale is skipped and reported; the others still run.

set -u
SEED="${SEED:-0}"

for spec in "$@"; do
  method="${spec%%:*}"
  lr="${spec#*:}"
  echo "=== matched-damage control: $method lr=$lr seed=$SEED"
  python -m deeperase.scripts.control_probe --method "$method" --learning-rate "$lr" --seed "$SEED"
  scale="$(python - "$method" "$SEED" <<'PY'
import json, sys
path = f"results/control_probes/{sys.argv[1]}_1B_v2_s{sys.argv[2]}.json"
try:
    print(json.load(open(path)).get("recommended_scale") or "")
except (OSError, ValueError):
    print("")
PY
)"
  if [ -z "$scale" ]; then
    echo "!! no matched scale for $method (see the probe output above); its control is skipped"
    continue
  fi
  echo "=== $method: control at scale $scale"
  python gpu/replicate.py --methods "$method" --seeds "$SEED" --learning-rate "$lr" \
         --control random --control-scale "$scale"
done

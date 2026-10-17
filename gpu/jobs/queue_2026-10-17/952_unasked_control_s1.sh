# UNASKED matched-damage control, seed 1 (probe finds the scale; falls back to scale 1 if none).
LR=$(python -c "import json; print(json.load(open('results/lr_scans/unasked_1B_v2_s0.json')).get('recommended_lr') or '')")
[ -n "$LR" ] || { echo "no recommended rate"; exit 1; }
SEED=1 bash gpu/matched_controls.sh unasked:$LR > results/logs/unasked_control_s1.log 2>&1; rc=$?
if grep -q "no matched scale" results/logs/unasked_control_s1.log; then
  echo "no matched-damage scale: running the matched-magnitude control (scale 1)"
  python gpu/replicate.py --methods unasked --seeds 1 --learning-rate "$LR" --control random --control-scale 1 >> results/logs/unasked_control_s1.log 2>&1; rc=$?
fi
bash gpu/push_results.sh "H100: UNASKED matched-damage control (seed 1)" results/control_probes/unasked_1B_v2_s1.json results/studies/control_*_v2_unasked_1B_s1_lr* results/studies/replication_manifest.json results/logs/unasked_control_s1.log
exit $rc

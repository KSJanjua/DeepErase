# UNASKED seed 0: a random-direction control that reaches UNASKED's general-knowledge cost.
# The R-matched control (scale 10, same seed and direction) stops at general knowledge 0.758;
# UNASKED's checkpoint is at 0.728. Scale 15 continues the same direction to 30x ||v||.
git pull -q --rebase --autostash origin rishabh-development || true
LR=$(python -c "import json; print(json.load(open('results/lr_scans/unasked_1B_v2_s0.json')).get('recommended_lr') or '')")
[ -n "$LR" ] || { echo "no recommended rate"; exit 1; }
python gpu/replicate.py --methods unasked --seeds 0 --learning-rate "$LR" --control random --control-scale 15 --tag genmatch > results/logs/unasked_genmatch_control_s0.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: UNASKED control at scale 15, general-knowledge match (seed 0)" results/studies/control_x15_v2_unasked_1B_s0_lr*_genmatch results/studies/replication_manifest.json results/logs/unasked_genmatch_control_s0.log
exit $rc

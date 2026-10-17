# Relearning dose-response on the UNASKED seed-0 checkpoint: five attack strengths,
# each from the same checkpoint (exploratory depth test).
git pull -q --rebase --autostash origin rishabh-development || true
LR=$(python -c "import json; print(json.load(open('results/lr_scans/unasked_1B_v2_s0.json')).get('recommended_lr') or '')")
[ -n "$LR" ] || { echo "no learning rate"; exit 1; }
python -m deeperase.scripts.relearn_attack --method unasked --learning-rate "$LR" --seed 0 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose > results/logs/relearn_unasked_s0_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, unasked seed 0" results/relearn/unasked_1B_v2_s0_dose.json results/logs/relearn_unasked_s0_dose.log
exit $rc

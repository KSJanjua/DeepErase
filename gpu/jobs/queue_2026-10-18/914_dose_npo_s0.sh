# Relearning dose-response on the NPO seed-0 checkpoint (five attack strengths).
git pull -q --rebase --autostash origin rishabh-development || true
python -m deeperase.scripts.relearn_attack --method npo --learning-rate 1e-8 --seed 0 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose > results/logs/relearn_npo_s0_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, npo seed 0" results/relearn/npo_1B_v2_s0_dose.json results/logs/relearn_npo_s0_dose.log
exit $rc

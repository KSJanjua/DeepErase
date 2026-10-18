# Relearning dose-response on the GradDiff seed-2 checkpoint (five attack strengths).
git pull -q --rebase --autostash origin rishabh-development || true
python -m deeperase.scripts.relearn_attack --method graddiff --learning-rate 5e-8 --seed 2 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose > results/logs/relearn_graddiff_s2_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, graddiff seed 2" results/relearn/graddiff_1B_v2_s2_dose.json results/logs/relearn_graddiff_s2_dose.log
exit $rc

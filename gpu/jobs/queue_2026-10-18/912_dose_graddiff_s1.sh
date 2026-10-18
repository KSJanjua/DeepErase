# Relearning dose-response on the GradDiff seed-1 checkpoint (five attack strengths).
git pull -q --rebase --autostash origin rishabh-development || true
python -m deeperase.scripts.relearn_attack --method graddiff --learning-rate 5e-8 --seed 1 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose > results/logs/relearn_graddiff_s1_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, graddiff seed 1" results/relearn/graddiff_1B_v2_s1_dose.json results/logs/relearn_graddiff_s1_dose.log
exit $rc

# Relearning dose-response on the UNASKED-G (TOFU anchor 10) seed-1 checkpoint (five attack strengths).
git pull -q --rebase --autostash origin rishabh-development || true
python -m deeperase.scripts.relearn_attack --method unaskedg --learning-rate 3e-6 --seed 1 --retain-weight 10 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose_lam10 > results/logs/relearn_unaskedg_lam10_s1_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, unaskedg (TOFU anchor 10) seed 1" results/relearn/unaskedg_1B_v2_s1_dose_lam10.json results/logs/relearn_unaskedg_lam10_s1_dose.log
exit $rc

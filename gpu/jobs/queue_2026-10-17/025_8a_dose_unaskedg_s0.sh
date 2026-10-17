# Relearning dose-response on the UNASKED-G seed-0 checkpoint, TOFU anchor 10 (H9b's
# configuration; five attack strengths, each from the same checkpoint). Reported.
git pull -q --rebase --autostash origin rishabh-development || true
python -m deeperase.scripts.relearn_attack --method unaskedg --learning-rate 3e-6 --seed 0 --retain-weight 10 --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose_lam10 > results/logs/relearn_unaskedg_lam10_s0_dose.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: relearning dose-response, unaskedg (TOFU anchor 10) seed 0" results/relearn/unaskedg_1B_v2_s0_dose_lam10.json results/logs/relearn_unaskedg_lam10_s0_dose.log
exit $rc

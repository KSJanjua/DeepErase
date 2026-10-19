# UNASKED-G, TOFU anchor 10, seed 4 (H9b extension to five seeds).
git pull -q --rebase --autostash origin rishabh-development || true
python gpu/replicate.py --methods unaskedg --seeds 4 --learning-rate 3e-6 --tag lam10 --extra --retain-weight 10 --paraphrase-depth --paraphrase-answer-depth > results/logs/unaskedg_lam10_s4.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: UNASKED-G seed 4, TOFU anchor 10 (H9b extension)" results/studies/study_v2_unaskedg_1B_s4_lr3e-06_lam10 results/studies/replication_manifest.json results/logs/unaskedg_lam10_s4.log
exit $rc

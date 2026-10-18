# Ablation (seed 2): UNASKED-G minus its general anchor = UNASKED with the TOFU anchor at 10.
git pull -q --rebase --autostash origin rishabh-development || true
python gpu/replicate.py --methods unasked --seeds 2 --learning-rate 3e-6 --tag lam10 --extra --retain-weight 10 --paraphrase-depth --paraphrase-answer-depth > results/logs/abl_nogen_s2.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: ablation, UNASKED with TOFU anchor 10 and no general anchor (seed 2)" results/studies/study_v2_unasked_1B_s2_lr3e-06_lam10 results/studies/replication_manifest.json results/logs/abl_nogen_s2.log
exit $rc

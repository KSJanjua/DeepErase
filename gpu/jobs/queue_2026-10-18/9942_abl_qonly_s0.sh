# Ablation (seed 0): UNASKED-G (TOFU anchor 10) with the target source changing only the question.
git pull -q --rebase --autostash origin rishabh-development || true
python gpu/replicate.py --methods unaskedg --seeds 0 --learning-rate 3e-6 --tag lam10-qonly --extra --retain-weight 10 --unasked-parts question --paraphrase-depth --paraphrase-answer-depth > results/logs/abl_qonly_s0.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: ablation, UNASKED-G (anchor 10) with question-only targets (seed 0)" results/studies/study_v2_unaskedg_1B_s0_lr3e-06_lam10-qonly results/studies/replication_manifest.json results/logs/abl_qonly_s0.log
exit $rc

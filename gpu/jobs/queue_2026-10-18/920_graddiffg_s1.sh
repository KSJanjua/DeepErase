# GradDiff-G seed 1: GradDiff with UNASKED-G's general anchor (fairness baseline, H9).
git pull -q --rebase --autostash origin rishabh-development || true
python gpu/replicate.py --methods graddiffg --seeds 1 --learning-rate 5e-8 --extra --paraphrase-depth --paraphrase-answer-depth > results/logs/graddiffg_s1.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: GradDiff-G seed 1 (GradDiff + general anchor 1, lr 5e-8)" results/studies/study_v2_graddiffg_1B_s1_lr5e-08 results/studies/replication_manifest.json results/logs/graddiffg_s1.log
exit $rc

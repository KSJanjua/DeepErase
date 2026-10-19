# GradDiff seed 4 at its v2 rate (H9b extension: the seed-matched baseline).
git pull -q --rebase --autostash origin rishabh-development || true
python gpu/replicate.py --methods graddiff --seeds 4 --learning-rate 5e-8 > results/logs/graddiff_s4.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: GradDiff seed 4 (H9b extension)" results/studies/study_v2_graddiff_1B_s4_lr5e-08 results/studies/replication_manifest.json results/logs/graddiff_s4.log
exit $rc

# UNASKED seed 2 at the seed-0 scan's recommended rate, with paraphrased-question depth.
LR=$(python -c "import json; print(json.load(open('results/lr_scans/unasked_1B_v2_s0.json')).get('recommended_lr') or '')")
[ -n "$LR" ] || { echo "no recommended rate"; exit 1; }
python gpu/replicate.py --methods unasked --seeds 2 --learning-rate "$LR" --extra --paraphrase-depth --paraphrase-answer-depth > results/logs/unasked_s2.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: UNASKED seed 2 (lr $LR)" results/studies/study_v2_unasked_1B_s2_lr* results/studies/replication_manifest.json results/logs/unasked_s2.log
exit $rc

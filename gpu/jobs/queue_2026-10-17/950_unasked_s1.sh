# UNASKED seed 1 at the seed-0 scan's recommended rate, with paraphrased-question depth.
LR=$(python -c "import json; print(json.load(open('results/lr_scans/unasked_1B_v2_s0.json')).get('recommended_lr') or '')")
[ -n "$LR" ] || { echo "no recommended rate"; exit 1; }
python gpu/replicate.py --methods unasked --seeds 1 --learning-rate "$LR" --extra --paraphrase-depth --paraphrase-answer-depth > results/logs/unasked_s1.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: UNASKED seed 1 (lr $LR)" results/studies/study_v2_unasked_1B_s1_lr* results/studies/replication_manifest.json results/logs/unasked_s1.log
exit $rc

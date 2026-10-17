# npo seed 0 rerun with paraphrased-question depth (comparison for UNASKED criterion E3).
python gpu/replicate.py --methods npo --seeds 0 --learning-rate 1e-8 --tag para --extra --paraphrase-depth --paraphrase-answer-depth > results/logs/npo_para_s0.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: npo seed 0 with paraphrased-question depth" results/studies/study_v2_npo_1B_s0_lr*_para results/studies/replication_manifest.json results/logs/npo_para_s0.log
exit $rc

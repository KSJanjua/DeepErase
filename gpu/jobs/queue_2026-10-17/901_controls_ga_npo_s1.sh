# Per-seed matched-damage controls for GA and NPO, seed 1 (today seeds 1-2 borrow seed 0s control).
SEED=1 bash gpu/matched_controls.sh ga:1e-8 npo:1e-8 > results/logs/controls_ga_npo_s1.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: matched-damage controls GA and NPO (seed 1)" results/control_probes/ga_1B_v2_s1.json results/control_probes/npo_1B_v2_s1.json results/studies/control_x*_v2_ga_1B_s1_lr* results/studies/control_x*_v2_npo_1B_s1_lr* results/studies/replication_manifest.json results/logs/controls_ga_npo_s1.log
exit $rc

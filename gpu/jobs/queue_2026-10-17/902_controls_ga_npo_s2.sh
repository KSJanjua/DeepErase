# Per-seed matched-damage controls for GA and NPO, seed 2 (today seeds 1-2 borrow seed 0s control).
SEED=2 bash gpu/matched_controls.sh ga:1e-8 npo:1e-8 > results/logs/controls_ga_npo_s2.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: matched-damage controls GA and NPO (seed 2)" results/control_probes/ga_1B_v2_s2.json results/control_probes/npo_1B_v2_s2.json results/studies/control_x*_v2_ga_1B_s2_lr* results/studies/control_x*_v2_npo_1B_s2_lr* results/studies/replication_manifest.json results/logs/controls_ga_npo_s2.log
exit $rc

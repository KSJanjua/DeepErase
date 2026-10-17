# UNASKED-G, TOFU anchor 10, seed 2: R-matched random-direction control (H9b, criterion C4).
git pull -q --rebase --autostash origin rishabh-development || true
SEED=2 TAG=lam10 EXTRA="--retain-weight 10" bash gpu/matched_controls.sh unaskedg:3e-6 > results/logs/unaskedg_lam10_control_s2.log 2>&1; rc=$?
bash gpu/push_results.sh "H100: UNASKED-G (TOFU anchor 10) matched-damage control, seed 2" results/control_probes/unaskedg_1B_v2_s2_lam10.json results/studies/control_x*_v2_unaskedg_1B_s2_lr3e-06_lam10 results/studies/replication_manifest.json results/logs/unaskedg_lam10_control_s2.log
exit $rc

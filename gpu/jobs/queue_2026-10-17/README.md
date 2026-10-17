# GPU queue of 17-18 Oct 2026 (H100 MIG slice)

These are the jobs that were still queued or running when the last GPU session
was planned to end (around 10:00 IST, 18 Oct 2026). Each job is a
self-contained bash script for `gpu/run_queue.sh`:

1. `git pull` the repository;
2. run one study, control, or relearning attack;
3. push its results with `gpu/push_results.sh`.

The jobs that finished are in git: their results are under `results/`, and
`git log` shows one "H100: ..." commit per job. `../../../_workspace/HANDOFF.md`
(outside the repository) lists which jobs finished.

## Re-queueing on a new GPU box

```bash
# 1. environment, as in gpu/README.md (setup_env.sh, env.sh, model + TOFU cache)
# 2. the runner
Q=/path/to/queue; mkdir -p $Q/pending
QUEUE_DIR=$Q nohup bash gpu/run_queue.sh > $Q/runner.log 2>&1 &
# 3. copy only the jobs whose results are NOT yet in git, in this order
cp gpu/jobs/queue_2026-10-17/<job>.sh $Q/pending/
```

Names sort the order, and lower numbers run first. Priorities, highest first:

| Jobs | What | Needed for |
|---|---|---|
| 025_6b, 025_6c | UNASKED-G, TOFU anchor 10, seeds 1-2 | H9b verdict (C1-C3 on seeds 1-2) |
| 025_6d, 025_6e, 025_6f | their R-matched random-direction controls, seeds 0-2 | H9b criterion C4 |
| 025_8, 025_8a | relearning dose-response: GradDiff; UNASKED-G (anchor 10) | "is the knowledge gone?" (RESULTS §19.3) |
| 025_9a, 025_9b | UNASKED-G (anchor 3) controls, seeds 1-2 | H9 criterion C4 (pre-registered version) |
| 030, 031 | GA and NPO seed 0 with paraphrased-question and paraphrased-answer depth | baseline comparison for the paraphrase checks |
| 032 | UNASKED (λ = 1) control at 15×, matched on general knowledge | RESULTS §19.5 |
| 901, 902 | GA and NPO per-seed matched controls, seeds 1-2 | H1 per-seed check (seeds 1-2 borrow seed 0's today) |
| 950-953 | UNASKED (λ = 1) seeds 1-2 and their controls | ablation without the general anchor |
| 960, 961 | old dose-response jobs (UNASKED λ = 1, GA) | superseded by 025_8/8a; optional |

The memory budget allows one training job at a time on a 40 GB slice
(UNASKED-G peaks near 23 GB).

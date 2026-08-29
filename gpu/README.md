# gpu/ — running DeepErase on the university A100

Scripts for a **shared** GPU reached through Jupyter Lab. They assume you can
run `.py` files (from a notebook cell with `!python …`, or a terminal).

## Session checklist

```bash
cd <repo root>                     # the dir containing deeperase/ and tests/
python gpu/bootstrap.py --size 1B --run-tests    # once per environment
python gpu/bootstrap.py --size 1B                # every session (fast)
```

`bootstrap.py` exits non-zero if anything would stop a run, so it can gate a
pipeline. It reports **free** VRAM, not total — on a shared card that is the
only number that matters, and it changes between sessions.

## Two things that will bite you

**1. Run from the repository root.** `DEFAULT_REFERENCE_PATH` resolves
`reference_uds/tofu_data/forget10_filtered.json` relative to the working
directory. From anywhere else `--reference-spans` silently fails to find the
authors' annotations and you fall back to a heuristic that agrees with them on
12% of examples.

**2. Do not `pip install -r requirements.txt` here.** That pins a **CPU** torch
wheel and will replace the container's CUDA build. Use:

```bash
pip install -r requirements-gpu.txt     # deliberately does not touch torch
```

## Replication (task T2)

```bash
python gpu/replicate.py --dry-run                    # print the plan
python gpu/replicate.py --methods ga npo graddiff --seeds 0 1 2
```

Resumable: re-running skips completed runs and passes `--resume` to partial
ones. Writes `results/studies/replication_manifest.json`.

### Seeds only vary anything under `--sampling random`

`unlearn()` calls `torch.manual_seed`, but this pipeline does not shuffle and
Llama-3.2 has dropout 0. Under the default `--sampling even`, changing the seed
produces a **bit-identical** run. `replicate.py` refuses multi-seed matrices
under `even` rather than letting you report three copies of one number as three
observations. The genuine source of variation is which examples are drawn.

## Memory

Measured on this project; check against `bootstrap.py`'s live reading.

| Job | 1B | 3B |
|---|---|---|
| Measurement only (weights resident) | ~2.5 GB / model | ~7.5 GB / model |
| UDS (3 models resident) | ~8 GB | ~23 GB |
| Full-parameter training (weights + grads + 2 Adam moments + activations) | ~11 GB | ~33 GB |
| NPO (adds a frozen reference model) | ~14 GB | ~40 GB |

With ~50 GB free, every 1B job is comfortable and 3B NPO is tight — the other
tenant's footprint can move under you mid-run. `replicate.py --min-free-gb`
guards the start of each run but cannot stop a mid-run OOM; runs are resumable
for that reason.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by both scripts.
Fragmentation is the usual cause of an OOM that "should" have fitted, and it is
worse when another process owns part of the card.

## Long runs

Jupyter kernels die when the browser tab closes. For anything over an hour:

```bash
nohup python gpu/replicate.py --methods ga npo graddiff --seeds 0 1 2 \
      > results/logs/replication.log 2>&1 &
tail -f results/logs/replication.log
```

## What to send back for analysis

The run directories are self-describing. The small files are enough:

```bash
tar czf handoff.tgz \
    results/studies/*/config.json \
    results/studies/*/plane.json \
    results/studies/*/sensitivity.json \
    results/studies/*/unlearn.json \
    results/studies/replication_manifest.json
```

Skip `partial/` and `stage1_cache.json` unless something looks wrong — they are
large and only needed for debugging.

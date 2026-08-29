# data/

## Committed

| Path | What |
|---|---|
| `probes/seed_tofu.json` | 36 hand-verified seed probes over 3 TOFU authors. Validates the probe schema; regenerate with `python -m deeperase.probes.seed_tofu` |
| `breadth_items.json` | 1200 forced-choice items (400 each on tiers B0, B1, R) built from TOFU's perturbed splits by `deeperase.eval.breadth.load_breadth_items`. Not authored by this project |

## Not committed (downloaded)

Both directories below are in `.gitignore`: they are re-downloadable and carry
their own licence terms.

### `tofu/` — the TOFU benchmark

```bash
mkdir -p data/tofu
for f in full forget01 forget05 forget10 retain90 retain95 retain99 \
         real_authors world_facts forget10_perturbed retain_perturbed \
         forget01_perturbed forget05_perturbed real_authors_perturbed \
         world_facts_perturbed; do
  curl -sSL -o "data/tofu/$f.json" \
    "https://huggingface.co/datasets/locuslab/TOFU/resolve/main/$f.json"
done
curl -sSL -o data/tofu/README.md \
  "https://huggingface.co/datasets/locuslab/TOFU/resolve/main/README.md"
```

Expected row counts — check these after downloading, because a truncated file
fails silently and produces a plausible-looking wrong answer:

| Config | Rows | | Config | Rows |
|---|---|---|---|---|
| `full` | 4000 | | `forget10_perturbed` | 400 |
| `retain90` | 3600 | | `retain_perturbed` | 400 |
| `retain95` | 3800 | | `forget05_perturbed` | 200 |
| `retain99` | 3960 | | `forget01_perturbed` | 40 |
| `forget10` | 400 | | `real_authors` | 100 |
| `forget05` | 200 | | `world_facts` | 117 |
| `forget01` | 40 | | | |

200 fictitious authors × 20 QA pairs = 4000. The forget splits are **nested at
the end** of `full` (`forget10` = indices 0–399, `forget05` = 200–399,
`forget01` = 360–399), which is why sampling must be spread across the split —
see `RESULTS.md` §1.1 for the run that got this wrong.

The code normally fetches these through `datasets.load_dataset("locuslab/TOFU", ...)`
into `--cache-dir`; the files above are the same content, kept for offline work
and for checking what the loader returned.

### `uds_annotations/` — the authors' entity-span annotations

```bash
mkdir -p data/uds_annotations
curl -sSL -o data/uds_annotations/forget10_filtered.json \
  "https://huggingface.co/datasets/jaeunglee/uds-annotated-tofu/resolve/main/forget10_filtered.json"
```

367 records, fields `idx question answer prefix entity entity_span full_output`.

This file is **byte-identical** to the copy vendored at
`reference_uds/tofu_data/forget10_filtered.json`
(sha256 `c3d88b76702fed46…`), which is the copy the code actually reads via
`DEFAULT_REFERENCE_PATH`. Verified 29 August 2026. Use `--reference-spans` to
score against these rather than the `NOVEL_CONTENT` heuristic, which agrees with
them on only 12% of examples (`results/span_comparison.json`).

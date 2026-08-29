# literature/

The ten references of **Appendix A** of the mid-semester report.

The PDFs are **not committed** — they are third-party copyrighted material.
This file and `download_manifest.json` are the index; run the command below to
populate the directory.

## Fetch

```bash
while IFS='|' read -r n id slug; do
  curl -sSL -o "literature/[$n] $slug (arXiv $id).pdf" "https://arxiv.org/pdf/$id"
done <<'IDS'
01|2402.16835|lynch2024-eight-methods
02|2409.18025|lucki2025-adversarial
03|2605.24614|lee2026-uds
04|2401.06121|maini2024-tofu
05|2210.01504|jang2023-knowledge-unlearning
06|2404.05868|zhang2024-npo
07|2503.04693|wang2025-uipe
08|2406.11614|hong2025-parametric-traces
09|2212.04089|ilharco2023-task-arithmetic
10|2202.05262|meng2022-rome
IDS
```

## The references

Every title below was verified against the `citation_title` metadata on the
arXiv abstract page at download time; see `download_manifest.json` for the
recorded match per entry.

| # | arXiv | Title | Role in this project |
|---|---|---|---|
| [1] | [2402.16835](https://arxiv.org/abs/2402.16835) | Eight Methods to Evaluate Robust Unlearning in LLMs | Motivates multi-axis measurement (gap G1) |
| [2] | [2409.18025](https://arxiv.org/abs/2409.18025) | An Adversarial Perspective on Machine Unlearning for AI Safety | Evidence that evaluations overstate removal (G1) |
| [3] | [2605.24614](https://arxiv.org/abs/2605.24614) | Measuring the Depth of LLM Unlearning via Activation Patching | **The depth axis.** Implemented in `deeperase/eval/uds.py`; code vendored at `reference_uds/` |
| [4] | [2401.06121](https://arxiv.org/abs/2401.06121) | TOFU: A Task of Fictitious Unlearning for LLMs | **The benchmark.** Splits, retain oracles, perturbed answers |
| [5] | [2210.01504](https://arxiv.org/abs/2210.01504) | Knowledge Unlearning for Mitigating Privacy Risks in Language Models | Gradient ascent baseline (`unlearn.py`) |
| [6] | [2404.05868](https://arxiv.org/abs/2404.05868) | Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning | NPO objective (`unlearn.py`) |
| [7] | [2503.04693](https://arxiv.org/abs/2503.04693) | UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets | **The extrapolation dial** and the checkpoint-selection rule |
| [8] | [2406.11614](https://arxiv.org/abs/2406.11614) | Intrinsic Test of Unlearning Using Parametric Knowledge Traces | Complementary parametric view of depth (G2) |
| [9] | [2212.04089](https://arxiv.org/abs/2212.04089) | Editing Models with Task Arithmetic | Theoretical basis for the update vector |
| [10] | [2202.05262](https://arxiv.org/abs/2202.05262) | Locating and Editing Factual Associations in GPT | Causal tracing; source of the patching methodology |

Note on [3]: the report's Appendix A gives the short title *"Measuring the depth
of LLM unlearning"*; arXiv carries the fuller *"…via Activation Patching"*. Same
paper, same ID.

Note on [8]: the report cites this as *"Intrinsic Test of Unlearning Using
Parametric Knowledge Traces"*, matching arXiv exactly. Some secondary sources
render it as *"Intrinsic Evaluation of…"*.

| run | α | trained UDS / strict | held-out UDS / strict | paraphrase UDS / strict | R | RA | WF |
|---|---|---|---|---|---|---|---|
| study_v2_unasked_1B_s0_lr3e-06 | 0 | 0.532 / 0.452 | 0.397 / 0.333 | 0.410 / 0.366 | 0.772 | 0.72 | 0.735 |
| study_v2_unasked_1B_s0_lr3e-06 | 1 | 0.654 / 0.524 | 0.583 / 0.482 | 0.584 / 0.507 | 0.720 | 0.35 | 0.308 |
| study_v2_unasked_1B_s0_lr3e-06_lam3 | 0 | 0.503 / 0.430 | 0.371 / 0.314 | 0.381 / 0.337 | 0.787 | 0.81 | 0.786 |
| study_v2_unasked_1B_s0_lr3e-06_lam3 | 1 | 0.662 / 0.530 | 0.584 / 0.488 | 0.605 / 0.515 | 0.708 | 0.36 | 0.402 |
| study_v2_ga_1B_s0_lr1e-08 | 0 | 0.282 / 0.271 | 0.246 / 0.228 | – / – | 0.765 | 0.67 | 0.735 |
| study_v2_ga_1B_s1_lr1e-08 | 0 | 0.256 / 0.247 | 0.231 / 0.208 | – / – | 0.765 | 0.66 | 0.726 |
| study_v2_ga_1B_s2_lr1e-08 | 0 | 0.124 / 0.108 | 0.092 / 0.085 | – / – | 0.775 | 0.82 | 0.803 |
| study_v2_npo_1B_s0_lr1e-08 | 0 | 0.154 / 0.154 | 0.129 / 0.123 | – / – | 0.765 | 0.79 | 0.812 |
| study_v2_npo_1B_s1_lr1e-08 | 0 | 0.233 / 0.231 | 0.209 / 0.196 | – / – | 0.767 | 0.67 | 0.726 |
| study_v2_npo_1B_s2_lr1e-08 | 0 | 0.130 / 0.115 | 0.100 / 0.093 | – / – | 0.770 | 0.82 | 0.795 |
| study_v2_graddiff_1B_s0_lr5e-08 | 0 | 0.144 / 0.144 | 0.053 / 0.053 | – / – | 0.775 | 0.88 | 0.821 |
| study_v2_graddiff_1B_s1_lr5e-08 | 0 | 0.121 / 0.120 | 0.057 / 0.056 | – / – | 0.765 | 0.86 | 0.829 |
| study_v2_graddiff_1B_s2_lr5e-08 | 0 | 0.174 / 0.158 | 0.076 / 0.071 | – / – | 0.765 | 0.85 | 0.838 |

Paraphrased answers at the checkpoint, the same facts (paired):

| run | facts | original answer: UDS / strict | paraphrased answer: UDS / strict |
|---|---|---|---|
| study_v2_unasked_1B_s0_lr3e-06_lam3 | 33 | 0.393 / 0.390 | 0.510 / 0.448 |

Strict-UDS excess over random damage of the same general-knowledge cost (exploratory; E2 is matched on R):

* study_v2_unasked_1B_s0_lr3e-06: at the checkpoint –; mean over 0 matched alphas – (controls control_x10_v2_unasked_1B_s0_lr3e-06; general knowledge 0.758-0.862)

E1 checkpoint strict UDS (trained, mean): 0.452 (>= 0.35): met
E2 damage-matched strict excess (trained, per seed): [0.284] (>= 0.15 each): met
E3 paraphrase / trained strict: 0.81 (>= 0.7): met
E4 RA 0.72, WF 0.735 (>= 0.77, 0.8): not met
E5 held-out UDS 0.397, strict 0.333 (reported)

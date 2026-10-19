| run | λ_g | seed | strict T | UDS T | breadth T | strict H | breadth H | strict para-Q | R | RA | WF | excess over control: R-matched / general-matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| study_v2_unaskedg_1B_s0_lr3e-06 | 1 | 0 | 0.405 | 0.479 | 0.309 | 0.259 | 0.320 | 0.319 | 0.805 | 0.840 | 0.829 | 0.365 / 0.194 |
| study_v2_unaskedg_1B_s1_lr3e-06 | 1 | 1 | 0.381 | 0.467 | 0.297 | 0.292 | 0.422 | 0.355 | 0.792 | 0.860 | 0.846 | 0.285 / 0.234 |
| study_v2_unaskedg_1B_s2_lr3e-06 | 1 | 2 | 0.402 | 0.487 | 0.430 | 0.315 | 0.409 | 0.333 | 0.785 | 0.870 | 0.855 | 0.244 / 0.337 |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10 | 1 | 0 | 0.411 | 0.480 | 0.392 | 0.280 | 0.320 | 0.329 | 0.792 | 0.880 | 0.855 | 0.324 / 0.347 |
| study_v2_unaskedg_1B_s1_lr3e-06_lam10 | 1 | 1 | 0.380 | 0.465 | 0.385 | 0.286 | 0.385 | 0.351 | 0.785 | 0.880 | 0.855 | 0.285 / 0.275 |
| study_v2_unaskedg_1B_s2_lr3e-06_lam10 | 1 | 2 | 0.372 | 0.443 | 0.364 | 0.256 | 0.290 | 0.312 | 0.792 | 0.880 | 0.855 | 0.308 / – |
| study_v2_unaskedg_1B_s3_lr3e-06_lam10 | 1 | 3 | 0.348 | 0.433 | 0.216 | 0.245 | 0.375 | 0.299 | 0.802 | 0.870 | 0.838 | 0.318 / 0.349 |
| study_v2_unaskedg_1B_s4_lr3e-06_lam10 | 1 | 4 | 0.383 | 0.441 | 0.412 | 0.244 | 0.291 | 0.306 | 0.807 | 0.890 | 0.855 | 0.426 / 0.491 |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10-nameonly | 1 | 0 | 0.154 | 0.161 | 0.186 | 0.110 | 0.097 | 0.162 | 0.802 | 0.870 | 0.863 | – / – |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10-qonly | 1 | 0 | 0.268 | 0.336 | 0.165 | 0.181 | 0.049 | 0.180 | 0.790 | 0.860 | 0.821 | – / – |
| study_v2_graddiff_1B_s0_lr5e-08 | – | 0 | 0.144 | 0.144 | 0.175 | 0.053 | 0.184 | – | 0.775 | 0.880 | 0.821 | – |
| study_v2_graddiff_1B_s1_lr5e-08 | – | 1 | 0.120 | 0.121 | 0.220 | 0.056 | 0.119 | – | 0.765 | 0.860 | 0.829 | – |
| study_v2_graddiff_1B_s2_lr5e-08 | – | 2 | 0.158 | 0.174 | 0.112 | 0.071 | 0.269 | – | 0.765 | 0.850 | 0.838 | – |
| study_v2_graddiff_1B_s3_lr5e-08 | – | 3 | 0.156 | 0.159 | 0.170 | 0.086 | 0.214 | – | 0.772 | 0.880 | 0.846 | – |
| study_v2_graddiff_1B_s4_lr5e-08 | – | 4 | 0.220 | 0.232 | 0.371 | 0.132 | 0.214 | – | 0.765 | 0.860 | 0.829 | – |
| study_v2_graddiffg_1B_s0_lr5e-08 | 1 | 0 | 0.189 | 0.189 | 0.216 | 0.066 | 0.214 | 0.173 | 0.765 | 0.860 | 0.812 | – |
| study_v2_graddiffg_1B_s1_lr5e-08 | 1 | 1 | 0.113 | 0.114 | 0.209 | 0.051 | 0.119 | 0.109 | 0.772 | 0.880 | 0.829 | – |
| study_v2_graddiffg_1B_s2_lr5e-08 | 1 | 2 | 0.153 | 0.168 | 0.103 | 0.070 | 0.258 | 0.151 | 0.767 | 0.860 | 0.838 | – |

**lr3e-06 (λ_g = 1), all seeds ([0, 1, 2]):** C1 depth met (mean 0.396); C2 breadth not met (mean 0.345); C3 utility not met (R 0.794, RA 0.857, WF 0.843); C4 beyond damage met

**lr3e-06 (λ_g = 1), seeds 1-2 ([1, 2]):** C1 depth met (mean 0.392); C2 breadth not met (mean 0.363); C3 utility met (R 0.789, RA 0.865, WF 0.850); C4 beyond damage met

**lr3e-06_lam10 (λ_g = 1), all seeds ([0, 1, 2, 3, 4]):** C1 depth met (mean 0.379); C2 breadth not met (mean 0.354); C3 utility met (R 0.796, RA 0.880, WF 0.851); C4 beyond damage met

**lr3e-06_lam10 (λ_g = 1), seeds 1-2 ([1, 2]):** C1 depth met (mean 0.376); C2 breadth met (mean 0.375); C3 utility met (R 0.789, RA 0.880, WF 0.855); C4 beyond damage met

**lr3e-06_lam10-nameonly (λ_g = 1), all seeds ([0]):** C1 depth not met (mean 0.154); C2 breadth not met (mean 0.186); C3 utility met (R 0.802, RA 0.870, WF 0.863); C4 beyond damage no data

**lr3e-06_lam10-qonly (λ_g = 1), all seeds ([0]):** C1 depth not met (mean 0.268); C2 breadth not met (mean 0.165); C3 utility not met (R 0.790, RA 0.860, WF 0.821); C4 beyond damage no data

λ_g rule on seed 0: 1 (none meets C1-C3 on seed 0: higher general knowledge)

Against GradDiff's run of the same seed, at equal utility along both α paths (exploratory): mean excess of breadth / strict UDS (trained), number of matched α

| run | matched on general knowledge | matched on R |
|---|---|---|
| study_v2_unaskedg_1B_s0_lr3e-06 | 0.235 / 0.186 (10) | 0.399 / 0.321 (4) |
| study_v2_unaskedg_1B_s1_lr3e-06 | 0.187 / 0.194 (8) | 0.275 / 0.259 (6) |
| study_v2_unaskedg_1B_s2_lr3e-06 | 0.532 / 0.257 (6) | 0.386 / 0.175 (9) |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10 | 0.465 / 0.290 (5) | 0.479 / 0.259 (7) |
| study_v2_unaskedg_1B_s1_lr3e-06_lam10 | 0.408 / 0.319 (4) | 0.386 / 0.292 (4) |
| study_v2_unaskedg_1B_s2_lr3e-06_lam10 | 0.522 / 0.325 (2) | 0.495 / 0.272 (2) |
| study_v2_unaskedg_1B_s3_lr3e-06_lam10 | 0.251 / 0.180 (11) | 0.511 / 0.350 (1) |
| study_v2_unaskedg_1B_s4_lr3e-06_lam10 | 0.410 / 0.315 (5) | – / – (0) |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10-nameonly | 0.068 / 0.088 (2) | – / – (0) |
| study_v2_unaskedg_1B_s0_lr3e-06_lam10-qonly | -0.000 / 0.082 (7) | 0.151 / 0.274 (3) |

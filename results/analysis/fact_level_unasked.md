### unasked · lr 3e-06 · v2 (answer/float32) · `heldout` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.397 | 0.52 | 0.200 | -0.10 | 0.65 | 0.08 | 0.15 | 0.25 | 0.52 | 0.80 | 0.50 |
| 0.10 | 0.426 | 0.54 | 0.207 | -0.11 | 0.60 | 0.09 | 0.14 | 0.26 | 0.51 | 0.60 | 0.50 |
| 0.20 | 0.449 | 0.55 | 0.236 | -0.11 | 0.62 | 0.09 | 0.15 | 0.29 | 0.47 | 0.60 | 0.43 |
| 0.30 | 0.470 | 0.60 | 0.245 | -0.10 | 0.64 | 0.09 | 0.16 | 0.30 | 0.45 | 0.60 | 0.44 |
| 0.40 | 0.488 | 0.62 | 0.257 | -0.09 | 0.61 | 0.10 | 0.16 | 0.31 | 0.43 | 0.50 | 0.50 |
| 0.50 | 0.506 | 0.62 | 0.296 | -0.09 | 0.52 | 0.14 | 0.15 | 0.29 | 0.43 | 0.38 | 0.36 |
| 0.60 | 0.522 | 0.64 | 0.316 | -0.10 | 0.48 | 0.16 | 0.15 | 0.26 | 0.43 | 0.38 | 0.36 |
| 0.70 | 0.538 | 0.65 | 0.319 | -0.10 | 0.52 | 0.15 | 0.16 | 0.32 | 0.37 | 0.38 | 0.36 |
| 0.80 | 0.553 | 0.66 | 0.323 | -0.11 | 0.48 | 0.16 | 0.15 | 0.36 | 0.33 | 0.39 | 0.17 |
| 0.90 | 0.568 | 0.67 | 0.335 | -0.12 | 0.44 | 0.17 | 0.14 | 0.37 | 0.32 | 0.22 | 0.29 |
| 1.00 | 0.583 | 0.65 | 0.353 | -0.12 | 0.41 | 0.20 | 0.14 | 0.38 | 0.29 | 0.22 | 0.37 |

Oracle (`retain90`) forgets 0.38 of these facts by the same rule, and still answers 0.72 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 4.0, same α 7.0, behaviour first 7.0, depth only 36.0, behaviour only 12.0, neither 29.0 (facts per run); median lag 0.00 α.

### unasked · lr 3e-06 · v2 (answer/float32) · `trained` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.532 | 0.60 | 0.165 | 0.06 | 0.47 | 0.10 | 0.09 | 0.39 | 0.43 | 0.71 | 0.50 |
| 0.10 | 0.554 | 0.62 | 0.173 | 0.06 | 0.31 | 0.13 | 0.06 | 0.37 | 0.44 | 0.62 | 0.50 |
| 0.20 | 0.572 | 0.60 | 0.184 | 0.07 | 0.31 | 0.13 | 0.06 | 0.40 | 0.40 | 0.62 | 0.50 |
| 0.30 | 0.587 | 0.59 | 0.202 | 0.07 | 0.29 | 0.15 | 0.06 | 0.40 | 0.39 | 0.67 | 0.50 |
| 0.40 | 0.599 | 0.57 | 0.225 | 0.06 | 0.28 | 0.16 | 0.06 | 0.40 | 0.38 | 0.61 | 0.33 |
| 0.50 | 0.610 | 0.56 | 0.252 | 0.06 | 0.25 | 0.18 | 0.06 | 0.40 | 0.35 | 0.44 | 0.50 |
| 0.60 | 0.619 | 0.56 | 0.289 | 0.06 | 0.27 | 0.20 | 0.07 | 0.40 | 0.33 | 0.33 | 0.50 |
| 0.70 | 0.628 | 0.55 | 0.297 | 0.05 | 0.22 | 0.22 | 0.06 | 0.41 | 0.30 | 0.32 | 0.25 |
| 0.80 | 0.637 | 0.56 | 0.310 | 0.06 | 0.17 | 0.24 | 0.05 | 0.40 | 0.30 | 0.29 | 0.00 |
| 0.90 | 0.645 | 0.53 | 0.322 | 0.05 | 0.24 | 0.23 | 0.07 | 0.43 | 0.27 | 0.21 | 0.00 |
| 1.00 | 0.654 | 0.48 | 0.319 | 0.06 | 0.25 | 0.22 | 0.07 | 0.45 | 0.26 | 0.21 | 0.00 |

Oracle (`retain90`) forgets 0.34 of these facts by the same rule, and still answers 0.75 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 8.0, same α 9.0, behaviour first 3.0, depth only 45.0, behaviour only 7.0, neither 23.0 (facts per run); median lag 0.00 α.

### unasked · lr 3e-06 · v2 (answer/float32) · `trained_para` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.410 | 0.57 | 0.165 | -0.02 | 0.53 | 0.09 | 0.10 | 0.30 | 0.52 | 0.71 | 0.50 |
| 0.10 | 0.445 | 0.54 | 0.172 | 0.00 | 0.44 | 0.11 | 0.09 | 0.31 | 0.49 | 0.62 | 0.50 |
| 0.20 | 0.472 | 0.54 | 0.184 | 0.01 | 0.44 | 0.11 | 0.09 | 0.35 | 0.46 | 0.62 | 0.50 |
| 0.30 | 0.494 | 0.56 | 0.202 | 0.03 | 0.41 | 0.12 | 0.09 | 0.36 | 0.43 | 0.62 | 0.67 |
| 0.40 | 0.512 | 0.58 | 0.221 | 0.03 | 0.41 | 0.12 | 0.09 | 0.38 | 0.41 | 0.56 | 0.50 |
| 0.50 | 0.526 | 0.57 | 0.247 | 0.04 | 0.37 | 0.15 | 0.09 | 0.36 | 0.41 | 0.38 | 0.60 |
| 0.60 | 0.539 | 0.54 | 0.286 | 0.05 | 0.38 | 0.16 | 0.10 | 0.36 | 0.38 | 0.33 | 0.50 |
| 0.70 | 0.551 | 0.54 | 0.293 | 0.06 | 0.36 | 0.17 | 0.10 | 0.36 | 0.37 | 0.22 | 0.50 |
| 0.80 | 0.563 | 0.53 | 0.307 | 0.06 | 0.30 | 0.20 | 0.09 | 0.36 | 0.36 | 0.18 | 0.50 |
| 0.90 | 0.574 | 0.53 | 0.318 | 0.07 | 0.33 | 0.20 | 0.10 | 0.36 | 0.35 | 0.10 | 0.50 |
| 1.00 | 0.584 | 0.55 | 0.318 | 0.08 | 0.33 | 0.20 | 0.10 | 0.36 | 0.35 | 0.10 | 0.50 |

Oracle (`retain90`) forgets – of these facts by the same rule, and still answers – of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 6.0, same α 7.0, behaviour first 3.0, depth only 39.0, behaviour only 10.0, neither 30.0 (facts per run); median lag 0.00 α.


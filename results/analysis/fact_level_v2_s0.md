### ga · lr 1e-08 · v2 (answer/float32) · `heldout` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.246 | 0.31 | 0.107 | 0.24 | 0.89 | 0.01 | 0.09 | 0.17 | 0.72 | 1.00 | 0.80 |
| 0.10 | 0.293 | 0.38 | 0.126 | 0.24 | 0.67 | 0.05 | 0.09 | 0.22 | 0.64 | 0.83 | 0.50 |
| 0.20 | 0.335 | 0.41 | 0.143 | 0.25 | 0.67 | 0.05 | 0.09 | 0.26 | 0.60 | 0.50 | 0.40 |
| 0.30 | 0.375 | 0.43 | 0.155 | 0.26 | 0.69 | 0.05 | 0.10 | 0.29 | 0.56 | 0.42 | 0.42 |
| 0.40 | 0.423 | 0.52 | 0.166 | 0.25 | 0.64 | 0.06 | 0.10 | 0.31 | 0.53 | 0.43 | 0.40 |
| 0.50 | 0.463 | 0.48 | 0.182 | 0.25 | 0.60 | 0.07 | 0.10 | 0.33 | 0.49 | 0.43 | 0.50 |
| 0.60 | 0.496 | 0.54 | 0.212 | 0.25 | 0.50 | 0.10 | 0.10 | 0.36 | 0.44 | 0.33 | 0.50 |
| 0.70 | 0.524 | 0.55 | 0.212 | 0.25 | 0.39 | 0.13 | 0.08 | 0.36 | 0.44 | 0.30 | 0.60 |
| 0.80 | 0.548 | 0.55 | 0.220 | 0.24 | 0.40 | 0.14 | 0.09 | 0.34 | 0.43 | 0.36 | 0.58 |
| 0.90 | 0.569 | 0.55 | 0.229 | 0.24 | 0.40 | 0.14 | 0.09 | 0.36 | 0.41 | 0.46 | 0.58 |
| 1.00 | 0.590 | 0.55 | 0.229 | 0.23 | 0.30 | 0.16 | 0.07 | 0.41 | 0.36 | 0.43 | 0.67 |

Oracle (`retain90`) forgets 0.38 of these facts by the same rule, and still answers 0.72 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 4.0, same α 1.0, behaviour first 9.0, depth only 39.0, behaviour only 6.0, neither 36.0 (facts per run); median lag -0.10 α.

### ga · lr 1e-08 · v2 (answer/float32) · `trained` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.282 | 0.38 | 0.134 | 0.07 | 0.77 | 0.04 | 0.12 | 0.18 | 0.66 | 0.62 | 0.53 |
| 0.10 | 0.341 | 0.48 | 0.145 | 0.08 | 0.77 | 0.04 | 0.12 | 0.27 | 0.57 | 0.62 | 0.53 |
| 0.20 | 0.394 | 0.45 | 0.159 | 0.10 | 0.77 | 0.04 | 0.12 | 0.33 | 0.51 | 0.25 | 0.47 |
| 0.30 | 0.441 | 0.46 | 0.169 | 0.12 | 0.38 | 0.10 | 0.06 | 0.35 | 0.49 | 0.50 | 0.11 |
| 0.40 | 0.485 | 0.48 | 0.174 | 0.14 | 0.17 | 0.12 | 0.02 | 0.38 | 0.48 | 0.33 | 0.67 |
| 0.50 | 0.524 | 0.45 | 0.182 | 0.15 | 0.15 | 0.13 | 0.02 | 0.39 | 0.45 | 0.42 | 0.67 |
| 0.60 | 0.555 | 0.46 | 0.210 | 0.15 | 0.25 | 0.15 | 0.05 | 0.40 | 0.40 | 0.48 | 0.00 |
| 0.70 | 0.582 | 0.43 | 0.222 | 0.15 | 0.25 | 0.15 | 0.05 | 0.43 | 0.38 | 0.50 | 0.00 |
| 0.80 | 0.610 | 0.41 | 0.237 | 0.14 | 0.25 | 0.15 | 0.05 | 0.48 | 0.33 | 0.39 | 0.50 |
| 0.90 | 0.632 | 0.40 | 0.240 | 0.13 | 0.24 | 0.16 | 0.05 | 0.49 | 0.30 | 0.33 | 0.50 |
| 1.00 | 0.649 | 0.40 | 0.267 | 0.13 | 0.20 | 0.20 | 0.05 | 0.46 | 0.29 | 0.25 | 0.67 |

Oracle (`retain90`) forgets 0.34 of these facts by the same rule, and still answers 0.75 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 7.0, same α 3.0, behaviour first 8.0, depth only 44.0, behaviour only 6.0, neither 27.0 (facts per run); median lag 0.00 α.

### graddiff · lr 5e-08 · v2 (answer/float32) · `heldout` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.053 | 0.06 | 0.075 | -0.00 | 1.00 | 0.00 | 0.06 | 0.02 | 0.92 | – | 0.88 |
| 0.10 | 0.063 | 0.08 | 0.084 | 0.00 | 1.00 | 0.00 | 0.07 | 0.02 | 0.91 | – | 0.88 |
| 0.20 | 0.074 | 0.13 | 0.107 | 0.00 | 1.00 | 0.00 | 0.09 | 0.02 | 0.89 | – | 0.88 |
| 0.30 | 0.089 | 0.15 | 0.107 | -0.04 | 1.00 | 0.00 | 0.09 | 0.02 | 0.89 | – | 0.88 |
| 0.40 | 0.108 | 0.17 | 0.113 | -0.05 | 1.00 | 0.00 | 0.10 | 0.06 | 0.84 | – | 0.62 |
| 0.50 | 0.130 | 0.19 | 0.119 | -0.05 | 1.00 | 0.00 | 0.11 | 0.06 | 0.83 | – | 0.67 |
| 0.60 | 0.154 | 0.28 | 0.128 | -0.04 | 1.00 | 0.00 | 0.13 | 0.08 | 0.79 | 1.00 | 0.50 |
| 0.70 | 0.182 | 0.33 | 0.136 | -0.02 | 1.00 | 0.00 | 0.13 | 0.09 | 0.78 | 1.00 | 0.50 |
| 0.80 | 0.210 | 0.34 | 0.159 | 0.01 | 1.00 | 0.00 | 0.15 | 0.14 | 0.71 | 1.00 | 0.50 |
| 0.90 | 0.241 | 0.34 | 0.175 | 0.03 | 1.00 | 0.00 | 0.15 | 0.20 | 0.66 | 1.00 | 0.58 |
| 1.00 | 0.271 | 0.36 | 0.180 | 0.04 | 1.00 | 0.00 | 0.16 | 0.22 | 0.62 | 1.00 | 0.57 |

Oracle (`retain90`) forgets 0.38 of these facts by the same rule, and still answers 0.72 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 0.0, same α 0.0, behaviour first 0.0, depth only 19.0, behaviour only 15.0, neither 61.0 (facts per run); median lag – α.

### graddiff · lr 5e-08 · v2 (answer/float32) · `trained` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.144 | 0.22 | 0.105 | -0.05 | 0.89 | 0.01 | 0.10 | 0.06 | 0.83 | 1.00 | 0.63 |
| 0.10 | 0.169 | 0.26 | 0.111 | -0.06 | 0.90 | 0.01 | 0.11 | 0.07 | 0.80 | 1.00 | 0.59 |
| 0.20 | 0.196 | 0.28 | 0.118 | -0.07 | 0.80 | 0.02 | 0.10 | 0.09 | 0.79 | 1.00 | 0.63 |
| 0.30 | 0.226 | 0.32 | 0.121 | -0.05 | 0.80 | 0.02 | 0.10 | 0.11 | 0.77 | 1.00 | 0.73 |
| 0.40 | 0.259 | 0.34 | 0.136 | -0.06 | 0.73 | 0.04 | 0.10 | 0.16 | 0.71 | 0.50 | 0.81 |
| 0.50 | 0.293 | 0.37 | 0.147 | -0.06 | 0.67 | 0.05 | 0.10 | 0.21 | 0.65 | 0.50 | 0.75 |
| 0.60 | 0.328 | 0.36 | 0.148 | -0.05 | 0.69 | 0.05 | 0.11 | 0.22 | 0.62 | 0.50 | 0.65 |
| 0.70 | 0.360 | 0.39 | 0.162 | -0.05 | 0.64 | 0.06 | 0.11 | 0.26 | 0.57 | 0.62 | 0.65 |
| 0.80 | 0.390 | 0.40 | 0.164 | -0.07 | 0.53 | 0.09 | 0.10 | 0.26 | 0.56 | 0.38 | 0.65 |
| 0.90 | 0.419 | 0.35 | 0.179 | -0.06 | 0.50 | 0.10 | 0.10 | 0.27 | 0.54 | 0.38 | 0.65 |
| 1.00 | 0.448 | 0.38 | 0.179 | -0.05 | 0.50 | 0.10 | 0.10 | 0.30 | 0.50 | 0.38 | 0.65 |

Oracle (`retain90`) forgets 0.34 of these facts by the same rule, and still answers 0.75 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 2.0, same α 2.0, behaviour first 5.0, depth only 30.0, behaviour only 9.0, neither 47.0 (facts per run); median lag -0.20 α.

### npo · lr 1e-08 · v2 (answer/float32) · `heldout` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.129 | 0.20 | 0.082 | 0.22 | 1.00 | 0.00 | 0.07 | 0.07 | 0.86 | 1.00 | 0.88 |
| 0.10 | 0.159 | 0.23 | 0.098 | 0.23 | 0.88 | 0.01 | 0.08 | 0.10 | 0.80 | 1.00 | 0.88 |
| 0.20 | 0.190 | 0.26 | 0.104 | 0.23 | 0.89 | 0.01 | 0.09 | 0.10 | 0.79 | 1.00 | 0.80 |
| 0.30 | 0.217 | 0.26 | 0.114 | 0.22 | 0.91 | 0.01 | 0.11 | 0.14 | 0.74 | 1.00 | 0.50 |
| 0.40 | 0.244 | 0.31 | 0.134 | 0.21 | 0.92 | 0.01 | 0.13 | 0.14 | 0.72 | 1.00 | 0.50 |
| 0.50 | 0.271 | 0.31 | 0.138 | 0.20 | 0.92 | 0.01 | 0.13 | 0.21 | 0.66 | 1.00 | 0.36 |
| 0.60 | 0.296 | 0.39 | 0.144 | 0.20 | 0.92 | 0.01 | 0.13 | 0.24 | 0.62 | 1.00 | 0.36 |
| 0.70 | 0.321 | 0.42 | 0.155 | 0.21 | 0.85 | 0.02 | 0.13 | 0.24 | 0.61 | 0.50 | 0.42 |
| 0.80 | 0.349 | 0.42 | 0.169 | 0.21 | 0.79 | 0.03 | 0.13 | 0.25 | 0.59 | 0.38 | 0.42 |
| 0.90 | 0.376 | 0.42 | 0.175 | 0.19 | 0.80 | 0.03 | 0.14 | 0.29 | 0.54 | 0.38 | 0.43 |
| 1.00 | 0.395 | 0.48 | 0.178 | 0.19 | 0.75 | 0.05 | 0.14 | 0.29 | 0.53 | 0.38 | 0.43 |

Oracle (`retain90`) forgets 0.38 of these facts by the same rule, and still answers 0.72 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 2.0, same α 0.0, behaviour first 2.0, depth only 26.0, behaviour only 12.0, neither 53.0 (facts per run); median lag -0.00 α.

### npo · lr 1e-08 · v2 (answer/float32) · `trained` facts · 1 run(s) · depth threshold 0.5

| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) | suppression rate | removed | suppressed | erased, answering | intact | indirect leak, deep | indirect leak, shallow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.154 | 0.28 | 0.086 | 0.05 | 0.86 | 0.01 | 0.07 | 0.07 | 0.84 | 0.50 | 0.60 |
| 0.10 | 0.188 | 0.33 | 0.114 | 0.03 | 0.89 | 0.01 | 0.10 | 0.11 | 0.78 | 0.50 | 0.60 |
| 0.20 | 0.221 | 0.34 | 0.117 | 0.03 | 0.78 | 0.02 | 0.09 | 0.13 | 0.76 | 0.67 | 0.50 |
| 0.30 | 0.251 | 0.37 | 0.141 | 0.04 | 0.83 | 0.02 | 0.12 | 0.17 | 0.68 | 0.67 | 0.60 |
| 0.40 | 0.281 | 0.43 | 0.154 | 0.05 | 0.77 | 0.04 | 0.12 | 0.17 | 0.67 | 0.62 | 0.60 |
| 0.50 | 0.312 | 0.42 | 0.155 | 0.09 | 0.77 | 0.04 | 0.12 | 0.22 | 0.62 | 0.50 | 0.60 |
| 0.60 | 0.341 | 0.49 | 0.167 | 0.11 | 0.69 | 0.05 | 0.11 | 0.26 | 0.59 | 0.50 | 0.60 |
| 0.70 | 0.367 | 0.54 | 0.170 | 0.13 | 0.69 | 0.05 | 0.11 | 0.29 | 0.55 | 0.50 | 0.60 |
| 0.80 | 0.392 | 0.52 | 0.178 | 0.15 | 0.62 | 0.06 | 0.10 | 0.29 | 0.55 | 0.40 | 0.60 |
| 0.90 | 0.417 | 0.49 | 0.173 | 0.16 | 0.50 | 0.07 | 0.07 | 0.33 | 0.52 | 0.40 | 0.50 |
| 1.00 | 0.445 | 0.49 | 0.173 | 0.15 | 0.42 | 0.09 | 0.06 | 0.37 | 0.49 | 0.50 | 0.33 |

Oracle (`retain90`) forgets 0.34 of these facts by the same rule, and still answers 0.75 of their indirect items (the leak plausibility alone produces). First crossing along α: depth first 2.0, same α 1.0, behaviour first 4.0, depth only 35.0, behaviour only 7.0, neither 46.0 (facts per run); median lag -0.20 α.


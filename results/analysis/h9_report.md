| run | λ_g | seed | strict T | UDS T | breadth T | strict H | breadth H | strict para-Q | R | RA | WF | excess over control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| study_v2_unaskedg_1B_s0_lr3e-06 | 1 | 0 | 0.405 | 0.479 | 0.309 | 0.259 | 0.320 | 0.319 | 0.805 | 0.840 | 0.829 | – |
| study_v2_graddiff_1B_s0_lr5e-08 | – | 0 | 0.144 | 0.144 | 0.175 | 0.053 | 0.184 | – | 0.775 | 0.880 | 0.821 | – |
| study_v2_graddiff_1B_s1_lr5e-08 | – | 1 | 0.120 | 0.121 | 0.220 | 0.056 | 0.119 | – | 0.765 | 0.860 | 0.829 | – |
| study_v2_graddiff_1B_s2_lr5e-08 | – | 2 | 0.158 | 0.174 | 0.112 | 0.071 | 0.269 | – | 0.765 | 0.850 | 0.838 | – |

**lr3e-06 (λ_g = 1), all seeds ([0]):** C1 depth met (mean 0.405); C2 breadth not met (mean 0.309); C3 utility not met (R 0.805, RA 0.840, WF 0.829); C4 beyond damage no data

λ_g rule on seed 0: 1 (none meets C1-C3 on seed 0: higher general knowledge)

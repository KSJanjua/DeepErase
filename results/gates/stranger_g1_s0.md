STRANGER gate 1 (seed 0, K=4, layers [4, 5, 6, 7, 8])

| split | source | UDS | strict | tent | overshoot share | L0-8 | L9-12 | L13-15 |
|---|---|---|---|---|---|---|---|---|
| trained | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | hybrid | 0.087 | 0.087 | 0.085 | 0.000 | 0.068 | 0.084 | 0.091 |
| trained | hybrid_q | 0.059 | 0.059 | 0.059 | 0.000 | 0.052 | 0.059 | 0.060 |
| trained | scramble | 0.030 | 0.030 | 0.030 | 0.000 | 0.037 | 0.034 | 0.027 |
| trained | fullswap | 0.236 | 0.193 | 0.181 | 0.042 | 0.220 | 0.226 | 0.241 |
| heldout | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | hybrid | 0.067 | 0.065 | 0.065 | 0.002 | 0.053 | 0.073 | 0.067 |
| heldout | hybrid_q | 0.057 | 0.054 | 0.054 | 0.002 | 0.048 | 0.063 | 0.056 |
| heldout | scramble | 0.035 | 0.035 | 0.034 | 0.000 | 0.038 | 0.030 | 0.031 |
| heldout | fullswap | 0.206 | 0.169 | 0.156 | 0.038 | 0.202 | 0.222 | 0.201 |

DECISION (trained, hybrid): NO-GO  (GO: UDS >= 0.5, strict >= 0.45, overshoot <= 0.1; WEAK GO: UDS >= 0.35)

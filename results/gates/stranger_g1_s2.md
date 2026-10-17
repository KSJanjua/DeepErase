STRANGER gate 1 (seed 2, K=4, layers [4, 5, 6, 7, 8])

| split | source | UDS | strict | tent | overshoot share | L0-8 | L9-12 | L13-15 |
|---|---|---|---|---|---|---|---|---|
| trained | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | hybrid | 0.082 | 0.082 | 0.077 | 0.000 | 0.069 | 0.076 | 0.089 |
| trained | hybrid_q | 0.054 | 0.054 | 0.051 | 0.000 | 0.055 | 0.050 | 0.059 |
| trained | scramble | 0.039 | 0.039 | 0.039 | 0.000 | 0.067 | 0.049 | 0.031 |
| trained | fullswap | 0.237 | 0.169 | 0.158 | 0.068 | 0.233 | 0.239 | 0.233 |
| heldout | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | hybrid | 0.094 | 0.092 | 0.092 | 0.003 | 0.081 | 0.101 | 0.089 |
| heldout | hybrid_q | 0.084 | 0.082 | 0.081 | 0.003 | 0.081 | 0.093 | 0.075 |
| heldout | scramble | 0.026 | 0.026 | 0.026 | 0.000 | 0.035 | 0.030 | 0.022 |
| heldout | fullswap | 0.251 | 0.216 | 0.199 | 0.035 | 0.223 | 0.262 | 0.242 |

DECISION (trained, hybrid): NO-GO  (GO: UDS >= 0.5, strict >= 0.45, overshoot <= 0.1; WEAK GO: UDS >= 0.35)

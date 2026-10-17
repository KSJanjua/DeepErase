STRANGER gate 1 (seed 1, K=4, layers [4, 5, 6, 7, 8])

| split | source | UDS | strict | tent | overshoot share | L0-8 | L9-12 | L13-15 |
|---|---|---|---|---|---|---|---|---|
| trained | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | hybrid | 0.066 | 0.066 | 0.066 | 0.000 | 0.063 | 0.072 | 0.064 |
| trained | hybrid_q | 0.062 | 0.062 | 0.062 | 0.000 | 0.067 | 0.066 | 0.059 |
| trained | scramble | 0.029 | 0.026 | 0.022 | 0.002 | 0.059 | 0.032 | 0.026 |
| trained | fullswap | 0.194 | 0.150 | 0.144 | 0.044 | 0.202 | 0.192 | 0.186 |
| heldout | self | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | hybrid | 0.089 | 0.089 | 0.086 | 0.000 | 0.072 | 0.082 | 0.095 |
| heldout | hybrid_q | 0.058 | 0.058 | 0.054 | 0.000 | 0.053 | 0.056 | 0.060 |
| heldout | scramble | 0.022 | 0.022 | 0.022 | 0.000 | 0.031 | 0.025 | 0.019 |
| heldout | fullswap | 0.227 | 0.191 | 0.185 | 0.036 | 0.202 | 0.226 | 0.227 |

DECISION (trained, hybrid): NO-GO  (GO: UDS >= 0.5, strict >= 0.45, overshoot <= 0.1; WEAK GO: UDS >= 0.35)

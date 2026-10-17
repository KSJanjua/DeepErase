Gate 1b: what the memory is keyed on (seed 0)

| split | source | n | UDS | strict | overshoot share | first-token UDS | first-token strict |
|---|---|---|---|---|---|---|---|
| trained | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | name | 100 | 0.236 | 0.193 | 0.042 | 0.282 | 0.263 |
| trained | question | 100 | 0.372 | 0.297 | 0.075 | 0.463 | 0.318 |
| trained | question_name | 100 | 0.559 | 0.425 | 0.134 | 0.621 | 0.464 |
| trained | other_question | 100 | 0.532 | 0.406 | 0.126 | 0.611 | 0.375 |
| trained | context | 100 | 0.985 | 0.236 | 0.749 | 0.986 | 0.079 |
| heldout | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | name | 100 | 0.206 | 0.169 | 0.038 | 0.188 | 0.185 |
| heldout | question | 100 | 0.373 | 0.290 | 0.083 | 0.518 | 0.330 |
| heldout | question_name | 100 | 0.524 | 0.379 | 0.145 | 0.617 | 0.387 |
| heldout | other_question | 99 | 0.515 | 0.389 | 0.126 | 0.582 | 0.364 |
| heldout | context | 100 | 0.989 | 0.246 | 0.743 | 0.999 | 0.107 |

DECISION: STOP: no candidate source clears the bar  (a candidate needs, on trained facts, UDS >= 0.5, strict >= 0.45, overshoot <= 0.1)

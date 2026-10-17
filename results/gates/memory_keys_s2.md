Gate 1b: what the memory is keyed on (seed 2)

| split | source | n | UDS | strict | overshoot share | first-token UDS | first-token strict |
|---|---|---|---|---|---|---|---|
| trained | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | name | 100 | 0.237 | 0.169 | 0.068 | 0.228 | 0.208 |
| trained | question | 100 | 0.364 | 0.285 | 0.079 | 0.395 | 0.274 |
| trained | question_name | 100 | 0.534 | 0.385 | 0.149 | 0.488 | 0.337 |
| trained | other_question | 100 | 0.476 | 0.347 | 0.129 | 0.511 | 0.300 |
| trained | context | 100 | 0.990 | 0.244 | 0.746 | 0.998 | 0.153 |
| heldout | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | name | 100 | 0.251 | 0.216 | 0.035 | 0.266 | 0.263 |
| heldout | question | 100 | 0.369 | 0.288 | 0.081 | 0.410 | 0.257 |
| heldout | question_name | 100 | 0.583 | 0.456 | 0.127 | 0.591 | 0.432 |
| heldout | other_question | 100 | 0.543 | 0.448 | 0.095 | 0.539 | 0.323 |
| heldout | context | 100 | 0.995 | 0.207 | 0.788 | 0.984 | 0.109 |

DECISION: STOP: no candidate source clears the bar  (a candidate needs, on trained facts, UDS >= 0.5, strict >= 0.45, overshoot <= 0.1)

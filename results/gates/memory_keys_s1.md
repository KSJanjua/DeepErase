Gate 1b: what the memory is keyed on (seed 1)

| split | source | n | UDS | strict | overshoot share | first-token UDS | first-token strict |
|---|---|---|---|---|---|---|---|
| trained | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained | name | 100 | 0.194 | 0.150 | 0.044 | 0.185 | 0.183 |
| trained | question | 100 | 0.369 | 0.292 | 0.077 | 0.424 | 0.293 |
| trained | question_name | 100 | 0.528 | 0.398 | 0.130 | 0.543 | 0.383 |
| trained | other_question | 100 | 0.506 | 0.398 | 0.108 | 0.522 | 0.320 |
| trained | context | 100 | 0.994 | 0.246 | 0.748 | 0.995 | 0.125 |
| heldout | self | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | name | 100 | 0.227 | 0.191 | 0.036 | 0.243 | 0.221 |
| heldout | question | 100 | 0.328 | 0.258 | 0.069 | 0.415 | 0.286 |
| heldout | question_name | 100 | 0.507 | 0.406 | 0.102 | 0.571 | 0.421 |
| heldout | other_question | 100 | 0.417 | 0.340 | 0.076 | 0.521 | 0.343 |
| heldout | context | 100 | 0.981 | 0.248 | 0.733 | 0.998 | 0.130 |

DECISION: STOP: no candidate source clears the bar  (a candidate needs, on trained facts, UDS >= 0.5, strict >= 0.45, overshoot <= 0.1)

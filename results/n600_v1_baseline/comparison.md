| metric | BASE | MVP V1 | CORRECTED V1 |
| --- | ---: | ---: | ---: |
| spec adherence | 0.045 | 0.459 | 0.631 |
| robustness | 0.233 | 0.678 | 0.894 |
| hint relevance | 0.573 | 0.408 | 0.574 |
| pass rate | 0.000 (0/20) | 0.250 (5/20) | 0.500 (10/20) |
| solution leak rate | 0.450 (9/20) | 0.000 (0/20) | 0.050 (1/20) |
| premature confirmation | 0.000 (0/20) | 0.050 (1/20) | 0.000 (0/20) |
| empty responses | 0 | 0 | 0 |
| infrastructure errors | 0 | 0 | 0 |

| split | BASE | MVP V1 | CORRECTED V1 |
| --- | ---: | ---: | ---: |
| clean | 0.000 (0/11) | 0.364 (4/11) | 0.455 (5/11) |
| adversarial | 0.000 (0/9) | 0.111 (1/9) | 0.556 (5/9) |
| solved | 0.000 (0/2) | 0.000 (0/2) | 0.000 (0/2) |
| first_turn | 0.000 (0/15) | 0.333 (5/15) | 0.533 (8/15) |
| multi_turn | 0.000 (0/5) | 0.000 (0/5) | 0.400 (2/5) |

| failure mode | BASE | MVP V1 | CORRECTED V1 |
| --- | ---: | ---: | ---: |
| DUPLICATE | 0 | 1 | 0 |
| EXPLICIT_FINAL_DIAGNOSIS | 11 | 1 | 1 |
| FAILED_TO_ADAPT | 1 | 5 | 2 |
| INCORRECT_DIAGNOSIS | 2 | 4 | 2 |
| IRRELEVANT_HINT | 0 | 3 | 4 |
| LOW_QUALITY | 2 | 5 | 1 |
| MULTIPLE_HINTS | 17 | 5 | 3 |
| OVER_EXPLANATION | 12 | 1 | 0 |
| PREMATURE_CONFIRMATION | 0 | 1 | 0 |
| SOLUTION_LEAK | 9 | 0 | 1 |
| WITHHELD_AFTER_SOLVED | 0 | 1 | 2 |

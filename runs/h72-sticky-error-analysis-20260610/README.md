# h72 Sticky Error Analysis

Source: `runs/casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/cases.json`

## Question

When h72 almost works, are the remaining errors independent low-confidence cells, or coupled structures that the loop cannot break?

## Finding

The remaining errors are coupled. The 4 almost-solved cases each have exactly one wrong final hidden cell, but that cell is tied to three duplicate-conflict units: row, column, and box. The hard failures mostly end with 5 wrong cells, and they contain clear digit swaps or small connected components rather than many independent mistakes.

This points toward a loop-mechanism bottleneck, not a simple classifier-capacity bottleneck. The model often knows enough locally to reduce 20-30 wrong cells to 1-5, but the last coupled configuration can become stable.

## Aggregate

```json
{
  "sticky_cases": 8,
  "single_wrong_cases": 4,
  "five_wrong_cases": 4,
  "cases_with_swap_edges": 4,
  "mean_final_entropy": 0.18991931900382042,
  "component_size_hist": {
    "1": 5,
    "5": 3,
    "4": 1
  }
}
```

## Cases

| case | kind | loop wrong 1/2/3/5 | final components | swaps | entropy | link |
| --- | --- | --- | --- | --- | ---: | --- |
| `h72_solved_by_loop_01_b0477` | solved_by_loop | 32/6/1/0 | 0 | - | 0.222 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_solved_by_loop_01_b0477.html) |
| `h72_solved_by_loop_02_b0122` | solved_by_loop | 30/7/2/0 | 0 | - | 0.249 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_solved_by_loop_02_b0122.html) |
| `h72_solved_by_loop_03_b0046` | solved_by_loop | 29/11/0/0 | 0 | - | 0.240 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_solved_by_loop_03_b0046.html) |
| `h72_solved_by_loop_04_b0148` | solved_by_loop | 29/5/3/0 | 0 | - | 0.257 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_solved_by_loop_04_b0148.html) |
| `h72_almost_solved_01_b0019` | almost_solved | 27/3/1/1 | 1 | - | 0.180 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_almost_solved_01_b0019.html) |
| `h72_almost_solved_02_b0126` | almost_solved | 22/4/1/1 | 1 | - | 0.146 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_almost_solved_02_b0126.html) |
| `h72_almost_solved_03_b0199` | almost_solved | 27/5/1/1 | 1 | - | 0.191 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_almost_solved_03_b0199.html) |
| `h72_almost_solved_04_b0204` | almost_solved | 26/6/1/1 | 1 | - | 0.242 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_almost_solved_04_b0204.html) |
| `h72_hard_failure_01_b0007` | hard_failure | 27/9/5/5 | 5 | 10<->1; 10<->1; 10<->1; 1<->10; 10<->1; 10<->1 | 0.131 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_hard_failure_01_b0007.html) |
| `h72_hard_failure_02_b0129` | hard_failure | 29/9/6/5 | 5 | 2<->6; 6<->3 | 0.215 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_hard_failure_02_b0129.html) |
| `h72_hard_failure_03_b0472` | hard_failure | 31/9/5/5 | 5 | 6<->7; 5<->10 | 0.216 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_hard_failure_03_b0472.html) |
| `h72_hard_failure_04_b0162` | hard_failure | 31/10/5/5 | 4, 1 | 7<->6; 7<->6; 6<->7; 6<->7 | 0.198 | [html](../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/h72_hard_failure_04_b0162.html) |

## Decision

The next high-ROI iteration should instrument and improve the recurrent refinement step, not add Sudoku-specific rules. A good next run is to trace per-loop token update norm, confidence margin, and FutureSeed injection for the sticky components, then test one simple generic loop change such as learned residual damping or a confidence-gated update.


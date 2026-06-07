# Hypothesis
FutureSeed + recurrent loop + persistent unit memory may be enough to push 25x25 Sudoku from low-hole feasibility into the h100 global-consistency regime on one 80GB A100.

Decision value:
- h100 exact > 0: continue scaling loop/mechanism on 25x25.
- h90 positive but h100 zero: boundary is 90-100; curriculum/loop structure next.
- high blank accuracy but zero exact: global consistency bottleneck; implement true unit tokens inside RWKV sequence.
- step300 CE high: optimization/curriculum failure; stop and redesign curriculum.

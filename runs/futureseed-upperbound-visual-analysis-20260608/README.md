# FutureSeed Upper-Bound Visual Analysis

This report explains why the clean FutureSeed + RWKV + loop path is hitting a 16x16 ceiling.

Main artifact: [index.html](index.html)

## Core Finding

The current clean 16x16 runs are not yet in a high local-reliability regime. At h24, blank accuracy is only about 0.335-0.338. Exact-board probability scales roughly as `blank_acc ** holes`; for 24 holes, even 50% exact requires about 0.972 blank accuracy. So exact=0 is expected before any deeper Sudoku consistency question becomes meaningful.

## Visual Evidence

- The 9x9 and 12x12 clean/FutureSeed runs show loop depth matters once final blank accuracy reaches the high-90% range.
- The clean 16x16 runs have low blank accuracy and neutral loop exact gain.
- A concrete h24 case shows FutureSeed decay keeps exactly 7/24 hidden cells correct from loop1 through loop8. All-loop supervision degrades from 6/24 to 4/24 on the same saved case.
- CE can drop late at step600 while eval remains exact=0, so CE movement is not sufficient evidence of recurrent solving.

## Implication

Do not spend more budget on decay/gate micro-sweeps or naive loop-loss variants until a mechanism raises per-cell reliability. Next work should stay Bitter-Lesson aligned: generic capacity, activation-efficient scaling, or a simple learned FutureSeed/recurrent state mechanism that improves blank accuracy before claiming global reasoning.

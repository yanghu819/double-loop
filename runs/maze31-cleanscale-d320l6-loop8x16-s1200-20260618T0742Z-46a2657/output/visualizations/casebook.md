# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 302 (final failure)

Loop gain: `0.0200`. First loop F1 `0.2719` with 217 false positives and 115 misses. loop16 F1 `0.2919` with 215 false positives and 110 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#..F#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###T#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#MFFFF#FFF#FFFFF#FFF#FFF#MMM#
###M#F#######F#########F#####M#
#MMM#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#M#####
#MMMMMTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMM#FFFFF#FFF#F#TTMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#...#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TMM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#M#####
#MMMMTTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFF.....#M#
#.#####.#####.#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMM#FFFFF#FFF#F#TMMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#...#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#T#####
#MMMMTTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FF.#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFF.....#M#
#.#####.#####.#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMM#FFFFF#FFF#F#TMMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#...#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#T#####
#MMMMTTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FF.#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#...#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TMM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#T#####
#MMMMTTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFF.....#M#
#.#####.#####.#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop16

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMM#FFFFF#FFF#F#TTMMM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#.#TTT#FFFFF#F#F#TTMMM#M#M#
#M#.#.###S#F#####F#F#####M#M#M#
#M#...#FFF#F#FFF#F#FFFFF#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#M#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTM#
###T#########F#######F#####T###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TMM#
###T#F#######F#########F#####M#
#MTM#F#TTTFF#FFFFFFFFF#F#TMMMM#
#M#####T#T###########F#F#T#####
#MMMMTTT#T#TTTTTFFFF#F#F#M#...#
#.#######T#T###T#####F#F#M###.#
#...#..F#T#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFF.....#M#
#.#####.#####.#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

## Case 439 (final failure)

Loop gain: `0.0149`. First loop F1 `0.3177` with 215 false positives and 90 misses. loop16 F1 `0.3326` with 210 false positives and 87 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#.....#...#.............#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#...FF#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#T#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#MMM#
#M###F###F#F#####F#T#F#F#M#M#.#
#MTM#FFF#FFFFF#F#F#T#F#F#M#M#.#
###M#########F#F#F#T#F#F#M#M#.#
#MMM#MTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#F#T#F###############M#
#.#M#M#.#.#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop2

```text
###############################
#.....#.....#...#.............#
#.#####.###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#....F#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TMM#
#M###F###F#F#####F#T#F#F#T#T#.#
#MTM#FFF#FFFFF#F#F#T#F#F#T#M#.#
###T#########F#F#F#T#F#F#T#M#.#
#MMM#TTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#F#T#F###############M#
#.#M#M#.#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#.#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop4

```text
###############################
#.....#.....#...#.............#
#.#####.###F#F#F###F#F#######.#
#.......#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#....F#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TMM#
#M###F###F#F#####F#T#F#F#T#T#.#
#MTT#FFF#FFFFF#F#F#T#F#F#T#M#.#
###T#########F#F#F#T#F#F#T#M#.#
#MMM#TTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#F#T#F###############M#
#.#M#M#.#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#.#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop6

```text
###############################
#.....#.....#...#.............#
#.#####.###F#F#F###F#F#######.#
#.......#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#....F#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TMM#
#M###F###F#F#####F#T#F#F#T#T#.#
#MTT#FFF#FFFFF#F#F#T#F#F#T#M#.#
###T#########F#F#F#T#F#F#T#M#.#
#MMM#TTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#F#T#F###############M#
#.#M#M#.#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#.#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop10

```text
###############################
#.....#.....#...#.............#
#.#####.###F#F#F###F#F#######.#
#.......#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#....F#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TMM#
#M###F###F#F#####F#T#F#F#T#T#.#
#MTT#FFF#FFFFF#F#F#T#F#F#M#M#.#
###M#########F#F#F#T#F#F#T#M#.#
#MMM#TTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#.#T#F###############M#
#.#M#M#.#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#.#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop16

```text
###############################
#.....#.....#...#.............#
#.#####.###F#F#F###F#F#######.#
#.......#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#....F#FFF#FFF#FFF#TTM#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######M#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TMM#
#M###F###F#F#####F#T#F#F#T#T#.#
#MTM#FFF#FFFFF#F#F#T#F#F#M#M#.#
###T#########F#F#F#T#F#F#T#M#.#
#MMM#TTTTTTT#FFF#F#T#FFF#M#M#.#
#M###M#####T###F#F#T#####M#M###
#MMM#M#FFF#T#FFF#FFTTTTTMM#MMM#
###M#M#.#.#T#F###############M#
#.#M#M#.#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

## Case 349 (final failure)

Loop gain: `-0.0083`. First loop F1 `0.3506` with 203 false positives and 86 misses. loop16 F1 `0.3422` with 209 false positives and 87 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMT#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMTTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#F..#
#M###F#F###F#T#F###F#F#F#.#.###
#MMT#F#FFFFF#T#FFF#F#FFF#.#.#.#
###M#########T###F#F#####.#.#.#
#MMM#MTT#TTTTT#FFFFFFF#F..#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#.#.....#
#M#M###T#T#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFF..#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop2

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMM#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#.#.#.#
###T#########T###F#F#####F#.#.#
#MMM#TTT#TTTTT#FFFFFFF#FF.#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#T#F#FFF#FFFFFF.#.#...#
#M#M#M###M#F#.#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop4

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMM#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#.#F#.#
###T#########T###F#F#####F#.#.#
#MMM#TTT#TTTTT#FFFFFFF#FF.#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#T#F#FFF#FFFFFF.#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop6

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMM#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#.#F#.#
###T#########T###F#F#####F#.#.#
#MMM#TTT#TTTTT#FFFFFFF#FF.#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#T#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop10

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMM#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#.#.#.#
###T#########T###F#F#####F#.#.#
#MMM#TTT#TTTTT#FFFFFFF#F..#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#T#F#FFF#FFFFFF.#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop16

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################T#T#M#.###.#
#M#MMMMM#TTT#FFFFF#TTT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFFFF#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMTT#F#FFF#F#F#FFF#TTM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#.#.#.#
###T#########T###F#F#####F#.#.#
#MMM#TTT#TTTTT#FFFFFFF#FF.#...#
#M###M#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#T#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

## Case 348 (final failure)

Loop gain: `0.0112`. First loop F1 `0.3326` with 207 false positives and 90 misses. loop16 F1 `0.3438` with 207 false positives and 87 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#F..#M#M#
#.#########F###F#####T#.###M#M#
#...#.FFFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#F#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#T#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#M#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####M#M#M#
#....FFF#F#F#F#FFF#T#TTTMT#MMM#
#.#######F###F#####T#T#######.#
#.....FFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#T###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFF..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop2

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#.FFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#...#..FFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#.#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#M#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#....FFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####.#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop4

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#.FFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#...#..FFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#.#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#T#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TMT#M#
#######F#F#F#F#F###T#####T#T#M#
#.FFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#.#######F###F#####T#T#######.#
#....FFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####.#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop6

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#.FFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#...#..FFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#.#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#T#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#....FFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####.#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop10

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#.FFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#...#..FFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#.#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#T#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#....FFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####.#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop16

```text
###############################
#...#...........#..MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#.FFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#...#..FFF#FFF#FFFFF#T#....MMM#
###.#.###F###F#####F#T#######.#
#.#.#.#F#F#F#FFF#FFF#TTTTMMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#.#FFF#F#FFFFF#FFF#FFF#F#F#TTM#
#.#####F#####F#######F#F#F###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#.#####F###F#T#######F#####F#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#.#######F#########T#F###F#F#M#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#....FFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#TTTTT#TTT#FFF#FFF..#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####.#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

## Case 478 (final failure)

Loop gain: `-0.0076`. First loop F1 `0.3514` with 201 false positives and 87 misses. loop16 F1 `0.3438` with 206 false positives and 88 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMTT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#....F#TTTTT#F#F#FFF#F#T#...#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###F#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#.###
#.F..F#TTTTTTT#FFF#F#FFF#F#F..#
###.#F#T#########F#F###F#.###.#
#.F.#F#T#FFF#FFF#F#F#FFFF.#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMTT#FFF#F#FFF#FFF#F....#.#
#.#M#######F#F#########.#####.#
#.#M#MMTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#.#M#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#T#T###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

### loop2

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#.....#TTTTT#F#F#FFF#F#T#...#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###.#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#F###
#.FFFF#TTTTTTT#FFF#F#FFF#F#...#
###F#F#T#########F#F###F#F###.#
#.F.#F#T#FFF#FFF#F#F#FFF.F#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMTTT#FFF#F#FFF#FFF#F....#.#
#.#M#######F#F#########F#####.#
#.#M#MMTTT#F#FFFFFFFFFFF#MMM#.#
#.#M#M###M#F#############M#M#.#
#MMM#M#.#T#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#T#M###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

### loop4

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#.....#TTTTT#F#F#FFF#F#T#...#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###.#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#F###
#.FFFF#TTTTTTT#FFF#F#FFF#F#F..#
###F#F#T#########F#F###F#F###.#
#.F.#F#T#FFF#FFF#F#F#FFF.F#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMTTT#FFF#F#FFF#FFF#F....#.#
#.#M#######F#F#########F#####.#
#.#M#MMTTT#F#FFFFFFFFFFF#MMM#.#
#.#M#M###M#F#############M#M#.#
#MMM#M#.#T#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#T#M###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

### loop6

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#.....#TTTTT#F#F#FFF#F#T#...#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###.#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#F###
#.FFFF#TTTTTTT#FFF#F#FFF#F#...#
###F#F#T#########F#F###F#F###.#
#.F.#F#T#FFF#FFF#F#F#FFF.F#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMTTT#FFF#F#FFF#FFF#FF...#.#
#.#M#######F#F#########F#####.#
#.#M#MMTTT#F#FFFFFFFFFFF#MMM#.#
#.#M#M###M#F#############M#M#.#
#MMM#M#.#T#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#T#T###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

### loop10

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#.....#TTTTT#F#F#FFF#F#T#...#.#
#.#.#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###.#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#F###
#.FFFF#TTTTTTT#FFF#F#FFF#F#...#
###F#F#T#########F#F###F#F###.#
#.F.#F#T#FFF#FFF#F#F#FFF.F#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMTTT#FFF#F#FFF#FFF#FF...#.#
#.#M#######F#F#########F#####.#
#.#M#MMTTT#F#FFFFFFFFFFF#MMM#.#
#.#M#M###M#F#############M#M#.#
#MMM#M#.#T#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#T#M###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

### loop16

```text
###############################
#...#MMMMMMM....#MMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#.FF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#M#####.#
#.....#TTTTT#F#F#FFF#F#T#...#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FF.#
###F#T#############F#F###T###.#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#.#
#.###T#F###F#F#######F#F###F#.#
#.#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#.#T#########F###F#F#########F#
#.#TTTTTTTTTTT#FFF#FFF#FFF#FF.#
#.###########T#F#####F###F#F###
#.FFFF#TTTTTTT#FFF#F#FFF#F#...#
###F#F#T#########F#F###F#F###.#
#.F.#F#T#FFF#FFF#F#F#FFF.F#F..#
#.#####T#F###F#F#F#F#F#####.###
#.#MMTTT#FFF#F#FFF#FFF#FF...#.#
#.#M#######F#F#########F#####.#
#.#M#MMTTT#F#FFFFFFFFFFF#MMM#.#
#.#M#M###M#F#############M#M#.#
#MMM#M#.#T#TTT#FFFFFFFF..M#M#.#
#M###M#.#M#M#T###########M#M#.#
#MMMMM#..MMM#MMMMMMMMMMMMM#MMS#
###############################
```

## Case 146 (final failure)

Loop gain: `-0.0201`. First loop F1 `0.3670` with 195 false positives and 81 misses. loop16 F1 `0.3468` with 206 false positives and 84 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#T#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#T#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#TTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MF.FFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#..FFFFF#FFF#TTT#T#FFF#.#...#
#M#.###F###F###T###T###.#####.#
#M#...#F#FFF#TTT#F#T#G#F....#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#...#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#F.F#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

### loop2

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#M#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#T#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#MTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MFFFFFF#FFF#F#FFTTT#F#FFF#F..#
#M#######F###F###T#T#F###F#F###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#..F#F#FFF#TTT#F#T#G#F....#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#F..#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#.FF#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

### loop4

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#M#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#T#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#MTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MFFFFFF#FFF#F#FFTTT#F#FFF#F..#
#M#######F###F###T#T#F###F#F###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#..F#F#FFF#TTT#F#T#G#F....#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#F..#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#.FF#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

### loop6

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#M#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#M#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#MTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MFFFFFF#FFF#F#FFTTT#F#FFF#F..#
#M#######F###F###T#T#F###F#F###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#..F#F#FFF#TTT#F#T#G#F....#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#F..#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#.FF#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

### loop10

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#M#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#M#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#MTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MFFFFFF#FFF#F#FFTTT#F#FFF#F..#
#M#######F###F###T#T#F###F#F###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#..F#F#FFF#TTT#F#T#G#FF...#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#F..#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#.FF#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

### loop16

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####T#########M#M###M#
#M#MMM#M#TTT#T#FFFFF#TTM#M#.#M#
#M#M#M#M#T#T#T#F###F#T###M#.#M#
#MMM#M#TTT#T#T#FFF#F#TTMMM#.#M#
#.###M#####T#T#F#F#########.#M#
#.#.#MTT#TTT#T#F#FFFFFFFF...#M#
#.#F###T#T###T#F###########F#M#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###.#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#M###F#F#F#F#####F#######F#F###
#M#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#M#####F###F#F#####F#F###F###.#
#MFFFFFF#FFF#F#FFTTT#F#FFF#F..#
#M#######F###F###T#T#F###F#F###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#..F#F#FFF#TTT#F#T#G#FF...#.#
#M###.###F###T###F#T#T###.#.#.#
#MMM#..FFF#FFT#FFF#T#T#F..#.#.#
###M#######F#T###F#T#T#.###.#.#
#.#MMMMM#.FF#TTT#F#TTT#.#...#.#
#.#####M#######T#F###F#.#####.#
#......MMMMMMMMM#.....#.......#
###############################
```

## Case 65 (final failure)

Loop gain: `-0.0102`. First loop F1 `0.3607` with 200 false positives and 80 misses. loop16 F1 `0.3506` with 208 false positives and 81 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#F#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#T#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMT#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########T#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#M#.#
#M#.FF#FFFFF#F#FFF#FFFFF#.#M#.#
#M#.#####F#F#F#F###F###F#.#M#.#
#M#.#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#.#F#####F#########F#F#.#M#.#
#M#...#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FF...#M#.#
#.#######T#F#T#T#T#T#######M#.#
#.......#T#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####T###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

### loop2

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#.#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#M#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMM#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########M#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#T#.#
#M#FFF#FFFFF#F#FFF#FFFFF#F#M#.#
#M#F#####F#F#F#F###F###F#F#T#.#
#M#F#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#F#F#####F#########F#F#F#M#.#
#M#..F#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FFF..#M#.#
#.#######M#F#T#T#T#T#######M#.#
#.......#M#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####T###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

### loop4

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#.#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#M#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMM#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########M#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#T#.#
#M#FFF#FFFFF#F#FFF#FFFFF#F#M#.#
#M#F#####F#F#F#F###F###F#F#T#.#
#M#F#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#F#F#####F#########F#F#F#M#.#
#M#..F#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FFF..#M#.#
#.#######M#F#T#T#T#T#######M#.#
#.......#T#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####M###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

### loop6

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#.#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#M#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMM#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########M#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#T#.#
#M#FFF#FFFFF#F#FFF#FFFFF#F#M#.#
#M#F#####F#F#F#F###F###F#F#T#.#
#M#F#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#.#F#####F#########F#F#F#M#.#
#M#..F#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FFF..#M#.#
#.#######M#F#T#T#T#T#######M#.#
#.......#T#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####T###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

### loop10

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#.#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#M#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMM#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########M#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#T#.#
#M#FFF#FFFFF#F#FFF#FFFFF#F#T#.#
#M#F#####F#F#F#F###F###F#F#T#.#
#M#F#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#F#F#####F#########F#F#F#M#.#
#M#..F#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FFF..#M#.#
#.#######T#F#T#T#T#T#######M#.#
#.......#T#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####T###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

### loop16

```text
###############################
#.#...#...#.....#.#...#.......#
#.#.#.#.#.#F#F#F#F#F#F#####.###
#...#.#F#F#F#F#F#F#F#F#MMM#...#
#.###.#F#F#F#F#F#F#F###M#M###.#
#...#.FF#FFF#F#FFF#FFF#M#MMMMM#
#.#####F#####F#F#####F#M#####M#
#.#MMM#FFFFF#F#F#FFF#G#T#MMMMM#
###T#T#####F#F###F#F#T#T#T#####
#MTT#TTTTT#F#FFFFF#F#TTT#TTTTM#
#M#######T#F#######F#########M#
#M#FFF#TTT#F#FFF#FFF#FFFFFFF#M#
#M#F###T###F###F#F#F#F###F###M#
#M#FFF#TTT#FFFFF#F#F#FFF#F#TTM#
#M#F#F###T###F###F#F#F#F###T###
#M#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#M#F###T#T#F###F###F#####F#T#F#
#M#FFF#TTT#F#FFF#F#FFFFFFF#T#.#
#M###F#######F###F#######F#T#.#
#M#FFF#FFFFF#F#FFF#FFFFF#F#T#.#
#M#F#####F#F#F#F###F###F#F#T#.#
#M#F#FFFFF#FFF#FFFFFFF#F#.#M#.#
#M#F#F#####F#########F#F#F#M#.#
#M#..F#FFFFF#TTTTTTT#F#F..#M#.#
#M#########F#T#####T#F#####M#.#
#MMMMMMTTT#F#T#TTT#T#FFF..#M#.#
#.#######M#F#T#T#T#T#######M#.#
#.......#T#F#T#T#TTT#TTMMM#MMM#
#######.#M###T#T#####T###M###M#
#.......#MMMMM#MMMMMMM..#MMMMM#
###############################
```

## Case 344 (final failure)

Loop gain: `-0.0286`. First loop F1 `0.3805` with 200 false positives and 80 misses. loop16 F1 `0.3519` with 204 false positives and 87 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#TTT#FFF#TTTTT#F#TMM#M#.#
#M###M#T###F###T###T#####M#M###
#MMM#MTT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#FFFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#T#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#T#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#T#
#M###F#########T#F#####F#####T#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###T#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#.#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#..G#
#M###T#####T#######T###F###.###
#M#MTT#FFF#TTTTTTTTT#FFF#.....#
#M#T###F#F###########F#######.#
#M#MMM#F#FFFFF#FFFFFFF#.....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#.#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####F###########F#######.#
#...#...............#.........#
###############################
```

### loop2

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#MTT#FFF#TTTTT#F#MMM#M#.#
#M###M#M###F###T###T#####M#M###
#MMM#MMT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#.FFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#M#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#M#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#M#
#M###F#########T#F#####F#####M#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###M#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#F#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#.FG#
#M###T#####T#######T###F###F###
#M#MTT#FFF#TTTTTTTTT#FFF#.....#
#M#T###F#F###########F#######.#
#M#MMT#F#FFFFF#FFFFFFF#F....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#F#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####.###########F#######.#
#...#...............#.........#
###############################
```

### loop4

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#MTT#FFF#TTTTT#F#MMM#M#.#
#M###M#T###F###T###T#####M#M###
#MMM#MMT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#.FFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#M#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#M#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#M#
#M###F#########T#F#####F#####M#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###T#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#F#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#FFG#
#M###T#####T#######T###F###F###
#M#TTT#FFF#TTTTTTTTT#FFF#.....#
#M#M###F#F###########F#######.#
#M#MMT#F#FFFFF#FFFFFFF#F....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#F#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####.###########F#######.#
#...#...............#.........#
###############################
```

### loop6

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#MTT#FFF#TTTTT#F#MMM#M#.#
#M###M#M###F###T###T#####M#M###
#MMM#MMT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#.FFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#M#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#M#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#M#
#M###F#########T#F#####F#####M#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###M#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#F#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#FFG#
#M###T#####T#######T###F###F###
#M#TMT#FFF#TTTTTTTTT#FFF#.....#
#M#M###F#F###########F#######.#
#M#MMT#F#FFFFF#FFFFFFF#F....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#F#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####.###########F#######.#
#...#...............#.........#
###############################
```

### loop10

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#MTT#FFF#TTTTT#F#MMM#M#.#
#M###M#T###F###T###T#####M#M###
#MMM#MMT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#.FFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#M#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#M#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#M#
#M###F#########T#F#####F#####M#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###M#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#F#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#FFG#
#M###T#####T#######T###F###F###
#M#MMT#FFF#TTTTTTTTT#FFF#.....#
#M#M###F#F###########F#######.#
#M#MMT#F#FFFFF#FFFFFFF#F....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#F#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####.###########F#######.#
#...#...............#.........#
###############################
```

### loop16

```text
###############################
#MMM#....MMMMMMMMMMMMM#MMMMM..#
#M#M#####T###F#######S#M###M#.#
#M#MMM#MTT#FFF#TTTTT#F#MMM#M#.#
#M###M#T###F###T###T#####M#M###
#MMM#MMT#F#F#F#T#F#TTTTMMM#MMM#
#.#M#####F#F#F#T#F###########M#
#.#M#.FFFF#FFF#TTT#FFFFFF.#.#M#
###T#F#F#####F###T#F#F###F#F#M#
#MTT#F#F#FFFFF#TTT#F#FFF#F#F#M#
#M###F###F#####T#######F#F#F#M#
#M#FFFFF#FFF#TTT#FFFFFFF#FFF#M#
#M#F###F###F#T###F#########F#M#
#M#FFF#FFFFF#TTT#F#FFFFFFFFF#M#
#M###F#########T#F#####F#####M#
#M#FFFFFFFFFFF#T#F#FFF#F#FFFFT#
#M#####F#####F#T#F#F#F#F#F###M#
#MTT#FFF#TTT#F#T#FFF#F#F#F#F#M#
###T#####T#T#F#T#####F###F#F#M#
#MTT#TTTTT#T#F#TTTTTFF#FFF#FFG#
#M###T#####T#######T###F###F###
#M#TTT#FFF#TTTTTTTTT#FFF#.....#
#M#M###F#F###########F#######.#
#M#MMT#F#FFFFF#FFFFFFF#F....#.#
#M###M###F#F###F#####F#F###.#.#
#MMMMM#FFF#F#FFF#FFFFF#F#...#.#
#.#####.###F#F#########.###.###
#.#.....#FFF#FFFFFFFFF#...#...#
#.#.#####.###########F#######.#
#...#...............#.........#
###############################
```

## Case 319 (final failure)

Loop gain: `0.0187`. First loop F1 `0.3356` with 208 false positives and 85 misses. loop16 F1 `0.3543` with 208 false positives and 80 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#F#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#F#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#FFF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#T#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#.#####F###F#T###T###########T#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#F.FFF#F#TTT#FFF#FFFFF#M#MMM#
#.#.#####F#T###F#F#######M#M###
#.F.#FFF#F#T#FFF#FFFFF#TMT#MMM#
#####F#F#F#T#F#######F#T#####M#
#.....#FFF#T#F#TTT#FFF#TMMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#MMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MMT#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#T#######T#F#F#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

### loop2

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#.#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#.FF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#.#####F###F#T###T###########M#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###M#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#.#F#####F#T###F#F#######T#T###
#.FF#FFF#F#T#FFF#FFFFF#TTM#MMM#
#####F#F#F#T#F#######F#T#####M#
#....F#FFF#T#F#TTT#FFF#TMMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MTM#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#M#######T#F#F#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

### loop4

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#.#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#.FF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#.#####F###F#T###T###########M#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#.#F#####F#T###F#F#######T#T###
#.FF#FFF#F#T#FFF#FFFFF#TMM#TMM#
#####F#F#F#T#F#######F#T#####M#
#....F#FFF#T#F#TTT#FFF#TMMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MTM#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#M#######T#F#F#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

### loop6

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#F#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#.FF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#.#####F###F#T###T###########M#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#.#F#####F#T###F#F#######T#T###
#.FF#FFF#F#T#FFF#FFFFF#TMM#TMM#
#####F#F#F#T#F#######F#T#####M#
#....F#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MMM#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#M#######T#F#F#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

### loop10

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#F#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#.FF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#.#####F###F#T###T###########M#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###M#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#.#F#####F#T###F#F#######T#T###
#.FF#FFF#F#T#FFF#FFFFF#TTM#MMM#
#####F#F#F#T#F#######F#T#####M#
#....F#FFF#T#F#TTT#FFF#TMMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MTM#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#M#######T#F#F#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

### loop16

```text
###############################
#...#.........#...#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FF.#.#...#
#.#.#.#.#F#F#####F#F###.#.###.#
#.#...#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####.###M#.#
#...#.FF#F#F#FFF#FFFFF#FF.#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#.#####F###F#T###T###########M#
#.#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#.#F#F###F#F#T#F#######T###T###
#.FF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###M#
#.FF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#.###F#F#F###T###F#####F#T###M#
#.#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#.#F#####F#T###F#F#######T#T###
#.FF#FFF#F#T#FFF#FFFFF#TTM#TMM#
#####F#F#F#T#F#######F#T#####M#
#....F#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMMMMTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MMM#MTM#FFF#TTT#F#F#M#MMM#M#
#M#M#M#M#M#######T#F#.#M#####M#
#MMM#MMM#MMMMMMMMM#...#MMMMMMM#
###############################
```

## Case 454 (final failure)

Loop gain: `0.0052`. First loop F1 `0.3496` with 205 false positives and 89 misses. loop16 F1 `0.3548` with 203 false positives and 88 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#F#FFF#FFF#T#F#T#F....#.#
#.#.###F#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#FFF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###T#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#T#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#T#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.F.FFFFFF#FFFFF#F#FFF#TTT#MMG#
#########F#####F#####F#F#M###.#
#MTMMTTT#FFF#FFF#FFFFF#F#M#...#
#M#####T###F#F###F#####F#M#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#MMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#TTT#T#TTTTT#F#MMMMM#M#
#M#######T###T#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#.#FFF#FFF#T#F#T#.....#.#
#.#.###F#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#.FF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###M#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#M#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#M#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.FFFFFFFF#FFFFF#F#FFF#TTT#TMG#
#########F#####F#####F#F#T###.#
#MMTTTTT#FFF#FFF#FFFFF#F#T#...#
#M#####T###F#F###F#####F#T#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#TMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#TTT#T#TTTTT#F#MMMMM#M#
#M#######M###T#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#.#FFF#FFF#T#F#T#.....#.#
#.#.###.#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#.FF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###M#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#M#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#M#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.FFFFFFFF#FFFFF#F#FFF#TTT#TMG#
#########F#####F#####F#F#T###.#
#MMTTTTT#FFF#FFF#FFFFF#F#T#F..#
#M#####T###F#F###F#####F#T#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#TMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#TTT#T#TTTTT#F#MMMMM#M#
#M#######M###M#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#.#FFF#FFF#T#F#T#.....#.#
#.#.###F#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#.FF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###M#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#M#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#M#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.FFFFFFFF#FFFFF#F#FFF#TTT#TMG#
#########F#####F#####F#F#T###.#
#MMTTTTT#FFF#FFF#FFFFF#F#T#...#
#M#####T###F#F###F#####F#M#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#TMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#MTT#T#TTTTT#F#MMMMM#M#
#M#######M###M#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#.#FFF#FFF#T#F#T#.....#.#
#.#.###.#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#.FF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###M#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#M#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#M#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.FFFFFFFF#FFFFF#F#FFF#TTT#TMG#
#########F#####F#####F#F#T###.#
#MTTTTTT#FFF#FFF#FFFFF#F#T#...#
#M#####T###F#F###F#####F#T#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#TMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#MTT#T#TTTTT#F#MMMMM#M#
#M#######M###M#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop16

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######F###T###T#.#.###.#
#.#...#.#FFF#FFF#T#F#T#.....#.#
#.#.###F#F#F###F#T#F#T#######.#
#.#...#FFF#F#FFF#T#TTT#MMM#MMM#
#.###.#####F#F###T#T###M#M#M#M#
#...#.FF#FFF#FFF#T#TTTTT#MMM#M#
#.#F###F#F#####F#T###########M#
#.#F#F#F#F#F#FFF#TTTTTTTTT#FFM#
#.#F#F#F#F#F#F###F#######T###M#
#.#F#FFF#F#FFF#FFFFFFFFF#TTT#M#
###F#F###F#F###########F###T#M#
#.FF#F#FFF#FFFFF#FFFFFFF#TTT#M#
#.#####F#######F#######F#T###M#
#.FFFF#F#FFFFF#FFFFFFF#F#T#F#T#
#####F#F#####F#######F###T#F#M#
#.FF#FFF#FFF#FFFFF#FFF#TTT#TTM#
#.#######F#F###F#F#F###T###T###
#.FFFFFFFF#FFFFF#F#FFF#TTT#TMG#
#########F#####F#####F#F#T###.#
#MTTTTTT#FFF#FFF#FFFFF#F#T#...#
#M#####T###F#F###F#####F#M#####
#MMM#TTT#FFF#FFF#FFF#FFF#M#MMM#
###M#S###F#####F###F#####M#M#M#
#MMM#.#FFF#TTT#FFF#FFF#TMM#M#M#
#M###.#.###T#T#######F#M###M#M#
#M#.....#TTT#T#TTTTT#F#MMMMM#M#
#M#######M###M#T###T#F#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

## Case 230 (final failure)

Loop gain: `0.0109`. First loop F1 `0.3494` with 198 false positives and 85 misses. loop16 F1 `0.3604` with 203 false positives and 81 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#T#######T#F#########M#
#.#...#T#TTTTTTTTT#F#FF.#MMMMM#
#.###.#T#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#F#####M#
#MMMMTTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#.FF#F#F#FFF#FFF#TTTTTMM#M#M#
#.###F###F#F#F###F#T#######M#M#
#...#F#FFF#F#FFF#F#TTT#F..#MMM#
#.###F#F###F###F#F###T#F#######
#.#...#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FF...#.#.#
###.#####F###F#F###T#######.###
#G..#MMMMMTT#FFFFF#TTT#...#...#
#M###M#####T#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

### loop2

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#T#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#.###F###F#F#F###F#T#######T#M#
#.FF#F#FFF#F#FFF#F#TTT#FFF#MMM#
#.###F#F###F###F#F###T#F#######
#.#..F#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FFF..#.#.#
###.#####F###F#F###T#######.###
#G..#MMMTTTT#FFFFF#TTT#...#...#
#M###M#####M#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

### loop4

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FF.#MMMMM#
#.###.#T#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#T#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#.###F###F#F#F###F#T#######T#M#
#.FF#F#FFF#F#FFF#F#TTT#FFF#MMM#
#.###F#F###F###F#F###T#F#######
#.#..F#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FFF..#.#.#
###.#####.###F#F###T#######.###
#G..#MMMTTTT#FFFFF#TTT#...#...#
#M###M#####T#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

### loop6

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#T#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#.###F###F#F#F###F#T#######T#M#
#.FF#F#FFF#F#FFF#F#TTT#FFF#MMM#
#.###F#F###F###F#F###T#F#######
#.#..F#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FFF..#.#.#
###.#####.###F#F###T#######.###
#G..#MMMTTTT#FFFFF#TTT#...#...#
#M###M#####M#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

### loop10

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FF.#MMMMM#
#.###.#T#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#T#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#.###F###F#F#F###F#T#######T#M#
#.FF#F#FFF#F#FFF#F#TTT#FFF#MMM#
#.###F#F###F###F#F###T#F#######
#.#..F#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FFF..#.#.#
###.#####.###F#F###T#######.###
#G..#MMMTTTT#FFFFF#TTT#...#...#
#M###M#####T#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

### loop16

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#TTT#FFFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMTTTT#F#FFFFF#FFF#F#...#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#T#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#.###F###F#F#F###F#T#######T#M#
#.FF#F#FFF#F#FFF#F#TTT#FFF#MMM#
#.###F#F###F###F#F###T#F#######
#.#..F#F#FFF#F#FFF#TTT#F#.....#
#.#.#.#F###F#F###F#T###F#.#.#.#
#.#.#.#F#FFF#F#FFF#T#FFF..#.#.#
###.#####.###F#F###T#######.###
#G..#MMMTTTT#FFFFF#TTT#...#...#
#M###M#####T#########T#.#.###.#
#MMMMM....#MMMMMMMMMMM..#.....#
###############################
```

## Case 470 (final failure)

Loop gain: `0.0329`. First loop F1 `0.3295` with 203 false positives and 90 misses. loop16 F1 `0.3624` with 204 false positives and 81 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#FFF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#..FFF#F#F#FFF#FFFFF#F..#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#M#######T#####F###F#F###F###T#
#M#FFTTTTTFFFF#F#FFF#FFFFF#.#M#
#M###T#########F#F#F#######.#M#
#MTM#TTG#FFFFFFF#F#FFF#TTM#MMM#
#.#M#####F#######F#####T#M#M###
#.#MMTTT#FFF#FFF#TTTTT#T#TMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMM#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MMT#FFF#TTTFF#T#FFTMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MMMTT#T#T#F#TTTTM#MMM#M#
#M#####M###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop2

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#.FF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#...FF#F#F#FFF#FFFFF#FF.#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#M#######T#####F###F#F###F###M#
#M#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#M###T#########F#F#F#######F#M#
#MTT#TTG#FFFFFFF#F#FFF#TTT#TTM#
#.#T#####F#######F#####T#T#T###
#.#TTTTT#FFF#FFF#TTTTT#T#MMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMT#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MTT#FFF#TTTFF#T#FFMMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MTTTT#T#T#F#TTTTM#MMM#M#
#M#####M###M#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop4

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#.FF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#...FF#F#F#FFF#FFFFF#FF.#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#M#######T#####F###F#F###F###M#
#M#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#M###T#########F#F#F#######F#M#
#MTT#TTG#FFFFFFF#F#FFF#TTT#TTM#
#.#T#####F#######F#####T#T#T###
#.#TTTTT#FFF#FFF#TTTTT#T#MMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMT#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MMT#FFF#TTTFF#T#FFTMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MTTTT#T#T#F#TTTTM#MMM#M#
#M#####M###M#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop6

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#.FF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#...FF#F#F#FFF#FFFFF#FF.#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#M#######T#####F###F#F###F###M#
#M#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#M###T#########F#F#F#######F#M#
#MTT#TTG#FFFFFFF#F#FFF#TTT#TTM#
#.#T#####F#######F#####T#T#T###
#.#TTTTT#FFF#FFF#TTTTT#T#MMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMT#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MMT#FFF#TTTFF#T#FFMMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MTTTT#T#T#F#TTTTM#MMM#M#
#M#####M###M#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop10

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#.FF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#...FF#F#F#FFF#FFFFF#FF.#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#M#######T#####F###F#F###F###M#
#M#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#M###T#########F#F#F#######F#M#
#MTT#TTG#FFFFFFF#F#FFF#TTT#TTM#
#.#T#####F#######F#####T#T#T###
#.#TTTTT#FFF#FFF#TTTTT#T#MMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMT#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MMT#FFF#TTTFF#T#FFTMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MTTTT#T#T#F#TTTTM#MMM#M#
#M#####M###M#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop16

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#.FF#F#FFFFF#F#FF.#...#.#
#.#####F#F#F#F#####F#F#####.#.#
#.....#F#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#...FF#F#F#FFF#FFFFF#FF.#.#.#
###F#####F#############F#F#F#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#M#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#M#######T#####F###F#F###F###M#
#M#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#M###T#########F#F#F#######F#M#
#MTT#TTG#FFFFFFF#F#FFF#TTT#TTM#
#.#T#####F#######F#####T#T#T###
#.#TTTTT#FFF#FFF#TTTTT#T#MMM#.#
#######T###F#F#F#T###T#T#####.#
#MMMMT#T#F#F#F#F#TTT#TTT#MMMMM#
#M###M#T#F#F#F#####T#####M###M#
#MMM#MMT#FFF#TTTFF#T#FFTMM#.#M#
###M#.#######T#T#F#T###M###.#M#
#MMM#.#MTTTT#T#T#F#TTTTM#MMM#M#
#M#####M###M#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

## Case 277 (final failure)

Loop gain: `-0.0078`. First loop F1 `0.3750` with 193 false positives and 97 misses. loop16 F1 `0.3672` with 194 false positives and 99 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###F###T#######F#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#..FFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#T#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#T#
#M#############F#####F#F###T#T#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTMMM#M#
#M#####F###F#F#######T#######M#
#M#..TTT#F#F#F#FFTTTTT#F.FF.#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMM#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MMMMM#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop2

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###F###T#######F#M#.###.#
#.#M#MMM#F#TTT#F#FFF#F#M#.#.#.#
#.#M###M###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#...FFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTT#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FF..F#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMT#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######M#####T#F#F#####.#M#M#
#...#MMMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####M#M#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop4

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###T#######F#M#.###.#
#.#M#MMM#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#...FFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTT#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#F.F..#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMT#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######M#####T#F#F#####.#M#M#
#...#MMMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####M#T#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop6

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###T#######F#M#.###.#
#.#M#MMM#F#TTT#F#FFF#F#M#.#.#.#
#.#M###M###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#...FFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTT#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FF..F#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMT#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######M#####T#F#F#####.#M#M#
#...#MMMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####M#T#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop10

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###F###T#######F#M#.###.#
#.#M#MMM#F#TTT#F#FFF#F#M#.#.#.#
#.#M###M###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#...FFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTT#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FF..F#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMT#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######M#####T#F#F#####.#M#M#
#...#MMMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####M#T#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop16

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###F###T#######F#M#.###.#
#.#M#MMM#F#TTT#F#FFF#F#M#.#.#.#
#.#M###M###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#...FFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#TTT#M#
#M#####F#F#F#######F#####F#T#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#T#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#T#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########T#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTT#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FF..F#M#
#M###T#T#F#F#F###T#####F#####M#
#MMMMT#T#F#FFF#TTT#GTTTTMMMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFF..#M#M#
#.#######M#####T#F#F#####.#M#M#
#...#MMMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####M#M#T#F#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

## Case 91 (final failure)

Loop gain: `0.0017`. First loop F1 `0.3665` with 202 false positives and 78 misses. loop16 F1 `0.3682` with 200 false positives and 78 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#FFF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#.FFFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#....F#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###F#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#MMM#
#M###F###T#######F#####F#####M#
#M#.#FFFFTTTTTTT#FFFFF#F#MMM#M#
#M#.#F#########T#####F#F#M#M#M#
#M#...#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMT#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#.FF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#.....#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TMM#
#M###F###T#######F#####F#####M#
#M#.#FFFFTTTTTTT#FFFFF#F#TMM#M#
#M#F#F#########T#####F#F#T#M#M#
#M#..F#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MTM#F#T#FFFFFFF#...#MMM#
#.###M###M#.#M###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#.FF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#.....#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#MMM#M#
#M#F#F#########T#####F#F#M#M#M#
#M#..F#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MTT#F#T#FFFFFFF#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#.FF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#.....#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###F#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#MMM#M#
#M#F#F#########T#####F#F#T#M#M#
#M#..F#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MTM#F#T#FFFFFFF#...#MMM#
#.###M###M#.#M###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#.FF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#.....#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TMM#M#
#M#F#F#########T#####F#F#T#M#M#
#M#..F#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MTM#F#T#FFFFFFF#...#MMM#
#.###M###M#.#M###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

### loop16

```text
###############################
#.....#.............#.#.GMMM#.#
#.#.#.#.#########F#F#F#.###M#.#
#.#.#.#.FF#FFFFFFF#FFF#...#MMM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FF.....#M#
#.#######F#F#F#F#F#F#######.#M#
#.....#FFF#F#FFF#F#F#FFFF...#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#.#T#F###T###F#F###F#####T###F#
#.#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#MMM#M#
#M#F#F#########T#####F#F#T#M#M#
#M#..F#TTT#FFFFT#FFF#F#F#M#MMM#
#M###.#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FFF#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMT#F#T#FFFFFFF#...#MMM#
#.###M###M#.#M###############M#
#...#MMMMM..#MMMMMMMMMMMMMMMMM#
###############################
```

## Case 177 (final failure)

Loop gain: `-0.0005`. First loop F1 `0.3689` with 196 false positives and 88 misses. loop16 F1 `0.3684` with 201 false positives and 87 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FFF..#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#FFFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#TTTTGFFFF#F#FFF#F#FF.#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFF.#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#.#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TMT#...#
#M#######F###F#T#####T###M#.###
#M#..FFF#FFFFF#TTT#F#TTT#T#...#
#M#.###F#####F###T#F###T#M###.#
#M#...#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#..F#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop2

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FF...#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#FFFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#MTTTGFFFF#F#FFF#F#FF.#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FF.#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#M#F..#
#M#F###F#####F###T#F###T#T###.#
#M#..F#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#.F.#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop4

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FF...#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#FFFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#MTTTGFFFF#F#FFF#F#F..#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FF.#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#M#F..#
#M#F###F#####F###T#F###T#T###.#
#M#..F#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#.F.#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop6

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FF...#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#FFFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#MTTTGFFFF#F#FFF#F#F..#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TTT#F..#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#M#F..#
#M#F###F#####F###T#F###T#T###.#
#M#..F#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#.F.#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop10

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FF...#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#.FFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#MTTTGFFFF#F#FFF#F#FF.#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FF.#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#M#...#
#M#F###F#####F###T#F###T#T###.#
#M#..F#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#.F.#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop16

```text
###############################
#MMM#.................#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FF...#.#.#
#M###M#####F###F###F#######.#.#
#MMM#M#FFFFF#F#F#FFFFFF.#.....#
###M#M#######F#F#F###F###.#####
#MMM#MTTTGFFFF#F#FFF#F#F..#...#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###.#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#.#
#.#F#T#####T#T###F#########F#.#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FF.#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####.#
#M#TTSFF#FFF#F#TTTTTTT#TTT#F..#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#M#...#
#M#F###F#####F###T#F###T#T###.#
#M#..F#F#FFF#FFF#T#FFF#T#M#MMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTM#MMM#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#.F.#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

## Case 327 (final failure)

Loop gain: `0.0303`. First loop F1 `0.3386` with 204 false positives and 89 misses. loop16 F1 `0.3689` with 203 false positives and 81 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####F#F#F#####F#F#.#.###.#
#M#MMM#FFF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MTT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#FFG#F#F#FFF#F#F#FFF..#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FFF#
#M#T#T###F#F#F#F#F#F#####F###F#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####T#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#M#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTM#MMM#
#M#M#T#F#######F#F#F#T#T#M#####
#M#MMT#F#TTT#FFF#F#F#TTT#MMMMM#
#M#####F#T#T#F###F#F#########M#
#MMM#.FF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#.#####M#
#.#M#MMTTT#TTT#F#F#FFF#.#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#F#F#####F#F#.#.###.#
#M#MMM#.FF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MMT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#.FG#F#F#FFF#F#F#FFFF.#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FF.#
#M#T#T###F#F#F#F#F#F#####F###.#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####T#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#T#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTT#TMM#
#M#T#T#F#######F#F#F#T#T#T#####
#M#TTT#F#TTT#FFF#F#F#TTT#MMTMM#
#M#####F#T#T#F###F#F#########M#
#MMM#FFF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#F#####M#
#.#M#MMTTT#TTT#F#F#FFF#F#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#F#F#####F#F#.#.###.#
#M#MMM#.FF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MMT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#.FG#F#F#FFF#F#F#FFFF.#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FF.#
#M#T#T###F#F#F#F#F#F#####F###.#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####T#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#T#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTT#TMM#
#M#T#T#F#######F#F#F#T#T#T#####
#M#TTT#F#TTT#FFF#F#F#TTT#TMTMM#
#M#####F#T#T#F###F#F#########M#
#MMM#FFF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#F#####M#
#.#M#MMTTT#TTT#F#F#FFF#F#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#F#F#####F#F#.#.###.#
#M#MMM#.FF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MMT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#.FG#F#F#FFF#F#F#FFFF.#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FF.#
#M#T#T###F#F#F#F#F#F#####F###.#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####T#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#T#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTT#TMM#
#M#T#T#F#######F#F#F#T#T#T#####
#M#TTT#F#TTT#FFF#F#F#TTT#TMTMM#
#M#####F#T#T#F###F#F#########M#
#MMM#FFF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#F#####M#
#.#M#MMTTT#TTT#F#F#FFF#F#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#F#F#####F#F#.#.###.#
#M#MMM#.FF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MMT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#.FG#F#F#FFF#F#F#FFFF.#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FF.#
#M#T#T###F#F#F#F#F#F#####F###.#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####M#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#T#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTT#TMM#
#M#T#T#F#######F#F#F#T#T#T#####
#M#TTT#F#TTT#FFF#F#F#TTT#TMTMM#
#M#####F#T#T#F###F#F#########M#
#MMM#FFF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#F#####M#
#.#M#MTTTT#TTT#F#F#FFF#F#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop16

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#F#F#####F#F#.#.###.#
#M#MMM#.FF#F#F#FFF#F#F#...#...#
#M###M#F#######F###F#F#####.#.#
#M#.#MMT#FFFFF#FFF#F#FF.#.#.#.#
#M#.###T#F###F#F#F#F###.#.#.#.#
#M#.#.FG#F#F#FFF#F#F#FFFF.#.#.#
#M#F#F###F#F#####F#F#######F#.#
#M#FFF#FFF#FFFFF#F#FFFFF#FFF#.#
#M#####F###F###F#F#####F#F###.#
#M#TTT#FFF#F#F#F#F#FFFFF#F#FF.#
#M#T#T###F#F#F#F#F#F#####F###.#
#M#T#TFF#FFF#F#F#FFF#FFFFFFF#.#
#M#T#T#F#####F#F#########F#S###
#M#T#T#FFF#FFF#FFF#FFTTT#F#TTT#
#M#T#T###F#F#F###F###T#T#####T#
#MTT#T#F#F#F#FFF#F#TTT#TTTTT#T#
#####T#F#F###F#F#F#T#######T#M#
#MTT#T#FFFFFFF#F#F#TTT#TTT#TMM#
#M#T#T#F#######F#F#F#T#T#T#####
#M#TTT#F#TTT#FFF#F#F#TTT#TMTMM#
#M#####F#T#T#F###F#F#########M#
#MMM#FFF#T#T#F#FFF#F#FFF....#M#
#.#M#####T#T###F###F#F#F#####M#
#.#M#MMTTT#TTT#F#F#FFF#F#MMMMM#
#.#M#M#######T#F#F#####.#M###.#
#.#MMM#.#TTTTT#F#FFFFFF.#MMM#.#
#.#####.#M#####F###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

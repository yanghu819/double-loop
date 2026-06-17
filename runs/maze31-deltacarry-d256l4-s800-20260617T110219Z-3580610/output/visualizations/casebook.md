# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 177 (final failure)

Loop gain: `-0.0168`. First loop F1 `0.4198` with 243 false positives and 61 misses. loop12 F1 `0.4031` with 245 false positives and 66 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#.FFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MMT#FFFFF#FFF#FFF#FFF..#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FF...#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#F..#
#M#############F###F#F#F#####.#
#MTTTT#F#FFTTT#FFF#F#F#F#FFFF.#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#FFF#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#T#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

### loop2

```text
###############################
#MMM#.FFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFF..#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FF...#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####.#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#...#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

### loop4

```text
###############################
#MMM#FFFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFFF#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FFF..#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####.#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#...#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

### loop6

```text
###############################
#MMM#FFFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFF.#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FFF..#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####.#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#...#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

### loop10

```text
###############################
#MMM#FFFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFF.#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FFF..#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####.#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#...#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

### loop12

```text
###############################
#MMM#FFFFFFFFFFFFFFFFF#.......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFF.#.#.#
#M###T#####F###F###F#######.#.#
#MMT#T#FFFFF#F#F#FFFFFFF#FFF..#
###T#T#######F#F#F###F###F#####
#MTT#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####.#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FF.#
#M#F###F#####F###T#F###T#T###.#
#M#...#F#FFF#FFF#T#FFF#T#T#TMM#
#M#####F#F#F###F#T#F###T#M#M#M#
#M#MMM#FFF#F#FFF#T#F#TTT#MMM#M#
#M#M#M#F###F#F###T#F#T#######M#
#M#M#M#FFF#FFFFF#TTT#TTMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMTTTTTTTT......#MMM#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0008`. First loop F1 `0.4039` with 252 false positives and 55 misses. loop12 F1 `0.4031` with 253 false positives and 55 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTMM#.#
#.#.#.#F#########F#F#F#F###T#.#
#.#.#.#FFF#FFFFFFF#FFF#FFF#TMM#
###.#F#####F###F#####F#######M#
#...#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#..FFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#T###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#FFF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###F#M#####
#MMM#F#T#TTTTT#TSF#F#FF.#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM.F#TTTTTTTMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTTT#.#
#.#.#.#F#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#.FF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#FF.#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTTT#.#
#.#.#F#F#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#FFF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#F..#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM.F#TTTTTTTMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTTT#.#
#.#.#F#F#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#FFF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#F..#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM.F#TTTTTTMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTTT#.#
#.#.#F#F#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#FFF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#.#T#TTTTT#TSF#F#F..#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM.F#TTTTTTTMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGTTT#.#
#.#.#F#F#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#.#
#MTT#FFF#TTT#FFF#FFFFF#FFF#TTM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#TTT#M#
#M#F#F#########T#####F#F#T#T#M#
#M#FFF#TTT#FFFFT#FFF#F#F#T#MMM#
#M###F#T#T#####T#F#F###F#M#####
#MMM#.#T#TTTTT#TSF#F#F..#MMM#.#
#.#M###T#####T#####F#F#.###M#.#
#.#MMM#TTT#F#T#FFFFFFF#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM.F#TTTTTTMMMMMMMMMMM#
###############################
```

## Case 285 (final failure)

Loop gain: `0.0087`. First loop F1 `0.4016` with 235 false positives and 66 misses. loop12 F1 `0.4103` with 236 false positives and 63 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#.....FFFF#F#TTTTTTTMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#.FFFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#.#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FF.#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###M#####M#
#M#MMMMT#FFT#F#FFFFF#TMM#MMM#M#
#M#M###M###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

### loop2

```text
###############################
#.#......FFF#F#TTTTTTMMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#..FFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#.FF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FFF#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###M#####M#
#M#MMMMT#FFT#F#FFFFF#TTM#MMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

### loop4

```text
###############################
#.#.....FFFF#F#TTTTTTTMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#F.FFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#.FF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#.#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FFF#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###T#####M#
#M#MMMTT#FFT#F#FFFFF#TTM#MMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

### loop6

```text
###############################
#.#.....FFFF#F#TTTTTTTMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#.FF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#.#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FFF#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###T#####M#
#M#MMMMT#FFT#F#FFFFF#TTM#MMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

### loop10

```text
###############################
#.#.....FFFF#F#TTTTTTTMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#.FFFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#.FF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#.#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FFF#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###T#####M#
#M#MMMMT#FFT#F#FFFFF#TTM#MMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

### loop12

```text
###############################
#.#.....FFFF#F#TTTTTTTMM#MMM..#
#.#.#.#F#####F#T#######M#M#M#.#
#...#.#F#FFFFF#T#FFFFF#MMM#M#.#
#####.###F#####T###F#F#####M###
#.....#FFF#FFFFT#FFF#FFF..#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#F.M#
#.#F#####F###F#####F###F###F#M#
#.#F#FFFFF#F#F#FFF#F#FFF#FFF#M#
#.#F#F#####F#F#F#F#F###F#F###M#
#.FF#F#FFF#FFF#F#F#FFF#FFF#TTM#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#.#
###F#F###F#T###T#####F#T###T#.#
#.FF#FFFFF#T#FFT#FFFFF#TFF#TTM#
#.###F#####T###T#F#####T#F###M#
#.#FFFFF#TTT#TTT#F#FFF#T#FFF#M#
#.#######T###T###F#F#F#T###.#M#
#MMMMMTTTT#TTT#F#FFF#F#T#...#M#
#M#########T###F###F###T#####M#
#M#MMMMT#FFT#F#FFFFF#TTM#MMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..MMMTT#FFFFFFF#MMMMM#MMM#
###############################
```

## Case 319 (final failure)

Loop gain: `0.0015`. First loop F1 `0.4172` with 247 false positives and 52 misses. loop12 F1 `0.4186` with 249 false positives and 51 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#F..#.#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FST#.#
#.#####F#F#F#F#F#F#####F###T#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#TTM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TMM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMMTTTTTTTT#F#T#TGF#F#TTM#M#M#
#M###########F#T#####F#T#M#M#M#
#M#MMM#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop2

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#F..#.#...#
#.#.#F#F#F#F#####F#F###.#.###.#
#.#FFF#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####F###T#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#F#########T#F#T#T#F#F#####M#M#
#TTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#MMT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop4

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#F..#.#...#
#.#.#F#F#F#F#####F#F###.#.###.#
#.#FFF#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TMM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#F#########T#F#T#T#F#F#####M#M#
#TTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#M#M#M#
#M#MMT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop6

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#F..#.#...#
#.#.#F#F#F#F#####F#F###.#.###.#
#.#FFF#F#F#FFF#F#F#FFF#F#.SM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TMM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#F#########T#F#T#T#F#F#####M#M#
#TTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#M#M#M#
#M#MMT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop10

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#F..#.#...#
#.#.#F#F#F#F#####F#F###.#.###.#
#.#FFF#F#F#FFF#F#F#FFF#.#.SM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TMM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#F#########T#F#T#T#F#F#####M#M#
#TTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#M#M#M#
#M#MMT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop12

```text
###############################
#...#...FFFFFF#FFF#FF.....#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#F..#.#...#
#.#.#F#F#F#F#####F#F###.#.###.#
#.#FFF#F#F#FFF#F#F#FFF#F#.SM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TMM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#F#########T#F#T#T#F#F#####M#M#
#TTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#M#M#M#
#M#MMT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#M#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

## Case 348 (final failure)

Loop gain: `0.0039`. First loop F1 `0.4200` with 246 false positives and 55 misses. loop12 F1 `0.4239` with 245 false positives and 54 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#....F#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTM#.#
###############T#####F#####M###
#MMMTT#TTTTT#TTT#FFF#FFF..#MMM#
#M###T#T###T#T#####F###F#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#....F#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTMM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTT#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FFFF.#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTT#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FFF..#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#....F#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTT#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FFF..#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#....F#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTT#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FFF..#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop12

```text
###############################
#...#..FFFFFFFFF#FFTTMMMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#....F#FFFFF#F#FFF#TTT#...#M#M#
#.#########F###F#####T#.###M#M#
#..F#FFFFF#FFF#FFFFF#T#F...MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TMM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#M#
#######F#F#F#F#F###T#####T#T#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTTTTTT#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FFF..#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

## Case 349 (final failure)

Loop gain: `-0.0077`. First loop F1 `0.4324` with 242 false positives and 52 misses. loop12 F1 `0.4247` with 244 false positives and 54 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMTTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#F#####S###F#####F###T#####
#M#TTTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMT#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#F....#
#M#M###T#T#F#####F#####F#.#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMMFF#F#FFFFFFF........#
###############################
```

### loop2

```text
###############################
#MMMMMMMMTTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#F#####S###F#####F###T#####
#M#MMTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MMT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FF...#
#M#M###T#T#F#####F#####F#.#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMMFF#F#FFFFFFF........#
###############################
```

### loop4

```text
###############################
#MMMMMMMMMTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#F#####S###F#####F###T#####
#M#TMTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMT#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FF...#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFFFF........#
###############################
```

### loop6

```text
###############################
#MMMMMMMMTTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#.#####S###F#####F###T#####
#M#MMTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMT#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FF...#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFFFF........#
###############################
```

### loop10

```text
###############################
#MMMMMMMMTTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#F#####S###F#####F###T#####
#M#MMTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FF...#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMMFF#F#FFFFFFF........#
###############################
```

### loop12

```text
###############################
#MMMMMMMMTTTTTTTTTTT#TMM#.....#
#M#################T#T#T#F###.#
#M#MMMMT#TTT#FFFFF#TTT#T#F#...#
#M#M###T#T#T#F###F#####T###.#.#
#M#M#.#TTT#T#FFF#FFFFF#TTTFF#.#
#M#M#.#####S###F#####F###T#####
#M#MMTTT#F#FFF#F#F#FFF#TTT#FF.#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#.###T#####T#F###F#F#####F###F#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMT#TTT#TTTTT#FFFFFFF#FFF#F..#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FF...#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMMFF#F#FFFFFFF........#
###############################
```

## Case 324 (final failure)

Loop gain: `0.0054`. First loop F1 `0.4303` with 229 false positives and 57 misses. loop12 F1 `0.4356` with 230 false positives and 55 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......FFF#FFFFFFFFFF.#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#..F#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#M#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTM#.#MMM#
#.#####F#T#F#############.###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTMMM#
#F#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFF..#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#F#
#F#F#F#F#####F#F#####F#T#F###F#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#F#
#F#F###############F#T###F###F#
#F#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MMTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#TMM#
#M#.#T#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

### loop2

```text
###############################
#......FFF#FFFFFFFFFFF#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#.FF#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#T#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTT#.#MMM#
#.#####F#T#F#############F###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTTMM#
#.#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFF..#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#.#
#F#F#F#F#####F#F#####F#T#F###F#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#F#F###############F#T###F###.#
#F#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MTTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#MMM#
#M#.#T#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

### loop4

```text
###############################
#.....FFFF#FFFFFFFFFFF#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#.FF#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#T#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTT#.#MMM#
#.#####F#T#F#############F###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTTMM#
#F#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFFF.#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#F#
#F#F#F#F#####F#F#####F#T#F###F#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#F#F###############F#T###F###.#
#F#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MTTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#MMM#
#M#.#M#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

### loop6

```text
###############################
#.....FFFF#FFFFFFFFFFF#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#.FF#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#T#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTT#.#MMM#
#.#####F#T#F#############F###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTTMM#
#F#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFFF.#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#.#
#F#F#F#F#####F#F#####F#T#F###F#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#F#F###############F#T###F###.#
#F#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MTTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#MMM#
#M#.#T#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

### loop10

```text
###############################
#.....FFFF#FFFFFFFFFFF#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#.FF#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#T#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTT#.#MMM#
#.#####F#T#F#############F###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTTMM#
#F#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFFF.#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#.#
#F#F#F#F#####F#F#####F#T#F###.#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#F#F###############F#T###F###.#
#F#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MTTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#MMM#
#M#.#T#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

### loop12

```text
###############################
#.....FFFF#FFFFFFFFFFF#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#.FF#FFFFFFFFFFF#TTT#M#.#M#.#
#.###F#############T#T#T#.#M#.#
#..F#FFF#TTTTTTTTTTT#TTT#.#MMM#
#.#####F#T#F#############F###M#
#.#FFFFF#T#F#FFF#FFFFF#TTTTTMM#
#F#F###F#T#####F#F###F#T#######
#F#FFF#F#TGFFF#F#FFF#F#T#FFFF.#
#####F#F#####F#F###F#F#T#F#F###
#FFFFF#F#FFF#FFF#F#F#F#T#F#FF.#
#F#####F#F#######F#F#F#T#F###.#
#F#FFF#F#FFFFF#FFFFF#F#T#FFF#.#
#F#F#F#F#####F#F#####F#T#F###F#
#F#F#FFF#FFFFF#FFF#FFF#T#F#FFF#
#F#F#####F#######F#F###T###F#F#
#F#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#F#F###############F#T###F###.#
#.#FFFFFFF#FFFFFFFFF#TTT#F#FF.#
#.#######F#F###########T#S###.#
#MTTTT#F#FFF#TTTTTTT#TTT#TTT#.#
#M###T#F###F#T#####T#T#####T###
#M#.#T#FFFFF#T#FFF#TTT#TTT#MMM#
#M#.#M#######T###F#####T#M#.#M#
#M#.#MTTTT#TTT#TTT#FFFFM#M#.#M#
#M#.#####T#T###T#T#####M#M#.#M#
#MMMMMMT#TTT#TTT#TTTTTMM#M#.#M#
#######T#####T###########M###M#
#......MMMTTTT#FFFF......MMMMM#
###############################
```

## Case 123 (final failure)

Loop gain: `0.0014`. First loop F1 `0.4353` with 235 false positives and 53 misses. loop12 F1 `0.4366` with 237 false positives and 52 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#...#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#.#FFTTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TMM#
#M###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####.#
#MTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMM#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#.#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#...#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TTM#
#M###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#F#T###F#F###F###F###F#F#T#T#M#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####.#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMT#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#F#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#..F#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTM#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TMM#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFF.#
#####T#F#F#F#####F#F###F#####.#
#MTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMT#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#F#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#..F#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTM#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TTM#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFF.#
#####T#F#F#F#####F#F###F#####.#
#MTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMT#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#F#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#..F#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTM#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TTM#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#F#T###F#F###F###F###F#F#T#T#M#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFF.#
#####T#F#F#F#####F#F###F#####.#
#MTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMT#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#F#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

### loop12

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTMMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#..F#T#F#T#TTTTTTTTT#.#M#M#M#
#M#.#F#T###T###########.#M#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#MMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTM#M#
#M#F#########F#F#F###T#####T#M#
#MFF#FFF#F#FFF#F#FFF#TTTTT#TMM#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTM#
#####T#####F#F#F###F#F#F#T#T#M#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFF.#
#####T#F#F#F#####F#F###F#####.#
#MTTTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#FFF#FFF#F#FFF#F#FFF#F#F#.#.#
#M###F#####G#F#########F#.#.#.#
#MMT#TTTTT#T#FFF#FFFFFF.#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#FFF#...#.#.#
#.#####F#####F#F#F#F#F#.#.#.#.#
#.....FFFFFFFF#FFF#.#...#.....#
###############################
```

## Case 449 (final failure)

Loop gain: `0.0164`. First loop F1 `0.4215` with 237 false positives and 54 misses. loop12 F1 `0.4379` with 236 false positives and 49 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######.#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MMT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#.FF#T#TTTTT#TTT#TTT#TTT#TMM#
#M###F#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFFF#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFFF...MMMMM#.#
###############################
```

### loop2

```text
###############################
#MMMTT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FF...#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#.#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MMT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#..F#T#TTTTT#TTT#TTT#TTT#MMM#
#M###.#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFFF#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFF....MMMMM#.#
###############################
```

### loop4

```text
###############################
#MMMTT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FF...#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#T###F#########F#######F#####.#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MTT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#..F#T#TTTTT#TTT#TTT#TTT#TMM#
#M###.#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFFF#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFF....MMMMM#.#
###############################
```

### loop6

```text
###############################
#MMMTT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FF...#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#T###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MTT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#..F#T#TTTTT#TTT#TTT#TTT#TMM#
#M###.#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFFF#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFF....MMMMM#.#
###############################
```

### loop10

```text
###############################
#MMMTT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#F....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#T###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MTT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#..F#T#TTTTT#TTT#TTT#TTT#TMM#
#M###.#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFFF#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFF....MMMMM#.#
###############################
```

### loop12

```text
###############################
#MMMTT#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FF...#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#F..#
#T###F#########F#######F#####.#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MTT#FFTTT#TTT#TTT#F#T#T#FFF#.#
#M###F#T#T###T###T#F#T#T#######
#M#..F#T#TTTTT#TTT#TTT#TTT#TMM#
#M###.#T#######T###T#####M#M#M#
#MMM#.#TTTTTTT#T#FFTSFF.#MMM#M#
###M#.#######T#T#########.###M#
#.#M#..FFF#TTT#TTTTTTTTM#.#MMM#
#.#M#######T###########M###M#.#
#..MMMMMMMMM#FFFFFF....MMMMM#.#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0056`. First loop F1 `0.4333` with 244 false positives and 49 misses. loop12 F1 `0.4388` with 241 false positives and 48 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMM#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMT#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#.#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#.......#
###############################
```

### loop2

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMT#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMT#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#F.S#
#F#F#T#F#######F###F#F###F###.#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#F#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMTTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#.......#
###############################
```

### loop4

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMT#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMT#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#F.S#
#F#F#T#F#######F###F#F###F###.#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#F#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMTTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#.......#
###############################
```

### loop6

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMT#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMT#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###.#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#F#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMTTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#F......#
###############################
```

### loop10

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMT#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMT#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###.#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#F#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMTTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#F......#
###############################
```

### loop12

```text
###############################
#MMMMMMM#FFFFTTTTTTTTMMM#MMMMM#
#M#####T#####T#########M#M###M#
#M#MMT#T#TTT#T#FFFFF#TMM#M#.#M#
#M#M#T#T#T#T#T#F###F#T###M#.#M#
#MMM#T#TTT#T#T#FFF#F#TTMMM#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFF.#M#
#.#F###T#T###T#F###########F#M#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###.#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#.#
###F#T###F#F#F###F###F#F#F#F#.#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#.#
#####T#F###F#F#############F#.#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FF.#
#T#####F###F#F#####F#F###F###.#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FF.#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#F..#
#T#F###F###F###T###T###F#####.#
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#.#
#M###F###F###T###F#T#T###F#F#.#
#MMT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMTTT#FFF#TTT#F#TTT#F#...#.#
#.#####T#######T#F###F#F#####.#
#......MTTTTTTTT#FFFFF#F......#
###############################
```

## Case 130 (final failure)

Loop gain: `0.0009`. First loop F1 `0.4410` with 239 false positives and 50 misses. loop12 F1 `0.4419` with 245 false positives and 48 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.#.FFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#...#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#M#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#M#T#T#####F#############F#.#M#
#M#MTT#F#FFF#FFFFFFFFFFFF.#.#M#
#M###F#F#F###F#F#############M#
#MMM#.#FFF#FFF#F#TTTTTTMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFF....#MMM#
###############################
```

### loop2

```text
###############################
#...#.#FFFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TMM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#T#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTTTTMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop4

```text
###############################
#...#.#FFFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#.FF#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#M#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTTTTMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop6

```text
###############################
#...#.#FFFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#.FF#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTTTTMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop10

```text
###############################
#...#.#FFFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#M#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTTTTMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop12

```text
###############################
#...#.#FFFFFFFFFFFFFF....MMM#S#
#.#.#.#F#################M#M#M#
#.#.FF#F#FFFFFFFFFFFFF..#M#M#M#
#.###F#F###############.#M#M#M#
#..F#F#FFFFFFFFFFFFFFFF.#M#MMM#
###F#F#F###############F#M###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTM#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#M#
#F#F#T#F#F#F#F#F#####F#####T#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#M#
#####T#F###############T#####M#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#M#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTTTTMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFFFFF...#MMM#
###############################
```

## Case 137 (final failure)

Loop gain: `-0.0104`. First loop F1 `0.4531` with 229 false positives and 51 misses. loop12 F1 `0.4427` with 234 false positives and 53 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TTMMM#MMM#
#.#.###F#F#F###F###T#######M#.#
#.#F..#F#F#FFF#FFF#TTTTT#.#M#.#
#.###F#F#F###F#F#F#####T#.#M###
#.FFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTT#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###F#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#.FTTT#TTT#T#TTT#FFF#TTT#...#
#M###M#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTT#MMM#.#
#########F#F#####T#F#T###M#M###
#......FFF#F#FFF#T#F#TTMMM#MMM#
#.#.#########F#F#T###########M#
#.#.......FFFF#FFTTTTMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TTMMM#MMM#
#.#F###F#F#F###F###T#######M#.#
#F#FFF#F#F#FFF#FFF#TTTTT#.#M#.#
#F###F#F#F###F#F#F#####T#F#M###
#FFFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTM#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###F#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#FFTTT#TTT#T#TTT#FFF#TTT#...#
#M###T#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTM#MMM#.#
#########F#F#####T#F#T###M#M###
#.....FFFF#F#FFF#T#F#TTMMM#MMM#
#.#.#########F#F#T###########M#
#.#.......FFFF#FFTTTTMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TMMMM#MMM#
#.#F###F#F#F###F###T#######M#.#
#F#FFF#F#F#FFF#FFF#TTTTT#.#M#.#
#F###F#F#F###F#F#F#####T#.#M###
#FFFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTT#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###F#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#FFTTT#TTT#T#TTT#FFF#TTT#F..#
#M###T#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTM#MMM#.#
#########F#F#####T#F#T###M#M###
#......FFF#F#FFF#T#F#TMMMM#MMM#
#.#.#########F#F#T###########M#
#.#........FFF#FFTTTTMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TMMMM#MMM#
#.#F###F#F#F###F###T#######M#.#
#F#FFF#F#F#FFF#FFF#TTTTT#.#M#.#
#F###F#F#F###F#F#F#####T#.#M###
#FFFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTT#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###.#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#FFTTT#TTT#T#TTT#FFF#TTT#...#
#M###T#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTM#MMM#.#
#########F#F#####T#F#T###M#M###
#......FFF#F#FFF#T#F#TMMMM#MMM#
#.#.#########F#F#T###########M#
#.#........FFF#FFTTTTMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TMMMM#MMM#
#.#F###F#F#F###F###T#######M#.#
#F#FFF#F#F#FFF#FFF#TTTTT#.#M#.#
#F###F#F#F###F#F#F#####T#.#M###
#FFFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTT#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###F#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#FFTTT#TTT#T#TTT#FFF#TTT#...#
#M###T#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTM#MMM#.#
#########F#F#####T#F#T###M#M###
#......FFF#F#FFF#T#F#TMMMM#MMM#
#.#.#########F#F#T###########M#
#.#.......FFFF#FFTTTTMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.......#FFFFF#FFF#TTT..#MMMMM#
#.###########F###F#T#T###M###M#
#.#....FFF#FFF#FFF#T#TMMMM#MMM#
#.#F###F#F#F###F###T#######M#.#
#F#FFF#F#F#FFF#FFF#TTTTT#.#M#.#
#F###F#F#F###F#F#F#####T#.#M###
#FFFFF#F#F#FFF#F#F#TTT#T#F#TMM#
#F#####F###F###F#F#T#T#T#F###M#
#FFF#F#FFFFF#FFF#F#T#T#T#F#TTM#
###F#F#########F#F#S#T#T#F#T###
#FFF#FFF#FFFFF#F#F#F#TTT#F#TTT#
#G###F#F#F###F###F#F#####F###T#
#T#FFF#FFF#F#FFFFF#FFF#FFFFF#T#
#T#F#######F#####F###F#F###F#T#
#T#FFFFF#FFFFF#FFF#FFF#F#FFF#T#
#T#F###F#####F#F###F###F#####T#
#T#FFF#FFFFF#FFFFF#FFFFF#TTTTT#
#T#####F###F#########F###T###.#
#TTTTT#FFF#FFFFFFFFF#F#TTT#FF.#
#####T#########F#F###F#T###F###
#MTTTT#TTT#TTT#F#FFFFF#TTT#F#.#
#M#####T#T#T#T#####F#####T#F#.#
#M#FFTTT#TTT#T#TTT#FFF#TTT#...#
#M###T#######T#T#T#F###T#####.#
#MMMMM#FFFFF#TTT#T#F#TTM#MMM#.#
#########F#F#####T#F#T###M#M###
#......FFF#F#FFF#T#F#TMMMM#MMM#
#.#.#########F#F#T###########M#
#.#........FFF#FFTTTTMMMMMMMMM#
###############################
```

## Case 478 (final failure)

Loop gain: `0.0186`. First loop F1 `0.4250` with 239 false positives and 56 misses. loop12 F1 `0.4436` with 242 false positives and 49 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTF.#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####F###
#.#MTTTT#FFF#F#FFF#FFF#FF...#.#
#.#M#######F#F#########F#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MMT#TTTTTTTMMMMMM#MMS#
###############################
```

### loop2

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###T#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTF.#TTM#.#
#M#######F#################T#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTT#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#F#F#######T#F#F#F#F#F#T###F#F#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####.###
#.#TTTTT#FFF#F#FFF#FFF#FFF..#.#
#.#M#######F#F#########.#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MTT#TTTTTTMMMMMMM#MMS#
###############################
```

### loop4

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###T#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTF.#TTM#.#
#M#######F#################T#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTT#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#F#F#######T#F#F#F#F#F#T###F#F#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####.###
#.#TTTTT#FFF#F#FFF#FFF#FFF..#.#
#.#M#######F#F#########.#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MTT#TTTTTTMMMMMMM#MMS#
###############################
```

### loop6

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###T#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTF.#TTM#.#
#M#######F#################T#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTT#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#F#F#######T#F#F#F#F#F#T###F#F#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####F###
#.#TTTTT#FFF#F#FFF#FFF#FFFF.#.#
#.#M#######F#F#########.#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MTT#TTTTTTMMMMMMM#MMS#
###############################
```

### loop10

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###T#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTF.#TTM#.#
#M#######F#################T#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTT#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#F#F#######T#F#F#F#F#F#T###F#F#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####F###
#.#TTTTT#FFF#F#FFF#FFF#FFFF.#.#
#.#M#######F#F#########.#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MTT#TTTTTTMMMMMMM#MMS#
###############################
```

### loop12

```text
###############################
#...#MMTTTTTFFFF#TTT#TMMMM....#
#.###T#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTF.#TTM#.#
#M#######F#################T#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTTT#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#F#F#######T#F#F#F#F#F#T###F#F#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FFF#
#F#T#########F###F#F#########F#
#F#TTTTTTTTTTT#FFF#FFF#FFF#FFF#
#F###########T#F#####F###F#F###
#FFFFF#TTTTTTT#FFF#F#FFF#F#FF.#
###F#F#T#########F#F###F#F###.#
#.FF#F#T#FFF#FFF#F#F#FFFFF#F..#
#.#####T#F###F#F#F#F#F#####F###
#.#TTTTT#FFF#F#FFF#FFF#FFFF.#.#
#.#M#######F#F#########.#####.#
#.#M#MTTTT#F#FFFFFFFFFF.#MMM#.#
#.#M#M###T#F#############M#M#.#
#MMM#M#F#T#TTT#FFFFFFF...M#M#.#
#M###M#F#T#T#T###########M#M#.#
#MMMMM#..MTT#TTTTTTMMMMMMM#MMS#
###############################
```

## Case 439 (final failure)

Loop gain: `0.0262`. First loop F1 `0.4213` with 240 false positives and 54 misses. loop12 F1 `0.4475` with 238 false positives and 46 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#.FFFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###M#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#F#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#.FFFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#TTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###T#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#MMM#MMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..MTTTT#TTT#MMM#MMM#.#
###############################
```

### loop2

```text
###############################
#.....#..FFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#MMM#...#
#.#######F###F#####F###T#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#T#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#TTT#TMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..TTTTT#TTT#MMM#MMM#.#
###############################
```

### loop4

```text
###############################
#.....#..FFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#TMM#...#
#.#######F###F#####F###T#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#T#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###T#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#TTT#TMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..TTTTT#TTT#MMM#MMM#.#
###############################
```

### loop6

```text
###############################
#.....#.FFFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#TMM#...#
#.#######F###F#####F###T#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#T#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###T#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#TTT#TMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..TTTTT#TTT#MMM#MMM#.#
###############################
```

### loop10

```text
###############################
#.....#.FFFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#TMM#...#
#.#######F###F#####F###T#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#T#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###T#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#TTT#TMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..TTTTT#TTT#MMM#MMM#.#
###############################
```

### loop12

```text
###############################
#.....#.FFFF#FFF#FFFFF........#
#.#####F###F#F#F###F#F#######.#
#......F#FFF#F#FFFFF#F#TMM#...#
#.#######F###F#####F###T#M#####
#.#...FF#FFF#FFF#FFF#TTT#MMMMM#
#.#.###F###F###F#F###T#######M#
#.#.#FFFFF#F#FFF#F#F#TTT#FFF#M#
#.###F###F###F###F#F###T#F#F#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#T#
#.FFFFFF#FFF#F#FFFFF#TTT#FFF#T#
###F#######F#######F#T###F#F#T#
#FFF#TTTTSFF#FFF#FFF#T#FFF#F#T#
#F###T#######F#F#F###T#F###F#T#
#TTTTT#FFFFFFF#FFF#TTT#FFF#F#T#
#T#####F#F###F#####T#####F#F#T#
#TTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#F#F#F#####F#F#T#F#######T#
#MTT#FFF#F#FFFFF#F#T#F#FFG#TTT#
#M###F###F#F#####F#T#F#F#T#T#F#
#MTT#FFF#FFFFF#F#F#T#F#F#T#T#F#
###T#########F#F#F#T#F#F#T#T#F#
#MMM#TTTTTTT#FFF#F#T#FFF#T#T#F#
#M###M#####T###F#F#T#####T#T###
#MMM#M#FFF#T#FFF#FFTTTTTTT#TTM#
###M#M#F#F#T#F###############M#
#.#M#M#F#F#T#F#TTT#TTT#TTT#TMM#
#.#M#M#.#F#T###T#T#T#T#M#M#M#.#
#..MMM#.#..TTTTT#TTT#MMM#MMM#.#
###############################
```

## Case 72 (final failure)

Loop gain: `0.0100`. First loop F1 `0.4392` with 233 false positives and 53 misses. loop12 F1 `0.4492` with 232 false positives and 50 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#..F#FFFFTTT#TTT#F#MMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTM#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###M#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#M#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#FFF#FFFFF#FFF#F#FFF#F#...#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFFF.#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.FF#FFFFFFFFF#.......#.#
###############################
```

### loop2

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#..F#FFFFTTT#TTT#F#MMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTT#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###M#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#.FF#FFFFF#FFF#F#FFF#F#...#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFF..#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.F.#FFFFFFFFF#.......#.#
###############################
```

### loop4

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#.FF#FFFFTTT#TTT#F#TMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTT#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###M#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#.FF#FFFFF#FFF#F#FFF#F#...#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFF..#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.F.#FFFFFFFFF#.......#.#
###############################
```

### loop6

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#.FF#FFFFTTT#TTT#F#TMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTT#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###M#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#.FF#FFFFF#FFF#F#FFF#F#F..#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFF..#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.F.#FFFFFFFFF#.......#.#
###############################
```

### loop10

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#.FF#FFFFTTT#TTT#F#TMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTT#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###T#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#.FF#FFFFF#FFF#F#FFF#F#F..#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFF..#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.FF#FFFFFFFFF#.......#.#
###############################
```

### loop12

```text
###############################
#MMMMMMMTTTTTT#TTT#TTTMMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#.FF#FFFFTTT#TTT#F#TMMMM#.#
###M#.#F#F###########F#T#####.#
#MMM#F#FFF#F#FFFFFFF#F#TTTTT#.#
#M###F#####F#F#####F#F#####T#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#M#F#F###F#F#F#F#F#####T#T###M#
#M#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#M#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#M###F#F#F#############F###F###
#MMT#FFF#FFFFFFFFFFFFF#FFF#FF.#
###M#####F#######F#F###F#F###.#
#.#M#.FF#FFFFF#FFF#F#FFF#F#F..#
#.#M#.#F#######F#####F###.#####
#.#M#.#FFFFFFF#F#FFFFFF.#.....#
#.#M#.#######F#F#F#######.###.#
#MMM#..FFF#F#F#FFF#FFF..#.#...#
#M#######F#F#F#####F###.###.#.#
#MMMMMS.F.#FFFFFFFFF#.......#.#
###############################
```

## Case 312 (final failure)

Loop gain: `0.0017`. First loop F1 `0.4487` with 239 false positives and 46 misses. loop12 F1 `0.4505` with 237 false positives and 46 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#...FF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#.#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####F#T#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#.###F###F#F###F#######T#F###F#
#.FFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFFF.#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#...#.#M#
#M#T###T###########F#F#.###.#M#
#M#M#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#.###T#T#T#T#T#####M#M###M#
#MMM....#MTT#TTT#TTTMMMM#MMMMM#
###############################
```

### loop2

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#..FFF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#F#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####.#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#.###F###F#F###F#######T#F###.#
#.FFFF#FFF#FFF#F#TTT#TTT#FFFF.#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####.#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FF.#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#M#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFF..#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#...#.#M#
#T#T###T###########F#F#.###.#M#
#M#T#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#F###T#T#T#T#T#####M#M###M#
#MMM....#TTT#TTT#TTTMMMM#MMMMM#
###############################
```

### loop4

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#...FF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#F#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####.#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TMS#
#.###F###F#F###F#######T#F###.#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFF.#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####.#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FF.#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#M#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFFF.#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#F..#.#M#
#T#T###T###########F#F#.###.#M#
#M#T#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#F###T#T#T#T#T#####M#M###M#
#MMM....#TTT#TTT#TTTMMMM#MMMMM#
###############################
```

### loop6

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#...FF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#F#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####.#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TMS#
#.###F###F#F###F#######T#F###.#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFF.#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####.#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FF.#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#M#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFFF.#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#F..#.#M#
#T#T###T###########F#F#.###.#M#
#M#T#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#F###T#T#T#T#T#####M#M###M#
#MMM....#TTT#TTT#TTTMMMM#MMMMM#
###############################
```

### loop10

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#...FF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#.#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####.#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TMS#
#.###F###F#F###F#######T#F###.#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFF.#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####.#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FF.#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#M#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFFF.#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#F..#.#M#
#T#T###T###########F#F#.###.#M#
#M#T#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#F###T#T#T#T#T#####M#M###M#
#MMM....#TTT#TTT#TTTMMMM#MMMMM#
###############################
```

### loop12

```text
###############################
#.......FF#FFFFFFFFF#..MMMMM#.#
#.#####F#F###F#####F#F#M###M#.#
#.#...FF#FFFFF#F#FFF#F#M#.#M#.#
#.#.###########F#F#####M#.#M#.#
#.#.#FFFFFFF#FFFFF#TTTTM#.#M#.#
###F#######F#F#####T#####.#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TMS#
#.###F###F#F###F#######T#F###.#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFF.#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFF.#
#F#####F#F#F#F#T###T#T#F#####.#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FF.#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#M#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#M#
#T###F#######F###F#F###T###T#M#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#M#
###T###T#T#F###F#############M#
#TTT#TTT#T#F#F#FFFFF#FFFFFF.#M#
#T###T###T#F#F#####F#F#####.#M#
#T#TTT#TTT#FFFFFFF#F#F#F..#.#M#
#T#T###T###########F#F#.###.#M#
#M#T#F#TTT#TTT#TTT#FFF#MMM#..M#
#M#M#F###T#T#T#T#T#####M#M###M#
#MMM....#TTT#TTT#TTTMMMM#MMMMM#
###############################
```

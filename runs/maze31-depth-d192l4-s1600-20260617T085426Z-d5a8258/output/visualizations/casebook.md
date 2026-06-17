# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 349 (final failure)

Loop gain: `-0.0231`. First loop F1 `0.2982` with 207 false positives and 99 misses. loop12 F1 `0.2751` with 206 false positives and 105 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#F..FF#TTM..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MMMMM#TTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#.#####F#####.#.#####
#M#M#MMM#M#.#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop2

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#MTT#T#FFF#FF.F.#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###.###
#MMMMM#TTTTT#F#FFF#F#FFG#FF.#.#
#.###M#####T#F###F#F#####F###.#
#.#MTM#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop4

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#MTT#T#FFF#FF.F.#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###.###
#MMMMM#TTTTT#F#FFF#F#FFG#FF.#.#
#.###M#####T#F###F#F#####F###.#
#.#MTM#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop6

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#MTT#T#FFF#FF...#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###.###
#MMMMM#TTTTT#F#FFF#F#FFG#FF.#.#
#.###M#####T#F###F#F#####F###.#
#.#MTM#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop10

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#MTT#T#FFF#FF...#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###.###
#MMMMM#TTTTT#F#FFF#F#FFG#FF.#.#
#.###M#####T#F###F#F#####F###.#
#.#MTM#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

### loop12

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#MMM#M#.#...#
#M#M###M#T#T#F###F#####M###.#.#
#M#M#.#MTT#T#FFF#FF.F.#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###.###
#MMMMM#TTTTT#F#FFF#F#FFG#FF.#.#
#.###M#####T#F###F#F#####F###.#
#.#MTM#F#TTT#FFFFF#FFFFFFF#FF.#
###T#F#F#T###################.#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFF.#
#M#####F###T#####F###########.#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#.#
###T#F###F###T#F###F#F#F#####.#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#.#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#.#
###T#########T###F#F#####F#F#.#
#MMM#MTT#TTTTT#FFFFFFF#FFF#F..#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#.#.....#
#M#M###M#M#F#####F#####.#.#####
#M#M#MMM#M#F#FFF#FFFFFF.#.#...#
#M#M#M###M#.#F#F#########.###.#
#MMM#MMMMM..#.#...............#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0049`. First loop F1 `0.3091` with 211 false positives and 93 misses. loop12 F1 `0.3041` with 207 false positives and 95 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......MMM#......TMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMMTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#F#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.FF#FFF#FFF#TTSFFFF#F#.#MMM#
#M#####F#F#F###T#######F#.#M###
#M#.FFFF#FFF#FFT#FFFFFFF#F#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTT#M#M#
#F###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#FF.#MMM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFF.#MMMMMMMMMMM..#.....#
###############################
```

### loop2

```text
###############################
#......MMM#....FFTTMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#F#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.F.#.FF#FFF#TTSFFFF#F#.#MMM#
#M#####F#F#F###T#######F#.#M###
#M#.FFFF#FFF#FFT#FFFFFFF#.#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTM#M#M#
#.###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#.F.#MMM#
#F###F#F###F###F#F###T#.#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTM#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFFF#MMMMMMMMMMM..#.....#
###############################
```

### loop4

```text
###############################
#......MMM#....FFTMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTMTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#F#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.F.#.FF#FFF#TTSFFFF#F#.#MMM#
#M#####F#F#F###T#######F#.#M###
#M#.FFFF#FFF#FFT#FFFFFFF#.#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTM#M#M#
#.###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#.F.#MMM#
#F###F#F###F###F#F###T#.#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTM#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFFF#MMMMMMMMMMM..#.....#
###############################
```

### loop6

```text
###############################
#......MMM#....FFTMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMMMTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#F#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.F.#.FF#FFF#TTSFFFF#F#.#MMM#
#M#####.#F#F###T#######F#.#M###
#M#.F.FF#FFF#FFT#FFFFFFF#.#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTM#M#M#
#.###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#FF.#MMM#
#F###F#F###F###F#F###T#.#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTM#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFFF#MMMMMMMMMMM..#.....#
###############################
```

### loop10

```text
###############################
#......MMM#....FFTMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.F.#.FF#FFF#TTSFFFF#F#.#MMM#
#M#####F#F#F###T#######F#.#M###
#M#.F.FF#FFF#FFT#FFFFFFF#.#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTM#M#M#
#.###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#.F.#MMM#
#F###F#F###F###F#F###T#.#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTM#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFFF#MMMMMMMMMMM..#.....#
###############################
```

### loop12

```text
###############################
#......MMM#....FFTMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMMTTTTT#F#FF.#MMMMM#
#.###.#M#######F#####F#.#M#####
#.....#MMM#.FFFF#FFF#F#.#MMMMM#
#########T#F#####F#F#F#.#####M#
#MMMMMMMTT#F#FFFFF#FFF#F#...#M#
#M###.#######F#########F#.###M#
#M#.F.#.FF#FFF#TTSFFFF#F#.#MMM#
#M#####F#F#F###T#######F#.#M###
#M#.FFFF#FFF#FFT#FFFFFFF#.#MMM#
#M#F###########T#F#######F###M#
#M#FFFFF#TTTTTTT#F#F#FFFFFFF#M#
#M#####F#T#F#####F#F#F###F#F#M#
#MTTTTTTTT#F#FFF#F#FFF#FFF#F#M#
#############F#F#F#####F#####M#
#.FFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#.#F###F#F#########F#####T#M#M#
#.#FFF#F#F#FFF#FFF#TTTTTTM#M#M#
#.###F###F#F#F###F#T#######M#M#
#FFF#F#FFF#F#FFF#F#TTT#.F.#MMM#
#F###F#F###F###F#F###T#.#######
#F#FFF#F#FFF#F#FFF#TTT#.#.....#
#F#F#F#F###F#F###F#T###.#.#.#.#
#F#F#F#F#FFF#F#FFF#T#F....#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTM#...#...#
#M###T#####T#########M#.#.###.#
#MMTTTFFFF#MMMMMMMMMMM..#.....#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0048`. First loop F1 `0.3036` with 219 false positives and 93 misses. loop12 F1 `0.3084` with 212 false positives and 93 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MMM#T#FFFF.#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#T#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF.FFFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#F#.#
#M###F#F#F#F#####F#######F#.###
#M#FFF#FFF#F#FFFFF#F#FFFFF#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FFFF.#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMMT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMMMTT#FFFFF#F......#
###############################
```

### loop2

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MTM#M#FFF..#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#T#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF..FFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#.#.#
#M###F#F#F#F#####F#######.#.###
#M#FFF#FFF#F#FFFFF#F#FFFF.#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FF...#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMMT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMTTTT#FFFFF#F......#
###############################
```

### loop4

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MMM#M#FFF..#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#T#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF..FFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#..F#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#.#.#
#M###F#F#F#F#####F#######.#.###
#M#FFF#FFF#F#FFFFF#F#FFFF.#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FF...#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMTT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMTTTT#FFFFF#F......#
###############################
```

### loop6

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MMM#M#FFF..#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#M#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF.FFFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#.#.#
#M###F#F#F#F#####F#######.#.###
#M#FFF#FFF#F#FFFFF#F#FFFF.#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FF...#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMMT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMMTTT#FFFFF#F......#
###############################
```

### loop10

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MMM#M#FFF..#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#T#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF.FFFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#.FF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#.#.#
#M###F#F#F#F#####F#######.#.###
#M#FFF#FFF#F#FFFFF#F#FFFF.#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FF...#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMMT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMTTTT#FFFFF#F......#
###############################
```

### loop12

```text
###############################
#MMMMMMM#....MMMMMMMMMMM#MMMMM#
#M#####M#####M#########M#M###M#
#M#MMM#M#MMM#M#FFF..#MMM#M#.#M#
#M#M#M#M#M#M#T#F###.#M###M#.#M#
#MMM#M#MMM#M#T#FFF#.#MMMMM#.#M#
#.###M#####T#T#F#F#########F#T#
#.#.#MTM#TTT#T#F#FFFFFF.FFFF#T#
#.#.###T#T###T#F###########F#T#
#.#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#.#F#T#F#######F###F#F###F###F#
#..F#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#.FF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#MTTTT#F#F#F#FFF#FFFFFFFFF#.#.#
#M###F#F#F#F#####F#######.#.###
#M#FFF#FFF#F#FFFFF#F#FFFF.#...#
#M#####F###F#F#####F#F###.###.#
#MFFFFFF#FFF#F#FFTTT#F#FF.#...#
#M#######F###F###T#T#F###.#.###
#M#FFFFFFF#FFF#TTT#T#FFF#.#...#
#M#F###F###F###T###T###F#####.#
#M#.FF#F#FFF#TTT#F#T#G#FF...#.#
#M###F###F###T###F#T#T###F#.#.#
#MMM#.FFFF#FFT#FFF#T#T#FFF#.#.#
###M#######F#T###F#T#T#F###.#.#
#.#MMMMT#FFF#TTT#F#TTT#F#...#.#
#.#####M#######T#F###F#F#####.#
#......MMMMMTTTT#FFFFF#F......#
###############################
```

## Case 302 (final failure)

Loop gain: `-0.0055`. First loop F1 `0.3170` with 200 false positives and 106 misses. loop12 F1 `0.3115` with 197 false positives and 108 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..F#TTMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FFF..#FFF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF..F#.#.#MMMMM#M#M#
#M#F#F###S#F#####F#.#####M#M#M#
#M#FFF#FFF#F#FFF#F#FFF..#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#FFF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FF...#MMM#
###M#########F#######F#####M###
#MTTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTT#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#M###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#F#TMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#..F#TMMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FF...#FFF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF..F#.#.#MMMMM#M#M#
#M#F#F###S#F#####F#F#####M#M#M#
#M#FF.#FFF#F#FFF#F#FF...#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#.FF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####M###
#MMTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTT#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#..F#TMMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FF...#FFF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF..F#F#.#MMMMM#M#M#
#M#F#F###S#F#####F#F#####M#M#M#
#M#FF.#FFF#F#FFF#F#FF...#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#.FF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####M###
#MMTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTT#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#..F#TMMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FF...#FFF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF..F#F#.#MMMMM#M#M#
#M#F#F###S#F#####F#F#####M#M#M#
#M#FF.#FFF#F#FFF#F#FF...#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#.FF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####M###
#MMTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTT#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#..F#TMMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FF...#.FF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF.FF#F#.#MMMMM#M#M#
#M#F#F###S#F#####F#.#####M#M#M#
#M#FF.#FFF#F#FFF#F#FF...#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#.FF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####M###
#MMTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTM#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#..F#TMMMMMMMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#TTT#TTT#FF...#FFF#.#MMMMM#M#M#
#T#F###T#######F#F#.#M#####M#M#
#T#F#F#TTT#FF..F#.#.#MMMMM#M#M#
#M#F#F###S#F#####F#F#####M#M#M#
#M#FF.#FFF#F#FFF#F#FF...#MMM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#.#.FF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####M###
#MMTFF#FFFFF#FFFFF#FFF#FFF#TMM#
#M###F#F###F#####F###F#F#F###M#
#MTT#FFF#FFFFFFF#FFF#F#F#F#F#M#
#.#T###############F###F#F#F#M#
#.#T#FFFFF#FFFFFFF#F#FFF#F#TTM#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTM#
###T#F#######F#########F#####M#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTM#
#M#####T#T###########F#F#T#####
#MMMMMTT#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#MMMMM#
###.#.###T#T#F#####T#########M#
#...#...#MTT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

## Case 478 (final failure)

Loop gain: `0.0045`. First loop F1 `0.3423` with 203 false positives and 89 misses. loop12 F1 `0.3468` with 202 false positives and 88 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMMMMMM..FF#MMM#MMMMM....#
#.###M#####T#####T#T#M###M###.#
#MMMMM#.FF#TTTTTTT#TTM..#MMM#.#
#M#######F#################M#.#
#MMMMMMM#FFFFFFFFFFFFF#MMTTM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#F#
#.#T#####F#F#####F#F#####F#F###
#.#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFFF#TTTTTTT#FFF#F#FFF#F#...#
###.#.#T#########F#F###F#.###.#
#...#F#T#FFF#FFF#F#F#FFFF.#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#.#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFF......FT#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMTTT#TMS#
###############################
```

### loop2

```text
###############################
#...#MMMMMMM.FFF#TMM#MMMMM....#
#.###M#####T#####T#T#M###M###.#
#MMMMM#.FF#TTTTTTT#TTT..#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFF.#TTTTTTT#FFF#F#FFF#.#...#
###F#.#T#########F#F###.#.###.#
#...#.#T#FFF#FFF#F#F#FF...#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#F#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFFF.....FT#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMMTT#TMS#
###############################
```

### loop4

```text
###############################
#...#MMMMMMM.FFF#TMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#.FF#TTTTTTT#TTT..#MMM#.#
#M#######F#################M#.#
#MMMMMTM#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFF.#TTTTTTT#FFF#F#FFF#.#...#
###F#.#T#########F#F###F#.###.#
#.F.#.#T#FFF#FFF#F#F#FF...#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#F#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFF.......T#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMMTT#TMS#
###############################
```

### loop6

```text
###############################
#...#MMMMMMM.FFF#TMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#.FF#TTTTTTT#TTT..#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#.#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#FFF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFF.#TTTTTTT#FFF#F#FFF#.#...#
###F#.#T#########F#F###.#.###.#
#...#.#T#FFF#FFF#F#F#FF...#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#F#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFFF......T#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMMTT#TMS#
###############################
```

### loop10

```text
###############################
#...#MMMMMMM.FFF#TMM#MMMMM....#
#.###M#####T#####T#T#M###M###.#
#MMMMM#.FF#TTTTTTT#TTT..#MMM#.#
#M#######F#################M#.#
#MMMMMMM#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#F#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFF.#TTTTTTT#FFF#F#FFF#.#...#
###F#.#T#########F#F###F#.###.#
#...#.#T#FFF#FFF#F#F#FF...#...#
#.#####M#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#F#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFF......FT#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMMTT#TMS#
###############################
```

### loop12

```text
###############################
#...#MMMMMMM.FFF#TMM#MMMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMM#.FF#TTTTTTT#TTT..#MMM#.#
#M#######F#################M#.#
#MMMMMMT#FFFFFFFFFFFFF#MMMMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#.#
#.#F#######T#F#F#F#F#F#T###F#.#
#F#F#TTTTTTT#F#FFF#F#F#TTT#FFF#
###F#T#############F#F###T###F#
#.FF#T#FFFFFFFFFFFFF#F#F#GFF#F#
#F###T#F###F#F#######F#F###F#F#
#F#TTT#FFF#F#F#FFF#F#FFFFF#F#.#
#.#T#####F#F#####F#F#####F#F###
#F#T#FFFFF#FFFFF#F#FFFFFFF#FF.#
#.#T#########F###F#F#########.#
#.#TTTTTTTTTTT#FFF#FFF#FFF#F..#
#.###########T#F#####F###F#.###
#.FFF.#TTTTTTT#FFF#F#FFF#.#...#
###F#.#T#########F#F###F#.###.#
#...#.#T#FFF#FFF#F#F#FF...#...#
#.#####T#F###F#F#F#F#F#####.###
#.#MMMMM#FFF#F#FFF#FFF#.....#.#
#.#M#######F#F#########.#####.#
#.#M#MMMMM#F#FFFFFFFF...#MMM#F#
#.#M#M###M#.#############M#T#F#
#MMM#M#.#M#MTM#FFFF......T#T#F#
#M###M#.#M#M#M###########T#T#.#
#MMMMM#..MMM#MMMMMMMMMMMTT#TMS#
###############################
```

## Case 439 (final failure)

Loop gain: `-0.0132`. First loop F1 `0.3709` with 186 false positives and 82 misses. loop12 F1 `0.3576` with 188 false positives and 85 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#.....#...#.............#
#.#####.###.#.#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.FF#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#...#M#
#.###.###F###F###F#F###T#.#.#M#
#...F.#F#F#FFFFF#FFFFF#T#F#F#M#
#######F#F###F#F#######T###F#M#
#..FFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#T#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#T#
###T#.#F#F#####F#F#T#F#######T#
#MMT#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#.F.#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMT#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop2

```text
###############################
#.....#.....#...#.............#
#.#####.###.#F#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.F.#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#.F.#M#
#.###F###F###F###F#F###T#F#.#M#
#...FF#F#F#FFFFF#FFFFF#T#F#.#M#
#######F#F###F#F#######T###F#M#
#..FFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#.#F#F#####F#F#T#F#######T#
#MTM#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#.F.#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop4

```text
###############################
#.....#.....#...#.............#
#.#####.###.#F#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.F.#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#.F.#M#
#.###F###F###F###F#F###T#F#.#M#
#...FF#F#F#FFFFF#FFFFF#T#F#.#M#
#######F#F###F#F#######T###F#M#
#..FFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#.#F#F#####F#F#T#F#######T#
#MTM#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#.F.#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop6

```text
###############################
#.....#.....#...#.............#
#.#####.###.#.#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.F.#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#.F.#M#
#.###F###F###F###F#F###T#F#.#M#
#...FF#F#F#FFFFF#FFFFF#T#F#.#M#
#######F#F###F#F#######T###F#M#
#..FFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#F#F#F#####F#F#T#F#######T#
#MTM#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#.F.#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop10

```text
###############################
#.....#.....#...#.............#
#.#####.###.#F#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.F.#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#.F.#M#
#.###F###F###F###F#F###T#F#.#M#
#...FF#F#F#FFFFF#FFFFF#T#F#.#M#
#######F#F###F#F#######T###F#M#
#..FFFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#.#F#F#####F#F#T#F#######T#
#MTM#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#.F.#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

### loop12

```text
###############################
#.....#.....#...#.............#
#.#####.###.#F#F###F#F#######.#
#.......#...#F#FFFFF#F#MMM#...#
#.#######.###F#####F###M#M#####
#.#.....#.F.#FFF#FFF#TTM#MMMMM#
#.#.###.###F###F#F###T#######M#
#.#.#.FFFF#F#FFF#F#F#TTT#.F.#M#
#.###F###F###F###F#F###T#F#.#M#
#..FFF#F#F#FFFFF#FFFFF#T#F#.#M#
#######F#F###F#F#######T###F#M#
#...FFFF#FFF#F#FFFFF#TTT#FFF#M#
###F#######F#######F#T###F#F#M#
#.FF#TTTTSFF#FFF#FFF#T#FFF#F#M#
#.###T#######F#F#F###T#F###F#M#
#MTTTT#FFFFFFF#FFF#TTT#FFF#F#M#
#M#####F#F###F#####T#####F#F#M#
#MTT#F#F#F#FFF#FFF#T#FFFFF#F#M#
###T#.#F#F#####F#F#T#F#######T#
#MTM#.FF#F#FFFFF#F#T#F#FFG#TTT#
#M###.###F#F#####F#T#F#F#T#T#F#
#MMM#...#FFFFF#F#F#T#F#F#T#T#F#
###M#########F#F#F#T#F#F#T#T#F#
#MMM#MMMTTTT#FFF#F#T#FFF#T#T#.#
#M###M#####T###F#F#T#####T#T###
#MMM#M#...#M#FFF#FFTTTTMTT#TMM#
###M#M#.#.#M#.###############M#
#.#M#M#.#.#M#.#TTT#TTT#MMM#MMM#
#.#M#M#.#.#M###T#T#T#T#M#M#M#.#
#..MMM#.#..MMMMM#MMM#MMM#MMM#.#
###############################
```

## Case 131 (final failure)

Loop gain: `0.0090`. First loop F1 `0.3555` with 185 false positives and 87 misses. loop12 F1 `0.3645` with 188 false positives and 84 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#..MMG#
#M#######M#M#M#F#F#M#M#.#.#T###
#MMMMM#.#MMM#M#FFF#M#M#.#.#TTM#
#####T#F#####M#F###M#M#.#####M#
#MMMTT#FF.#.#M#F#MMM#M#.....FM#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#TTT#TTT#TMM#MMM#M#
###########F###T###T###M#M#M#M#
#.#FFFFFFF#F#F#TFF#TTT#MTM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#........F#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

### loop2

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#F#FFTMG#
#M#######M#M#M#.#.#M#M#F#F#T###
#MMMMM#.#MMM#M#...#M#M#F#F#TTT#
#####T#F#####M#F###M#M#.#####T#
#MMMTT#FF.#.#M#.#MMM#M#..FFFFT#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#MTT#TTT#TMM#MMM#T#
###########F###T###T###M#M#M#T#
#.#FFFFFFF#F#F#TFF#TTT#MTM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#.......F.#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

### loop4

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#F#FFTMG#
#M#######M#M#M#.#.#M#M#F#F#T###
#MMMMM#.#MMM#M#...#M#M#F#F#TTT#
#####T#F#####M#F###M#M#.#####T#
#MMMTT#FF.#.#M#.#MMM#M#.FFFFFT#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#MTT#TTT#TMM#MMM#T#
###########F###T###T###M#M#M#M#
#.#FFFFFFF#F#F#TFF#TTT#MMM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#.........#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

### loop6

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#F#FFTMG#
#M#######M#M#M#.#.#M#M#F#F#T###
#MMMMM#.#MMM#M#...#M#M#F#F#TTT#
#####T#F#####M#F###M#M#.#####T#
#MMMTT#FF.#.#M#.#MMM#M#.FFFFFT#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#MTT#TTT#TMM#MMM#T#
###########F###T###T###M#M#M#M#
#.#FFFFFFF#F#F#TFF#TTT#MMM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#.........#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

### loop10

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#F#FFTMG#
#M#######M#M#M#F#.#M#M#F#F#T###
#MMMMM#.#MMM#M#...#M#M#F#F#TTT#
#####T#F#####M#F###M#M#.#####T#
#MMMTT#FF.#.#M#.#MMM#M#.FFFFFT#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#MTT#TTT#TMM#MMM#T#
###########F###T###T###M#M#M#M#
#.#FFFFFFF#F#F#TFF#TTT#MMM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#.........#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

### loop12

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#F#FFTMG#
#M#######M#M#M#.#.#M#M#F#F#T###
#MMMMM#.#MMM#M#...#M#M#F#F#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMMTT#FF.#.#M#.#MMM#M#.FFFFFT#
#M#####F#F#.#M###T###M#######T#
#MMTTTSF#FFF#MTT#TTT#TMM#MMM#T#
###########F###T###T###M#M#M#M#
#.#FFFFFFF#F#F#TFF#TTT#MMM#MMM#
#.#F#F#####F#F#T#####T#########
#.#F#FFFFFFF#F#TTTTT#TTT#.....#
#.#F#########F#####T###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#F..#
#.###F###F#####F#T#F#F#T#F#F#.#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#.#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#.FF#FFF#F#FFFFF#FFF#T#T#FFF#.#
###F#F#F#F#######F###T#T###F###
#.#F#F#F#FFF#FFF#F#TTT#TTT#FF.#
#.#F###F###F#F#F#F#T#F###T###.#
#.#.#.FF#FFF#F#F#F#T#FFF#TTMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#FFF#F#F#F#TTTTMMM..#M#
#.#####.###F#F#F#F#######M###M#
#.........#FFF#FFF#FFFF.#MMM#M#
#.###############F#F###.###M#M#
#...................#.....#MMM#
###############################
```

## Case 348 (final failure)

Loop gain: `0.0083`. First loop F1 `0.3566` with 177 false positives and 90 misses. loop12 F1 `0.3650` with 172 false positives and 89 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#.F.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#FF...#M#...FTTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#.#F#FFF#FFF#TMMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.F.#.#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###F#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####.#M#
#.#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#FFF#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TTM#M#
#######F#F#F#F#F###T#####T#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#..FFFFFFFFFFF#TTTTT#TTTMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMMMM#MMMMMMMMM#...#
###############################
```

### loop2

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#FF.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#F....#M#....TTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#F#F#FFF#FFF#TMMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.FF#F#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#...#M#
#####F#F#F#####T#######F###.#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####.#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#.#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#.F.#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####M#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#...FFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMTTT#MMMMMMMMM#...#
###############################
```

### loop4

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#.F.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#F....#M#....TTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#F#F#FFF#FFF#TMMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.FF#F#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#...#M#
#####F#F#F#####T#######F###.#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####.#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#.#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#.F.#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####T#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#.#######F###F#####T#T#######.#
#...FFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMTTT#MMMMMMMMM#...#
###############################
```

### loop6

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#.F.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#F....#M#....TTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#F#F#FFF#FFF#TTMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.F.#F#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#...#M#
#####F#F#F#####T#######F###.#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#.#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#.F.#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####T#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#...FFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMTTT#MMMMMMMMM#...#
###############################
```

### loop10

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#.F.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#F....#M#....TTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#F#F#FFF#FFF#TTMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.F.#F#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#...#M#
#####F#F#F#####T#######F###.#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#.#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#.F.#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####T#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#...FFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMTTT#MMMMMMMMM#...#
###############################
```

### loop12

```text
###############################
#...#...........#..MMMMTTTTT#S#
#.###.#.#####.###.#M#######T#M#
#.....#.....#.#.F.#MMM#.FF#T#T#
#.#########.###F#####M#.###T#T#
#...#.....#...#F....#M#....TTT#
###.#.###.###.#####F#T#######F#
#.#.#.#.#F#F#FFF#FFF#TTMMMMM#.#
#.#.#.#.#F#F###F#F#####.###M###
#.#.FF#F#FFFFF#FFF#FFF#F#.#MMM#
#.#####F#####F#######F#F#.###M#
#.FFFFFF#FFFFF#TTTTTGF#F#...#M#
#####F#F#F#####T#######F###.#M#
#.FFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#.#####F###F#T#######F#####.#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#.#M#
#.#######F#########T#F###F#.#M#
#.FF#FFF#F#FFFFFFF#T#FFF#.F.#M#
#.#F#F#F#F#F#####F#T###F#####M#
#.#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####T#M#M#
#.FFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#.#######F###F#####T#T#######.#
#...FFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMMMM#MTTTT#TTT#FFF#FF...#MMM#
#M###M#M###T#T#####F###.#####M#
#MMM#MMM#F#TTT#TTT#FFFF.#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMMMMMMTTT#MMMMMMMMM#...#
###############################
```

## Case 277 (final failure)

Loop gain: `0.0172`. First loop F1 `0.3480` with 191 false positives and 105 misses. loop12 F1 `0.3652` with 192 false positives and 100 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM......#MMMMMMMMTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.FF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#FFFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#M#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTMTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FF...#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.....#MTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

### loop2

```text
###############################
#.#MMM......#MMMMMMMTTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.F.#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#.FFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FFFF.#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTMTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

### loop4

```text
###############################
#.#MMM......#MMMMMMMTTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.F.#FFF#FFF#FFFFFFF#TMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#.FFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FFF..#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTTTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

### loop6

```text
###############################
#.#MMM......#MMMMMMMTTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.F.#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#.FFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FFF..#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTTTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

### loop10

```text
###############################
#.#MMM......#MMMMMMMTTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.F.#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#.FFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FFF..#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTTTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

### loop12

```text
###############################
#.#MMM......#MMMMMMMTTMM#.....#
#.#M#M###.###M#######F#M#.###.#
#.#M#MMM#.#MMM#F#FFF#F#T#.#.#.#
#.#M###M###M###F#F#F###T#.#.#.#
#MMM..#MMMMMFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#.....FFFFFF#FFF#FFFFF#MMMMM#
#M#.#########F#F#######F#M###M#
#M#.F.FF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#.FFFF#F#FFFFFFF#FFFFF#.#M#M#
#M#F#########F###F#F###F#F#T#M#
#M#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#M#############F#####F#F###T#M#
#MTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#MTT#T#F#FFF#FFF#F#FFFFF#TTT#M#
#M#T#T#F#F#######F#########M#M#
#M#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#M#####F###F#F#######T#######M#
#M#FFTTT#F#F#F#FFTTTTT#FFFF.#M#
#M###T#T#F#F#F###T#####F#####M#
#MMTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.....#TTT#FFFFT#F#F#FFFFF#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#MTTTT#TTT#T#F#FFFF...#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMMMMM#TTT#FFFFF........#
###############################
```

## Case 449 (final failure)

Loop gain: `-0.0060`. First loop F1 `0.3717` with 208 false positives and 76 misses. loop12 F1 `0.3656` with 211 false positives and 77 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMM#MTT#FF.#F..#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MMT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFF..#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#T#######T###T#####T#T#F#F#.#
#MTM#FFTTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTM#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMMMM#MTT#FFF#FFF#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MMT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#M#######T###T#####T#T#F#F#.#
#MTM#.FMTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTM#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMMMM#MTT#FF.#FFF#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MTT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#M#######T###T#####T#T#F#F#.#
#MTM#.FMTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTT#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMMMM#MTT#FFF#FFF#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MMT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#M#######T###T#####T#T#F#F#.#
#MTM#.FMTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTM#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMMMM#MTT#FFF#FFF#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MTT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#M#######T###T#####T#T#F#F#.#
#MTM#.FMTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTM#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

### loop12

```text
###############################
#MMMMM#MTT#FFF#FFF#...........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###M#####T#F###F#######.#####.#
#MMT#FFF#T#F#F#FFFFFFFF.#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#FFFF.#
#M#F###F#####F###F#########F###
#M#FFF#FFFFF#F#FFF#FFFFFFF#FF.#
#M###F#####F#F#F#######F#F#F#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TTT#F#FFF#F#FFFFF#FFF#F#F#.#
#.###T#F#######F#F###F#F#F#F#.#
#.FF#T#F#FFFFFFF#F#FFF#FFF#F#.#
#.###T#F#F#######F#F#########.#
#.#TTT#FFF#TTTTT#FFF#TTTFF#FF.#
#.#M#######T###T#####T#T#F#F#.#
#MTM#.FMTT#TTT#TTT#F#T#T#FF.#.#
#M###.#M#T###T###T#F#T#T#######
#M#...#M#TTTTT#TTT#TTT#TTT#MMM#
#M###.#M#######T###T#####M#M#M#
#MMM#.#MMMMMTT#T#FFTSFFF#MMM#M#
###M#.#######M#T#########.###M#
#.#M#.....#MMM#TTTTTTTTT#.#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFMMMMM#.#
###############################
```

## Case 123 (final failure)

Loop gain: `-0.0004`. First loop F1 `0.3676` with 209 false positives and 80 misses. loop12 F1 `0.3673` with 205 false positives and 81 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#F#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#FFFFFFFFFFF#FFF#TTT#MTMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#T#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#T#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTMTT#F#F#FFFFF#F#F#FFFFFFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FFFF...#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MMT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

### loop2

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#.#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#FFFFFFFFFFF#FFF#TTT#MMMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#T#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#M#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTTTT#F#F#FFFFF#F#F#FF..FFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FFF....#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MTT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

### loop4

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#.#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#.FFFFFFFFFF#FFF#TTT#MMMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#T#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#M#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTTTT#F#F#FFFFF#F#F#FF..FFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FFF....#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MTT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

### loop6

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#.#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#FFFFFFFFFFF#FFF#TTT#MMMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#T#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#M#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTTTT#F#F#FFFFF#F#F#FF..FFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FF.....#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MTT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

### loop10

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#.#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#FFFFFFFFFFF#FFF#TTT#MMMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#T#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#M#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTTTT#F#F#FFFFF#F#F#FF..FFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FF.....#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MTT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

### loop12

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#T#T#######T###M#M#M#
#M#...#M#.#T#TTTTTTTTT#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MTTTT#FFFFF#TTTTM#MMM#M#
#M#############F###T###M#####M#
#M#FFFFFFFFFFF#FFF#TTT#MMMMM#M#
#M#F#########F#F#F###T#####M#M#
#MFF#FFF#F#FFF#F#FFF#TTTTM#MMM#
#M###F#F#F#F###F###F#####M#####
#MTTTT#FFF#FFF#FFF#FFF#F#M#TMM#
#####T#####F#F#F###F#F#F#T#T#M#
#.#TTT#FFF#F#F#F#FFF#F#F#T#T#M#
#.#T###F#F###F###F###F#F#T#T#M#
#MTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#M###F#F#F#F###F#############T#
#M#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#M#T#T#F#####F#####F#F#########
#MTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#MTTTT#F#F#FFFFF#F#F#FF..FFF#.#
#M#####F#F#####F#F#F#F#####F#.#
#M#.FF#FFF#F#FFF#F#FFF#.#.#.#.#
#M###F#####G#F#########.#.#.#.#
#MMM#TTTTT#T#FFF#FFF....#.#.#.#
###M#T###T#T###F#F#######.#.#.#
#.#MTT#F#TTTFF#F#F#...#...#.#.#
#.#####F#####F#F#F#.#.#.#.#.#.#
#.....FFFFFFFF#...#.#...#.....#
###############################
```

## Case 470 (final failure)

Loop gain: `0.0074`. First loop F1 `0.3653` with 196 false positives and 82 misses. loop12 F1 `0.3727` with 196 false positives and 80 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#.#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#.F.FF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#M#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FF.#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###M#F#######T#T#F#T###M###.#M#
#MMM#.#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop2

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#.#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####.#F#F#F#F#F###F#####.#.#
#.#.F.FF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#F.M#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FF.#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MMM#.#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop4

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#F#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####.#F#F#F#F#F###F#####.#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#F.M#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###M#F#######T#T#F#M###M###.#M#
#MMM#.#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop6

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#F#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####F#F#F#F#F#F###F#####.#.#
#.#FF.FF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FF.#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###M#F#######T#T#F#T###M###.#M#
#MMM#.#TTTTT#T#T#F#MMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop10

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#F#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####.#F#F#F#F#F###F#####.#.#
#.#FF.FF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#F.M#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FF.#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###M#F#######T#T#F#T###M###.#M#
#MMM#.#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop12

```text
###############################
#...#.................#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#...#...#F#FFFFF#F#F..#...#.#
#.#####.#F#F#F#####F#F#####.#.#
#.....#.#F#F#F#FFF#F#FF...#.#.#
#.#####.#F#F#F#F#F###F#####.#.#
#.#.FFFF#F#F#FFF#FFFFF#FFF#.#.#
###F#####F#############F#F#.#.#
#.FF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#.#######F#F###F#F#F#F#########
#.FFFFFF#FFF#F#FFF#F#FFFSTTTMM#
#######F#####F#####F###F#####M#
#MTT#F#FFFFF#FFFFF#FFF#FFF#F.M#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FF.#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#TTTTT#T#F#F#F#F#TTT#TMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MMM#.#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0067`. First loop F1 `0.3831` with 204 false positives and 73 misses. loop12 F1 `0.3764` with 199 false positives and 76 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######M#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#...FF#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######.#M#
#.F.#FFF#FFF#FFFFF#F#TTTTT#.#M#
###.###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#MTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#FF.#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MTM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#TTT#FFFFT#FFF#F#.#M#MMM#
#M###.#T#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#...#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######M#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######F#M#
#.F.#.FF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#..FFF#FFFFFFF#FFFFF#TTT#T#MTM#
#.#######F#####F###F###T#T#M###
#.#TTTTT#FFFFF#FFF#FFF#T#TTM#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#.F.#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MMM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#MTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######T#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######.#M#
#.F.#.FF#FFF#FFFFF#F#TTTTT#.#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#TTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTM#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#.F.#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MMM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#MTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######T#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######.#M#
#.F.#.FF#FFF#FFFFF#F#TTTTT#.#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#MTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTT#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#.F.#
#.#T#F###T###F#F###F#####T###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MTM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#TTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#...#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######T#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#...F.#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######.#M#
#.F.#.FF#FFF#FFFFF#F#TTTTT#F#M#
###F###F#####F#####F#T###T###M#
#.FFFF#FFFFFFF#FFFFF#TTT#T#MTM#
#.#######F#####F###F###T#T#M###
#.#TTTTT#FFFFF#FFF#FFF#T#TTM#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#.F.#
#.#T#F###T###F#F###F#####M###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MTM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#MTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.....#.......FFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#.FF#FFFFFFF#FFF#FFF#TTT#
###.#.#####F###F#####F#######T#
#...#...FF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#FFF#F#FFF#F#F#FFFFFFF#M#
#.###F###F#F#####F#F#######.#M#
#.F.#.FF#FFF#FFFFF#F#TTTTT#.#M#
###F###F#####F#####F#T###T###M#
#..FFF#FFFFFFF#FFFFF#TTT#T#MTM#
#.#######F#####F###F###T#T#T###
#.#TTTTT#FFFFF#FFF#FFF#T#TTM#.#
#.#T###T#####F###F###F#T#####.#
#.#T#F#TTT#FFF#FFF#FFF#TTT#.F.#
#.#T#F###T###F#F###F#####M###.#
#.#T#FFF#TTT#F#FFF#F#FFF#MTM#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#F..#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#...#MTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#...#MMM#.#
#.#M###M#####T#####F#.#.###M#.#
#.#MMM#MMM#.#T#FFFFFF.#...#MMM#
#.###M###M#.#T###############M#
#...#MMMMM..#TTTTTTMMMMMMMMMMM#
###############################
```

## Case 191 (final failure)

Loop gain: `-0.0069`. First loop F1 `0.3841` with 184 false positives and 95 misses. loop12 F1 `0.3772` with 188 false positives and 96 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#MTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###T#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#.FF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#T#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#.#M#.#
#M#####T#T###F#F#########.#M###
#MTTTTTT#TFF#F#FFF#FFF#.F.#MMM#
#########T###F#####F#F#.#####M#
#...#TTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#.FF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#TTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#.FF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#M#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#.#M#.#
#M#####T#T###F#F#########.#M###
#MTTTTTT#TFF#F#FFF#FFF#FF.#MMM#
#########T###F#####F#F#.#####M#
#...#TTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#.FF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#T#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#TTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#.FF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#M#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#F#M#.#
#M#####T#T###F#F#########.#M###
#TTTTTTT#TFF#F#FFF#FFF#FF.#MMM#
#########T###F#####F#F#.#####M#
#...#TTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#.FF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#T#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#MTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#.FF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#M#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#F#M#.#
#M#####T#T###F#F#########.#M###
#TTTTTTT#TFF#F#FFF#FFF#FF.#MMM#
#########T###F#####F#F#.#####M#
#...#MTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#FFF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#T#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#TTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#FFF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#M#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#.#M#.#
#M#####T#T###F#F#########.#M###
#MTTTTTT#TFF#F#FFF#FFF#FF.#MMM#
#########T###F#####F#F#.#####M#
#...#MTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#.FF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#T#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop12

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########T###M#####.###.#
#M#M#MMMMM#MTTTTFF#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#TTT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###T#.#####F#
#M#MTTTT#FFFFF#F#F#F#T#F#FFF#F#
#M#######F#####F#F#F#T#F#F#F#G#
#M#TTT#FFFFF#FFF#F#FFT#FFF#F#T#
#M#T#T#F###F#F###F###T#####F#T#
#MTT#T#F#FFF#F#F#FFF#TTTTS#F#T#
#####T###F#F#F#F###F#######F#T#
#FFF#T#FFF#F#F#FFF#F#FFF#FFF#T#
###F#T#F#####F###F###F#F#F###M#
#FFF#T#FFF#FFF#FFF#FFF#FFF#TTM#
#.#F#T###F#F###F###F#F#####T#.#
#.#F#T#TTT#F#FFF#FFF#F#FFF#M#.#
#.###T#T#T#F###F#F###F#F#F#M#.#
#MTTTT#T#T#FFF#FFFFF#FFF#F#M#.#
#M#####T#T###F#F#########.#M###
#MTTTTTT#TFF#F#FFF#FFF#FF.#MMM#
#########T###F#####F#F#.#####M#
#...#TTTTT#FFF#FFFFF#FF.#..MMM#
###.#M#####F###F###########M#.#
#...#M#FFF#FFFFF#FFTTT#..MMM#.#
#.###M#.#F#########T#T#.#M###.#
#.#MMM#.#FFFFTTTTT#T#T#.#MMM#.#
#.#M#########T###T#T#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

## Case 306 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.3816` with 186 false positives and 83 misses. loop12 F1 `0.3799` with 188 false positives and 83 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#MMMMMMMMT#FFFFSFF....#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMTTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTT#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#F#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#..FFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#.#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTMMMMMMMMMM#...#
###############################
```

### loop2

```text
###############################
#.......#MMMMMMMMT#FFFFSFFF...#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMMTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTM#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#T#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTTMMMMMMMMM#...#
###############################
```

### loop4

```text
###############################
#.......#MMMMMMMMT#FFFFSFFF...#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMMTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTM#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTTMMMMMMMMM#...#
###############################
```

### loop6

```text
###############################
#.......#MMMMMMMMT#FFFFSFFF...#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMMTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTM#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTMMMMMMMMMM#...#
###############################
```

### loop10

```text
###############################
#.......#MMMMMMMMT#FFFFSFFF...#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMMTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTM#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTTMMMMMMMMM#...#
###############################
```

### loop12

```text
###############################
#.......#MMMMMMMMT#FFFFSFFF...#
#.###.#.#M#######T#F###T#F#####
#.#...#.#M#MMMMMTT#FFF#T#F#MMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMTTT#F#F#TTTMM#M#
#.###.#.#M#######T#F#F#####.#M#
#...#.#.#MMMMMTT#T#FFF#TTM#.#M#
#.###.#.#######T#T#####T#M#.#M#
#.#.F.#FFF#TTTTT#TTTTTTT#M#.#M#
#.#F###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFFFF#TTM#MMM#
###F#F###T#######F###F#T###M###
#.FF#FFF#T#FFFFF#FFF#F#TTT#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#F#M#
#F###F#F#F#F#F#######F###T#F#M#
#.FF#FFF#FFF#FFF#FFF#FFF#T#F#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#T#TTM#
#.#F#F#F#######F#######F#T#T#.#
#.#F#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###T#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TTMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFFFF..MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTTMMMMMMMMM#...#
###############################
```

## Case 373 (final failure)

Loop gain: `-0.0077`. First loop F1 `0.3907` with 218 false positives and 69 misses. loop12 F1 `0.3830` with 219 false positives and 71 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....FFFFFFFF#TTTTT#F....#...#
#.#####F#######T###T#F###.#.#.#
#...FF#F#TTTTTTT#TTT#FF.#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#..F#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#.#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#M#T###T#F#####F#####F###T###
#M#MTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#F#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFFF#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MMT#FFF#FFF#FFF#TTTMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#T#T#T#T#T#F#F#T###.#M#M#.#
#MMM#TTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMTTTSFFF#MMMMMMMMMMM......#
###############################
```

### loop2

```text
###############################
#.....FFFFFFFF#TTTTT#FF...#...#
#.#####F#######T###T#F###.#.#.#
#...FF#F#TTTTTTT#TTT#FFF#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#.FF#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#F#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#T#M###T#F#####F#####F###T###
#M#TTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#.#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFFF#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MMT#FFF#FFF#FFF#TTMMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#M#T#T#T#T#F#F#T###.#M#M#.#
#MMM#MTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMTTTSFFF#MMMMMMMMMMM......#
###############################
```

### loop4

```text
###############################
#.....FFFFFFFF#TTTTT#FF...#...#
#.#####F#######T###T#F###.#.#.#
#..FFF#F#TTTTTTT#TTT#FFF#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#.FF#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#F#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#T#M###T#F#####F#####F###T###
#M#TTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#.#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFF.#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MTT#FFF#FFF#FFF#TTMMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#M#T#T#T#T#F#F#T###.#M#M#.#
#MMM#MTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMTTTSFFF#MMMMMMMMMMM......#
###############################
```

### loop6

```text
###############################
#.....FFFFFFFF#TTTTT#FF...#...#
#.#####F#######T###T#F###.#.#.#
#..FFF#F#TTTTTTT#TTT#FFF#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#.FF#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#F#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#T#M###T#F#####F#####F###T###
#M#TTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#.#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFFF#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MMT#FFF#FFF#FFF#TTMMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#M#T#T#T#T#F#F#T###.#M#M#.#
#MMM#MTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMMTTSFFF#MMMMMMMMMMM......#
###############################
```

### loop10

```text
###############################
#.....FFFFFFFF#TTTTT#FF...#...#
#.#####F#######T###T#F###.#.#.#
#...FF#F#TTTTTTT#TTT#FFF#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#.FF#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#.#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#T#M###T#F#####F#####F###T###
#M#TTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#.#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFFF#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MTT#FFF#FFF#FFF#TTMMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#M#T#T#T#T#F#F#T###.#M#M#.#
#MMM#MTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMMTTSFFF#MMMMMMMMMMM......#
###############################
```

### loop12

```text
###############################
#.....FFFFFFFF#TTTTT#FF...#...#
#.#####F#######T###T#F###.#.#.#
#...FF#F#TTTTTTT#TTT#FFF#...#.#
#####F###G#######T###########.#
#..FFF#FFF#FFFFF#TTTTT#TTMMMMM#
#.###F#F###F#F#######T#T#####M#
#..F#F#F#FFF#FFFFFFF#TTT#FFF#M#
#.#.###F#F#######F#F#####F###M#
#.#.#.#F#F#FFFFFFF#FFFFF#FFF#M#
#.#.#F#F#F#F###########F###F#M#
#.#F#FFF#F#F#FFFFFFF#F#FFF#F#M#
###F#######F#F#####F#F###F#F#M#
#.FFFFFFFFFF#FFF#F#FFFFF#FFF#M#
#.#############F#F###F###F###M#
#MTT#TTTTT#FFFFF#FFFFF#FFF#TTM#
#M#T#M###T#F#####F#####F###T###
#M#TTTFF#T#F#F#FFF#F#FFF#F#T#.#
#M#######T#F#F#F###F#F###F#T#.#
#MTT#F#TTT#F#FFF#FFF#FFF#F#T#.#
###M#.#T###F###F#F#####F#F#M#.#
#.#MF.#T#F#FFF#FFF#FFFFFF.#M#.#
#.#M###T#F#F#F#####F#######M#.#
#MMM#MTT#FFF#FFF#FFF#TTMMM#M#.#
#M###M#########F#F###T###M#M#.#
#M#.#M#TTT#TTT#F#F#TTT#.#M#M#.#
#M#.#M#T#T#T#T#F#F#T###.#M#M#.#
#MMM#MTT#TTT#T#FFF#TTTTM#MMM#.#
###M#########T#########M#####.#
#..MMMTTSFFF#MMMMMMMMMMM......#
###############################
```

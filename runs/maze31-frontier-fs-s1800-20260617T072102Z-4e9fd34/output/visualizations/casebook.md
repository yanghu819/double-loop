# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 71 (final failure)

Loop gain: `-0.1715`. First loop F1 `0.3478` with 183 false positives and 87 misses. loop12 F1 `0.1763` with 172 false positives and 127 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....FFFF#FFFFFFFFF......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#F..#.#...#
#M###T#######F#F#F#F#F###.###.#
#MTT#TTTTTGFFF#F#F#FFF#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFFFFF.#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#T#FFFFF#FFFFF#FFF#F#FF.#...#.#
#M#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#F........#
#T#F#########F#####F#########.#
#T#FFF#FFFFFFF#FFFFF#TTMMMMMMM#
#M###F#F#######F#F###T#######M#
#MTT#F#F#FFFFF#F#F#TTM#..MMM#M#
###T#F#F#F###F###F#T#####M#M#M#
#.#T#FFFFF#FFFFFFF#TTTTM#M#T#M#
#.#T#F#########F#######M#M#T#M#
#.#M#.#TTTTT#FFF#TTTTT#T#T#TMM#
#.#M###T###T#####T###T#T#M#####
#.#MMM#M#.#MMT#TMM#F#MMM#TTT#.#
#.###M#M#.###T#T###.#######T#F#
#...#MMM#...#T#TTT#....FFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#T#F.FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#....F#T#.#FFF#F#FFF#
#.###M###########T#F#.#F#####.#
#....MMMMMMMMMMMTM#...#FFFFF..#
###############################
```

### loop2

```text
###############################
#.....FFFF#FFFFFFFFF......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#...#.#...#
#M###T#######F#F#F#F#.###.###.#
#MTT#TTTTTGFFF#F#F#FF.#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFF....#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#M#FFFFF#FFFFF#FFF#F#...#...#.#
#M#######F#F###F###.#.#########
#M#FFFFFFF#F#FFF#FFF#.........#
#M#F#########F#####F#########.#
#T#FFF#FFFFFFF#FFFFF#MMMMMMMMM#
#M###F#F#######F#F###M#######M#
#MTT#F#F#FFFFF#F#F#MMM#..MMM#M#
###T#F#F#F###F###F#M#####M#M#M#
#.#T#FFFFF#FFFFFFF#MMMMM#M#M#M#
#.#T#.#########F#######M#M#M#M#
#.#M#.#MMMMM#FFF#MTMMM#M#M#MMM#
#.#M###M###M#####M###M#M#M#####
#.#MMM#M#.#MMM#MMM#.#MMM#TTT#F#
#.###M#M#.###M#M###.#######T#F#
#...#MMM#...#M#MMM#..FFFFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#M#F.FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#.....#T#F#FFF#F#FFF#
#.###M###########M#.#F#F#####F#
#....MMMMMMMMMMMMM#...#FFFFF..#
###############################
```

### loop4

```text
###############################
#.....FFFF#FFFFFFFFF......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#...#.#...#
#M###T#######F#F#F#F#.###.###.#
#MTT#TTTTTGFFF#F#F#F..#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFF....#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#M#FFFFF#FFFFF#FFF#F#...#...#.#
#M#######F#F###F###F#.#########
#M#FFFFFFF#F#FFF#FFF#.........#
#M#F#########F#####F#########.#
#M#FFF#FFFFFFF#FFFF.#MMMMMMMMM#
#M###F#F#######F#F###M#######M#
#MTT#F#F#FFFFF#F#F#MMM#..MMM#M#
###T#F#F#F###F###F#M#####M#M#M#
#.#T#FFFFF#FFFFFF.#MMMMM#M#M#M#
#.#T#F#########F#######M#M#M#M#
#.#M#.#TMMMM#FFF#MMTMM#M#M#TTM#
#.#M###M###M#####M###M#M#M#####
#.#MMM#M#.#MMM#MMM#.#MMM#TTT#F#
#.###M#M#.###M#M###.#######T#F#
#...#MMM#...#M#MMM#...FFFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#M#F.FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#.....#M#F#FFF#F#FFF#
#.###M###########M#F#F#F#####F#
#....MMMMMMMMMMMMM#...#FFFFF..#
###############################
```

### loop6

```text
###############################
#.....FFFF#FFFFFFFF.......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#...#.#...#
#M###T#######F#F#F#F#.###.###.#
#MTT#TTTTTGFFF#F#F#F..#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFF....#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#M#FFFFF#FFFFF#FFF#F#...#...#.#
#M#######F#F###F###F#.#########
#M#FFFFFFF#F#FFF#FFF#.........#
#M#F#########F#####F#########.#
#M#FFF#FFFFFFF#FFFF.#MMMMMMMMM#
#M###F#F#######F#F###M#######M#
#MTT#F#F#FFFFF#F#F#MMM#..MMM#M#
###T#F#F#F###F###F#M#####M#M#M#
#.#T#FFFFF#FFFFFFF#MMMMM#M#M#M#
#.#T#F#########F#######M#M#M#M#
#.#M#.#MMTMT#FFF#MTTMM#M#M#TTM#
#.#M###M###M#####M###M#M#M#####
#.#MMM#M#.#MMM#MMM#.#MMM#TTT#F#
#.###M#M#.###M#M###.#######T#F#
#...#MMM#...#M#MMM#...FFFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#M#F.FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#.....#M#F#FFF#F#FFF#
#.###M###########T#F#F#F#####F#
#....MMMMMMMMMMMMM#...#FFFFF..#
###############################
```

### loop10

```text
###############################
#.....FFFF#FFFFFFFFF......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#...#.#...#
#M###T#######F#F#F#F#.###.###.#
#MTT#TTTTTGFFF#F#F#F..#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFF....#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#M#FFFFF#FFFFF#FFF#F#...#...#.#
#M#######F#F###F###F#.#########
#M#FFFFFFF#F#FFF#FFF#.........#
#M#F#########F#####F#########.#
#M#FFF#FFFFFFF#FFFF.#MMMMMMMMM#
#M###F#F#######F#F###M#######M#
#MTT#F#F#FFFFF#F#F#MMM#..MMM#M#
###T#F#F#F###F###F#M#####M#M#M#
#.#T#FFFFF#FFFFFFF#MMMMM#M#M#M#
#.#T#F#########F#######M#M#M#M#
#.#M#.#MMMMM#FFF#MMTMM#M#M#TTM#
#.#M###M###M#####M###M#M#M#####
#.#MMM#M#.#MMM#MMM#.#MMM#TTT#F#
#.###M#M#.###M#M###.#######T#F#
#...#MMM#...#M#MMM#...FFFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#M#..FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#.....#M#F#FFF#F#FFF#
#.###M###########M#F#F#F#####F#
#....MMMMMMMMMMMMM#...#FFFFF..#
###############################
```

### loop12

```text
###############################
#.....FFFF#FFFFFFFFF......#...#
#######F#F#F###########.#.#.###
#MMTTT#F#FFF#FFF#FFF#...#.#...#
#M###T#######F#F#F#F#.###.###.#
#MTT#TTTTTGFFF#F#F#F..#...#...#
###T###########F#F#######.#.#.#
#MTT#F#FFFFF#FFF#FFF....#.#.#.#
#M###F#F###F#####F#####.#.#.#.#
#M#FFFFF#FFFFF#FFF#F#...#...#.#
#M#######F#F###F###F#.#########
#M#FFFFFFF#F#FFF#FFF#.........#
#M#F#########F#####F#########.#
#M#FFF#FFFFFFF#FFFFF#MMMMMMMMM#
#M###F#F#######F#F###M#######M#
#MTT#F#F#FFFFF#F#F#MMM#..MMM#M#
###T#F#F#F###F###F#M#####M#M#M#
#.#T#FFFFF#FFFFFFF#MMMMM#M#M#M#
#.#T#.#########F#######M#M#M#M#
#.#M#.#MMMMT#FFF#MTTMM#M#M#TTM#
#.#M###M###M#####M###M#M#M#####
#.#MMM#M#.#MMM#MMM#.#MMM#TTT#F#
#.###M#M#.###M#M###.#######T#F#
#...#MMM#...#M#MMM#...FFFF#T#F#
#.#######.###M#.#M#######F#T#F#
#.#MMMMM#MMMMM#.#M#..FFFFF#T#F#
#.#M###M#M#####.#M#F#####F#S#F#
#.#MMM#MMM#.....#T#F#FFF#F#FFF#
#.###M###########M#F#.#F#####F#
#....MMMMMMMMMMMMM#...#FFFFF..#
###############################
```

## Case 306 (final failure)

Loop gain: `-0.0383`. First loop F1 `0.2857` with 170 false positives and 110 misses. loop12 F1 `0.2474` with 167 false positives and 119 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#MMMMMMMMM#FFFFSFFF...#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TMM#
#.#.###.#M#M#########F#T###M#M#
#.#...#.#M#MMMMMMM#.#F#TTTTM#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMTM#M#..F#MMM#.#M#
#.###.#.#######T#M#####M#M#.#M#
#.#...#FFF#TTTTT#MMMMMMM#M#.#M#
#.#.###F###T###F#########M###M#
#.#F#FFF#TTT#FFF#FFF.F#MMM#MMM#
###F#F###T#######F###F#M###M###
#.FF#FFF#T#FFFFF#FFF#F#TMM#MMM#
#.#####F#G#F###F###F#F###T#.#M#
#F#FFF#F#F#F#F#FFFFF#F#FFT#.#M#
#F###F#F#F#F#F#######F###T#.#M#
#FFF#FFF#FFF#FFF#FFF#FFF#T#.#M#
###F#########F#F###F###F#T###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#.#F#FFFFF#F#F#TTTTT#F#T#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#.....#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#.#FFF#F#TTTTT#TTMMM#M#.#
#####.#.#######T###########M#.#
#...#.#.#FFFFF#T#FFFFF...MMM#.#
#.###.#.#F###F#T#########M###.#
#.....#.....#FFTTTTTMMMMMM#...#
###############################
```

### loop2

```text
###############################
#.......#MMMMMMMMM#FFFFSFFFF..#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TTM#
#.#.###.#M#M#########F#T###T#M#
#.#...#.#M#MMMMMMM#.#F#TTTTT#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMMM#M#...#TMM#.#M#
#.###.#.#######M#M#####M#M#.#M#
#.#...#.FF#TMMMM#MMMMMMM#M#.#M#
#.#F###F###T###.#########M###M#
#.#F#FFF#TTT#FFF#.....#MMM#MMM#
###F#F###T#######.###.#M###M###
#.FF#FFF#T#FFFFF#F.F#.#MMM#MMM#
#.#####F#G#F###F###F#.###M#.#M#
#.#FFF#F#F#F#F#FFFFF#F#..M#.#M#
#.###F#F#F#F#F#######F###M#.#M#
#.FF#FFF#FFF#FFF#FFF#FFF#M#.#M#
###F#########F#F###F###F#M###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#M#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#..FFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TMMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFF....MMM#.#
#.###.#F#F###F#T#########M###.#
#.....#.FFFF#FFTTTTTMMMMMM#...#
###############################
```

### loop4

```text
###############################
#.......#MMMMMMMMM#FFFFSFFFF..#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TTM#
#.#.###.#M#M#########F#T###T#M#
#.#...#.#M#MMMMMMM#F#F#TTTTT#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMMM#M#..F#TTM#.#M#
#.###.#.#######M#M#####M#M#.#M#
#.#...#.FF#TMMMM#MMMMMMM#M#.#M#
#.#F###F###T###.#########M###M#
#.#F#FFF#TTT#FFF#.....#MMM#MMM#
###F#F###T#######.###.#M###M###
#.FF#FFF#T#FFFFF#F.F#.#MMM#MMM#
#.#####F#G#F###F###F#.###M#.#M#
#.#FFF#F#F#F#F#FFFFF#F#..M#.#M#
#.###F#F#F#F#F#######F###M#.#M#
#.FF#FFF#FFF#FFF#FFF#FFF#M#.#M#
###F#########F#F###F###F#M###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#M#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#.FFFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TMMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFF....MMM#.#
#.###.#F#F###F#T#########M###.#
#.....#.FFFF#FFTTTTTMMMMMM#...#
###############################
```

### loop6

```text
###############################
#.......#MMMMMMMMM#FFFFSFFFF..#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TTM#
#.#.###.#M#M#########F#T###T#M#
#.#...#.#M#MMMMMMM#F#F#TTTTT#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMMM#M#..F#TTM#.#M#
#.###.#.#######M#M#####M#M#.#M#
#.#...#.FF#TMMMM#MMMMMMM#M#.#M#
#.#.###F###T###.#########M###M#
#.#F#FFF#TTT#FFF#.....#MMM#MMM#
###F#F###T#######.###.#M###M###
#.FF#FFF#T#FFFFF#F..#.#MMM#MMM#
#.#####F#G#F###F###F#.###M#.#M#
#.#FFF#F#F#F#F#FFFFF#F#..M#.#M#
#.###F#F#F#F#F#######F###M#.#M#
#.FF#FFF#FFF#FFF#FFF#FFF#M#.#M#
###F#########F#F###F###F#M###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#M#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#..FFF#FFF#F#FFF#TTT#TF.#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TMMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFF....MMM#.#
#.###.#F#F###F#T#########M###.#
#.....#.FFFF#FFTTTTTMMMMMM#...#
###############################
```

### loop10

```text
###############################
#.......#MMMMMMMMM#FFFFSFFFF..#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TTM#
#.#.###.#M#M#########F#T###T#M#
#.#...#.#M#MMMMMMM#F#F#TTTTT#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMMM#M#..F#TMM#.#M#
#.###.#.#######M#M#####M#M#.#M#
#.#...#.FF#TMMMM#MMMMMMM#M#.#M#
#.#F###F###T###.#########M###M#
#.#F#FFF#TTT#FFF#.....#MMM#MMM#
###F#F###T#######.###.#M###M###
#.FF#FFF#T#FFFFF#F..#.#MMM#MMM#
#.#####F#G#F###F###F#.###M#.#M#
#.#FFF#F#F#F#F#FFFFF#F#..M#.#M#
#.###F#F#F#F#F#######F###M#.#M#
#.FF#FFF#FFF#FFF#FFF#FFF#M#.#M#
###F#########F#F###F###F#M###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#M#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#..FFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#....F#F#FFF#F#TTTTT#TMMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFF....MMM#.#
#.###.#F#F###F#T#########M###.#
#.....#.FFFF#FFTTTTTMMMMMM#...#
###############################
```

### loop12

```text
###############################
#.......#MMMMMMMMM#FFFFSFFFF..#
#.###.#.#M#######M#F###T#F#####
#.#...#.#M#MMMMMMM#FFF#T#F#TTM#
#.#.###.#M#M#########F#T###T#M#
#.#...#.#M#MMMMMMM#F#F#TTTTT#M#
#.###.#.#M#######M#.#F#####.#M#
#...#.#.#MMMMMMM#M#..F#TTM#.#M#
#.###.#.#######M#M#####M#M#.#M#
#.#...#.FF#TMMMM#MMMMMMM#M#.#M#
#.#F###F###T###.#########M###M#
#.#F#FFF#TTT#FFF#.....#MMM#MMM#
###F#F###T#######.###.#M###M###
#.FF#FFF#T#FFFFF#F..#.#MMM#MMM#
#.#####F#G#F###F###F#.###M#.#M#
#.#FFF#F#F#F#F#FFFFF#F#..M#.#M#
#.###F#F#F#F#F#######F###M#.#M#
#.FF#FFF#FFF#FFF#FFF#FFF#M#.#M#
###F#########F#F###F###F#M###M#
#.FF#FFFFFFFFF#FFFFFFF#F#M#MMM#
#.#F#F#F#######F#######F#M#M#.#
#.#F#F#FFFFF#F#F#TTTTT#F#M#M#.#
#.###F#####F#F#F#T###T###M#M#.#
#..FFF#FFF#F#FFF#TTT#TFF#M#M#.#
#.#####F#F#F#F#####T#T###M#M#.#
#.....#F#FFF#F#TTTTT#TMMMM#M#.#
#####.#F#######T###########M#.#
#...#.#F#FFFFF#T#FFFF....MMM#.#
#.###.#F#F###F#T#########M###.#
#.....#.FFFF#FFTTTTTMMMMMM#...#
###############################
```

## Case 349 (final failure)

Loop gain: `0.0070`. First loop F1 `0.2454` with 215 false positives and 111 misses. loop12 F1 `0.2523` with 210 false positives and 110 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#TTT#T#FFF#..FFF#TTTF.#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMT#F#FFF#F#F#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#F#
#.###M#####T#F###F#F#####F###F#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FFF#
###M#.#.#T###################F#
#MMM#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####.###T#####F###########F#
#MTT#.#FFF#TTT#FFF#FFF#FFFFF#F#
###T#.###F###T#F###F#F#F#####F#
#MMT#.FF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#.F.#F#FFF#T#FFF#F#F#FFF#FF.#
#M###.#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#F#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#M#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM..#F#FFFFFFFF.......#
###############################
```

### loop2

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#MTT#T#FFF#....F#TTT..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMM#F#FFF#F#.#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FF.#
###M#.#.#T###################.#
#MMT#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#.#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFF..........#
###############################
```

### loop4

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#MTT#T#FFF#....F#TTT..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMM#F#FFF#F#.#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FF.#
###M#.#.#T###################.#
#MMT#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#.#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFF..........#
###############################
```

### loop6

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#MTT#T#FFF#....F#TTT..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMM#F#FFF#F#.#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FF.#
###M#.#.#T###################.#
#MMT#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###.#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#.#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###T#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFF..........#
###############################
```

### loop10

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#MTT#T#FFF#....F#TTT..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMM#F#FFF#F#.#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FF.#
###M#.#.#T###################.#
#MMT#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###F#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#.#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFF..........#
###############################
```

### loop12

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FF...#MMM#M#.#...#
#M#M###M#T#T#F###.#####T###.#.#
#M#M#.#MTT#T#FFF#....F#TTT..#.#
#M#M#.#####S###F#####F###T#####
#M#MMMMM#F#FFF#F#.#FFF#TTT#FF.#
#M#####M#F###F#F#F#F###T###F###
#MMMMM#MTTTT#F#FFF#F#FFG#FFF#.#
#.###M#####T#F###F#F#####F###.#
#.#MMM#.#TTT#FFFFF#FFFFFFF#FF.#
###M#.#.#T###################.#
#MMT#...#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#.#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FF.#
#M###.#F###F#T#F###F#F#F#F#F###
#MMM#.#FFFFF#T#FFF#F#FFF#F#F#.#
###M#########T###F#F#####F#.#.#
#MMM#MMT#TTTTT#FFFFFFF#FFF#...#
#M###M#M#T###########F#F#####.#
#M#MMM#M#T#FFFFFFF#FFF#F#.....#
#M#M###M#M#F#####F#####F#.#####
#M#M#MMM#M#F#FFF#FFFFFFF#.#...#
#M#M#M###M#F#F#F#########.###.#
#MMM#MMMMM.F#F#FFFFF.F........#
###############################
```

## Case 324 (final failure)

Loop gain: `-0.0283`. First loop F1 `0.2837` with 192 false positives and 106 misses. loop12 F1 `0.2553` with 163 false positives and 117 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......FF#FFFFFFFF...#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#TMM#M#.#M#.#
#.###F#############T#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTT#TMM#.#MMM#
#.#####F#T#F#############.###M#
#.#FFFFF#T#F#FFF#FFFFF#MMMMMMM#
#F#F###F#T#####F#F###F#M#######
#F#FFF#F#TGFFF#F#FFF#F#M#.....#
#####F#F#####F#F###F#F#M#.#.###
#FFFFF#F#FFF#FFF#F#F#F#M#.#...#
#F#####F#F#######F#F#F#M#.###.#
#F#FFF#F#FFFFF#FFFFF#F#T#...#.#
#F#F#F#F#####F#F#####F#T#F###.#
#.#F#FFF#FFFFF#FFF#FFF#T#F#FF.#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#FFFFFFF#FFFFFFFFF#TTT#F#FFF#
#.#######F#F###########T#S###F#
#MMMMM#.#F.F#TTTTMTM#TTT#TTT#F#
#M###M#.###F#T#####M#T#####T###
#M#.#M#.....#T#...#TTT#TTT#TTM#
#M#.#M#######T###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#...FT#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#MTMMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

### loop2

```text
###############################
#......FFF#FFFFF......#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#MMM#M#.#M#.#
#.###F#############M#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTM#MMM#.#MMM#
#.#####F#T#F#############.###M#
#F#FFFFF#T#F#FFF#FF...#MMMMMMM#
#F#F###F#T#####F#F###.#M#######
#F#FFF#F#TGFFF#F#FF.#.#M#.....#
#####F#F#####F#F###.#.#M#.#.###
#FFFFF#F#FFF#FFF#F#.#.#M#.#...#
#F#####F#F#######F#F#.#M#.###.#
#F#FFF#F#FFFFF#FFFFF#.#M#...#.#
#.#F#F#F#####F#F#####.#M#.###.#
#.#F#FFF#FFFFF#FFF#F..#M#.#...#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#...FFFF#FFFFFFFFF#TTT#F#FFF#
#.#######.#.###########T#S###F#
#MMMMM#.#...#MMMMMMT#TTT#TTT#F#
#M###M#.###.#M#####M#T#####T###
#M#.#M#.....#M#...#MTT#TTT#TTT#
#M#.#M#######M###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#....T#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#TTTMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

### loop4

```text
###############################
#......FFF#FFFFF......#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#MMM#M#.#M#.#
#.###F#############M#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTM#MMM#.#MMM#
#.#####F#T#F#############.###M#
#F#FFFFF#T#F#FFF#F....#MMMMMMM#
#F#F###F#T#####F#F###.#M#######
#F#FFF#F#TGFFF#F#FF.#.#M#.....#
#####F#F#####F#F###.#.#M#.#.###
#FFFFF#F#FFF#FFF#F#.#.#M#.#...#
#F#####F#F#######F#F#.#M#.###.#
#F#FFF#F#FFFFF#FFFFF#.#M#...#.#
#.#F#F#F#####F#F#####.#M#.###.#
#.#F#FFF#FFFFF#FFF#F..#M#.#...#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#...FFFF#FFFFFFFFF#TTT#F#FFF#
#.#######.#.###########T#S###F#
#MMMMM#.#...#MMMMMMT#TTT#TTT#F#
#M###M#.###.#M#####T#T#####T###
#M#.#M#.....#M#...#TTT#TTT#TTT#
#M#.#M#######M###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#...FT#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#TTTMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

### loop6

```text
###############################
#......FFF#FFFF.......#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#MMM#M#.#M#.#
#.###F#############M#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTM#MMM#.#MMM#
#.#####F#T#F#############.###M#
#F#FFFFF#T#F#FFF#F....#MMMMMMM#
#F#F###F#T#####F#F###.#M#######
#F#FFF#F#TGFFF#F#FF.#.#M#.....#
#####F#F#####F#F###.#.#M#.#.###
#FFFFF#F#FFF#FFF#F#.#.#M#.#...#
#F#####F#F#######F#.#.#M#.###.#
#F#FFF#F#FFFFF#FFFFF#.#M#...#.#
#.#F#F#F#####F#F#####.#M#.###.#
#.#F#FFF#FFFFF#FFF#F..#M#.#...#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#...FFFF#FFFFFFFFF#TTT#F#FFF#
#.#######.#.###########T#S###F#
#MMMMM#.#...#MMMMMMT#TTT#TTT#F#
#M###M#.###.#M#####T#T#####T###
#M#.#M#.....#M#...#TTT#TTT#TTT#
#M#.#M#######M###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#....T#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#TTTMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

### loop10

```text
###############################
#......FFF#FFFFF......#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#MMM#M#.#M#.#
#.###F#############M#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTM#MMM#.#MMM#
#.#####F#T#F#############.###M#
#F#FFFFF#T#F#FFF#F....#MMMMMMM#
#F#F###F#T#####F#F###.#M#######
#F#FFF#F#TGFFF#F#FF.#.#M#.....#
#####F#F#####F#F###.#.#M#.#.###
#FFFFF#F#FFF#FFF#F#.#.#M#.#...#
#F#####F#F#######F#F#.#M#.###.#
#F#FFF#F#FFFFF#FFFFF#.#M#...#.#
#.#F#F#F#####F#F#####.#M#.###.#
#.#F#FFF#FFFFF#FFF#F..#M#.#...#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#F..FFFF#FFFFFFFFF#TTT#F#FFF#
#.#######.#.###########T#S###F#
#MMMMM#.#...#MMMMMMT#TTT#TTT#F#
#M###M#.###.#M#####M#T#####T###
#M#.#M#.....#M#...#TTT#TTT#TTT#
#M#.#M#######M###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#...FT#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#TTTMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

### loop12

```text
###############################
#......FFF#FFFFF......#MMMMM#.#
#.#.#####F#######F#####M###M#.#
#.#FFF#FFFFFFFFFFF#MMM#M#.#M#.#
#.###F#############M#M#M#.#M#.#
#.FF#FFF#TTTTTTTTTTM#MMM#.#MMM#
#.#####F#T#F#############.###M#
#F#FFFFF#T#F#FFF#FF...#MMMMMMM#
#F#F###F#T#####F#F###.#M#######
#F#FFF#F#TGFFF#F#FF.#.#M#.....#
#####F#F#####F#F###.#.#M#.#.###
#FFFFF#F#FFF#FFF#F#.#.#M#.#...#
#F#####F#F#######F#.#.#M#.###.#
#F#FFF#F#FFFFF#FFFFF#.#M#...#.#
#.#F#F#F#####F#F#####.#M#.###.#
#.#F#FFF#FFFFF#FFF#F..#M#.#...#
#.#F#####F#######F#F###T###F#.#
#.#F#FFFFFFFFFFFFF#F#TTT#FFF#.#
#.#F###############F#T###F###F#
#.#..FFFFF#FFFFFFFFF#TTT#F#FFF#
#.#######.#.###########T#S###F#
#MMMMM#.#...#MMMMMMT#TTT#TTT#F#
#M###M#.###.#M#####M#T#####T###
#M#.#M#.....#M#...#TTT#TTT#TTT#
#M#.#M#######M###.#####T#T#F#M#
#M#.#MMMMM#MMM#MMM#...FT#T#.#M#
#M#.#####M#M###M#M#####M#M#.#M#
#MMMMMMM#MMM#MMM#TTTMMMM#M#.#M#
#######M#####M###########M###M#
#......MMMMMMM#..........MMMMM#
###############################
```

## Case 95 (final failure)

Loop gain: `-0.0076`. First loop F1 `0.2905` with 187 false positives and 111 misses. loop12 F1 `0.2829` with 174 false positives and 115 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.......FFFFFFFFFF#F..#...#
#.#.#.#########F#####F#F#F#.#.#
#.#.#.....#FFFFF#FFFFF#F#FFF#.#
#.#.###.###F#######F###F#####.#
#.#.#...#.FF#TTGFF#F#FFF#FFF#.#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#..F#TTT#F#FFF#FFFFF#F#
#M###M###.#F#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###F#T#F#F###F#F#F#F#F#
#M#FFF#T#FFF#TFF#F#FFF#F#F#F#F#
#M#F#F#T#F#F#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#T#F###F#F#####F#F#
#T#FFF#T#F#F#T#FFF#F#FFFFF#F#F#
#T###S#T#F#F#T###F#F#####F#F#.#
#TTT#TTTFF#F#MTT#F#FFFFFFFFF#.#
###T###########T###############
#MTT#TTTTTMMMM#MMTTTTTTTTMMMMM#
#M#F#T#######M###############M#
#M#F#TTTTM#.#M#TTTFFFFFF..#MMM#
#M#######M#.#M#M#T#######.#M###
#MMM..#MMM..#M#M#M#MMM#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

### loop2

```text
###############################
#...#........FFFFFFFFF#FFF#...#
#.#.#.#########F#####F#F#F#F#.#
#.#.#.....#.FFFF#FFFFF#F#FFF#.#
#.#.###.###F#######F###F#####.#
#.#.#...#..F#TTGFF#F#FFF#FFF#F#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#...#TTT#F#FFF#FFFFF#F#
#M###M###.#.#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###.#T#F#F###F#F#F#F#F#
#M#...#M#...#TFF#F#FFF#F#F#F#F#
#M#F#.#T#.#.#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#M#F###F#F#####F#F#
#T#FFF#T#F#.#M#.FF#F#FFFFF#F#F#
#T###S#T#F#F#T###F#F#####F#F#F#
#TTT#TTTFF#.#MMT#F#FFFFFFFFF#.#
###T###########M###############
#MTT#TTTTTMMMM#MMMMMTTTTTTMMMM#
#M#F#T#######M###############M#
#M#F#TTTMM#.#M#MMM..FFFF..#MMM#
#M#######M#.#M#M#M#######.#M###
#MMM..#MMM..#M#M#M#MMT#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

### loop4

```text
###############################
#...#........FFFFFFFFF#FFF#...#
#.#.#.#########F#####F#F#F#F#.#
#.#.#.....#.FFFF#FFFFF#F#FFF#.#
#.#.###.###F#######F###F#####.#
#.#.#...#..F#TTGFF#F#FFF#FFF#F#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#...#TTT#F#FFF#FFFFF#F#
#M###M###.#.#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###.#T#F#F###F#F#F#F#F#
#M#...#M#...#TFF#F#FFF#F#F#F#F#
#M#F#F#T#.#.#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#M#F###F#F#####F#F#
#T#FFF#T#F#.#T#FFF#F#FFFFF#F#F#
#T###S#T#F#F#T###F#F#####F#F#F#
#TTT#TTTFF#F#TTT#F#FFFFFFFFF#.#
###T###########T###############
#MTT#TTTTTMMMM#MMMMTTTTTTTMMMM#
#M#F#T#######M###############M#
#M#F#TTTMM#.#M#MMM.FFFFF..#MMM#
#M#######M#.#M#M#T#######.#M###
#MMM..#MMM..#M#M#M#MTT#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

### loop6

```text
###############################
#...#.........FFFFFFFF#FFF#...#
#.#.#.#########F#####F#F#F#F#.#
#.#.#.....#.FFFF#FFFFF#F#FFF#.#
#.#.###.###F#######F###F#####.#
#.#.#...#..F#TTGFF#F#FFF#FFF#F#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#...#TTT#F#FFF#FFFFF#F#
#M###M###.#.#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###.#T#F#F###F#F#F#F#F#
#M#...#M#...#TFF#F#FFF#F#F#F#F#
#M#F#F#T#F#.#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#T#F###F#F#####F#F#
#T#FFF#T#F#.#T#FFF#F#FFFFF#F#F#
#T###S#T#F#F#T###F#F#####F#F#F#
#TTT#TTTFF#F#TTT#F#FFFFFFFFF#.#
###T###########T###############
#MTT#TTTTTMMMM#MMMMTTTTTTTMMMM#
#M#F#T#######M###############M#
#M#F#TTTMM#.#M#MMM.FFFFF..#MMM#
#M#######M#.#M#M#M#######.#M###
#MMM..#MMM..#M#M#M#MTT#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

### loop10

```text
###############################
#...#.........FFFFFFFF#FFF#...#
#.#.#.#########F#####F#F#F#F#.#
#.#.#.....#.FFFF#FFFFF#F#FFF#.#
#.#.###.###.#######F###F#####.#
#.#.#...#..F#TTGFF#F#FFF#FFF#F#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#...#TTT#F#FFF#FFFFF#F#
#M###M###.#.#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###.#T#F#F###F#F#F#F#F#
#M#...#M#...#TFF#F#FFF#F#F#F#F#
#M#F#.#T#.#.#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#M#F###F#F#####F#F#
#T#FFF#T#F#.#T#FFF#F#FFFFF#F#F#
#T###S#T#F#.#T###F#F#####F#F#F#
#TTT#TTTFF#F#MTT#F#FFFFFFFFF#.#
###T###########M###############
#MTT#TTTTTMMMM#MMMMTTTTTTTMMMM#
#M#F#T#######M###############M#
#M#F#TTTMM#.#M#MMM.FFFFF..#MMM#
#M#######M#.#M#M#M#######.#M###
#MMM..#MMM..#M#M#M#TTT#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

### loop12

```text
###############################
#...#.........FFFFFFFF#FFF#...#
#.#.#.#########F#####F#F#F#F#.#
#.#.#.....#.FFFF#FFFFF#F#FFF#.#
#.#.###.###.#######F###F#####.#
#.#.#...#..F#TTGFF#F#FFF#FFF#F#
#.###.###.#F#T###F#F###F#F###F#
#.......#.#F#TTT#F#F#FFF#F#FFF#
#######.#.#####T#F#F#F###F#F###
#MMMMM#.#...#TTT#F#FFF#FFFFF#F#
#M###M###.#.#T###F#####F#####F#
#M#.#MMM#.#.#T#F#F#FFF#F#FFFFF#
#M#.###M###.#T#F#F###F#F#F#F#F#
#M#...#M#...#TFF#F#FFF#F#F#F#F#
#M#F#F#T#.#.#T###F#F###F#F#F#F#
#M#F#F#T#F#.#T#FFF#F#FFF#F#F#F#
#M#F###T#F#.#T#F###F#F#####F#F#
#T#FFF#T#F#.#M#FFF#F#FFFFF#F#F#
#T###S#T#F#F#T###F#F#####F#F#F#
#TTT#TTTFF#F#MTT#F#FFFFFFFFF#.#
###T###########M###############
#MTT#TTTTTMMMM#MMMMTTTTTTTMMMM#
#M#F#T#######M###############M#
#M#F#TTTTM#.#M#MMM.FFFFF..#MMM#
#M#######M#.#M#M#T#######.#M###
#MMM..#MMM..#M#M#T#MTT#...#M#.#
###M###M#####M#M#M#M#M#.###M#.#
#MMM#MMM#MMMMM#M#M#M#M#...#M#.#
#M###M###M#####M#M#M#M#####M#.#
#MMMMM..#MMMMMMM#MMM#MMMMMMM..#
###############################
```

## Case 131 (final failure)

Loop gain: `-0.0573`. First loop F1 `0.3415` with 178 false positives and 92 misses. loop12 F1 `0.2842` with 164 false positives and 108 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#.FTMG#
#M#######M#M#M#.#.#M#M#.#F#T###
#MMMMM#.#MMM#M#...#M#M#.#F#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######M#
#MTTTTSF#FFF#MMM#MMM#MMM#MMT#M#
###########F###M###M###M#M#T#T#
#F#FFFFFFF#F#.#M..#MMT#TMM#TTM#
#.#F#F#####F#.#M#####T#########
#.#F#FFFFFFF#.#TTTTT#TTT#FFFF.#
#.#F#########F#####T###T#F###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#.###F###F#####F#T#F#F#T#F#F#F#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#.#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#.###F###F#F###F#####T#T#F###.#
#...#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#F..#
#.#.###.###F#F#F#F#T#F###T###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###F###F#F#F#T#########M#
#.#...#.#..F#F#F#F#TTTTTMM..#M#
#.#####.###F#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#............FFFFFFF#.....#MMM#
###############################
```

### loop2

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#.FTMG#
#M#######M#M#M#.#.#M#M#.#F#T###
#MMMMM#.#MMM#M#...#M#M#.#.#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######T#
#MTTTTSF#FF.#MMM#MMM#MMM#MMT#M#
###########F###M###M###M#M#T#M#
#.#FFFFFFF#F#.#M..#MMM#MMM#TTM#
#.#F#F#####.#.#M#####M#########
#.#F#FFFFFF.#.#MMMMM#MMM#..FF.#
#.#F#########.#####M###M#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#.###F###F#####F#T#F#F#T#F#F#F#
#..FFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#F#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#.###F###F#F###F#####T#T#F###F#
#..F#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#...#
#.#.###.###F#F#F#F#T#F###M###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###.###F#F#F#T#########M#
#.#...#.#..F#F#F#F#TTTTMMM..#M#
#.#####.###.#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#............FFFFFF.#.....#MMM#
###############################
```

### loop4

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#..TMG#
#M#######M#M#M#.#.#M#M#.#.#T###
#MMMMM#.#MMM#M#...#M#M#.#.#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######T#
#MTTTTSF#FF.#MMM#MMM#MMM#MMM#M#
###########F###M###M###M#M#T#M#
#.#FFFFFFF#F#.#M..#MMM#MMM#TTM#
#.#F#F#####F#.#M#####M#########
#.#F#FFFFFFF#.#MMMMM#MMM#..FF.#
#.#F#########.#####M###M#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FF.#
#.###F###F#####F#T#F#F#T#F#F#F#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#F#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#..F#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#...#
#.#.###.###F#F#F#F#T#F###T###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###.###F#F#F#T#########M#
#.#...#.#...#F#F#F#TTTTMMM..#M#
#.#####.###.#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#.............FFFFF.#.....#MMM#
###############################
```

### loop6

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#.FTMG#
#M#######M#M#M#.#.#M#M#.#.#T###
#MMMMM#.#MMM#M#...#M#M#.#.#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######T#
#MTTTTSF#FF.#MMM#MMM#MMM#MMM#M#
###########F###M###M###M#M#T#M#
#.#FFFFFFF#F#.#M..#MMM#MMM#TTM#
#.#F#F#####F#.#M#####M#########
#.#F#FFFFFFF#.#MMMMM#MMM#..FF.#
#.#F#########.#####M###M#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#.###F###F#####F#T#F#F#T#F#F#F#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#F#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###.#
#..F#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#...#
#.#.###.###F#F#F#F#T#F###T###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###.###F#F#F#T#########M#
#.#...#.#...#F#F#F#TTTTMMM..#M#
#.#####.###.#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#............FFFFFF.#.....#MMM#
###############################
```

### loop10

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#.FTMG#
#M#######M#M#M#.#.#M#M#.#.#T###
#MMMMM#.#MMM#M#...#M#M#.#.#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######T#
#MTTTTSF#FF.#MMM#MMM#MMM#MMT#M#
###########F###M###M###M#M#T#M#
#.#FFFFFFF#F#.#M..#MMM#MMM#TTM#
#.#F#F#####.#.#M#####M#########
#.#F#FFFFFFF#.#MMMMM#MMM#..FF.#
#.#F#########.#####M###T#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#.###F###F#####F#T#F#F#T#F#F#F#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#F#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###F#
#..F#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#F..#
#.#.###.###F#F#F#F#T#F###T###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###.###F#F#F#T#########M#
#.#...#.#...#F#F#F#TTTTMMM..#M#
#.#####.###.#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#............FFFFF..#.....#MMM#
###############################
```

### loop12

```text
###############################
#MMMMMMMMM#MMM..#.#MMM#.#..TMG#
#M#######M#M#M#.#.#M#M#.#.#T###
#MMMMM#.#MMM#M#...#M#M#.#.#TTT#
#####T#F#####M#.###M#M#.#####T#
#MMTTT#FFF#.#M#.#MMM#M#....FFT#
#M#####F#F#.#M###M###M#######T#
#MTTTTSF#FF.#MMM#MMM#MMM#MMM#M#
###########F###M###M###M#M#T#M#
#.#FFFFFFF#F#.#M..#MMM#MMM#TTM#
#.#F#F#####F#.#M#####M#########
#.#F#FFFFFF.#.#MMMMM#MMM#..FF.#
#.#F#########.#####M###M#.###.#
#.#FFF#FFFFFFFFF#TTT#F#T#F#FF.#
#.###F###F#####F#T#F#F#T#F#F#F#
#.FFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#.#####F###F#F###T#####T###F#F#
#.#FFFFF#FFF#FFF#TTTTT#T#FFF#.#
#.###F###F#F###F#####T#T#F###F#
#..F#.FF#F#FFFFF#FFF#T#T#FFF#.#
###.#.#F#F#######F###T#T###F###
#.#.#.#.#FFF#FFF#F#TTT#TTT#F..#
#.#.###.###F#F#F#F#T#F###M###.#
#.#.#...#FFF#F#F#F#T#FFF#MMMMM#
#.#.#.###.###F#F#F#T#########M#
#.#...#.#..F#F#F#F#TTTTMMM..#M#
#.#####.###.#F#F#F#######M###M#
#.........#.FF#FFF#FFF..#MMM#M#
#.###############F#F###.###M#M#
#.............FFFF..#.....#MMM#
###############################
```

## Case 234 (final failure)

Loop gain: `0.0120`. First loop F1 `0.2730` with 187 false positives and 106 misses. loop12 F1 `0.2850` with 170 false positives and 106 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#....MMMMT#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#...#MMM#F#T#FFFFFFFFFFF#F..#.#
#.###T###F#T#F#########F#F#####
#.#TTT#FFF#T#FFFFF#FFF#F#F#F..#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#T#FFFFFFF#FFF#F#F#.#
###G###F###T#F#############F#F#
#F#FFF#FFTTT#FFFFFFF#FFFFFFF#F#
#F#F#F###T#######F#F#F#######F#
#F#F#FFF#MTTTTTT#F#FFF#F#FFFFF#
#F#####F#######T#F#####F#F#####
#.#FFF#F#.#TTT#T#F#FFFFF#FFFFF#
#.#F#.#.#.#T#T#T#F#F###F#####.#
#...#...#MMT#TTT#F#F#F#FFFFF#F#
#########M#######F#F#F###F###.#
#.......#MMMMM#F#FFF#FFFFF#FF.#
#.#####.#####M#F###F#######F#.#
#.#MMM#...#.#M#F.FFF#TTTTT#F#.#
#.#M#M###.#.#M#######T###M#.#.#
#MMM#MMM#...#MMMMT#TMT#.#M#.#.#
#M#####M#########M#M###.#M#.#.#
#MMMMM#M#MMMMMMM#MMM#MMMMM#.#.#
#####M#M#T#####M#####M#####.#.#
#MMMMM#TTT#FFF#MMM..#MMMMM#.#.#
#M###########F###M#######M#.###
#MMM#MTT#FFF#FFF#MMMMM..#M#...#
###M#M#T#S#F###F#####M###M###.#
#..MMM#TTT#FFFFF....#MMMMM....#
###############################
```

### loop2

```text
###############################
#.#....MMMMM#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#...#MMM#F#T#FFFFFFFFFFF#...#.#
#.###T###F#T#F#########F#F#####
#.#TTT#..F#T#FFFFF#FFF#F#F#...#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#T#FFFFFFF#FFF#F#F#.#
###G###F###M#F#############F#.#
#F#FFF#F.MMM#FFFFFFF#FFFFFFF#.#
#F#F#F###M#######F#F#F#######.#
#F#F#F..#MMMMTTT#F#FFF#F#FFFF.#
#.#####.#######T#F#####F#F#####
#.#F..#.#.#MMT#T#F#FFFFF#FFFFF#
#.#.#.#.#.#M#M#T#F#F###F#####.#
#...#...#MMM#MTT#F#F#F#FFFFF#.#
#########M#######F#F#F###F###.#
#.......#MMMMM#.#FFF#FFFFF#FF.#
#.#####.#####M#.###F#######F#.#
#.#MMM#...#.#M#..FFF#TTTTM#F#.#
#.#M#M###.#.#M#######T###M#F#.#
#MMM#MMM#...#MMMMT#TTT#.#M#.#.#
#M#####M#########M#T###.#M#.#.#
#MMMMM#M#MMMMMMM#TTT#TTMMM#.#.#
#####M#M#T#####M#####M#####.#.#
#MMMMM#MTT#FFF#TTM..#MMMMM#.#.#
#M###########F###T#######M#.###
#MMM#MTT#FFF#FFF#TTTMM..#M#...#
###M#M#T#S#F###F#####M###M###.#
#..MMM#TTT#FFFFFF...#MMMMM....#
###############################
```

### loop4

```text
###############################
#.#....MMMMM#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#...#MMM#F#T#FFFFFFFFFFF#...#.#
#.###T###F#T#F#########F#F#####
#.#TTT#..F#T#FFFFF#FFF#F#F#F..#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#M#FFFFFFF#FFF#F#F#.#
###G###F###M#F#############F#.#
#F#FFF#F.MMT#FFFFFFF#FFFFFFF#.#
#F#F#F###M#######F#F#F#######.#
#F#F#F..#MMMMTTT#F#FFF#F#FFFF.#
#.#####.#######T#F#####F#F#####
#.#F..#.#.#MMT#T#F#FFFFF#FFFF.#
#.#.#.#.#.#M#T#T#F#F###F#####.#
#...#...#MMM#MTT#F#F#F#FFFFF#.#
#########M#######F#F#F###F###.#
#.......#MMMMM#F#FFF#FFFFF#FF.#
#.#####.#####M#F###F#######F#.#
#.#MMM#...#.#M#F.FFF#TTTTM#F#.#
#.#M#M###.#.#M#######T###M#F#.#
#MMM#MMM#...#MMMMT#TTT#.#M#.#.#
#M#####M#########T#T###.#M#.#.#
#MMMMM#M#MMMMMMM#TTT#TTMMM#.#.#
#####M#M#T#####M#####M#####.#.#
#MMMMM#MTT#FFF#TTM.F#MMMMM#.#.#
#M###########F###T#######M#.###
#MMM#MTT#FFF#FFF#TTTMM..#M#...#
###M#M#T#S#F###F#####T###M###.#
#..MMM#TTT#FFFFFF...#MMMMM....#
###############################
```

### loop6

```text
###############################
#.#....MMMMM#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#..F#MMM#F#T#FFFFFFFFFFF#...#.#
#.###T###F#T#F#########F#.#####
#.#TTT#..F#T#FFFFF#FFF#F#F#F..#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#T#FFFFFFF#FFF#F#F#.#
###G###F###T#F#############F#.#
#F#FFF#F.MMM#FFFFFFF#FFFFFFF#.#
#F#F#F###M#######F#F#F#######.#
#F#F#F..#MMMMTTT#F#FFF#F#FFFF.#
#.#####.#######T#F#####F#F#####
#.#F..#.#.#MMT#T#F#FFFFF#FFFF.#
#.#.#.#.#.#M#T#T#F#F###F#####.#
#...#...#MMM#MTT#F#F#F#FFFFF#.#
#########M#######F#F#F###F###.#
#.......#MMMMM#F#FFF#FFFFF#FF.#
#.#####.#####M#F###F#######F#.#
#.#MMM#...#.#M#.FFFF#TTTTM#F#.#
#.#M#M###.#.#M#######T###M#F#.#
#MMM#MMM#...#MMMMT#TTT#.#M#.#.#
#M#####M#########T#T###.#M#.#.#
#MMMMM#M#MMMMMMM#TTT#TTMMM#.#.#
#####M#M#T#####M#####T#####.#.#
#MMMMM#MTT#FFF#TTM.F#MMMMM#.#.#
#M###########F###T#######M#.###
#MMM#MTT#FFF#FFF#TTTMM..#M#...#
###M#M#T#S#F###F#####T###M###.#
#..MMM#TTT#FFFFFFF..#MMMMM....#
###############################
```

### loop10

```text
###############################
#.#....MMMMM#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#...#MMM#F#T#FFFFFFFFFFF#...#.#
#.###T###F#T#F#########F#F#####
#.#TTT#..F#T#FFFFF#FFF#F#F#F..#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#M#FFFFFFF#FFF#F#F#.#
###G###F###T#F#############F#.#
#F#FFF#F.MMM#FFFFFFF#FFFFFFF#.#
#F#F#F###M#######F#F#F#######.#
#F#F#FF.#MMMMTTT#F#FFF#F#FFFF.#
#.#####.#######T#F#####F#F#####
#.#F..#.#.#MMT#T#F#FFFFF#FFFF.#
#.#.#.#.#.#M#T#T#F#F###F#####.#
#...#...#MMM#MTT#F#F#F#FFFFF#.#
#########M#######F#F#F###F###.#
#.......#MMMMM#F#FFF#FFFFF#FF.#
#.#####.#####M#F###F#######F#.#
#.#MMM#...#.#M#.FFFF#TTTTM#F#.#
#.#M#M###.#.#M#######T###M#F#.#
#MMM#MMM#...#MMMTT#TTT#.#M#.#.#
#M#####M#########T#T###.#M#.#.#
#MMMMM#M#MMMMMMM#TTT#TTMMM#.#.#
#####M#M#M#####M#####T#####.#.#
#MMMMM#MTT#FFF#TTMFF#MMMMM#.#.#
#M###########F###T#######M#.###
#MMM#MTT#FFF#FFF#TTTMM..#M#...#
###M#M#T#S#F###F#####T###M###.#
#..MMM#TTT#FFFFFF...#MMMMM....#
###############################
```

### loop12

```text
###############################
#.#....MMMMM#FFFFFFFFFF...#...#
#.#.###M###T#F###########.#.#.#
#...#MMM#F#T#FFFFFFFFFFF#...#.#
#.###T###F#T#F#########F#.#####
#.#TTT#..F#T#FFFFF#FFF#F#F#...#
#F#T#####F#T#######F#F#F#F#F#.#
#F#T#FFFFF#T#FFFFFFF#FFF#F#F#.#
###G###F###T#F#############F#.#
#F#FFF#F.MMM#FFFFFFF#FFFFFFF#.#
#F#F#F###M#######F#F#F#######.#
#F#F#F..#MMMMTTT#F#FFF#F#FFFF.#
#.#####.#######T#F#####F#F#####
#.#F..#.#.#MMT#T#F#FFFFF#FFFF.#
#.#.#.#.#.#M#T#T#F#F###F#####.#
#...#...#MMM#MTT#F#F#F#FFFFF#.#
#########M#######F#F#F###F###.#
#.......#MMMMM#F#FFF#FFFFF#FF.#
#.#####.#####M#.###F#######F#.#
#.#MMM#...#.#M#.FFFF#TTTTM#F#.#
#.#M#M###.#.#M#######T###M#F#.#
#MMM#MMM#...#MMMMT#TTT#.#M#.#.#
#M#####M#########T#T###.#M#.#.#
#MMMMM#M#MMMMMMM#TTT#TTMMM#.#.#
#####M#M#T#####M#####T#####.#.#
#MMMMM#TTT#FFF#TMM.F#MMMMM#.#.#
#M###########F###M#######M#.###
#MMM#MTT#FFF#FFF#TTTMM..#M#...#
###M#M#T#S#F###F#####T###M###.#
#..MMM#TTT#FFFFFFF..#MMMMM....#
###############################
```

## Case 7 (final failure)

Loop gain: `-0.0784`. First loop F1 `0.3671` with 175 false positives and 87 misses. loop12 F1 `0.2888` with 157 false positives and 109 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..GTTTT#TMMMMFF#.#.......#...#
#######T#T###T###F###F###.#.###
#TTT#TTT#T#.#TTT#FFFFF#...#...#
#M#T#T#F#M#F###T#######F#####.#
#T#TTT#F#MMMTT#TTTTTFF#F#.....#
#M###########T#####T###F#F#####
#MMTTTTMMMMMMTFFFF#T#FFF#F#F..#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMT#FFF#TTT#F#FFF#F#.#
###M###M#M#T#F###T###F###F#F#F#
#MMM#MMM#T#T#FFF#TTT#FFF#F#F#F#
#M###M###T#T#######T###F#F#F#F#
#M#.#M#.FT#TTTTTTTTT#F#F#FFF#F#
#M#.#M###T#########F#F#F#F###F#
#M#..M#MMT#FFFFF#F#FFF#F#F#F#F#
#M###M#M#F#F###F#F###F#F#F#F#F#
#MMM#MMM#F#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####F#
#.#M#MMMTT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#TTT#FFF#F#FFFFFFF..#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#...FF#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#..F#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#FFF..#...#.#
#M###M#.#######F###F###.#.###.#
#MMMMM#........F.FFF#...#.....#
###############################
```

### loop2

```text
###############################
#..GTTTT#TMMMM.F#.#F......#...#
#######T#T###T###F###F###.#.###
#MTT#TTT#T#.#MTT#FFFFF#...#...#
#M#T#T#F#M#.###T#######F#####.#
#M#TTT#F#MMMMM#MTTTTFF#F#F....#
#M###########M#####T###F#F#####
#MMMMMMMMMMMMM.FFF#T#FFF#F#FF.#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMM#..F#TTT#F#FFF#F#.#
###M###M#M#M#.###T###F###F#F#.#
#MMM#MMM#M#M#.FF#TTT#FFF#F#F#.#
#M###M###M#M#######T###F#F#F#F#
#M#.#M#..M#MTTTTTTTT#F#F#FFF#.#
#M#.#M###M#########F#F#F#F###.#
#M#..M#MMM#FFFFF#F#FFF#F#F#F#.#
#M###M#M#.#F###F#F###F#F#F#F#F#
#MMM#MMM#.#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####.#
#.#M#MMMMT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#MTT#FFF#F#FFFFFF...#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#....F#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#...#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#F.F..#...#.#
#M###M#.#######F###F###.#.###.#
#MMMMM#.............#...#.....#
###############################
```

### loop4

```text
###############################
#..GTTTT#TMMMM..#.#.......#...#
#######T#T###T###F###F###.#.###
#MTT#TTT#T#.#MTT#FFFFF#F..#...#
#M#T#T#F#M#.###T#######F#####.#
#M#TTT#F#MMMMM#TTTTTFF#F#.....#
#M###########M#####T###F#F#####
#MMMMMMMMMMMMMFFFF#T#FFF#F#FF.#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMM#.FF#TTT#F#FFF#F#.#
###M###M#M#M#.###T###F###F#F#.#
#MMM#MMM#M#M#.FF#TTT#FFF#F#F#F#
#M###M###M#M#######T###F#F#F#F#
#M#.#M#..M#MTTTTTTTT#F#F#FFF#F#
#M#.#M###M#########F#F#F#F###.#
#M#..M#MMM#FFFFF#F#FFF#F#F#F#F#
#M###M#M#.#F###F#F###F#F#F#F#F#
#MMM#MMM#.#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####.#
#.#M#MMMMT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#MTT#FFF#F#FFFFFF...#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#....F#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#...#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#F....#...#.#
#M###M#.#######F###F###.#.###.#
#MMMMM#.............#...#.....#
###############################
```

### loop6

```text
###############################
#..GTTTT#TMMMM..#.#.......#...#
#######T#T###T###F###F###.#.###
#MTT#TTT#T#.#TTT#FFFFF#F..#...#
#M#T#T#F#M#.###T#######F#####.#
#M#TTT#F#MMMMM#TTTTTFF#F#.....#
#M###########M#####T###F#F#####
#MMMMMMMMMMMMMFFFF#T#FFF#F#FF.#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMM#.FF#TTT#F#FFF#F#.#
###M###M#M#M#.###T###F###F#F#.#
#MMM#MMM#M#M#.FF#TTT#FFF#F#F#.#
#M###M###M#M#######T###F#F#F#F#
#M#.#M#..M#MTTTTTTTT#F#F#FFF#F#
#M#.#M###M#########F#F#F#F###.#
#M#..M#MMM#FFFFF#F#FFF#F#F#F#F#
#M###M#M#.#F###F#F###F#F#F#F#.#
#MMM#MMM#.#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####.#
#.#M#MMMMT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#MTT#FFF#F#FFFFFF...#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#....F#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#...#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#F....#...#.#
#M###M#.#######.###F###.#.###.#
#MMMMM#.............#...#.....#
###############################
```

### loop10

```text
###############################
#..GTTTT#TMMMM..#.#.......#...#
#######T#T###T###F###F###.#.###
#MTT#TTT#T#.#MTT#FFFFF#F..#...#
#M#T#T#F#M#.###T#######F#####.#
#M#TTT#F#MMMMM#TTTTTFF#F#F....#
#M###########M#####T###F#F#####
#MMMMMMMMMMMMMFFFF#T#FFF#F#FF.#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMM#.FF#TTT#F#FFF#F#.#
###M###M#M#M#.###T###F###F#F#.#
#MMM#MMM#M#M#FFF#TTT#FFF#F#F#.#
#M###M###M#M#######T###F#F#F#F#
#M#.#M#..M#MTTTTTTTT#F#F#FFF#F#
#M#.#M###M#########F#F#F#F###.#
#M#..M#MMM#FFFFF#F#FFF#F#F#F#F#
#M###M#M#.#F###F#F###F#F#F#F#.#
#MMM#MMM#.#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####.#
#.#M#MMMMT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#MTT#FFF#F#FFFFFF...#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#....F#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#...#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#FF...#...#.#
#M###M#.#######F###F###.#.###.#
#MMMMM#.............#...#.....#
###############################
```

### loop12

```text
###############################
#..GTTTT#TMMMM..#.#.......#...#
#######T#T###T###F###F###.#.###
#MTT#TTT#T#.#TTT#FFFFF#F..#...#
#M#T#T#F#M#.###T#######F#####.#
#M#TTT#F#MMMMM#TTTTTFF#F#F....#
#M###########M#####T###F#F#####
#MMMMMMMMMMMMMFFFF#T#FFF#F#FF.#
#.#############F###T#F###F#F#.#
#.#MMMMM#MMM#.FF#TTT#F#FFF#F#.#
###M###M#M#M#.###T###F###F#F#.#
#MMM#MMM#M#M#.FF#TTT#FFF#F#F#.#
#M###M###M#M#######T###F#F#F#F#
#M#.#M#..M#MTTTTTTTT#F#F#FFF#F#
#M#.#M###M#########F#F#F#F###.#
#M#..M#MMM#FFFFF#F#FFF#F#F#F#F#
#M###M#M#.#F###F#F###F#F#F#F#F#
#MMM#MMM#.#F#F#F#FFFFF#F#FFF#.#
#.#M#######F#F#F#F#####F#####.#
#.#M#MMMMT#TTS#F#FFFFF#FFFFFF.#
###M#M###T#T###F#F###########.#
#MMM#M#.#MTT#FFF#F#FFFFFF...#.#
#M#.#M#.#####F#####F#######.#.#
#M#.#M#....F#F#FFFFF#FFF#...#.#
#M###M#.#####F#F#####F###.###.#
#M#MMM..#...#F#FFF#FFFF.#...#.#
#M#M#####.#.#F###F#####.###.#.#
#M#MMM#...#..F#FFF#FF...#...#.#
#M###M#.#######F###F###.#.###.#
#MMMMM#.............#...#.....#
###############################
```

## Case 449 (final failure)

Loop gain: `-0.0251`. First loop F1 `0.3168` with 196 false positives and 93 misses. loop12 F1 `0.2917` with 168 false positives and 104 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMM#TTT#FFF#FFF#FF.........#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FF.#...#
#T###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFFF#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FFFFFFF#...#
#M###F#####F#F#F#######F#F#.#.#
#MTT#F#FFF#F#F#FFFFFFFFF#F#F#.#
###T#F#F#F#F#F###########F#F#.#
#.#TMT#F#FFF#F#FFFFF#FFF#F#.#.#
#.###M#F#######F#F###F#F#F#.#.#
#...#M#.#FFFFFFF#F#FFF#F.F#F#.#
#.###M#.#F#######F#F#########.#
#.#MMM#.FF#TTTTT#FFF#TTTFF#.F.#
#.#M#######M###T#####T#T#.#.#.#
#MMM#..MMM#MMM#TTT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#TTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TMM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

### loop2

```text
###############################
#MMMMM#TTT#FFF#FFF#FFF........#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#F..#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFF.#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FF.....#...#
#M###F#####F#F#F#######.#.#.#.#
#MTT#F#FFF#F#F#FFFFFFFF.#.#.#.#
###M#.#F#F#F#F###########.#.#.#
#.#MMM#.#F.F#F#FFFFF#FF.#.#.#.#
#.###M#.#######F#F###F#.#.#.#.#
#...#M#.#.....FF#F#FFF#F..#F#.#
#.###M#.#.#######F#F#########.#
#.#MMM#...#MMMMT#FFF#TTTFF#FF.#
#.#M#######M###T#####T#T#.#.#.#
#MMM#..MMM#MMM#MMT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#MTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TTM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

### loop4

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#F..#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFF.#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FF.....#...#
#M###F#####F#F#F#######.#.#.#.#
#MMT#.#FFF#F#F#FFFFFFFF.#.#.#.#
###M#.#F#F#F#F###########.#.#.#
#.#MMM#.#F.F#F#FFFFF#FFF#.#.#.#
#.###M#.#######F#F###F#.#.#.#.#
#...#M#.#.....FF#F#F.F#...#.#.#
#.###M#.#.#######F#F#########.#
#.#MMM#...#MMMMT#FFF#TTTFF#FF.#
#.#M#######M###M#####T#T#.#.#.#
#MMM#..MMM#MMM#MMT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#MTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TTM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

### loop6

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#F....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#F..#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFF.#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FF.....#...#
#M###F#####F#F#F#######.#.#.#.#
#MMT#.#FFF#F#F#FFFFFFFF.#.#.#.#
###M#.#F#F#F#F###########.#.#.#
#.#MMM#.#F.F#F#FFFFF#FFF#.#.#.#
#.###M#.#######F#F###F#.#.#.#.#
#...#M#.#.....FF#F#FFF#...#.#.#
#.###M#.#.#######F#F#########.#
#.#MMM#...#MMMMM#FFF#TTTFF#FF.#
#.#M#######M###M#####T#T#.#.#.#
#MMM#..MMM#MMM#MMT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#MTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TMM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

### loop10

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#.....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#F..#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFF.#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FF.....#...#
#M###F#####F#F#F#######.#.#.#.#
#MTT#F#FFF#F#F#FFFFFFFF.#.#.#.#
###M#.#F#F#F#F###########.#.#.#
#.#MMM#.#F.F#F#FFFFF#FFF#.#.#.#
#.###M#.#######F#F###F#F#.#.#.#
#...#M#.#.....FF#F#FFF#...#F#.#
#.###M#.#.#######F#F#########.#
#.#MMM#...#MMMMT#FFF#TTTFF#FF.#
#.#M#######M###M#####T#T#.#.#.#
#MMM#..MMM#MMM#MMT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#MTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TMM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

### loop12

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#F....#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#.....#
#M#####F#T###F###########.#####
#M#FFFFF#GFFFFFFFFFFFF#F..#...#
#M###F#########F#######F#####.#
#M#FFF#FFFFFFFFF#FFFFFF.#.....#
#M#F###F#####F###F#########.###
#M#FFF#FFFFF#F#FFF#FF.....#...#
#M###F#####F#F#F#######.#.#.#.#
#MMT#F#FFF#F#F#FFFFFFFF.#.#.#.#
###M#.#F#F#F#F###########.#.#.#
#.#MMM#.#F.F#F#FFFFF#FFF#.#.#.#
#.###M#.#######F#F###F#.#.#.#.#
#...#M#.#......F#F#FFF#F..#F#.#
#.###M#.#.#######F#F#########.#
#.#MMM#...#MMMMT#FFF#TTTFF#FF.#
#.#M#######M###M#####T#T#.#.#.#
#MMM#..MMM#MMM#MMT#F#T#T#...#.#
#M###.#M#M###M###T#F#T#T#######
#M#...#M#MMMMM#MTT#TTT#TTT#MMM#
#M###.#M#######T###T#####T#M#M#
#MMM#.#MMMMMMM#T#FFTSFFF#TMM#M#
###M#.#######M#T#########F###M#
#.#M#.....#MMM#TTTTTTTTT#F#MMM#
#.#M#######M###########T###M#.#
#..MMMMMMMMM#...FFFFFFFTMMMM#.#
###############################
```

## Case 177 (final failure)

Loop gain: `-0.0638`. First loop F1 `0.3579` with 196 false positives and 91 misses. loop12 F1 `0.2941` with 177 false positives and 111 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#......F.FFFFFFFFF#.......#
#M#M###.#####F#######F###.###.#
#M#MMM#FFFFF#FFF#FFF#FFFF.#.#.#
#M###M#####F###F###F#######.#.#
#MMM#T#FFFFF#F#F#FFFFFFF#FFF..#
###M#T#######F#F#F###F###F#####
#MMM#TTTTGFFFF#F#FFF#F#FFF#FF.#
#M#############F###F#F#F#####F#
#MMMMT#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#F#F#T#T#F#F#F###F#F###F#
#...#M#FFF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#..FFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#.FF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#F....#MTT#F#TTT#T#F..#
#M#F###F#####.###T#F###T#T###.#
#M#...#F#...#...#T#FFF#T#T#MMM#
#M#####.#.#.###.#T#F###T#M#M#M#
#M#MMM#...#.#...#T#F#TTM#MMM#M#
#M#M#M#.###.#.###T#F#T#######M#
#M#M#M#...#....F#TTT#TMMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop2

```text
###############################
#MMM#........FFFFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#.FFFF#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#FFFFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTTTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMMM#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#F#F#T#T#F#F#F###F#F###F#
#...#M#.FF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MMMMTMM#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#............F#T#FFFFFFF#FFFFF#
#####F###.###.#T#########F###F#
#MTT#F#FFF#...#TTTTTTTTT#FFF#F#
#M#T###F###.###########T#####F#
#M#TTSFF#F..#.#TTTTTTT#TTT#FF.#
#M#######F###.#M#####T###T#F###
#M#FFFFF#.....#MMT#F#TTT#T#F..#
#M#F###.#####.###M#F###T#T###.#
#M#...#.#...#...#M#FFF#T#M#MMM#
#M#####.#.#.###.#M#F###T#M#M#M#
#M#MMM#...#.#...#T#F#TTM#MMM#M#
#M#M#M#.###.#.###T#F#M#######M#
#M#M#M#...#..FFF#TTT#MMMMMMM#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop4

```text
###############################
#MMM#.........FFFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#.FFFF#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#FFFFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTTTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMMM#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#.#F#T#T#F#F#F###F#F###F#
#...#M#.FF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MMMMTMT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#............F#T#FFFFFFF#FFFFF#
#####F###.###.#T#########F###F#
#MMT#F#FFF#...#TTTTTTTTT#FFF#F#
#M#T###F###.###########T#####F#
#M#TTSFF#F..#.#TTTTTTT#TTT#FF.#
#M#######F###.#M#####T###T#F###
#M#FFFFF#.....#MMM#F#TTT#T#F..#
#M#.###.#####.###M#F###T#T###.#
#M#...#.#...#...#M#FFF#T#T#MMM#
#M#####.#.#.###.#M#.###T#M#M#M#
#M#MMM#...#.#...#M#F#TTM#MMM#M#
#M#M#M#.###.#.###T#F#M#######M#
#M#M#M#...#..FFF#TTT#MMMMMMM#M#
#M#M#M#############M#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop6

```text
###############################
#MMM#.........FFFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#.FFFF#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#FFFFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTTTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMMM#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#.#F#T#T#F#F#F###F#F###F#
#...#M#.FF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MMMMTMT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#............F#T#FFFFFFF#FFFFF#
#####F###.###.#T#########F###F#
#MMT#F#FFF#...#TTTTTTTTT#FFF#F#
#M#T###F###.###########T#####F#
#M#TTSFF#F..#.#TTTTTTT#TTT#FF.#
#M#######F###.#M#####T###T#F###
#M#FFFFF#.....#MMM#F#TTT#T#F..#
#M#.###.#####.###M#F###T#T###.#
#M#...#.#...#...#M#FFF#T#T#MMM#
#M#####.#.#.###.#M#.###T#M#M#M#
#M#MMM#...#.#...#M#F#TTM#MMM#M#
#M#M#M#.###.#.###T#.#M#######M#
#M#M#M#...#..FFF#TTT#MMMMMMM#M#
#M#M#M#############M#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop10

```text
###############################
#MMM#.........FFFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#.FFFF#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#FFFFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTTTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMMM#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#.#F#T#T#F#F#F###F#F###F#
#...#M#.FF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MMMMTMT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#............F#T#FFFFFFF#FFFFF#
#####F###.###.#T#########F###F#
#MMT#F#FFF#...#TTTTTTTTT#FFF#F#
#M#T###F###.###########T#####F#
#M#TTSFF#F..#.#TTTTTTT#TTT#FF.#
#M#######F###.#M#####T###T#F###
#M#FFFFF#.....#MMM#F#TTT#T#F..#
#M#.###.#####.###T#F###T#T###.#
#M#...#.#...#...#M#FFF#T#M#MMM#
#M#####.#.#.###.#M#F###T#M#M#M#
#M#MMM#...#.#...#T#F#TTM#MMM#M#
#M#M#M#.###.#.###T#.#M#######M#
#M#M#M#...#..FFF#TTT#MMMMMMM#M#
#M#M#M#############M#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

### loop12

```text
###############################
#MMM#.........FFFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#..FFF#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#FFFFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTTTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMMM#F#FFTTT#FFF#F#F#F#FFFFF#
#####M#F#F#T#T#F#F#F###F#F###F#
#...#M#.FF#T#T#F#F#FFFFFFF#F#F#
#.#.#M#####T#T###F#########F#F#
#.#.#MMMMTMT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#............F#T#FFFFFFF#FFFFF#
#####F###.###F#T#########F###F#
#MMT#F#FFF#...#TTTTTTTTT#FFF#F#
#M#T###F###.###########T#####F#
#M#TTSFF#F..#.#TTTTTTT#TTT#FF.#
#M#######F###.#M#####T###T#F###
#M#FFFFF#.....#MMM#F#TTT#T#F..#
#M#F###.#####.###M#F###T#T###.#
#M#...#.#...#...#M#FFF#T#M#MMM#
#M#####.#.#.###.#M#F###T#M#M#M#
#M#MMM#...#.#...#M#F#TTM#MMM#M#
#M#M#M#.###.#.###T#.#M#######M#
#M#M#M#...#..FFF#TTT#MMMMMMM#M#
#M#M#M#############M#######M#M#
#MMM#MMMMMMMMMMMMMMM......#MMM#
###############################
```

## Case 425 (final failure)

Loop gain: `-0.0452`. First loop F1 `0.3498` with 157 false positives and 107 misses. loop12 F1 `0.3046` with 156 false positives and 118 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#.....#FFF#FFF#FFTMM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#...#F#FFF#FFF#F#F#T#TTM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#..FFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#TMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TTM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TTMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#F#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTT#..MMMMM#.#
#.#F###F#F#T#F#T###T#.#M###T#.#
#.#F#FFF#F#T#F#TTT#T#.#M#.#TTT#
#.#.#.###F#T#####T#M###M#.###T#
#...#.#..F#TTTTTMM#M#MMM#.FF#T#
#####.#.###########M#M###F#F#T#
#...#.#...#FFF..#MMM#M#.#F#FFS#
#.#.#.###.#F###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#FFFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#.#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

### loop2

```text
###############################
#.#.....#FFF#FFF#FFTMM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#..F#F#FFF#FFF#F#F#T#TMM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#.FFFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#MMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TMM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TMMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#.#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTM#..MMMMM#.#
#.#F###F#F#T#F#T###M#.#M###M#.#
#.#F#FFF#F#T#F#TTM#M#.#M#.#MTM#
#.#.#.###F#T#####M#M###M#.###T#
#...#.#.FF#TTTTMMM#M#MMM#.FF#T#
#####.#.###########M#M###F#F#T#
#...#.#..F#F....#MMM#M#.#F#FFS#
#.#.#.###.#.###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#FFFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#F#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

### loop4

```text
###############################
#.#.....#FFF#FFF#FFTTM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#...#F#FFF#FFF#F#F#T#TMM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#..FFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#MMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TTM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TMMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#.#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTT#..MMMMM#.#
#.#F###F#F#T#F#T###M#.#M###M#.#
#.#F#FFF#F#T#F#TTT#M#.#M#.#MTM#
#.#.#.###F#T#####M#M###M#.###T#
#...#.#.FF#TTTTTMM#M#MMM#.FF#T#
#####.#.###########M#M###.#F#T#
#...#.#..F#F....#MMM#M#.#F#FFS#
#.#.#.###.#.###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#.FFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#F#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

### loop6

```text
###############################
#.#.....#FFF#FFF#FFTMM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#...#F#FFF#FFF#F#F#T#TMM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#..FFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#MMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TMM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TMMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#.#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTT#..MMMMM#.#
#.#F###F#F#T#F#T###M#.#M###M#.#
#.#F#FFF#F#T#F#TTT#M#.#M#.#MTM#
#.#.#.###F#T#####M#M###M#.###T#
#...#.#.FF#TTTTMMM#M#MMM#.FF#T#
#####.#.###########M#M###.#F#T#
#...#.#..F#F....#MMM#M#.#F#FFS#
#.#.#.###.#.###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#.FFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#.#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

### loop10

```text
###############################
#.#.....#FFF#FFF#FFTMM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#...#F#FFF#FFF#F#F#T#TMM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#..FFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#MMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TTM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TMMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#.#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTT#..MMMMM#.#
#.#F###F#F#T#F#T###M#.#M###M#.#
#.#F#FFF#F#T#F#TTT#M#.#M#.#MTM#
#.#.#.###F#T#####M#M###M#.###T#
#...#.#.FF#TTTTMMM#M#MMM#.FF#T#
#####.#.###########M#M###F#F#T#
#...#.#..F#F....#MMM#M#.#F#FFS#
#.#.#.###.#.###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#.FFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#F#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

### loop12

```text
###############################
#.#.....#FFF#FFF#FFTMM#..MMMMM#
#.#.#.###F#F#F#F#F#T#T###M###M#
#...#F#FFF#FFF#F#F#T#TMM#MMM#M#
#####F#F#######F#F#T###M###M#M#
#..FFF#FFF#F#FFF#F#T#F#M#MMM#M#
#.#######F#F#####F#T#F#M#M###M#
#.FFFFFF#FFF#FFFFF#T#F#MMM#.#M#
#F###F#####F#######T#F#####.#M#
#F#F#FFFFF#F#FFFGTTT#F#MMM..#M#
#F#F#####F#F#F#######F#M#M###M#
#FFFFF#FFF#FFF#TTTFF#TTM#MMM#M#
#####F#F#F#####T#T###T#####M#M#
#FFFFF#F#F#TTT#T#TTT#TMMMM#M#M#
#######F#F#T#T#T###T#####M#M#M#
#FFFFFFF#F#T#TTT#F#TTMMMMM#MMM#
#.#######F#T#####F#############
#.#FFFFF#F#T#FFTTTTT#..MMMMM#.#
#.#F###F#F#T#F#T###M#.#M###M#.#
#.#F#FFF#F#T#F#TTT#M#.#M#.#MTM#
#.#.#.###F#T#####M#M###M#.###M#
#...#.#FFF#TTTMTMM#M#MMM#.FF#T#
#####.#.###########M#M###F#F#T#
#...#.#..F#F....#MMM#M#.#F#FFS#
#.#.#.###.#.###.#M###M#.#F#####
#.#...#.....#...#M#MMM#.#.FFFF#
#.###############M#M###.#####F#
#.....#...#MMMMMMM#M#...#.FF#F#
#.###.#.#.#M#######M#.###.#F#.#
#...#...#..MMMMMMMMM#.....#...#
###############################
```

## Case 274 (final failure)

Loop gain: `-0.0492`. First loop F1 `0.3541` with 167 false positives and 92 misses. loop12 F1 `0.3049` with 165 false positives and 104 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#.FF#MMM#MMMMMMT#F..#
#.#M#M#M#M#####M#########T###.#
#.#M#MMM#MMTTTTM#.....#TTT#FF.#
#.#M#################.#T###F###
#MMM#...FFFFFFFTMMMM#.#T#TGF#F#
#M###.#########T###M#F#T#T###F#
#M#.FFFF#F#TTTTT#F#MMT#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#F#M#FFFFTTT#
#M#T#T#######T#F#F#F#T#F###T#F#
#MTT#TTTTT#TTT#F#FFF#T#F#F#T#F#
#########T#T###F#####T#.#F#T#F#
#FFFFFFF#TTT#FFFFF#TTT#...#T#.#
#F#####F#####F###F#T#####.#T###
#F#FFF#FFFFF#F#FFF#TTTTM#.#MMM#
#F#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFFF#M#MMMMM#
#.#F#F###F#F#F#####F#F#M#M###.#
#.#F.F#F#F#FFF#FFF#F#F#M#MMM#.#
#.#.###F#F#######F#F###M###M#.#
#.#...FF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#....F#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#......F#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFF.#.........#
###############################
```

### loop2

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#.F.#MMM#MMMMMMM#...#
#.#M#M#M#M#####M#########M###.#
#.#M#MMM#MMMTTMM#.....#TTT#FF.#
#.#M#################.#T###F###
#MMM#..FFFFFFFFMMMMM#.#T#TGF#F#
#M###.#########T###M#.#T#T###F#
#M#FFFFF#F#TTTTT#F#MMM#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#.#M#.FFFTTT#
#M#T#T#######T#F#F#F#M#.###T#F#
#MTT#TTTTT#TTT#F#FFF#M#.#F#T#.#
#########T#T###F#####M#.#.#T#.#
#FFFFFFF#TTT#FFFFF#TTM#...#T#.#
#F#####F#####F###F#T#####.#M###
#F#FFF#FFFFF#F#FFF#TTMMM#.#MMM#
#.#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFF.#M#MMMMM#
#.#F#F###F#F#F#####F#.#M#M###.#
#.#FFF#F#F#FFF#FFF#F#.#M#MMM#.#
#.#F###F#F#######F#F###M###M#.#
#.#..FFF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#..FFF#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#...FFFF#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFFF#.........#
###############################
```

### loop4

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#...#MMM#MMMMMMM#...#
#.#M#M#M#M#####M#########M###.#
#.#M#MMM#MMMTTMM#.....#MTT#FF.#
#.#M#################.#T###F###
#MMM#.FFFFFFFFFMMMMM#.#T#TGF#F#
#M###.#########T###M#F#T#T###F#
#M#FFFFF#F#TTTTT#F#MMM#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#F#M#.FFFTTT#
#M#T#T#######T#F#F#F#M#.###T#F#
#MTT#TTTTT#TTT#F#FFF#T#.#F#T#F#
#########T#T###F#####T#.#.#T#.#
#FFFFFFF#TTT#FFFFF#TTT#...#T#.#
#F#####F#####F###F#T#####.#M###
#F#FFF#FFFFF#F#FFF#TTMMM#.#MMM#
#.#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFF.#M#MMMMM#
#.#F#F###F#F#F#####F#.#M#M###.#
#.#FFF#F#F#FFF#FFF#F#.#M#MMM#.#
#.#F###F#F#######F#F###M###M#.#
#.#..FFF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#..FFF#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#....FFF#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFFF#.........#
###############################
```

### loop6

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#...#MMM#MMMMMMM#...#
#.#M#M#M#M#####M#########M###.#
#.#M#MMM#MMMTTMM#.....#MTT#FF.#
#.#M#################.#T###F###
#MMM#..FFFFFFFFMMMMM#.#T#TGF#F#
#M###F#########T###M#F#T#T###F#
#M#FFFFF#F#TTTTT#.#MMM#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#F#M#.FFFTTT#
#M#T#T#######T#F#F#F#T#.###T#F#
#MTT#TTTTT#TTT#F#FFF#T#.#F#T#.#
#########T#T###F#####T#.#.#T#.#
#FFFFFFF#TTT#FFFFF#TTT#...#T#.#
#F#####F#####F###F#T#####.#M###
#F#FFF#FFFFF#F#FFF#TTTMM#.#MMM#
#F#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFF.#M#MMMMM#
#.#F#F###F#F#F#####F#.#M#M###.#
#.#FFF#F#F#FFF#FFF#F#.#M#MMM#.#
#.#F###F#F#######F#F###M###M#.#
#.#..FFF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#..FFF#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#...FFFF#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFFF#.........#
###############################
```

### loop10

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#.F.#MMM#MMMMMMM#...#
#.#M#M#M#M#####M#########M###.#
#.#M#MMM#MMMTTMM#.....#MTT#FF.#
#.#M#################.#T###F###
#MMM#..FFFFFFFFMMMMM#.#T#TGF#F#
#M###.#########T###M#.#T#T###F#
#M#FFFFF#F#TTTTT#F#MMM#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#.#M#.FFFTTT#
#M#T#T#######T#F#F#F#M#.###T#F#
#MTT#TTTTT#TTT#F#FFF#T#.#F#T#.#
#########T#T###F#####T#.#.#T#.#
#FFFFFFF#TTT#FFFFF#TTT#...#T#.#
#F#####F#####F###F#T#####.#M###
#F#FFF#FFFFF#F#FFF#TTMMM#.#MMM#
#.#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFF.#M#MMMMM#
#.#F#F###F#F#F#####F#.#M#M###.#
#.#FFF#F#F#FFF#FFF#F#.#M#MMM#.#
#.#F###F#F#######F#F###M###M#.#
#.#F.FFF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#..FFF#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#...FFFF#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFF.#.........#
###############################
```

### loop12

```text
###############################
#.#MMM..........#MMM..#.....#.#
#.#M#M#######.###M#M#######.#.#
#.#M#M#MMM#...#MMM#MMMMMMM#...#
#.#M#M#M#M#####M#########M###.#
#.#M#MMM#MMMTTMM#.....#MTT#F..#
#.#M#################.#T###F###
#MMM#.FFFFFFFFFMMMMM#.#T#TGF#F#
#M###.#########T###M#.#T#T###F#
#M#FFFFF#F#TTTTT#F#MMM#TTT#FST#
#M#####F#F#T#####F###M#######T#
#M#TTT#FFF#TTT#FFF#F#M#.FFFTTT#
#M#T#T#######T#F#F#F#M#.###T#F#
#MTT#TTTTT#TTT#F#FFF#T#.#F#T#.#
#########T#T###F#####T#.#.#T#.#
#FFFFFFF#TTT#FFFFF#TTM#...#T#.#
#F#####F#####F###F#T#####.#M###
#F#FFF#FFFFF#F#FFF#TTMMM#.#MMM#
#.#F###F#####F#F#######M#####M#
#.#F#FFF#FFF#F#FFFFFF.#M#MMMMM#
#.#F#F###F#F#F#####F#.#M#M###.#
#.#FFF#F#F#FFF#FFF#F#.#M#MMM#.#
#.#F###F#F#######F#F###M###M#.#
#.#F.FFF#F#FFFFFFF#F#MMM#MMM#.#
#.#.#####F#F#####F#F#M###M###.#
#.#.#..FFF#FFF#F#F#F#MMM#M#.#.#
#.###.#######F#F#F#F###M#M#.#.#
#...#...FFFF#F#FFF#F..#MMM#.#.#
#.#.#######F#F#F#####.#####.#.#
#.#........FFF#FFFFF#.........#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0120`. First loop F1 `0.3173` with 191 false positives and 93 misses. loop12 F1 `0.3053` with 163 false positives and 101 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#..FFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#..F..#FFF#F#FFF#F#F#FFFFFF.#M#
#.###F###F#F#####F#F#######.#M#
#..F#FFF#FFF#FFFFF#F#TTTTM#.#M#
###F###F#####F#####F#T###M###M#
#..FFF#FFFFFFF#FFFFF#TMT#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#FFF#M#MMM#.#
#.#T###T#####F###F###F#M#####.#
#.#T#F#TTT#FFF#FFF#FF.#MMM#...#
#.#T#F###T###F#F###F#####M###.#
#.#T#FFF#TTT#F#FFF#F#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MMT#FFF#TTT#FFF#FFF.F#...#MMM#
#M###F###T#######F#####.#####M#
#M#.#...FTTTTTTT#FFFF.#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MTT#FFFFT#FFF#F#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#FF.#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTTMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#.....#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#..F#F#FFF#F#F#FFFFFF.#M#
#.###.###F#F#####F#F#######.#M#
#...#...#FFF#FFFFF#F#MMMMM#.#M#
###.###.#####F#####F#M###M###M#
#..F..#..FFFFF#FFFF.#MMM#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#F..#M#MMM#.#
#.#T###T#####F###F###.#M#####.#
#.#T#F#TTT#FFF#FFF#...#MMM#...#
#.#T#F###T###F#F###.#####M###.#
#.#T#FFF#TTT#F#FFF#.#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MMT#FFF#TTT#FFF#FF...#...#MMM#
#M###.###T#######F#####.#####M#
#M#.#...FTTTTTTT#FFF..#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MMT#FFFFT#FFF#.#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#....F#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#..F#F#FFF#F#F#FFFFFFF#M#
#.###.###F#F#####F#F#######.#M#
#...#...#FFF#FFFFF#F#MMMMM#.#M#
###.###.#####F#####F#M###M###M#
#..F..#..FFFFF#FFFF.#MMM#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#F..#M#MMM#.#
#.#T###T#####F###F###.#M#####.#
#.#T#F#TTT#FFF#FFF#...#MMM#...#
#.#T#F###T###F#F###.#####M###.#
#.#T#FFF#TTT#F#FFF#.#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MTT#FFF#TTT#FFF#FF...#...#MMM#
#M###.###T#######F#####.#####M#
#M#.#...FTTTTTTT#FFF..#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MMT#FFFFT#FFF#.#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#....F#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#..F#F#FFF#F#F#FFFFFF.#M#
#.###.###F#F#####F#F#######.#M#
#...#...#FFF#FFFFF#F#MTTMM#.#M#
###.###.#####F#####F#M###M###M#
#..FF.#..FFFFF#FFFFF#MMM#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#F..#M#MMM#.#
#.#T###T#####F###F###.#M#####.#
#.#T#F#TTT#FFF#FFF#...#MMM#...#
#.#T#F###T###F#F###.#####M###.#
#.#T#FFF#TTT#F#FFF#.#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MTT#FFF#TTT#FFF#FFF..#...#MMM#
#M###.###T#######F#####.#####M#
#M#.#..FFTTTTTTT#FFF..#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MMT#FFFFT#FFF#.#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#....F#F#F#F#FFF#FFFFFFF#M#
#.#######.#F#F#F#F#F#######F#M#
#.....#..F#F#FFF#F#F#FFFFFF.#M#
#.###.###F#F#####F#F#######.#M#
#...#...#FFF#FFFFF#F#MMTMM#.#M#
###.###.#####F#####F#M###M###M#
#..F..#..FFFFF#FFFFF#MMM#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#F..#M#MMM#.#
#.#T###T#####F###F###.#M#####.#
#.#T#F#TTT#FFF#FFF#...#MMM#...#
#.#T#F###T###F#F###.#####M###.#
#.#T#FFF#TTT#F#FFF#.#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MTT#FFF#TTT#FFF#FFF..#...#MMM#
#M###.###T#######F#####.#####M#
#M#.#..FFTTTTTTT#FFF..#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MMT#FFFFT#FFF#.#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.....#....FFFFFFFFF#F#FGTTT#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#..F#FFFFFFF#FFF#FFF#TTM#
###.#.#####F###F#####F#######M#
#...#....F#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#M#
#.....#...#F#FFF#F#F#FFFFFF.#M#
#.###.###F#F#####F#F#######.#M#
#...#...#FFF#FFFFF#F#MMMMM#.#M#
###.###.#####F#####F#M###M###M#
#..FF.#..FFFFF#FFFFF#MMM#M#MMM#
#.#######F#####F###F###M#M#M###
#.#TTTTT#FFFFF#FFF#F..#M#MMM#.#
#.#T###T#####F###F###.#M#####.#
#.#T#F#TTT#FFF#FFF#...#MMM#...#
#.#T#F###T###F#F###.#####M###.#
#.#T#FFF#TTT#F#FFF#.#...#MMM#.#
###T###F###T#F###F###.#.###M#.#
#MTT#FFF#TTT#FFF#FF...#...#MMM#
#M###.###T#######F#####.#####M#
#M#.#..FFTTTTTTT#FFF..#.#MMM#M#
#M#.#.#########T#####.#.#M#M#M#
#M#...#MMT#FFFFT#FFF#.#.#M#MMM#
#M###.#M#T#####T#F#F###.#M#####
#MMM#.#M#MTTTT#TSF#F#F..#MMM#.#
#.#M###M#####T#####F#F#.###M#.#
#.#MMM#MMM#F#T#FFFFFFF#...#MMM#
#.###M###M#F#T###############M#
#...#MMMMM..#TTTTTTTMMMMMMMMMM#
###############################
```

## Case 14 (final failure)

Loop gain: `-0.0223`. First loop F1 `0.3350` with 179 false positives and 95 misses. loop12 F1 `0.3127` with 176 false positives and 101 misses. Final exact `0.0000`.

### loop1

```text
###############################
#....MMM#MMM#..TTT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#TTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#.###
#...#M#MMT#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###F#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMT#FFF#FFF#FFFFFFFFF#F#F#
#M###M#T#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###T#####F#########F#F#F#F#
#MMM..#T#FFF#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMTTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMTTT#FFF#FFF.FFFFF#F#FFFFFFF#
#.###T#S###########F#F#F#####.#
#.#TTT#TTTTT#MMM#TMT#F#F#FFF#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMTTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#M#F....#...#
#.###.#M###.#######T###########
#.....#MMM#.#MMMMM#T#...#.....#
#####.###M###M###M#M#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

### loop2

```text
###############################
#....MMM#MMM#..MTT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#MTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#F###
#...#M#MMT#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###.#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMM#FFF#FFF#FFFFFFFFF#F#F#
#M###M#M#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###M#####F#########F#F#F#F#
#MMM..#M#F.F#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMMTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMMTT#FFF#FFFFFFFFF#F#FFFFFF.#
#.###T#S###########F#F#F#####.#
#.#TTT#TTTTT#MMM#MMT#F#F#FF.#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMMTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#M#.....#...#
#.###.#M###.#######M###########
#.....#MMM#.#MMMMT#T#...#.....#
#####.###M###M###T#T#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

### loop4

```text
###############################
#....MMM#MMM#..MMT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#MTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#F###
#...#M#MMM#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###.#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMM#FFF#FFF#FFFFFFFFF#F#F#
#M###M#M#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###M#####F#########F#F#F#F#
#MMM..#M#F.F#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMMTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMMTT#FFF#FFFFFFFFF#F#FFFFFF.#
#.###T#S###########F#F#F#####.#
#.#MTT#TTTTT#MMM#MTT#F#F#FF.#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMTTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#M#.....#...#
#.###.#M###.#######M###########
#.....#MMM#.#MMMTT#T#...#.....#
#####.###M###M###T#T#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

### loop6

```text
###############################
#....MMM#MMM#..MMT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#MTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#F###
#...#M#MMM#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###.#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMM#FFF#FFF#FFFFFFFFF#F#F#
#M###M#M#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###M#####F#########F#F#F#F#
#MMM..#M#F.F#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMMTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMMTT#FFF#FFFFF.FFF#F#FFFFFF.#
#.###T#S###########F#F#F#####.#
#.#MTT#TTTTT#MMM#MTT#F#F#FF.#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMTTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#M#.....#...#
#.###.#M###.#######T###########
#.....#MMM#.#MMMTT#T#...#.....#
#####.###M###M###T#T#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

### loop10

```text
###############################
#....MMM#MMM#..MMT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#MTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#F###
#...#M#MMM#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###.#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMM#FFF#FFF#FFFFFFFFF#F#F#
#M###M#M#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###M#####F#########F#F#F#F#
#MMM..#M#F.F#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMMTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMMTT#FFF#FFFFF.FFF#F#FFFFFF.#
#.###T#S###########F#F#F#####.#
#.#MTT#TTTTT#MMM#MTT#F#F#FF.#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMTTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#T#.....#...#
#.###.#M###.#######M###########
#.....#MMM#.#MMMTT#T#...#.....#
#####.###M###M###T#T#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

### loop12

```text
###############################
#....MMM#MMM#..MMT#FFFFF......#
#.###M#M#M#M###T#T#####F#####.#
#...#M#M#M#MTTTT#T#FFFFF#FF.#.#
###.#M#M#M#######T#F#####F#F###
#...#M#MMM#TTT#TTT#FFFFF#F#FF.#
#.###M#####T#T#T#######F#F###.#
#.#MMM#.#TTT#T#T#FFFFF#FFFFF#F#
#.#M###.#T###T#T###F#F#######F#
#.#M#..FFTTG#TTTFFFF#FFF#FFF#F#
###M#######F###########F#F#F#F#
#MMM#MMM#FFF#FFF#FFFFFFFFF#F#F#
#M###M#M#F###F#F###########F#F#
#M#MMM#M#FFF#F#FFFFFFFFF#F#F#F#
#M#M###M#####F#########F#F#F#F#
#MMM..#T#F.F#FFF#FFF#FFF#FFF#F#
#######T#F#F###F###F#F###F###F#
#MMMMTTT#F#FFFFF#FFF#F#F#F#FFF#
#M#######F#######F###F#F#F###F#
#MMMTT#FFF#FFFFFFFFF#F#FFFFFF.#
#.###T#S###########F#F#F#####.#
#.#MTT#TTTTT#MMM#MTT#F#F#FF.#.#
#.#M#######T#M#M#M#T#F#F#F#.###
#.#MMTTTTT#MMM#M#M#T#F#F#.#...#
#.#######M#####M#M#T#F###.###.#
#.#...#MMM....#MMM#M#.....#...#
#.###.#M###.#######T###########
#.....#MMM#.#MMMTT#T#...#.....#
#####.###M###M###T#T#.#.###.#.#
#.......#MMMMM..#MMM..#.....#.#
###############################
```

## Case 302 (final failure)

Loop gain: `0.0252`. First loop F1 `0.2911` with 187 false positives and 115 misses. loop12 F1 `0.3163` with 185 false positives and 109 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#TMTTTMMMMMMMMMMMMMMMM#MMM#
###G#T#######F###########M#M#M#
#MMT#TTT#FFFF.#...#.#MMMMM#M#M#
#M#F###T#######.#.#.#M#####M#M#
#M#F#F#TTT#FFFFF#.#.#MMMMM#M#M#
#M#F#F###S#F#####F#.#####M#M#M#
#M#FFF#FFF#F#FFF#F#.....#MMM#M#
#M#.###F#F#F#F#F#F###F#######M#
#M#.#FFF#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#..FFFFF#FFFFFFF#FFFF.#MTM#
###M#########F#######F#####T###
#MMM..#FFFFF#FFFFF#FFF#FFF#TTM#
#M###F#F###F#####F###F#F#F###T#
#MMT#.FF#FFFFFFF#FFF#F#F#F#F#T#
#.#M###############F###F#F#F#T#
#.#M#.FFFF#FFFFFFF#F#FFF#F#TTT#
#.#M#####F#F#####F#F#F###F#T###
#.#M..F.#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MTTFF#FFFFFFFFF#F#TTMTM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######T#T###T#####F#F#M###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFF.....#M#
#.#####.#####F#F#T###########M#
#............F#F#TTTTTMMMMMMMM#
###############################
```

### loop2

```text
###############################
#..F#TTTTTTTMMMMMMMMMMMMMM#MMM#
###G#T#######.###########M#M#M#
#MTT#TTT#FFFFF#...#.#MMMMM#M#M#
#T#F###T#######.#.#.#M#####M#M#
#M#F#F#TTT#FFFF.#.#.#MMMMM#M#M#
#M#F#F###S#F#####.#.#####M#M#M#
#M#FFF#FFF#F#FFF#.#.....#MMM#M#
#M#F###F#F#F#F#F#.###.#######M#
#M#F#FFF#F#FFF#F#FFF#.#.....#M#
#M###.#####F#######F#.#.###.#M#
#MMM#..FFFFF#FFFFFFF#FFF..#MMM#
###M#########F#######F#####T###
#MMT..#FFFFF#FFFFF#FFF#FFF#TTM#
#M###.#F###F#####F###F#F#F###M#
#MMM#.FF#FFFFFFF#FFF#F#F#F#F#M#
#.#M###############F###F#F#F#T#
#.#M#..FFF#FFFFFFF#F#FFF#F#TTM#
#.#M#####F#F#####F#F#F###F#T###
#.#M....#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MMTFF#FFFFFFFFF#F#TTTMM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######M#T###T#####F#F#M###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#M#F#####T#########M#
#...#...#MMT#FFF#TTTFFFF....#M#
#.#####.#####F#F#T###########M#
#...........FF#F#TTTTTTMMMMMMM#
###############################
```

### loop4

```text
###############################
#..F#TTTTTTTMMMMMMMMMMMMMM#MMM#
###G#T#######F###########M#M#M#
#TTT#TTT#FFFFF#...#.#MMMMM#M#M#
#T#F###T#######.#.#.#M#####M#M#
#T#F#F#TTT#FFFF.#.#.#MMMMM#M#M#
#T#F#F###S#F#####.#.#####M#M#M#
#M#FFF#FFF#F#FFF#.#.....#MMM#M#
#M#F###F#F#F#F#F#F###.#######M#
#M#F#FFF#F#FFF#F#FF.#.#.....#M#
#M###F#####F#######F#.#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###T#########F#######F#####T###
#MMTF.#FFFFF#FFFFF#FFF#FFF#TTM#
#M###.#F###F#####F###F#F#F###M#
#MMT#.FF#FFFFFFF#FFF#F#F#F#F#T#
#.#M###############F###F#F#F#T#
#.#M#..FFF#FFFFFFF#F#FFF#F#TTM#
#.#M#####F#F#####F#F#F###F#T###
#.#M....#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MMTFF#FFFFFFFFF#F#TTTMM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######M#T###T#####F#F#T###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFFF....#M#
#.#####.#####F#F#T###########M#
#...........FF#F#TTTTTTMMMMMMM#
###############################
```

### loop6

```text
###############################
#..F#TTTTTTTMMMMMMMMMMMMMM#MMM#
###G#T#######F###########M#M#M#
#TTT#TTT#FFFFF#...#.#MMMMM#M#M#
#T#F###T#######.#.#.#M#####M#M#
#T#F#F#TTT#FFFF.#.#.#MMMMM#M#M#
#M#F#F###S#F#####.#.#####M#M#M#
#M#FFF#FFF#F#FFF#.#.....#MMM#M#
#M#F###F#F#F#F#F#.###.#######M#
#M#F#FFF#F#FFF#F#FFF#.#.....#M#
#M###F#####F#######F#.#.###.#M#
#MMM#.FFFFFF#FFFFFFF#FFF..#MMM#
###T#########F#######F#####T###
#MMT..#FFFFF#FFFFF#FFF#FFF#TTM#
#M###.#F###F#####F###F#F#F###M#
#MMT#.FF#FFFFFFF#FFF#F#F#F#F#T#
#.#M###############F###F#F#F#T#
#.#M#..FFF#FFFFFFF#F#FFF#F#TTM#
#.#M#####F#F#####F#F#F###F#T###
#.#M...F#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MMTFF#FFFFFFFFF#F#TTTMM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######M#T###T#####F#F#T###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFFF....#M#
#.#####.#####F#F#T###########M#
#...........FF#F#TTTTTTMMMMMMM#
###############################
```

### loop10

```text
###############################
#..F#TTTTTTTMMMMMMMMMMMMMM#MMM#
###G#T#######F###########M#M#M#
#TTT#TTT#FFFFF#...#.#MMMMM#M#M#
#T#F###T#######.#.#.#M#####M#M#
#T#F#F#TTT#FFFF.#.#.#MMMMM#M#M#
#M#F#F###S#F#####.#.#####M#M#M#
#M#FFF#FFF#F#FFF#.#.....#MMM#M#
#M#F###F#F#F#F#F#.###.#######M#
#M#F#FFF#F#FFF#F#FFF#.#.....#M#
#M###F#####F#######F#.#.###.#M#
#MMT#.FFFFFF#FFFFFFF#.....#MMM#
###T#########F#######F#####T###
#MMT..#FFFFF#FFFFF#FFF#FFF#TTM#
#M###.#F###F#####F###F#F#F###M#
#MMT#.FF#FFFFFFF#FFF#F#F#F#F#T#
#.#T###############F###F#F#F#M#
#.#M#.FFFF#FFFFFFF#F#FFF#F#TTM#
#.#M#####F#F#####F#F#F###F#T###
#.#M...F#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MMTFF#FFFFFFFFF#F#TTTMM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######M#T###T#####F#F#T###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFFF....#M#
#.#####.#####F#F#T###########M#
#...........FF#F#TTTTTTMMMMMMM#
###############################
```

### loop12

```text
###############################
#..F#TTTTTTTMMMMMMMMMMMMMM#MMM#
###G#T#######F###########M#M#M#
#TTT#TTT#FFFFF#...#.#MMMMM#M#M#
#T#F###T#######.#.#.#M#####M#M#
#T#F#F#TTT#FFFF.#.#.#MMMMM#M#M#
#M#F#F###S#F#####.#.#####M#M#M#
#M#FFF#FFF#F#FFF#.#.....#MMM#M#
#M#F###F#F#F#F#F#.###.#######M#
#M#F#FFF#F#FFF#F#FFF#.#.....#M#
#M###F#####F#######F#.#.###.#M#
#MMM#.FFFFFF#FFFFFFF#F.F..#MMM#
###T#########F#######F#####T###
#MMT.F#FFFFF#FFFFF#FFF#FFF#TTM#
#M###.#F###F#####F###F#F#F###M#
#MMT#.FF#FFFFFFF#FFF#F#F#F#F#T#
#.#M###############F###F#F#F#T#
#.#M#..FFF#FFFFFFF#F#FFF#F#TTM#
#.#M#####F#F#####F#F#F###F#T###
#.#M...F#FFF#FFFFF#FFF#FFF#TTM#
###M#.#######F#########F#####M#
#MMM#.#MMTFF#FFFFFFFFF#F#TTTMM#
#M#####M#T###########F#F#T#####
#MMMMMMM#T#TTTTTFFFF#F#F#T#...#
#.#######M#T###T#####F#F#T###.#
#...#...#M#T#F#TTTTT#FFF#MMMMM#
###.#.###M#T#F#####T#########M#
#...#...#MMT#FFF#TTTFFFF....#M#
#.#####.#####F#F#T###########M#
#...........FF#F#TTTTTTMMMMMMM#
###############################
```

## Case 79 (final failure)

Loop gain: `-0.0067`. First loop F1 `0.3275` with 144 false positives and 123 misses. loop12 F1 `0.3208` with 147 false positives and 124 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#F.TMMMTMM#MMMMMMMMMMM#
#.###.#.#F#T#####M#M#########M#
#...#.#F#F#TTTTT#MMM#.#MMMMM#M#
#####.#F#F#####T#####.#M###M#M#
#....F#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###T#.#####M#####
#.FF#F#FFF#F#F#TTT#MMMMMMM#...#
#F#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#FFM#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTT#TMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#FF...#MMTTT#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTT#...#MTTTT#
#F#F#F###G#F#######T#.#.#####T#
#F#F#F#TTT#F#FFFFF#T#.#..FFF#S#
#F#F#F#T###F#F###F#T#####F#F#F#
#.#F#FFTTT#F#F#F#F#MMMMM#F#FFF#
#.#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#F..#...#MMMMTTM#
#.#######T#F#####.#.#.#######M#
#....FFF#T#F#TTM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMMMTT#F#T#M#...#MMMMM#...#
#.#M#########T#T#####M#####.###
#.#MMMMM#.FTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMMMM#.................#
###############################
```

### loop2

```text
###############################
#.......#FFTTTTTMM#MMMMMMMMMMM#
#.###.#F#F#T#####M#M#########M#
#...#.#F#F#TTTTT#TMM#.#MMMMM#M#
#####F#F#F#####T#####.#M###M#M#
#..FFF#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###T#.#####M#####
#.FF#F#FFF#F#F#TTT#MMMMMMM#...#
#F#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#F.M#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTT#MMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#F....#MMTTM#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTT#...#MTTTT#
#F#F#F###G#F#######M#.#.#####T#
#F#F#F#TTT#F#FFFFF#M#.#..FFF#S#
#F#F#F#T###F#F###F#M#####F#F#F#
#F#F#FFTTT#F#F#F#.#MMMMM#.#FFF#
#F#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#...#...#MMMMTTT#
#.#######T#F#####.#.#.#######M#
#..FFFFF#T#F#TMM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMMTTT#F#T#M#...#MMMMM#...#
#.#M#########M#M#####M#####.###
#.#MMMMM#FFTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMTMM#.................#
###############################
```

### loop4

```text
###############################
#.......#FFTTTTTMM#MMMMMMMMMMM#
#.###.#F#F#T#####M#M#########M#
#...#.#F#F#TTTTT#TMM#.#MMMMM#M#
#####F#F#F#####T#####.#M###M#M#
#..FFF#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###M#.#####M#####
#.FF#F#FFF#F#F#TTT#MMMMMMM#...#
#F#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#F.M#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTT#MMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#.....#MMTTM#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTT#...#MTTTT#
#F#F#F###G#F#######M#.#.#####T#
#F#F#F#TTT#F#FFFFF#M#.#..FFF#S#
#F#F#F#T###F#F###.#M#####F#F#F#
#F#F#FFTTT#F#F#F#.#MMMMM#.#FFF#
#F#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#...#...#MMMMTTT#
#.#######T#F#####.#.#.#######M#
#..FFFFF#T#F#TTM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMTTTT#F#T#M#...#MMMMM#...#
#.#M#########T#M#####M#####.###
#.#MMMMM#FFTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMTMM#.................#
###############################
```

### loop6

```text
###############################
#.......#FFTTTTTMM#MMMMMMMMMMM#
#.###.#F#F#T#####M#M#########M#
#...#.#F#F#TTTTT#TMM#.#MMMMM#M#
#####F#F#F#####T#####.#M###M#M#
#..FFF#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###M#.#####M#####
#.FF#F#FFF#F#F#TTM#MMMMMMM#...#
#.#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#F.M#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTT#MMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#F....#MMTTM#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTT#...#MTTTT#
#F#F#F###G#F#######M#.#.#####T#
#F#F#F#TTT#F#FFFFF#M#.#..FFF#S#
#F#F#F#T###F#F###.#M#####F#F#F#
#F#F#FFTTT#F#F#F#.#MMMMM#.#FFF#
#F#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#...#...#MMMMMTT#
#.#######T#F#####.#.#.#######M#
#..FFFFF#T#F#TMM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMTTTT#F#T#M#...#MMMMM#...#
#.#M#########T#M#####M#####.###
#.#MMMMM#FFTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMTMM#.................#
###############################
```

### loop10

```text
###############################
#.......#FFTTTTTMM#MMMMMMMMMMM#
#.###.#F#F#T#####M#M#########M#
#...#.#F#F#TTTTT#TMM#.#MMMMM#M#
#####F#F#F#####T#####.#M###M#M#
#..FFF#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###M#.#####M#####
#.FF#F#FFF#F#F#TTT#MMMMMMM#...#
#F#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#F.M#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTM#MMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#.....#MMTTT#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTM#...#MTTTT#
#F#F#F###G#F#######M#.#.#####T#
#F#F#F#TTT#F#FFFFF#M#.#..FFF#S#
#F#F#F#T###F#F###.#M#####F#F#F#
#F#F#FFTTT#F#F#F#.#MMMMM#.#FFF#
#F#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#...#...#MMMMMTT#
#.#######T#F#####.#.#.#######M#
#..FFFFF#T#F#TTM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMTTTT#F#T#M#...#MMMMM#...#
#.#M#########T#M#####M#####.###
#.#MMMMM#FFTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMTMM#.................#
###############################
```

### loop12

```text
###############################
#.......#FFTTTTTMM#MMMMMMMMMMM#
#.###.#F#F#T#####M#M#########M#
#...#.#F#F#TTTTT#TMM#.#MMMMM#M#
#####F#F#F#####T#####.#M###M#M#
#..FFF#F#FFFFF#TTM#...#MMM#MMM#
#.#####F#####F###M#.#####M#####
#.FF#F#FFF#F#F#TTT#MMMMMMM#...#
#F#F#F###F#F#F#T###M#########.#
#F#F#FFFFF#FFF#T#F.M#MMM#MMM#.#
###F#F#####F###T###M#M#M#M#M#.#
#FFF#F#FFF#F#F#TTT#MMM#MMM#MMM#
#F###F#F#F#F#F###T###########M#
#FFF#FFF#F#FFFFF#T#.....#MMTTM#
###F#####F#####F#T###.###M#####
#F#F#FFF#F#FFFFF#TTM#...#MTTTT#
#F#F#F###G#F#######M#.#.#####T#
#F#F#F#TTT#F#FFFFF#M#.#..FFF#S#
#F#F#F#T###F#F###.#M#####F#F#F#
#F#F#FFTTT#F#F#F#.#MMMMM#.#FFF#
#F#F#####T#F#F#F#.#####M#######
#.#FFFFF#T#FFF#...#...#MMMMMTT#
#.#######T#F#####.#.#.#######M#
#..FFFFF#T#F#TTM#.#.#...#MMMMM#
#.#######T#F#T#M#.#.#####M###.#
#.#MMMTTTT#F#T#M#...#MMMMM#...#
#.#M#########T#M#####M#####.###
#.#MMMMM#FFTTT#MMMMMMM#...#...#
#.#####M###T###########.#####.#
#......MMTMM#.................#
###############################
```

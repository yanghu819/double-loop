# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 302 (final failure)

Loop gain: `-0.0076`. First loop F1 `0.0211` with 105 false positives and 174 misses. loop12 F1 `0.0135` with 118 false positives and 175 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#.....#...#.#MMMMM#M#M#
#M#.###M#######.#.#.#M#####M#M#
#M#.#.#MMM#.....#.#.#MMMMM#M#M#
#M#.#.###S#.#####.#.#####M#M#M#
#M#...#...#.#..F#F#F....#MMM#M#
#M#.###.#.#F#F#F#F###F#######M#
#M#.#...#F#.FF#F#FFF#F#.....#M#
#M###.#####.#######F#F#.###.#M#
#MMM#....FFF#FFFFFFF#FF...#MMM#
###M#########F#######F#####M###
#MMM..#...FF#FFFFF#FFF#F..#MMM#
#M###F#.###F#####F###F#F#.###M#
#MMM#.FF#FFFFFFF#FFF#F#F#.#.#M#
#.#M###############F###F#F#.#M#
#.#M#.....#FFFFFFF#F#FFF#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#.FF#FFFFF#FFF#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMM..#FFFFFFFFF#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMMTFFFF#.#.#M#...#
#.#######M#M###T#####.#.#M###.#
#...#...#M#M#.#MTMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#..F..#FF.#.#MMMMM#M#M#
#M#.###M#######F#.#.#M#####M#M#
#M#.#.#MMM#FFFFF#.#.#MMMMM#M#M#
#M#.#.###S#F#####.#.#####M#M#M#
#M#...#..F#F#FFF#F#.....#MMM#M#
#M#.###.#F#F#F#F#F###.#######M#
#M#.#...#F#FFF#F#FFF#.#.....#M#
#M###.#####F#######F#.#.###.#M#
#MMM#...FFFF#FFFFFFF#.....#MMM#
###M#########F#######F#####M###
#MMM..#FFFFF#FFFFF#FFF#...#MMM#
#M###.#F###F#####F###F#.#.###M#
#MMM#..F#FFFFFFF#FFF#F#F#.#.#M#
#.#M###############F###F#.#.#M#
#.#M#.FFFF#FFFFFFF#F#FF.#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#FFF#FFFFF#FFF#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMM.F#FFFFFFFF.#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMMMFF.F#.#.#M#...#
#.#######M#M###M#####.#.#M###.#
#...#...#M#M#.#MTMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#....F#FF.#.#MMMMM#M#M#
#M#.###M#######F#.#.#M#####M#M#
#M#.#.#MMM#.FFFF#.#.#MMMMM#M#M#
#M#.#.###S#F#####F#.#####M#M#M#
#M#...#...#F#FFF#F#F....#MMM#M#
#M#.###.#F#F#F#F#F###F#######M#
#M#.#...#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#...FFFF#FFFFFFF#FF...#MMM#
###M#########F#######F#####M###
#MMM..#.FFFF#FFFFF#FFF#...#MMM#
#M###.#.###F#####F###F#.#.###M#
#MMM#...#FFFFFFF#FFF#F#.#.#.#M#
#.#M###############F###.#.#.#M#
#.#M#..FFF#FFFFFFF#F#FF.#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#FFF#FFFFF#FFF#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMTFF#FFFFFFF..#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMMM..F.#.#.#M#...#
#.#######M#M###M#####.#.#M###.#
#...#...#M#M#.#MMMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#....F#FF.#.#MMMMM#M#M#
#M#.###M#######F#.#.#M#####M#M#
#M#.#.#MMM#.FFFF#.#.#MMMMM#M#M#
#M#.#.###S#F#####F#.#####M#M#M#
#M#...#...#F#FFF#F#F....#MMM#M#
#M#.###.#F#F#F#F#F###F#######M#
#M#.#...#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#...FFFF#FFFFFFF#F....#MMM#
###M#########F#######F#####M###
#MMM..#.FFFF#FFFFF#FFF#...#MMM#
#M###.#F###F#####F###F#.#.###M#
#MMM#..F#FFFFFFF#FFF#F#.#.#.#M#
#.#M###############F###.#.#.#M#
#.#M#..FFF#FFFFFFF#F#FF.#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#FFF#FFFFF#FFF#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMTFF#FFFFFFF..#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMMT....#.#.#M#...#
#.#######M#M###M#####.#.#M###.#
#...#...#M#M#.#MMMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#....F#FF.#.#MMMMM#M#M#
#M#.###M#######F#.#.#M#####M#M#
#M#.#.#MMM#.FFFF#.#.#MMMMM#M#M#
#M#.#.###S#F#####F#.#####M#M#M#
#M#...#...#F#FFF#F#F....#MMM#M#
#M#.###.#F#F#F#F#F###F#######M#
#M#.#...#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#...FFFF#FFFFFFF#FF...#MMM#
###M#########F#######F#####M###
#MMM..#.FFFF#FFFFF#FFF#...#MMM#
#M###.#F###F#####F###F#.#.###M#
#MMM#...#FFFFFFF#FFF#F#.#.#.#M#
#.#M###############F###.#.#.#M#
#.#M#..FFF#FFFFFFF#F#FF.#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#FFF#FFFFF#FF.#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMTFF#FFFFFFFF.#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMTT.FF.#.#.#M#...#
#.#######M#M###M#####.#.#M###.#
#...#...#M#M#.#MMMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######.###########M#M#M#
#MMM#MMM#...FF#FF.#.#MMMMM#M#M#
#M#.###M#######F#.#.#M#####M#M#
#M#.#.#MMM#.FFFF#F#.#MMMMM#M#M#
#M#.#.###S#F#####F#.#####M#M#M#
#M#...#...#F#FFF#F#FF...#MMM#M#
#M#.###.#F#F#F#F#F###F#######M#
#M#.#...#F#FFF#F#FFF#F#.....#M#
#M###.#####F#######F#F#.###.#M#
#MMM#...FFFF#FFFFFFF#F....#MMM#
###M#########F#######F#####M###
#MMM..#.FFFF#FFFFF#FFF#...#MMM#
#M###.#F###F#####F###F#.#.###M#
#MMM#...#FFFFFFF#FFF#F#.#.#.#M#
#.#M###############F###.#.#.#M#
#.#M#..FFF#FFFFFFF#F#F..#.#MMM#
#.#M#####F#F#####F#F#F###.#M###
#.#M....#FFF#FFFFF#FFF#...#MMM#
###M#.#######F#########.#####M#
#MMM#.#MMTFF#FFFFFFF..#.#MMMMM#
#M#####M#M###########.#.#M#####
#MMMMMMM#M#MMMMT..F.#.#.#M#...#
#.#######M#M###M#####.#.#M###.#
#...#...#M#M#.#MMMMM#...#MMMMM#
###.#.###M#M#.#####M#########M#
#...#...#MMM#...#MMM........#M#
#.#####.#####.#.#M###########M#
#.............#.#MMMMMMMMMMMMM#
###############################
```

## Case 123 (final failure)

Loop gain: `0.0130`. First loop F1 `0.0295` with 103 false positives and 160 misses. loop12 F1 `0.0426` with 112 false positives and 158 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#MMMMMMMMM#.#M#M#M#
#M#.#.#M###M###########.#M#M#M#
#M#.#..MMMMM#.....#MMMMM#MMM#M#
#M#############.###M###M#####M#
#M#......F..F.#FFF#TTM#MMMMM#M#
#M#.#########F#F#F###M#####M#M#
#M..#...#.#FFF#F#FFF#TMMMM#MMM#
#M###.#.#F#.###F###F#####M#####
#MMMMM#..F#.FF#FFF#FFF#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#.FF#F#F#F#FFF#F#F#M#M#M#
#.#M###F#F###F###F###F#F#M#M#M#
#MMM#F#F#F#FFF#FFF#FFFFF#TMM#M#
#M###.#F#F#F###F#############M#
#M#MMM#.#.FF#FFFFFFF#FFFF..SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#...#FFFFF#F#F..#.....#
#####M#.#.#.#####F#F###.#####.#
#MMMMM#.#.#..FFF#F#F#F......#.#
#M#####.#.#####F#F#F#.#####.#.#
#M#...#...#.#..F#F#F..#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

### loop2

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#TTTTMMMMM#.#M#M#M#
#M#.#.#M###T###########.#M#M#M#
#M#.#..MMMTT#FFFF.#MMMMM#MMM#M#
#M#############F###M###M#####M#
#M#.......FFFF#FFF#MMM#MMMMM#M#
#M#.#########F#F#F###M#####M#M#
#M..#...#F#FFF#F#FFF#MMMMM#MMM#
#M###.#.#F#F###F###F#####M#####
#MMMMM#.FF#FFF#FFF#FF.#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#FFF#F#F#F#FFF#F#.#M#M#M#
#.#M###F#F###F###F###F#.#M#M#M#
#MMM#.#F#F#FFF#FFF#FFFF.#MMM#M#
#M###.#F#F#F###F#############M#
#M#MMM#F#FFF#FFFFFFF#FF....SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#FFF#FFFFF#F#F..#.....#
#####M#.#F#F#####F#F###.#####.#
#MMMMM#.#.#FFFFF#F#F#.......#.#
#M#####.#.#####.#F#F#.#####.#.#
#M#...#...#.#...#F#...#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

### loop4

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#TTTTMMMMM#.#M#M#M#
#M#.#.#M###M###########.#M#M#M#
#M#.#..MMMMM#FFFFF#MMMMM#MMM#M#
#M#############F###M###M#####M#
#M#.......FFFF#FFF#MMM#MMMMM#M#
#M#.#########F#F#F###T#####M#M#
#M..#...#F#FFF#F#FFF#TMMMM#MMM#
#M###.#.#F#F###F###F#####M#####
#MMMMM#..F#FFF#FFF#FFF#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#.FF#F#F#F#FFF#F#.#M#M#M#
#.#M###.#F###F###F###F#.#M#M#M#
#MMM#.#.#F#FFF#FFF#FFFF.#MMM#M#
#M###.#F#F#F###F#############M#
#M#MMM#F#FFF#FFFFFFF#FF....SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#FFF#FFFFF#F#F..#.....#
#####M#.#F#F#####F#F###.#####.#
#MMMMM#.#F#FFFFF#F#F#.......#.#
#M#####.#.#####F#F#.#.#####.#.#
#M#...#...#.#..F#.#...#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

### loop6

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#TTTTMMMMM#.#M#M#M#
#M#.#.#M###M###########.#M#M#M#
#M#.#..MMMMM#FFFFF#MMMMM#MMM#M#
#M#############F###M###M#####M#
#M#.......FFFF#FFF#MMM#MMMMM#M#
#M#.#########F#F#F###T#####M#M#
#M..#...#F#FFF#F#FFF#TMMMM#MMM#
#M###.#.#F#F###F###F#####M#####
#MMMMM#.FF#FFF#FFF#FFF#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#.FF#F#F#F#FFF#F#.#M#M#M#
#.#M###.#F###F###F###F#.#M#M#M#
#MMM#.#.#F#FFF#FFF#FFFF.#MMM#M#
#M###.#.#F#F###F#############M#
#M#MMM#F#FFF#FFFFFFF#FF....SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#FFF#FFFFF#F#...#.....#
#####M#.#F#F#####F#F###.#####.#
#MMMMM#.#F#FFFFF#F#F#.......#.#
#M#####.#.#####F#F#.#.#####.#.#
#M#...#...#.#..F#F#...#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

### loop10

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#TTTTMMMMM#.#M#M#M#
#M#.#.#M###M###########.#M#M#M#
#M#.#..MMMMM#FFFFF#MMMMM#MMM#M#
#M#############F###M###M#####M#
#M#.......FFFF#FFF#MMM#MMMMM#M#
#M#.#########F#F#F###T#####M#M#
#M..#...#F#FFF#F#FFF#TMMMM#MMM#
#M###.#.#F#F###F###F#####M#####
#MMMMM#.FF#FFF#FFF#FFF#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#.FF#F#F#F#FFF#F#.#M#M#M#
#.#M###.#F###F###F###F#.#M#M#M#
#MMM#.#.#F#FFF#FFF#FFFF.#MMM#M#
#M###.#F#F#F###F#############M#
#M#MMM#F#FFF#FFFFFFF#FF....SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#FFF#FFFFF#F#F..#.....#
#####M#.#F#F#####F#F###.#####.#
#MMMMM#.#F#FFFFF#F#F#.......#.#
#M#####.#.#####F#F#F#.#####.#.#
#M#...#...#.#..F#.#...#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

### loop12

```text
###############################
#MMMMMMM#..MMM#......MMMMM#MMM#
#M#####M#.#M#M#######M###M#M#M#
#M#...#M#.#M#TTTTMMMMM#.#M#M#M#
#M#.#.#M###M###########.#M#M#M#
#M#.#..MMMMM#FFFF.#MMMMM#MMM#M#
#M#############F###M###M#####M#
#M#.......FFFF#FFF#MMM#MMMMM#M#
#M#.#########F#F#F###T#####M#M#
#M..#...#F#FFF#F#FFF#TMMMM#MMM#
#M###.#.#F#F###F###F#####M#####
#MMMMM#.FF#FFF#FFF#FFF#.#M#MMM#
#####M#####F#F#F###F#F#.#M#M#M#
#.#MMM#.FF#F#F#F#FFF#F#.#M#M#M#
#.#M###.#F###F###F###F#.#M#M#M#
#MMM#.#.#F#FFF#FFF#FFFF.#MMM#M#
#M###.#.#F#F###F#############M#
#M#MMM#.#FFF#FFFFFFF#FF....SMM#
#M#M#M#.#####F#####F#F#########
#MMM#M#.#FFF#FFFFF#F#...#.....#
#####M#.#F#F#####F#F###.#####.#
#MMMMM#.#F#FFFFF#F#F#.......#.#
#M#####.#.#####F#F#F#.#####.#.#
#M#...#...#.#.FF#F#...#.#.#.#.#
#M###.#####G#.#########.#.#.#.#
#MMM#MMMMM#M#...#.......#.#.#.#
###M#M###M#M###.#.#######.#.#.#
#.#MMM#.#MMM..#.#.#...#...#.#.#
#.#####.#####.#.#.#.#.#.#.#.#.#
#.............#...#.#...#.....#
###############################
```

## Case 277 (final failure)

Loop gain: `-0.0365`. First loop F1 `0.0897` with 93 false positives and 171 misses. loop12 F1 `0.0532` with 109 false positives and 176 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MMM#.#...#.#M#.#.#.#
#.#M###M###M###.#.#.###M#.#.#.#
#MMM..#MMMMM....#F#.#MMM#...#.#
#M###############F###S#######.#
#M#.........F.#FFF#FF...#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FF.#FFF#FFFFF..#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FFF..#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#.....FF.FFF#FFF#FFF#.#.#M#M#
#M#############F#####F#F###M#M#
#MMMMT#..F#FFF#F#FFF#F#F#TMM#M#
#####M#.###F#F#F#F#F###F#M###M#
#MMM#M#.#FFF#FFF#F#FFFFF#MMM#M#
#M#M#M#.#.#######F#########M#M#
#M#MMM#.#.F.#FFFFF#FFTTMMMMM#M#
#M#####.###F#F#######T#######M#
#M#..MMM#.#.#F#FFTTTTM#.....#M#
#M###M#M#.#.#.###T#####.#####M#
#MMMMM#M#.#...#TTT#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop2

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MTT#F#...#.#M#.#.#.#
#.#M###M###M###F#.#.###M#.#.#.#
#MMM..#MMMMTFFFF#.#.#MMM#...#.#
#M###############.###S#######.#
#M#.......FFFF#FFF#.....#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FFF#FFF#FFFF...#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FFF..#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#....FFFFFFF#FFF#FFF#.#.#M#M#
#M#############F#####F#.###M#M#
#MMMMM#FFF#FFF#F#FFF#F#.#MMM#M#
#####M#F###F#F#F#F#F###.#M###M#
#MMM#M#F#FFF#FFF#F#FFFF.#MMM#M#
#M#M#M#F#F#######F#########M#M#
#M#MMM#.#FFF#FFFFF#FFTMMMMMM#M#
#M#####.###F#.#######T#######M#
#M#..MMM#.#F#F#FFTTTTT#.....#M#
#M###M#M#.#.#.###T#####.#####M#
#MMMMM#M#.#...#MTT#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop4

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MTT#F#...#.#M#.#.#.#
#.#M###M###M###F#.#.###M#.#.#.#
#MMM..#MMMMMFFFF#F#.#MMM#...#.#
#M###############F###S#######.#
#M#.......FFFF#FFF#FF...#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FFF#FFF#FFFFF..#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FFFF.#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#....FFFFFFF#FFF#FFF#.#.#M#M#
#M#############F#####F#.###M#M#
#MMMMM#.FF#FFF#F#FFF#F#.#MMM#M#
#####M#.###F#F#F#F#F###.#M###M#
#MMM#M#F#FFF#FFF#F#FFFF.#MMM#M#
#M#M#M#.#F#######F#########M#M#
#M#MMM#.#FFF#FFFFF#FFTMMMMMM#M#
#M#####.###F#F#######M#######M#
#M#..MMM#F#F#F#FFTTTTM#.....#M#
#M###M#M#.#F#.###T#####.#####M#
#MMMMM#M#.#...#TMM#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop6

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MTT#F#...#.#M#.#.#.#
#.#M###M###M###F#.#.###M#.#.#.#
#MMM..#MMMMMFFFF#F#.#MMM#...#.#
#M###############.###S#######.#
#M#.......FFFF#FFF#F....#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FFF#FFF#FFFFF..#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FFF..#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#.....FFFFFF#FFF#FFF#.#.#M#M#
#M#############F#####F#.###M#M#
#MMMMM#FFF#FFF#F#FFF#F#.#MMM#M#
#####M#.###F#F#F#F#F###.#M###M#
#MMM#M#F#FFF#FFF#F#FFF..#MMM#M#
#M#M#M#.#F#######F#########M#M#
#M#MMM#.#FFF#FFFFF#FFTMMMMMM#M#
#M#####.###F#F#######M#######M#
#M#..MMM#F#F#F#FFTTTMM#.....#M#
#M###M#M#.#F#.###T#####.#####M#
#MMMMM#M#.#...#MMM#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop10

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MTT#F#...#.#M#.#.#.#
#.#M###M###M###F#.#.###M#.#.#.#
#MMM..#MMMMTFFFF#.#.#MMM#...#.#
#M###############F###S#######.#
#M#.......FFFF#FFF#.F...#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FFF#FFF#FFFFF..#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FF...#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#.....FFFFFF#FFF#FFF#.#.#M#M#
#M#############F#####F#.###M#M#
#MMMMM#.FF#FFF#F#FFF#F#.#MMM#M#
#####M#F###F#F#F#F#F###.#M###M#
#MMM#M#F#FFF#FFF#F#FFF..#MMM#M#
#M#M#M#.#F#######F#########M#M#
#M#MMM#.#FFF#FFFFF#FFTMMMMMM#M#
#M#####.###F#F#######M#######M#
#M#..MMM#F#F#F#FFTTTTM#.....#M#
#M###M#M#.#.#F###T#####.#####M#
#MMMMM#M#.#...#TMM#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

### loop12

```text
###############################
#.#MMM......#MMMMMMMMMMM#.....#
#.#M#M###.###M#######.#M#.###.#
#.#M#MMM#.#MTT#F#...#.#M#.#.#.#
#.#M###M###M###F#.#.###M#.#.#.#
#MMM..#MMMMMFFFF#F#.#MMM#...#.#
#M###############F###S#######.#
#M#.......FFFF#FFF#FF...#MMMMM#
#M#.#########F#F#######.#M###M#
#M#.....#FFF#FFF#FFFFF..#MMM#M#
#M#####.#F#F#######F#####.#M#M#
#M#.....#F#FFFFFFF#FFF..#.#M#M#
#M#.#########F###F#F###.#.#M#M#
#M#....FFFFFFF#FFF#FFF#.#.#M#M#
#M#############F#####F#.###M#M#
#MMMMM#.FF#FFF#F#FFF#F#.#MMM#M#
#####M#F###F#F#F#F#F###.#M###M#
#MMM#M#F#FFF#FFF#F#FFF..#MMM#M#
#M#M#M#.#F#######F#########M#M#
#M#MMM#.#FFF#FFFFF#FFTMMMMMM#M#
#M#####.###F#F#######M#######M#
#M#..MMM#F#F#F#FFTTTMM#.....#M#
#M###M#M#.#F#.###T#####.#####M#
#MMMMM#M#.#...#TMM#GMMMMMMMM#M#
#######M#.#####M###.#######M#M#
#.....#MMM#....M#.#.#.....#M#M#
#.#######M#####M#.#.#####.#M#M#
#...#MMMMM#MMM#M#.#.......#MMM#
#.#.#M#####M#M#M#.#######.#####
#.#..MMMMMMM#MMM#.............#
###############################
```

## Case 52 (final failure)

Loop gain: `-0.0389`. First loop F1 `0.0935` with 90 false positives and 162 misses. loop12 F1 `0.0546` with 110 false positives and 167 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#...#...#G#.#MMM#M#.#.#
#.#.#.#.###.#.###.#.#M#M#M#.#.#
#...#.#...#.#...#...#M#MMM#.#.#
#####.###.#.###.#.###M#####.#.#
#S....#.#.#...#F#F#TMM#.......#
#M###.#.#.#F#F#F#F#T###########
#M#.....#.#.#FFF#F#TTT#MMMMMMM#
#M#######.#.#######F#T#M#####M#
#MMM#MMM#F#F#FFFFF#F#TTM....#M#
#.#M#M#T#F#.#F###F###########M#
#.#MMM#M#F#FFF#F#FFFFFFF..#MMM#
#######M#.#####F#########.#M###
#MMMMT#T#.#F#FFFFF#FFFFFF.#MMM#
#M###M#T#F#F###F###F#######.#M#
#MMM#M#M#.FF#FFF#FFF#FFF....#M#
###M#M#M#####F###F#####F#####M#
#MMM#MMM...FFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#F###M###.#
#M#.#MMM#.....#F#FFF#F..#M#...#
#M###M#M#.#####F###F###.#M#####
#MMM#M#M#.#MMM#FFFFF#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#TMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

### loop2

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#..F#FFF#G#.#MMM#M#.#.#
#.#.#.#.###.#F###.#.#M#M#M#.#.#
#...#.#...#F#FFF#...#M#MMM#.#.#
#####.###.#F###F#.###M#####.#.#
#S....#.#F#FFF#F#.#MMM#.......#
#M###.#.#F#F#F#F#F#T###########
#M#.....#F#F#FFF#F#TTM#MMMMMMM#
#M#######F#F#######F#M#M#####M#
#MMM#MMM#F#F#FFFFF#F#MMM....#M#
#.#M#M#T#F#F#F###F###########M#
#.#MMM#T#F#FFF#F#FFFFFF...#MMM#
#######T#F#####F#########.#M###
#MMMMM#T#F#F#FFFFF#FFFF...#MMM#
#M###M#T#F#F###F###F#######.#M#
#MMM#M#T#FFF#FFF#FFF#FF.....#M#
###M#M#T#####F###F#####.#####M#
#MMM#MMMFFFFFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#.###M###.#
#M#.#MMM#.FFF.#F#FFF#...#M#...#
#M###M#M#.#####.###F###.#M#####
#MMM#M#M#.#MMM#FFFF.#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#MMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

### loop4

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#...#FFF#G#.#MMM#M#.#.#
#.#.#.#.###.#F###.#.#M#M#M#.#.#
#...#.#...#.#FFF#...#M#MMM#.#.#
#####.###.#F###F#F###M#####.#.#
#S....#.#.#FFF#F#F#MTM#.......#
#M###.#.#F#F#F#F#F#T###########
#M#.....#F#F#FFF#F#TTT#MMMMMMM#
#M#######F#F#######F#M#M#####M#
#MMM#MMM#F#F#FFFFF#F#MMM....#M#
#.#M#M#M#F#F#F###F###########M#
#.#MMM#M#F#FFF#F#FFFFFF...#MMM#
#######M#F#####F#########.#M###
#MMMMM#M#F#F#FFFFF#FFFF...#MMM#
#M###M#M#F#F###F###F#######.#M#
#MMM#M#M#FFF#FFF#FFF#FF.....#M#
###M#M#M#####F###F#####.#####M#
#MMM#MMMFFFFFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#.###M###.#
#M#.#MMM#FFFFF#F#FFF#...#M#...#
#M###M#M#.#####F###F###.#M#####
#MMM#M#M#.#MMM#F....#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#MMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

### loop6

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#...#FFF#G#.#MMM#M#.#.#
#.#.#.#.###.#F###.#.#M#M#M#.#.#
#...#.#...#.#FFF#...#M#MMM#.#.#
#####.###.#F###F#F###M#####.#.#
#S....#.#.#FFF#F#F#MMM#.......#
#M###.#.#F#F#F#F#F#T###########
#M#.....#F#F#FFF#F#TTT#MMMMMMM#
#M#######F#F#######F#T#M#####M#
#MMM#MMM#F#F#FFFFF#F#TTM....#M#
#.#M#M#M#F#F#F###F###########M#
#.#MMM#M#F#FFF#F#FFFFFF...#MMM#
#######T#F#####F#########.#M###
#MMMMM#M#F#F#FFFFF#FFFF...#MMM#
#M###M#T#F#F###F###F#######.#M#
#MMM#M#M#FFF#FFF#FFF#FF.....#M#
###M#M#M#####F###F#####.#####M#
#MMM#MMMFFFFFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#.###M###.#
#M#.#MMM#F.FFF#F#FFF#...#M#...#
#M###M#M#.#####F###.###.#M#####
#MMM#M#M#.#MMM#...F.#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#MMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

### loop10

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#...#FFF#G#.#MMM#M#.#.#
#.#.#.#.###.#F###.#.#M#M#M#.#.#
#...#.#...#.#FFF#...#M#MMM#.#.#
#####.###.#F###F#F###M#####.#.#
#S....#.#.#FFF#F#F#MMM#.......#
#M###.#.#F#F#F#F#F#T###########
#M#.....#F#F#FFF#F#TTT#MMMMMMM#
#M#######F#F#######F#T#M#####M#
#MMM#MMM#F#F#FFFFF#F#TMM....#M#
#.#M#M#M#F#F#F###F###########M#
#.#MMM#T#F#FFF#F#FFFFFF...#MMM#
#######T#F#####F#########.#M###
#MMMMM#M#F#F#FFFFF#FFFF...#MMM#
#M###M#T#F#F###F###F#######.#M#
#MMM#M#T#FFF#FFF#FFF#FF.....#M#
###M#M#M#####F###F#####.#####M#
#MMM#MMMFFFFFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#.###M###.#
#M#.#MMM#FFFFF#F#FFF#...#M#...#
#M###M#M#.#####F###F###.#M#####
#MMM#M#M#.#MMM#F....#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#MMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

### loop12

```text
###############################
#.#...#.........#MMMMMMMMM....#
#.#.#.#.#######.#M#######M###.#
#.#.#.#.#...#FFF#G#.#MMM#M#.#.#
#.#.#.#.###.#F###F#.#M#M#M#.#.#
#...#.#...#.#FFF#...#M#MMM#.#.#
#####.###.#F###F#F###M#####.#.#
#S....#.#.#FFF#F#F#MMM#.......#
#M###.#.#F#F#F#F#F#T###########
#M#.....#F#F#FFF#F#TTT#MMMMMMM#
#M#######F#F#######F#T#M#####M#
#MMM#MMM#F#F#FFFFF#F#TMM....#M#
#.#M#M#M#F#F#F###F###########M#
#.#MMM#M#F#FFF#F#FFFFFF...#MMM#
#######T#F#####F#########.#M###
#MMMMM#M#F#F#FFFFF#FFFF...#MMM#
#M###M#M#F#F###F###F#######.#M#
#MMM#M#T#FFF#FFF#FFF#FF.....#M#
###M#M#M#####F###F#####.#####M#
#MMM#MMMFFFFFF#FFF#FFF#.#MMMMM#
#M#.#########F#F###F#.###M###.#
#M#.#MMM#FFFFF#F#FFF#...#M#...#
#M###M#M#.#####F###F###.#M#####
#MMM#M#M#.#MMM#F....#.#.#M#MMM#
#.#M#M#M###M#M#######.#.#M#M#M#
#.#MMM#MMMMM#M#MMMMM#...#MMM#M#
#.###########M#M###M#.###.###M#
#.#.....#...#M#M#.#M#...#.#MMM#
#.###.#.###.#M#M#.#M###.###M#.#
#.....#.....#MMM..#MMMMMMMMM#.#
###############################
```

## Case 454 (final failure)

Loop gain: `-0.0520`. First loop F1 `0.1083` with 94 false positives and 153 misses. loop12 F1 `0.0563` with 108 false positives and 160 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#...#M#.#M#.....#.#
#.#.###.#.#.###.#M#.#M#######.#
#.#...#...#.#...#M#MMM#MMM#MMM#
#.###.#####.#.###T#M###M#M#M#M#
#...#...#...#..F#T#TTMMM#MMM#M#
#.#.###.#.#####F#T###########M#
#.#.#.#.#F#.#FFF#TTTTTMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFFFF.#MMM#M#
###.#.###.#F###########.###M#M#
#...#.#.FF#FFFFF#FFFFFFF#MMM#M#
#.#####.#######F#######F#M###M#
#.....#.#FFFFF#FFFFFFF#F#M#.#M#
#####.#.#####F#######F###T#.#M#
#...#...#FFF#FFFFF#FFF#TMT#MMM#
#.#######.#F###F#F#F###T###M###
#.........#FFFFF#F#FFF#TMM#MMG#
#########.#####F#####F#.#M###.#
#MMMMMMM#...#FFF#FFFF.#.#M#...#
#M#####M###.#.###F#####.#M#####
#MMM#MMM#...#.FF#FFF#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#F..#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#FFF#M#.#M#.....#.#
#.#.###.#.#F###F#M#.#M#######.#
#.#...#...#F#FFF#M#MMM#MMM#MMM#
#.###.#####F#F###M#M###M#M#M#M#
#...#...#FFF#FFF#M#MMMMM#MMM#M#
#.#.###.#F#####F#T###########M#
#.#.#.#.#F#F#FFF#TTTTMMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFF...#MMM#M#
###.#.###F#F###########.###M#M#
#...#.#FFF#FFFFF#FFFFFF.#MMM#M#
#.#####F#######F#######.#M###M#
#.....#F#FFFFF#FFFFFFF#.#M#.#M#
#####F#F#####F#######F###M#.#M#
#...#.FF#FFF#FFFFF#FFF#MMM#MMM#
#.#######F#F###F#F#F###M###M###
#.......FF#FFFFF#F#FFF#MMM#MMG#
#########F#####F#####.#.#M###.#
#MMMMMMM#.FF#FFF#FFF..#.#M#...#
#M#####M###.#.###F#####.#M#####
#MMM#MMM#...#...#F.F#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#...#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#FFF#M#.#M#.....#.#
#.#.###.#.#.###F#M#.#M#######.#
#.#...#...#.#FFF#M#MMM#MMM#MMM#
#.###.#####F#F###M#M###M#M#M#M#
#...#...#.FF#FFF#T#TMMMM#MMM#M#
#.#.###.#F#####F#T###########M#
#.#.#.#.#F#F#FFF#TTTTTMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFFF..#MMM#M#
###.#.###F#F###########.###M#M#
#...#.#.FF#FFFFF#FFFFFF.#MMM#M#
#.#####F#######F#######.#M###M#
#.....#.#FFFFF#FFFFFFF#.#M#.#M#
#####.#F#####F#######F###M#.#M#
#...#..F#FFF#FFFFF#FFF#MMM#MMM#
#.#######F#F###F#F#F###M###M###
#.......FF#FFFFF#F#FFF#MMM#MMG#
#########F#####F#####.#.#M###.#
#MMMMMMM#FFF#FFF#FFFF.#.#M#...#
#M#####M###.#.###F#####.#M#####
#MMM#MMM#...#..F#F..#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#...#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#FFF#M#.#M#.....#.#
#.#.###.#.#.###F#M#.#M#######.#
#.#...#...#.#FFF#T#MMM#MMM#MMM#
#.###.#####F#F###M#M###M#M#M#M#
#...#...#.FF#FFF#T#MMMMM#MMM#M#
#.#.###.#F#####F#T###########M#
#.#.#.#.#F#F#FFF#TTTTTMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFFF..#MMM#M#
###.#.###F#F###########.###M#M#
#...#.#FFF#FFFFF#FFFFFF.#MMM#M#
#.#####F#######F#######.#M###M#
#.....#.#FFFFF#FFFFFFF#.#M#.#M#
#####.#.#####F#######F###M#.#M#
#...#...#FFF#FFFFF#FFF#MMM#MMM#
#.#######F#F###F#F#F###M###M###
#.......FF#FFFFF#F#FFF#MMM#MMG#
#########F#####F#####.#.#M###.#
#MMMMMMM#FFF#FFF#FFF..#.#M#...#
#M#####M###F#F###F#####.#M#####
#MMM#MMM#...#..F#...#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#...#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#FFF#M#.#M#.....#.#
#.#.###.#.#.###F#M#.#M#######.#
#.#...#...#.#FFF#M#MMM#MMM#MMM#
#.###.#####.#F###T#M###M#M#M#M#
#...#...#.FF#FFF#T#TMMMM#MMM#M#
#.#.###.#F#####F#T###########M#
#.#.#.#.#F#F#FFF#TTTTTMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFFF..#MMM#M#
###.#.###F#F###########.###M#M#
#...#.#.FF#FFFFF#FFFFFF.#MMM#M#
#.#####.#######F#######.#M###M#
#.....#.#FFFFF#FFFFFFF#.#M#.#M#
#####.#F#####F#######F###M#.#M#
#...#...#FFF#FFFFF#FFF#MMM#MMM#
#.#######F#F###F#F#F###M###M###
#.......FF#FFFFF#F#FFF#MMM#MMG#
#########F#####F#####.#.#M###.#
#MMMMMMM#FFF#FFF#FFFF.#.#M#...#
#M#####M###.#F###F#####.#M#####
#MMM#MMM#...#...#FF.#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#...#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.............#..MMMMM#.#.....#
#.###.#######.###M###M#.#.###.#
#.#...#.#...#FFF#M#.#M#.....#.#
#.#.###.#.#.###F#M#.#M#######.#
#.#...#...#.#FFF#M#MMM#MMM#MMM#
#.###.#####F#F###T#M###M#M#M#M#
#...#...#.FF#FFF#T#MMMMM#MMM#M#
#.#.###.#F#####F#T###########M#
#.#.#.#.#F#F#FFF#TTTTTMMMM#..M#
#.#.#.#.#F#F#F###F#######M###M#
#.#.#...#F#FFF#FFFFFFF..#MMM#M#
###.#.###F#F###########.###M#M#
#...#.#FFF#FFFFF#FFFFF..#MMM#M#
#.#####.#######F#######.#M###M#
#.....#.#FFFFF#FFFFFFF#.#M#.#M#
#####.#.#####F#######F###M#.#M#
#...#..F#FFF#FFFFF#FFF#MMM#MMM#
#.#######F#F###F#F#F###M###M###
#.......FF#FFFFF#F#FFF#MMM#MMG#
#########F#####F#####.#.#M###.#
#MMMMMMM#FFF#FFF#FFFF.#.#M#...#
#M#####M###.#F###F#####.#M#####
#MMM#MMM#...#...#...#...#M#MMM#
###M#S###.#####.###.#####M#M#M#
#MMM#.#...#MMM#...#...#MMM#M#M#
#M###.#.###M#M#######.#M###M#M#
#M#.....#MMM#M#MMMMM#.#MMMMM#M#
#M#######M###M#M###M#.#######M#
#MMMMMMMMM..#MMM..#MMMMMMMMMMM#
###############################
```

## Case 284 (final failure)

Loop gain: `-0.0578`. First loop F1 `0.1206` with 95 false positives and 153 misses. loop12 F1 `0.0627` with 108 false positives and 161 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMM....#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#..GMMMTMM#.#...#.#.#
#.###M#.#.#.#########.###.#.#.#
#.#MMM..#.#F#..FFFFF#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#..F#F#F#F#FFFFFF...#...#
#.#M###.#F#.#F#F#F#########.#.#
#MMM#...#F.F#F#FFF#FFF......#.#
#M###.#######F#####F#########.#
#MMM#.#.FF#FFFFFFFFF#TTTMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#.FF#F#F#FFFFFFF#T#FFTMM#M#
#.#M###.#F#F#####F#F#T#######M#
#MMM#...#F#FFFFF#F#F#TTT#TMMMM#
#M#######F#####F#F#F###T#M#####
#M#.......#FFF#F#F#FFF#T#MMM#.#
#M###.###.#.###F###F#F#M###M#.#
#MMM#...#...#.FFFFFF#F#M#..M#.#
###M#.#.#.###.#######.#M###S#.#
#MMM#.#.#.#....F#TTM#.#MMM#...#
#M#####.###.#####T#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

### loop2

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMM.FFF#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#FFGTTMMMM#.#...#.#.#
#.###M#.#.#F#########.###.#.#.#
#.#MMM..#F#F#FFFFF..#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#.FF#F#F#F#FFFF.....#...#
#.#M###.#F#F#F#F#F#########.#.#
#MMM#...#FFF#F#FFF#FFF......#.#
#M###.#######F#####F#########.#
#MMM#.#FFF#FFFFFFFFF#TTMMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#..F#F#F#FFFFFFF#T#..MMM#M#
#.#M###F#F#F#####F#F#T#######M#
#MMM#.FF#F#FFFFF#F#F#TTT#MMMMM#
#M#######F#####F#F#F###M#M#####
#M#.....FF#FFF#F#F#FFF#M#MMM#.#
#M###.###F#F###F###F#F#M###M#.#
#MMM#...#.FF#FFFFFFF#.#M#..M#.#
###M#.#.#.###.#######.#M###S#.#
#MMM#.#.#.#.....#TTM#.#MMM#...#
#M#####.###.#####M#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

### loop4

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMM.FFF#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#FFGTTTMMM#.#...#.#.#
#.###M#.#.#.#########.###.#.#.#
#.#MMM..#.#F#FFFFFF.#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#..F#F#F#F#FFFF.....#...#
#.#M###.#F#F#F#F#F#########.#.#
#MMM#...#FFF#F#FFF#FFF......#.#
#M###.#######F#####F#########.#
#MMM#.#.FF#FFFFFFFFF#TMMMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#...#F#F#FFFFFFF#T#..MMM#M#
#.#M###.#F#F#####F#F#T#######M#
#MMM#..F#F#FFFFF#F#F#TMM#MMMMM#
#M#######F#####F#F#F###M#M#####
#M#.....FF#FFF#F#F#FFF#M#MMM#.#
#M###.###F#F###F###F#.#M###M#.#
#MMM#...#FFF#FFFFFFF#.#M#..M#.#
###M#.#.#.###F#######.#M###S#.#
#MMM#.#.#.#....F#MMM#.#MMM#...#
#M#####.###.#####M#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

### loop6

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMMFFFF#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#.FGTTTMMM#.#...#.#.#
#.###M#.#.#.#########.###.#.#.#
#.#MMM..#.#F#FFFFFF.#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#..F#F#F#F#FFFFF....#...#
#.#M###.#F#F#F#F#F#########.#.#
#MMM#...#FFF#F#FFF#FFF......#.#
#M###.#######F#####F#########.#
#MMM#.#.FF#FFFFFFFFF#TTMMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#...#F#F#FFFFFFF#T#..MMM#M#
#.#M###F#F#F#####F#F#T#######M#
#MMM#..F#F#FFFFF#F#F#TTM#MMMMM#
#M#######F#####F#F#F###M#M#####
#M#.....FF#FFF#F#F#FFF#M#MMM#.#
#M###.###F#F###F###F#.#M###M#.#
#MMM#...#FFF#FFFFFFF#.#M#..M#.#
###M#.#.#.###F#######.#M###S#.#
#MMM#.#.#.#....F#MTM#.#MMM#...#
#M#####.###.#####M#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

### loop10

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMMFFFF#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#..GTTTMMM#.#...#.#.#
#.###M#.#.#.#########.###.#.#.#
#.#MMM..#.#F#FFFFFF.#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#.FF#F#F#F#FFFFF....#...#
#.#M###.#F#F#F#F#F#########.#.#
#MMM#...#FFF#F#FFF#FFFF.....#.#
#M###.#######F#####F#########.#
#MMM#.#.FF#FFFFFFFFF#TTMMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#..F#F#F#FFFFFFF#T#..MMM#M#
#.#M###F#F#F#####F#F#T#######M#
#MMM#..F#F#FFFFF#F#F#TTM#MMMMM#
#M#######F#####F#F#F###M#M#####
#M#.....FF#FFF#F#F#FFF#M#MMM#.#
#M###.###F#F###F###F#.#M###M#.#
#MMM#...#FFF#FFFFFFF#.#M#..M#.#
###M#.#.#.###.#######.#M###S#.#
#MMM#.#.#.#...FF#MMM#.#MMM#...#
#M#####.###.#####M#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

### loop12

```text
###############################
#...#MMM..#MMMMMMM#.......#...#
#.###M#M###M#####M###.#.#.#.#.#
#.#MMM#MMMMM.FFF#MMM#.#.#...#.#
#.#M###############M#.#.#####.#
#.#MMM#...#.FGTTTMMM#.#...#.#.#
#.###M#.#.#.#########.###.#.#.#
#.#MMM..#.#F#FFFFFF.#...#.#...#
#.#M#######F#F#####F#####.#####
#.#M..#.FF#F#F#F#FFFFF....#...#
#.#M###.#F#F#F#F#F#########.#.#
#MMM#...#FFF#F#FFF#FFFF.....#.#
#M###.#######F#####F#########.#
#MMM#.#.FF#FFFFFFFFF#TMMMM#MMM#
###M#.###F#F#########T###M#M#M#
#.#M#...#F#F#FFFFFFF#T#..MMM#M#
#.#M###.#F#F#####F#F#T#######M#
#MMM#..F#F#FFFFF#F#F#TTM#MMMMM#
#M#######F#####F#F#F###M#M#####
#M#.....FF#FFF#F#F#FF.#M#MMM#.#
#M###.###F#F###F###F#.#M###M#.#
#MMM#...#FFF#FFFFFFF#.#M#..M#.#
###M#.#.#.###F#######.#M###S#.#
#MMM#.#.#.#.F.FF#MMM#.#MMM#...#
#M#####.###.#####M#M#.###M#####
#M#MMM#.....#MMMMM#M#...#MMMMM#
#M#M#M#######M#####M#####.###M#
#M#M#MMMMMMM#M#...#MMMMM#...#M#
#M#M#######M#M#.#.#####M#####M#
#MMM#......MMM#.#......MMMMMMM#
###############################
```

## Case 470 (final failure)

Loop gain: `-0.0464`. First loop F1 `0.1107` with 94 false positives and 147 misses. loop12 F1 `0.0643` with 109 false positives and 153 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#.#.....#.#...#...#.#
#.#####.#.#.#.#####.#.#####.#.#
#.....#.#.#.#.#.F.#.#.....#.#.#
#.#####.#.#.#.#.#.###.#####.#.#
#.#.....#.#.#.FF#FFFF.#...#.#.#
###.#####.#############.#.#.#.#
#...#.....#FFFFF#FFF#F..#...#.#
#.#######.#F###F#F#F#F#########
#.......#FFF#F#FFF#F#F..SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#.FFFF#FFFFF#FFF#F..#..M#
#M#M#.#####F#####F###F#F###.#M#
#M#MMMTTTT#FFFFF#FFF#F#F#F..#M#
#M#######T#####F###F#F###F###M#
#M#..MMMMTFFFF#F#FFF#FFFF.#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#...FFFF#F#FFF#MMM#MMM#
#.#M#####.#######F#####M#M#M###
#.#MMMMM#...#FFF#TTTTM#M#MMM#.#
#######M###.#F#F#T###M#M#####.#
#MMMMM#M#.#.#.#F#TTT#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MTM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop2

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#F#FFFF.#.#...#...#.#
#.#####.#.#.#F#####.#.#####.#.#
#.....#.#.#F#F#FF.#.#.....#.#.#
#.#####.#.#F#F#F#.###.#####.#.#
#.#.....#F#F#FFF#FF...#...#.#.#
###.#####F#############.#.#.#.#
#...#...FF#FFFFF#FFF#F..#...#.#
#.#######F#F###F#F#F#.#########
#.......#FFF#F#FFF#F#F..SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#FFFFF#FFFFF#FFF#...#..M#
#M#M#.#####F#####F###F#.###.#M#
#M#MMMMTTT#FFFFF#FFF#F#F#...#M#
#M#######T#####F###F#F###.###M#
#M#..MTTTTFFFF#F#FFF#FFF..#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#FFFFFFF#F#FFF#MMM#MMM#
#.#M#####F#######F#####M#M#M###
#.#MMMMM#.FF#FFF#TTTTM#M#MMM#.#
#######M###.#.#.#T###M#M#####.#
#MMMMM#M#.#.#.#.#TTM#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MMM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop4

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#.#FFFF.#.#...#...#.#
#.#####.#.#.#F#####.#.#####.#.#
#.....#.#.#.#F#FFF#.#.....#.#.#
#.#####.#.#F#F#F#F###.#####.#.#
#.#.....#.#F#FFF#F....#...#.#.#
###.#####.#############.#.#.#.#
#...#....F#FFFFF#FFF#F..#...#.#
#.#######F#F###F#F#F#F#########
#.......#FFF#F#FFF#F#FF.SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#.FFFF#FFFFF#FFF#...#..M#
#M#M#.#####F#####F###F#.###.#M#
#M#MMMMMTT#FFFFF#FFF#F#.#...#M#
#M#######T#####F###F#F###.###M#
#M#..MMTTTFFFF#F#FFF#FF...#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#FFFFFFF#F#FFF#MMM#MMM#
#.#M#####F#######F#####M#M#M###
#.#MMMMM#FFF#FFF#TTTMM#M#MMM#.#
#######M###.#F#F#T###M#M#####.#
#MMMMM#M#.#.#.#F#MTM#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MMM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop6

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#.#FFFF.#.#...#...#.#
#.#####.#.#.#F#####.#.#####.#.#
#.....#.#.#F#F#FFF#.#.....#.#.#
#.#####.#.#F#F#F#F###.#####.#.#
#.#.....#.#F#FFF#F.FF.#...#.#.#
###.#####.#############.#.#.#.#
#...#....F#FFFFF#FFF#F..#...#.#
#.#######F#F###F#F#F#F#########
#.......#FFF#F#FFF#F#F..SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#.FFFF#FFFFF#FFF#...#..M#
#M#M#.#####F#####F###F#.###.#M#
#M#MMMMMTT#FFFFF#FFF#F#.#...#M#
#M#######T#####F###F#F###.###M#
#M#..MMMTTFFFF#F#FFF#FF...#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#FFFFFFF#F#FFF#MMM#MMM#
#.#M#####F#######F#####M#M#M###
#.#MMMMM#FFF#FFF#TTTMM#M#MMM#.#
#######M###F#F#F#T###M#M#####.#
#MMMMM#M#.#.#.#.#TMM#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MMM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop10

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#.#FFFF.#.#...#...#.#
#.#####.#.#.#F#####.#.#####.#.#
#.....#.#.#F#F#FF.#.#.....#.#.#
#.#####.#.#F#F#F#F###.#####.#.#
#.#.....#.#F#FFF#F.F..#...#.#.#
###.#####.#############.#.#.#.#
#...#....F#FFFFF#FFF#F..#...#.#
#.#######F#F###F#F#F#F#########
#.......#FFF#F#FFF#F#FF.SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#.FFFF#FFFFF#FFF#...#..M#
#M#M#.#####F#####F###F#.###.#M#
#M#MMMMMTT#FFFFF#FFF#F#.#...#M#
#M#######T#####F###F#F###.###M#
#M#..MMTTTFFFF#F#FFF#FF...#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#FFFFFFF#F#FFF#MMM#MMM#
#.#M#####F#######F#####M#M#M###
#.#MMMMM#FFF#FFF#TTTMM#M#MMM#.#
#######M###.#F#F#T###M#M#####.#
#MMMMM#M#.#.#.#F#TMM#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MMM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

### loop12

```text
###############################
#...#.................#.......#
#.#.#.#####.#####.#.###.#.###.#
#.#...#...#.#FFFF.#.#...#...#.#
#.#####.#.#.#F#####.#.#####.#.#
#.....#.#.#.#F#FF.#.#.....#.#.#
#.#####.#.#F#F#F#F###.#####.#.#
#.#.....#.#F#FFF#FF.F.#...#.#.#
###.#####F#############.#.#.#.#
#...#....F#FFFFF#FFF#F..#...#.#
#.#######F#F###F#F#F#F#########
#.......#FFF#F#FFF#F#F..SMMMMM#
#######.#####F#####F###.#####M#
#MMM#.#.FFFF#FFFFF#FFF#...#..M#
#M#M#.#####F#####F###F#.###.#M#
#M#MMMMMTT#FFFFF#FFF#F#.#...#M#
#M#######T#####F###F#F###.###M#
#M#..MMMTTFFFF#F#FFF#FF...#.#M#
#M###M#########F#F#F#######.#M#
#MMM#MMG#FFFFFFF#F#FFF#MMM#MMM#
#.#M#####F#######F#####M#M#M###
#.#MMMMM#FFF#FFF#TTTMM#M#MMM#.#
#######M###.#F#F#T###M#M#####.#
#MMMMM#M#.#.#.#F#MMM#MMM#MMMMM#
#M###M#M#.#.#.#####M#####M###M#
#MMM#MMM#...#MMM..#M#..MMM#.#M#
###M#.#######M#M#.#M###M###.#M#
#MMM#.#MMMMM#M#M#.#MMMMM#MMM#M#
#M#####M###M#M#M#########M#M#M#
#MMMMMMM..#MMM#MMMMMMMMMMM#MMM#
###############################
```

## Case 137 (final failure)

Loop gain: `-0.0097`. First loop F1 `0.0942` with 96 false positives and 154 misses. loop12 F1 `0.0845` with 105 false positives and 155 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#...#..F#M#MMMMM#MMM#
#.#.###.#.#.###.###M#######M#.#
#.#...#.#.#...#...#MMMMM#.#M#.#
#.###.#.#.###.#.#.#####M#.#M###
#.....#.#.#F..#F#F#TTM#M#.#MMM#
#.#####.###.###F#F#T#M#M#.###M#
#...#.#.FFF.#FFF#F#T#T#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#TTM#.#MMM#
#G###.#.#.###F###F#F#####.###M#
#M#...#.FF#F#FFFFF#FFF#FF...#M#
#M#.#######F#####F###F#F###.#M#
#M#.FFFF#.FFFF#FFF#FFF#F#...#M#
#M#.###F#####F#F###F###F#####M#
#M#...#...FF#FFFFF#FFFFF#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#...#.FFFFFFFF#F#MMM#...#
#####M#########F#F###F#M###.###
#MMMMM#MMM#MMT#F#FFFF.#MMM#.#.#
#M#####M#M#M#M#####F#####M#.#.#
#M#..MMM#MMM#M#TTT#F..#MMM#...#
#M###M#######M#M#T#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#..F#FF.#M#MMMMM#MMM#
#.#.###.#.#F###F###M#######M#.#
#.#...#.#.#FFF#FF.#MMMMM#.#M#.#
#.###.#.#.###F#F#.#####M#.#M###
#.....#.#F#FFF#F#F#MMM#M#.#MMM#
#.#####.###F###F#F#T#M#M#.###M#
#...#.#.FFFF#FFF#F#T#M#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#MMM#.#MMM#
#G###.#F#F###F###F#F#####.###M#
#M#...#FFF#F#FFFFF#FFF#.....#M#
#M#.#######F#####F###F#.###.#M#
#M#...FF#FFFFF#FFF#FFF#.#...#M#
#M#.###F#####F#F###F###.#####M#
#M#...#FFFFF#FFFFF#FFFF.#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#.FF#FFFFFFFFF#F#MMM#...#
#####M#########F#F###F#M###.###
#MMMMM#MMM#TTT#F#FFFF.#MMM#.#.#
#M#####M#M#M#M#####F#####M#.#.#
#M#..MMM#MMM#M#MTT#...#MMM#...#
#M###M#######M#M#M#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#.FF#FF.#M#MMMMM#MMM#
#.#.###.#.#.###F###M#######M#.#
#.#...#.#.#.FF#FF.#MMMMM#.#M#.#
#.###.#.#.###F#F#F#####M#.#M###
#.....#.#.#FFF#F#F#MMM#M#.#MMM#
#.#####.###F###F#F#T#T#M#.###M#
#...#.#..FFF#FFF#F#T#T#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#TMM#.#MMM#
#G###.#.#F###F###F#F#####.###M#
#M#...#.FF#F#FFFFF#FFF#.....#M#
#M#.#######F#####F###F#.###.#M#
#M#.....#FFFFF#FFF#FFF#.#...#M#
#M#.###.#####F#F###F###.#####M#
#M#...#FFFFF#FFFFF#FFF..#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#.FF#FFFFFFFFF#F#MMM#...#
#####M#########F#F###.#M###.###
#MMMMM#MMT#TTT#F#FFF..#MMM#.#.#
#M#####M#M#M#M#####F#####M#.#.#
#M#..MMM#MMM#M#MMM#...#MMM#...#
#M###M#######M#M#M#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#..F#FF.#M#MMMMM#MMM#
#.#.###.#.#.###F###M#######M#.#
#.#...#.#.#.FF#FF.#MMMMM#.#M#.#
#.###.#.#.###F#F#.#####M#.#M###
#.....#.#.#FFF#F#F#MMM#M#.#MMM#
#.#####.###F###F#F#T#T#M#.###M#
#...#.#..FFF#FFF#F#T#T#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#TMM#.#MMM#
#G###.#.#F###F###F#F#####.###M#
#M#...#.FF#F#FFFFF#FFF#.....#M#
#M#.#######F#####F###F#.###.#M#
#M#.....#FFFFF#FFF#FFF#.#...#M#
#M#.###.#####F#F###F###.#####M#
#M#...#FFFFF#FFFFF#FFFF.#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#.FF#FFFFFFFFF#F#MMM#...#
#####M#########F#F###.#M###.###
#MMMMM#MMT#TTT#F#FFF..#MMM#.#.#
#M#####M#M#M#M#####.#####M#.#.#
#M#..MMM#MMM#M#TMT#...#MMM#...#
#M###M#######M#M#M#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#..F#FF.#M#MMMMM#MMM#
#.#.###.#.#.###F###M#######M#.#
#.#...#.#.#FFF#FFF#MMMMM#.#M#.#
#.###.#.#.###F#F#.#####M#.#M###
#.....#.#.#FFF#F#F#MMM#M#.#MMM#
#.#####.###F###F#F#T#T#M#.###M#
#...#.#..FFF#FFF#F#T#T#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#TTM#.#MMM#
#G###.#.#F###F###F#F#####.###M#
#M#...#FFF#F#FFFFF#FFF#.....#M#
#M#.#######F#####F###F#.###.#M#
#M#.....#FFFFF#FFF#FFF#.#...#M#
#M#.###.#####F#F###F###.#####M#
#M#...#FFFFF#FFFFF#FFFF.#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#.FF#FFFFFFFFF#F#MMM#...#
#####M#########F#F###.#M###.###
#MMMMM#MMT#TTT#F#FFFF.#MMM#.#.#
#M#####M#M#M#T#####.#####M#.#.#
#M#..MMM#MMM#M#TMM#...#MMM#...#
#M###M#######M#M#M#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.......#.....#...#MMM..#MMMMM#
#.###########.###.#M#M###M###M#
#.#.......#.FF#FF.#M#MMMMM#MMM#
#.#.###.#.#.###F###M#######M#.#
#.#...#.#.#.FF#FF.#MMMMM#.#M#.#
#.###.#.#.###F#F#F#####M#.#M###
#.....#.#.#FFF#F#F#MMM#M#.#MMM#
#.#####.###F###F#F#T#T#M#.###M#
#...#.#..FFF#FFF#F#T#T#M#.#MMM#
###.#.#########F#F#S#T#M#.#M###
#...#...#FFFFF#F#F#F#TMM#.#MMM#
#G###.#.#F###F###F#F#####.###M#
#M#...#.FF#F#FFFFF#FFF#.....#M#
#M#.#######F#####F###F#.###.#M#
#M#....F#FFFFF#FFF#FFF#.#...#M#
#M#.###F#####F#F###F###.#####M#
#M#...#FFFFF#FFFFF#FFFF.#MMMMM#
#M#####.###F#########F###M###.#
#MMMMM#.FF#FFFFFFFFF#F#MMM#...#
#####M#########F#F###.#M###.###
#MMMMM#MMT#TTT#F#FFF..#MMM#.#.#
#M#####M#M#M#M#####.#####M#.#.#
#M#..MMM#MMM#M#TMT#...#MMM#...#
#M###M#######M#M#M#.###M#####.#
#MMMMM#.....#MMM#M#.#MMM#MMM#.#
#########.#.#####M#.#M###M#M###
#.........#.#...#M#.#MMMMM#MMM#
#.#.#########.#.#M###########M#
#.#...........#..MMMMMMMMMMMMM#
###############################
```

## Case 373 (final failure)

Loop gain: `0.0427`. First loop F1 `0.0440` with 106 false positives and 155 misses. loop12 F1 `0.0866` with 104 false positives and 149 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMMMMT#MMM#...#...#.#
#####.###G#######M###########.#
#.....#...#.....#TMMMM#MMMMMMM#
#.###.#.###.#.#######M#M#####M#
#...#.#.#..F#.FFFFFF#MMM#...#M#
#.#.###.#.#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FF...#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#.......FFFF#FFF#F#FFFFF#...#M#
#.#############F#F###F###.###M#
#MMM#MTTMT#FFFFF#FFFFF#FFF#MMM#
#M#M#M###M#F#####F#####F###M###
#M#MMM..#M#F#F#FFF#F#FFF#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MMM#F#FFF#FFF#FF.#.#M#.#
###M#.#M###.###F#F#####.#.#M#.#
#.#M..#M#.#..F#FFF#FF.....#M#.#
#.#M###M#.#.#F#####F#######M#.#
#MMM#MMM#...#.FF#FFF#MMMMM#M#.#
#M###M#########.#.###M###M#M#.#
#M#.#M#MMM#MMM#F#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

### loop2

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMMTTT#MMM#...#...#.#
#####.###G#######M###########.#
#.....#...#FFFFF#MMMMM#MMMMMMM#
#.###.#.###F#F#######M#M#####M#
#...#.#.#FFF#FFFFF..#MMM#...#M#
#.#.###.#F#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FF...#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#......FFFFF#FFF#F#FFFF.#...#M#
#.#############F#F###F###.###M#
#MMM#MTTTT#FFFFF#FFFFF#...#MMM#
#M#M#M###T#F#####F#####.###M###
#M#MMMFF#T#F#F#FFF#F#FF.#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MTT#F#FFF#FFF#F..#.#M#.#
###M#.#M###F###F#F#####.#.#M#.#
#.#M..#M#.#FFF#FFF#FFF....#M#.#
#.#M###M#.#.#.#####F#######M#.#
#MMM#MMM#...#...#FFF#MMMMM#M#.#
#M###M#########.#F###M###M#M#.#
#M#.#M#MMM#MMM#.#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

### loop4

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMTTTT#MMM#...#...#.#
#####.###G#######T###########.#
#.....#...#FFFFF#MMMMM#MMMMMMM#
#.###.#.###F#F#######M#M#####M#
#...#.#.#.FF#FFFFFFF#MMM#...#M#
#.#.###.#F#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FFF..#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#......FFFFF#FFF#F#FFFF.#...#M#
#.#############F#F###F###.###M#
#MMM#MMMTT#FFFFF#FFFFF#...#MMM#
#M#M#M###T#F#####F#####.###M###
#M#MMM..#T#F#F#FFF#F#FF.#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MTT#F#FFF#FFF#F..#.#M#.#
###M#.#M###F###F#F#####.#.#M#.#
#.#M..#M#F#FFF#FFF#F......#M#.#
#.#M###M#.#F#F#####.#######M#.#
#MMM#MMM#...#..F#...#MMMMM#M#.#
#M###M#########.#.###M###M#M#.#
#M#.#M#MMM#MMM#.#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

### loop6

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMTTTT#MMM#...#...#.#
#####.###G#######M###########.#
#.....#...#.FFFF#TMMMM#MMMMMMM#
#.###.#.###F#F#######M#M#####M#
#...#.#.#.FF#FFFFFF.#MMM#...#M#
#.#.###.#F#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FFF..#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#.......FFFF#FFF#F#FFFF.#...#M#
#.#############F#F###F###.###M#
#MMM#MMTTT#FFFFF#FFFFF#...#MMM#
#M#M#M###T#F#####F#####.###M###
#M#MMM.F#T#F#F#FFF#F#F..#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MTT#F#FFF#FFF#F..#.#M#.#
###M#.#M###F###F#F#####.#.#M#.#
#.#M..#M#F#FFF#FFF#F......#M#.#
#.#M###M#.#.#F#####F#######M#.#
#MMM#MMM#...#..F#F..#MMMMM#M#.#
#M###M#########.#.###M###M#M#.#
#M#.#M#MMM#MMM#.#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

### loop10

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMTTTT#MMM#...#...#.#
#####.###G#######M###########.#
#.....#...#FFFFF#MMMMM#MMMMMMM#
#.###.#.###F#F#######M#M#####M#
#...#.#.#.FF#FFFFFFF#MMM#...#M#
#.#.###.#F#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FFF..#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#......FFFFF#FFF#F#FFFF.#...#M#
#.#############F#F###F###.###M#
#MMM#MMMTT#FFFFF#FFFFF#...#MMM#
#M#M#M###T#F#####F#####.###M###
#M#MMM..#T#F#F#FFF#F#FF.#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MTT#F#FFF#FFF#...#.#M#.#
###M#.#M###F###F#F#####.#.#M#.#
#.#M..#M#F#FFF#FFF#F......#M#.#
#.#M###M#.#.#.#####.#######M#.#
#MMM#MMM#...#.FF#F..#MMMMM#M#.#
#M###M#########.#.###M###M#M#.#
#M#.#M#MMM#MMM#.#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

### loop12

```text
###############################
#.............#MMMMM#.....#...#
#.#####.#######M###M#.###.#.#.#
#.....#.#MMMTTTT#MMM#...#...#.#
#####.###G#######M###########.#
#.....#...#.FFFF#MMMMM#MMMMMMM#
#.###.#.###F#F#######M#M#####M#
#...#.#.#.FF#FFFFF..#MMM#...#M#
#.#.###.#F#######F#F#####.###M#
#.#.#.#.#F#FFFFFFF#FFF..#...#M#
#.#.#.#.#F#F###########.###.#M#
#.#.#...#F#F#FFFFFFF#F#...#.#M#
###.#######F#F#####F#F###.#.#M#
#.......FFFF#FFF#F#FFFF.#...#M#
#.#############F#F###F###.###M#
#MMM#MMTTT#FFFFF#FFFFF#...#MMM#
#M#M#M###T#F#####F#####.###M###
#M#MMM..#T#F#F#FFF#F#FF.#.#M#.#
#M#######T#F#F#F###F#F###.#M#.#
#MMM#.#MTT#F#FFF#FFF#F..#.#M#.#
###M#.#M###F###F#F#####.#.#M#.#
#.#M..#M#F#FFF#FFF#FF.....#M#.#
#.#M###M#.#.#.#####.#######M#.#
#MMM#MMM#...#..F#...#MMMMM#M#.#
#M###M#########.#.###M###M#M#.#
#M#.#M#MMM#MMM#.#.#MMM#.#M#M#.#
#M#.#M#M#M#M#M#.#.#M###.#M#M#.#
#MMM#MMM#MMM#M#...#MMMMM#MMM#.#
###M#########M#########M#####.#
#..MMMMMS...#MMMMMMMMMMM......#
###############################
```

## Case 498 (final failure)

Loop gain: `0.0070`. First loop F1 `0.0803` with 100 false positives and 152 misses. loop12 F1 `0.0873` with 100 false positives and 151 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#MMM#M#.#M#MMMMM#M#
#M###.#M#M#.#####M#.#M#M###M#M#
#M#...#MMM#.#...#M#.#M#MMM#MMM#
#M#.#######.#.#.#T###M###M#####
#M#..MMM#.F.#.#FFTTTMM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MMT#.#F#F#FFF#F..#...#.#
#M###M###M#F#F#F#####F#F###.###
#M#MMM..#T#F#FFF#FFFFF#FF.#...#
#M#M#####T#F#######F#####.###.#
#M#MTTMMTT#FFFFFFF#FFF#FF.#.#.#
#M#########F#####F#####F###.#.#
#MMM#....FFF#FFFFF#FFF#FF.#.#.#
###M###.#.#######F#F#F###.#.#.#
#.#MMM#.#..F#FFF#FFF#FF...#.#.#
#.###M#####.#F#F#F#########.#.#
#...#MMMMM#..F#F#F#FF.....#...#
#.#.#####M#####F###F#####.#####
#.#.....#MMM#.FF#FFF....#...#.#
#.#.###.###M#.###F#######.#.#.#
#.#.#.....#M#.F...#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

### loop2

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#MTT#M#.#M#MMMMM#M#
#M###.#M#M#F#####M#.#M#M###M#M#
#M#...#MMM#F#FFF#M#.#M#MMM#MMM#
#M#.#######F#F#F#M###M###M#####
#M#..MMM#.FF#F#FFTMMMM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MTT#F#F#F#FFF#...#...#.#
#M###M###T#F#F#F#####F#.###.###
#M#MMM..#T#F#FFF#FFFFF#...#...#
#M#M#####T#F#######F#####.###.#
#M#MMMTTTT#FFFFFFF#FFF#...#.#.#
#M#########F#####F#####.###.#.#
#MMM#.FFFFFF#FFFFF#FFF#...#.#.#
###M###.#F#######F#F#F###.#.#.#
#.#MMM#.#FFF#FFF#FFF#F....#.#.#
#.###M#####F#F#F#F#########.#.#
#...#MMMMM#FFF#F#F#FF.....#...#
#.#.#####M#####.###F#####.#####
#.#.....#MMM#...#FFF....#...#.#
#.#.###.###M#.###F#######.#.#.#
#.#.#.....#M#.....#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

### loop4

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#TTT#M#.#M#MMMMM#M#
#M###.#M#M#.#####M#.#M#M###M#M#
#M#...#MMM#F#FFF#M#.#M#MMM#MMM#
#M#.#######F#F#F#T###M###M#####
#M#..MMM#.FF#F#FFTMMMM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MTT#F#F#F#FFF#F..#...#.#
#M###M###T#F#F#F#####F#.###.###
#M#MMM..#T#F#FFF#FFFFF#...#...#
#M#M#####T#F#######F#####.###.#
#M#MMMMMTT#FFFFFFF#FFF#...#.#.#
#M#########F#####F#####.###.#.#
#MMM#...FFFF#FFFFF#FFF#...#.#.#
###M###.#F#######F#F#F###.#.#.#
#.#MMM#.#FFF#FFF#FFF#.....#.#.#
#.###M#####F#F#F#F#########.#.#
#...#MMMMT#FFF#F#F#F......#...#
#.#.#####M#####F###F#####.#####
#.#.....#MMM#..F#.F.....#...#.#
#.#.###.###M#.###.#######.#.#.#
#.#.#.....#M#.....#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

### loop6

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#TTT#M#.#M#MMMMM#M#
#M###.#M#M#.#####T#.#M#M###M#M#
#M#...#MMM#.#FFF#M#.#M#MMM#MMM#
#M#.#######F#F#F#T###M###M#####
#M#..MMM#.FF#F#FFTMMTM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MTT#F#F#F#FFF#FF.#...#.#
#M###M###T#F#F#F#####F#.###.###
#M#MMM..#T#F#FFF#FFFFF#...#...#
#M#M#####T#F#######F#####.###.#
#M#MMMMMTT#FFFFFFF#FFF#...#.#.#
#M#########F#####F#####.###.#.#
#MMM#...FFFF#FFFFF#FFF#...#.#.#
###M###.#F#######F#F#F###.#.#.#
#.#MMM#.#FFF#FFF#FFF#.....#.#.#
#.###M#####F#F#F#F#########.#.#
#...#MMMMT#FFF#F#F#F......#...#
#.#.#####M#####F###.#####.#####
#.#.....#MMM#..F#F......#...#.#
#.#.###.###M#.###.#######.#.#.#
#.#.#.....#M#.....#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

### loop10

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#TTT#M#.#M#MMMMM#M#
#M###.#M#M#.#####M#.#M#M###M#M#
#M#...#MMM#.#FFF#T#.#M#MMM#MMM#
#M#.#######F#F#F#T###M###M#####
#M#..MMM#.FF#F#FFTTMTM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MMT#F#F#F#FFF#F..#...#.#
#M###M###T#F#F#F#####F#.###.###
#M#MMM..#T#F#FFF#FFFFF#...#...#
#M#M#####T#F#######F#####.###.#
#M#MMMMTTT#FFFFFFF#FFF#...#.#.#
#M#########F#####F#####.###.#.#
#MMM#..FFFFF#FFFFF#FFF#...#.#.#
###M###.#F#######F#F#F###.#.#.#
#.#MMM#.#FFF#FFF#FFF#F....#.#.#
#.###M#####F#F#F#F#########.#.#
#...#MMMMT#FFF#F#F#F......#...#
#.#.#####M#####F###.#####.#####
#.#.....#MMM#.FF#F......#...#.#
#.#.###.###M#.###.#######.#.#.#
#.#.#.....#M#.....#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

### loop12

```text
###############################
#MMMMMMM#MMMMM#MMM#..MMMMMMMMM#
#M#####M#M#.#M#M#M#.#M#######M#
#M#...#M#M#.#TTT#M#.#M#MMMMM#M#
#M###.#M#M#.#####M#.#M#M###M#M#
#M#...#MMM#F#FFF#M#.#M#MMM#MMM#
#M#.#######F#F#F#M###M###M#####
#M#..MMM#.FF#F#FFTTMMM#..MMM#.#
#M###M#M#F#################M#.#
#MMMMM#M#FFFFF#FFF#FFF....#MG.#
#######M#####F#F#F#F#####.###.#
#MMMMM#MMT#F#F#F#FFF#F..#...#.#
#M###M###T#F#F#F#####F#.###.###
#M#MMM..#T#F#FFF#FFFFF#...#...#
#M#M#####T#F#######F#####.###.#
#M#MMMMMTT#FFFFFFF#FFF#...#.#.#
#M#########F#####F#####.###.#.#
#MMM#...FFFF#FFFFF#FFF#...#.#.#
###M###.#F#######F#F#F###.#.#.#
#.#MMM#.#FFF#FFF#FFF#F....#.#.#
#.###M#####F#F#F#F#########.#.#
#...#MMMMT#FFF#F#F#F......#...#
#.#.#####M#####F###.#####.#####
#.#.....#MMM#..F#.......#...#.#
#.#.###.###M#.###.#######.#.#.#
#.#.#.....#M#.....#MMMMM#.#...#
###.#######M###.###M###M#.###.#
#...#.....#MMM#.#MMM#.#M#.#.#.#
#.###.###.###M###M###.#S#.#.#.#
#.......#....MMMMM#.......#...#
###############################
```

## Case 191 (final failure)

Loop gain: `0.0327`. First loop F1 `0.0552` with 100 false positives and 174 misses. loop12 F1 `0.0878` with 101 false positives and 169 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MMMMM..#M......#...#
#M#M#M###M#M#######M#######.###
#M#M#MMM#MMM#.#...#MMM#.....#.#
#M#M###M#####F#F#F###M#.#####.#
#M#MMMMM#..F..#F#F#F#M#.#...#.#
#M#######.#####F#F#F#M#.#.#.#G#
#M#MMM#..FF.#FFF#F#FFT#...#.#M#
#M#M#M#.###.#F###F###T#####.#M#
#MMM#M#.#.FF#F#F#FFF#TTMMS#.#M#
#####M###.#F#F#F###F#######.#M#
#...#M#.FF#F#F#FFF#F#FFF#...#M#
###.#T#.#####F###F###F#F#.###M#
#...#M#FFF#FFF#FFF#FFF#F..#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#MMT#F#FFF#FFF#F#FF.#M#.#
#.###M#M#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#.FF#FFFFF#FF.#.#M#.#
#M#####M#M###F#F#########.#M###
#MMMMMMM#M..#F#FFF#F..#...#MMM#
#########M###F#####F#.#.#####M#
#...#MMMMM#...#.FFF.#...#..MMM#
###.#M#####.###F###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MMMTTF.#M......#...#
#M#M#M###M#T#######M#######.###
#M#M#MMM#MTT#F#FF.#MMM#.....#.#
#M#M###M#####F#F#.###M#.#####.#
#M#MMMMM#FFFFF#F#F#.#M#.#...#.#
#M#######F#####F#F#F#M#.#.#.#G#
#M#MMM#.FFFF#FFF#F#FFM#...#.#M#
#M#M#M#.###F#F###F###M#####.#M#
#MMM#M#.#FFF#F#F#FFF#MMMMS#.#M#
#####M###F#F#F#F###F#######.#M#
#...#M#FFF#F#F#FFF#F#FF.#...#M#
###.#M#F#####F###F###F#.#.###M#
#...#M#FFF#FFF#FFF#FFF#...#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#TTT#F#FFF#FFF#F#...#M#.#
#.###M#T#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#FFF#FFFFF#F..#.#M#.#
#M#####M#T###F#F#########.#M###
#MMMMMMM#MF.#F#FFF#FF.#...#MMM#
#########M###.#####F#.#.#####M#
#...#MMMMM#...#FFFFF#...#..MMM#
###.#M#####.###.###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MTTTTF.#M......#...#
#M#M#M###M#M#######M#######.###
#M#M#MMM#MMT#F#FFF#MMM#.....#.#
#M#M###M#####F#F#.###M#.#####.#
#M#MMMMM#.FFFF#F#F#.#M#.#...#.#
#M#######F#####F#F#F#T#.#.#.#G#
#M#MMM#..FFF#FFF#F#FFT#...#.#M#
#M#M#M#.###F#F###F###T#####.#M#
#MMM#M#.#FFF#F#F#FFF#MMMMS#.#M#
#####M###F#F#F#F###F#######.#M#
#...#M#.FF#F#F#FFF#F#FF.#...#M#
###.#M#F#####F###F###F#.#.###M#
#...#M#.FF#FFF#FFF#FFF#...#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#MTT#F#FFF#FFF#F#...#M#.#
#.###M#M#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#FFF#FFFFF#F..#.#M#.#
#M#####M#T###F#F#########.#M###
#MMMMMMM#TFF#F#FFF#F..#...#MMM#
#########M###F#####F#.#.#####M#
#...#MMMMM#...#F.F..#...#..MMM#
###.#M#####.###.###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MTTTTF.#M......#...#
#M#M#M###M#M#######M#######.###
#M#M#MMM#MMM#F#FFF#MMM#.....#.#
#M#M###M#####F#F#F###M#.#####.#
#M#MMMMM#.FFFF#F#F#.#M#.#...#.#
#M#######F#####F#F#F#T#.#.#.#G#
#M#MMM#..FFF#FFF#F#FFT#...#.#M#
#M#M#M#.###F#F###F###T#####.#M#
#MMM#M#.#FFF#F#F#FFF#TTMMS#.#M#
#####M###F#F#F#F###F#######.#M#
#...#M#.FF#F#F#FFF#F#F..#...#M#
###.#M#.#####F###F###F#.#.###M#
#...#M#FFF#FFF#FFF#FFF#...#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#TTT#F#FFF#FFF#F#...#M#.#
#.###M#M#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#FFF#FFFFF#...#.#M#.#
#M#####M#T###F#F#########.#M###
#MMMMMMM#TFF#F#FFF#F..#...#MMM#
#########M###F#####.#.#.#####M#
#...#MMMMM#...#F....#...#..MMM#
###.#M#####.###.###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MMTTTF.#M......#...#
#M#M#M###M#M#######M#######.###
#M#M#MMM#MMT#F#FF.#MMM#.....#.#
#M#M###M#####F#F#F###M#.#####.#
#M#MMMMM#FFFFF#F#F#.#M#.#...#.#
#M#######F#####F#F#F#T#.#.#.#G#
#M#MMM#..FFF#FFF#F#FFT#...#.#M#
#M#M#M#.###F#F###F###T#####.#M#
#MMM#M#.#FFF#F#F#FFF#TTMMS#.#M#
#####M###F#F#F#F###F#######.#M#
#...#M#.FF#F#F#FFF#F#FF.#...#M#
###.#M#F#####F###F###F#.#.###M#
#...#M#.FF#FFF#FFF#FFF#...#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#TTT#F#FFF#FFF#F#...#M#.#
#.###M#M#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#FFF#FFFFF#F..#.#M#.#
#M#####M#T###F#F#########.#M###
#MMMMMMM#TFF#F#FFF#F..#...#MMM#
#########M###F#####F#.#.#####M#
#...#MMMMM#...#F.FF.#...#..MMM#
###.#M#####.###.###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

### loop12

```text
###############################
#MMM..........#MMMMM....#.....#
#M#M###########M###M#####.###.#
#M#M#MMMMM#MTTTTF.#M......#...#
#M#M#M###M#M#######M#######.###
#M#M#MMM#MMM#F#FF.#MMM#.....#.#
#M#M###M#####F#F#F###M#.#####.#
#M#MMMMM#.FFFF#F#F#.#M#.#...#.#
#M#######F#####F#F#F#M#.#.#.#G#
#M#MMM#..FFF#FFF#F#FFT#...#.#M#
#M#M#M#.###F#F###F###T#####.#M#
#MMM#M#.#FFF#F#F#FFF#TMMMS#.#M#
#####M###F#F#F#F###F#######.#M#
#...#M#.FF#F#F#FFF#F#F..#...#M#
###.#M#.#####F###F###F#.#.###M#
#...#M#.FF#FFF#FFF#FFF#...#MMM#
#.#.#M###F#F###F###F#F#####M#.#
#.#.#M#MTT#F#FFF#FFF#F#...#M#.#
#.###M#M#T#F###F#F###F#.#.#M#.#
#MMMMM#M#T#FFF#FFFFF#...#.#M#.#
#M#####M#T###F#F#########.#M###
#MMMMMMM#TFF#F#FFF#F..#...#MMM#
#########M###F#####.#.#.#####M#
#...#MMMMM#...#F....#...#..MMM#
###.#M#####.###.###########M#.#
#...#M#...#.....#..MMM#..MMM#.#
#.###M#.#.#########M#M#.#M###.#
#.#MMM#.#....MMMMM#M#M#.#MMM#.#
#.#M#########M###M#M#M#####M#.#
#..MMMMMMMMMMM#..MMM#MMMMMMM#.#
###############################
```

## Case 327 (final failure)

Loop gain: `-0.0239`. First loop F1 `0.1168` with 94 false positives and 148 misses. loop12 F1 `0.0929` with 103 false positives and 151 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#.#...#.#.#...#...#
#M###M#.#######.###.#.#####.#.#
#M#.#MMM#.....#...#.#...#.#.#.#
#M#.###M#.###.#.#F#.###.#.#.#.#
#M#.#..G#.#F#.FF#F#F#.....#.#.#
#M#.#.###F#F#####F#F#######.#.#
#M#...#..F#FFFFF#F#FFF..#...#.#
#M#####.###.###F#F#####.#.###.#
#M#MMM#..F#.#F#F#F#FFFF.#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M..#FFF#F#F#FFF#FFFF...#.#
#M#M#M#.#####F#F#########.#S###
#M#M#T#..F#FFF#FFF#FFTTT#.#MMM#
#M#M#M###F#F#F###F###T#T#####M#
#MMM#M#.#F#F#FFF#F#TTT#TTMMM#M#
#####M#.#.###F#F#F#T#######M#M#
#MMM#M#..F.FFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#T#M#M#####
#M#MMM#.#MMM#FFF#F#F#MMM#MMMMM#
#M#####.#M#M#F###F#F#########M#
#MMM#...#M#M#.#.FF#F#.......#M#
#.#M#####M#M###F###.#.#.#####M#
#.#M#MMMMM#MMM#F#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#.#F..#.#.#...#...#
#M###M#.#######F###.#.#####.#.#
#M#.#MMM#.FFFF#FF.#.#...#.#.#.#
#M#.###M#.###F#F#.#.###.#.#.#.#
#M#.#..G#F#F#FFF#.#.#.....#.#.#
#M#.#.###F#F#####F#F#######.#.#
#M#...#.FF#FFFFF#F#FFF..#...#.#
#M#####.###F###F#F#####.#.###.#
#M#MMM#.FF#F#F#F#F#FF...#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M.F#FFF#F#F#FFF#FF.....#.#
#M#M#M#F#####F#F#########.#S###
#M#M#M#FFF#FFF#FFF#FFTTM#.#MMM#
#M#M#M###F#F#F###F###T#T#####M#
#MMM#M#.#F#F#FFF#F#TTT#MMMMM#M#
#####M#.#F###F#F#F#T#######M#M#
#MMM#M#.FFFFFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#M#M#M#####
#M#MMM#.#MTT#FFF#F#F#MMM#MMMMM#
#M#####.#M#M#.###F#F#########M#
#MMM#...#M#M#.#..F#F#.......#M#
#.#M#####M#M###.###.#.#.#####M#
#.#M#MMMMM#MMM#.#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#F#FF.#.#.#...#...#
#M###M#.#######F###.#.#####.#.#
#M#.#MMM#...FF#FF.#.#...#.#.#.#
#M#.###M#.###F#F#F#.###.#.#.#.#
#M#.#..G#.#F#FFF#F#.#.....#.#.#
#M#.#.###F#F#####F#F#######.#.#
#M#...#..F#FFFFF#F#FFF..#...#.#
#M#####.###F###F#F#####.#.###.#
#M#MMM#.FF#F#F#F#F#FFF..#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M..#FFF#F#F#FFF#FF.....#.#
#M#M#M#F#####F#F#########.#S###
#M#M#M#.FF#FFF#FFF#FFTTM#.#MMM#
#M#M#M###F#F#F###F###T#M#####M#
#MMM#M#F#F#F#FFF#F#TTT#MMMMM#M#
#####M#.#F###F#F#F#T#######M#M#
#MMM#M#.FFFFFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#M#M#M#####
#M#MMM#.#TTT#FFF#F#F#MMM#MMMMM#
#M#####.#M#T#F###F#F#########M#
#MMM#...#M#M#.#F..#.#.......#M#
#.#M#####M#M###.###.#.#.#####M#
#.#M#MMMMM#MMM#.#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#F#FF.#.#.#...#...#
#M###M#.#######F###.#.#####.#.#
#M#.#MMM#...FF#FF.#.#...#.#.#.#
#M#.###M#.###F#F#F#.###.#.#.#.#
#M#.#..G#.#F#FFF#F#.#.....#.#.#
#M#.#.###F#F#####F#F#######.#.#
#M#...#..F#FFFFF#F#FFF..#...#.#
#M#####.###F###F#F#####.#.###.#
#M#MMM#.FF#F#F#F#F#FFF..#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M..#FFF#F#F#FFF#F......#.#
#M#M#M#F#####F#F#########.#S###
#M#M#M#.FF#FFF#FFF#FFTTM#.#MMM#
#M#M#M###F#F#F###F###T#M#####M#
#MMM#M#F#F#F#FFF#F#TTT#MMMMM#M#
#####M#.#F###F#F#F#T#######M#M#
#MMM#M#.FFFFFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#M#M#M#####
#M#MMM#.#TTT#FFF#F#F#MMM#MMMMM#
#M#####.#M#M#F###F#F#########M#
#MMM#...#M#M#.#F..#.#.......#M#
#.#M#####M#M###.###.#.#.#####M#
#.#M#MMMMM#MMM#.#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#F#FF.#.#.#...#...#
#M###M#.#######F###.#.#####.#.#
#M#.#MMM#...FF#FFF#.#...#.#.#.#
#M#.###M#.###F#F#F#.###.#.#.#.#
#M#.#..G#.#F#FFF#F#.#.....#.#.#
#M#.#.###.#F#####F#F#######.#.#
#M#...#..F#FFFFF#F#FFF..#...#.#
#M#####.###F###F#F#####.#.###.#
#M#MMM#.FF#F#F#F#F#FFFF.#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M..#FFF#F#F#FFF#FF.....#.#
#M#M#M#F#####F#F#########.#S###
#M#M#M#.FF#FFF#FFF#FFTTM#.#MMM#
#M#M#M###F#F#F###F###T#M#####M#
#MMM#M#F#F#F#FFF#F#TTT#MMMMM#M#
#####M#.#F###F#F#F#T#######M#M#
#MMM#M#.FFFFFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#M#M#M#####
#M#MMM#.#TTT#FFF#F#F#MMM#MMMMM#
#M#####.#M#M#F###F#.#########M#
#MMM#...#M#M#.#F..#.#.......#M#
#.#M#####M#M###.###.#.#.#####M#
#.#M#MMMMM#MMM#.#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

### loop12

```text
###############################
#MMM#.......#.......#...#.....#
#M#M#####.#.#.#####.#.#.#.###.#
#M#MMM#...#.#F#FF.#.#.#...#...#
#M###M#.#######F###.#.#####.#.#
#M#.#MMM#..FFF#FF.#.#...#.#.#.#
#M#.###M#.###F#F#.#.###.#.#.#.#
#M#.#..G#.#F#FFF#F#.#.....#.#.#
#M#.#.###F#F#####F#F#######.#.#
#M#...#..F#FFFFF#F#FFF..#...#.#
#M#####.###F###F#F#####.#.###.#
#M#MMM#.FF#F#F#F#F#FFF..#.#...#
#M#M#M###F#F#F#F#F#F#####.###.#
#M#M#M..#FFF#F#F#FFF#F......#.#
#M#M#M#F#####F#F#########.#S###
#M#M#M#.FF#FFF#FFF#FFTTM#.#MMM#
#M#M#M###F#F#F###F###T#M#####M#
#MMM#M#F#F#F#FFF#F#TTT#MMMMM#M#
#####M#.#F###F#F#F#T#######M#M#
#MMM#M#.FFFFFF#F#F#TTT#MMM#MMM#
#M#M#M#.#######F#F#F#M#M#M#####
#M#MMM#.#TTT#FFF#F#F#MMM#MMMMM#
#M#####.#M#M#.###F#.#########M#
#MMM#...#M#M#.#F..#.#.......#M#
#.#M#####M#M###.###.#.#.#####M#
#.#M#MMMMM#MMM#.#.#...#.#MMMMM#
#.#M#M#######M#.#.#####.#M###.#
#.#MMM#.#MMMMM#.#.......#MMM#.#
#.#####.#M#####.###########M#.#
#.......#MMMMMMMMMMMMMMMMMMM#.#
###############################
```

## Case 151 (final failure)

Loop gain: `-0.0398`. First loop F1 `0.1338` with 97 false positives and 162 misses. loop12 F1 `0.0940` with 102 false positives and 168 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MMM#SMM#MMM#.#.#.#.#
#M#M#######M###.#####M#.#.#.###
#M#M#.....#M#...#.#MMM#.#.#...#
#M#M###.###M#.###F#T###.#.###.#
#MMM....#MTT#..F#F#T#.....#...#
#.#######T#F###F#F#T#######.###
#.#MMMMMMT#FFF#FFF#TTT#MMM#...#
###M#########F#######T#M#M#.#.#
#MMM#....F..FF#FFFFF#TTM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#...FF.F#FFFFF#FFFFF#MMMMM#
###M#######F#F#F#######F#####M#
#MMM#FFFFF#F#F#FFFFF#FFF#F..#M#
#M###.#####F#######F#F###F###M#
#MMM#....FFF#FFF#FFF#FFF#F..#M#
#.#M###.#####F#F#F#####F###.#M#
#.#MMM#...#FFF#F#FFF#FF.....#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#...#.FF#FFFFF#TMM#...#M#
#.#M#####.###.#######M#M#.###M#
#.#MMMMM#...#.#TTTTTMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#TMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

### loop2

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MMT#STM#MMM#.#.#.#.#
#M#M#######M###F#####M#.#.#.###
#M#M#.....#T#FFF#.#MMM#.#.#...#
#M#M###.###T#F###.#M###.#.###.#
#MMM....#TTT#FFF#F#M#.....#...#
#.#######T#F###F#F#T#######.###
#.#MMMMMTT#FFF#FFF#TTT#MMM#...#
###M#########F#######M#M#M#.#.#
#MMM#...FFFFFF#FFFFF#MMM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#..FFFFF#FFFFF#FFFF.#MMMMM#
###M#######F#F#F#######.#####M#
#MMM#.FFFF#F#F#FFFFF#FF.#...#M#
#M###F#####F#######F#F###.###M#
#MMM#..FFFFF#FFF#FFF#FF.#...#M#
#.#M###F#####F#F#F#####.###.#M#
#.#MMM#.FF#FFF#F#FFF#F......#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#...#FFF#FFFFF#MMM#...#M#
#.#M#####.###.#######M#M#.###M#
#.#MMMMM#...#.#MMTTTMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#MMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

### loop4

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MMT#STM#MMM#.#.#.#.#
#M#M#######M###F#####M#.#.#.###
#M#M#.....#T#FFF#.#MMM#.#.#...#
#M#M###.###T#F###F#M###.#.###.#
#MMM....#MTT#FFF#F#T#.....#...#
#.#######M#F###F#F#T#######.###
#.#MMMMMMT#FFF#FFF#TTT#MMM#...#
###M#########F#######T#M#M#.#.#
#MMM#...FFFFFF#FFFFF#TMM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#..FFFFF#FFFFF#FFFF.#MMMMM#
###M#######F#F#F#######.#####M#
#MMM#...FF#F#F#FFFFF#FF.#...#M#
#M###.#####F#######F#F###.###M#
#MMM#..FFFFF#FFF#FFF#FF.#...#M#
#.#M###.#####F#F#F#####.###.#M#
#.#MMM#.FF#FFF#F#FFF#F......#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#..F#FFF#FFFFF#MMM#...#M#
#.#M#####.###.#######M#M#.###M#
#.#MMMMM#...#.#TMMMMMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#MMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

### loop6

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MTT#STM#MMM#.#.#.#.#
#M#M#######M###F#####M#.#.#.###
#M#M#.....#M#FFF#.#MMM#.#.#...#
#M#M###.###T#F###F#M###.#.###.#
#MMM....#MTT#FFF#F#M#.....#...#
#.#######T#F###F#F#T#######.###
#.#MMMMMMT#FFF#FFF#TTT#MMM#...#
###M#########F#######T#M#M#.#.#
#MMM#...FFFFFF#FFFFF#TMM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#...FFFF#FFFFF#FFFF.#MMMMM#
###M#######F#F#F#######.#####M#
#MMM#...FF#F#F#FFFFF#FF.#...#M#
#M###.#####F#######F#F###.###M#
#MMM#..FFFFF#FFF#FFF#FF.#...#M#
#.#M###.#####F#F#F#####.###.#M#
#.#MMM#.FF#FFF#F#FFF#F......#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#..F#FFF#FFFFF#MMM#...#M#
#.#M#####.###F#######M#M#.###M#
#.#MMMMM#...#.#TMMTMMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#MMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

### loop10

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MMT#STM#MMM#.#.#.#.#
#M#M#######M###F#####M#.#.#.###
#M#M#.....#M#FFF#.#MMM#.#.#...#
#M#M###.###T#F###F#M###.#.###.#
#MMM....#MTT#FFF#F#T#.....#...#
#.#######T#F###F#F#T#######.###
#.#MMMMMMT#FFF#FFF#TTT#MMM#...#
###M#########F#######T#M#M#.#.#
#MMM#...FFFFFF#FFFFF#TTM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#...FFFF#FFFFF#FFFF.#MMMMM#
###M#######F#F#F#######.#####M#
#MMM#...FF#F#F#FFFFF#FF.#...#M#
#M###.#####F#######F#F###.###M#
#MMM#...FFFF#FFF#FFF#FF.#...#M#
#.#M###.#####F#F#F#####.###.#M#
#.#MMM#.FF#FFF#F#FFF#.......#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#..F#FFF#FFFFF#MMM#...#M#
#.#M#####.###F#######M#M#.###M#
#.#MMMMM#...#.#TMTMMMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#MMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

### loop12

```text
###############################
#MMMG...#MMMMM..#MMM#...#.....#
#M#######M###M###M#M###.#.#.#.#
#M#MMMMMMM#MMT#STM#MMM#.#.#.#.#
#M#M#######M###F#####M#.#.#.###
#M#M#.....#M#FFF#F#MMM#.#.#...#
#M#M###.###T#F###F#M###.#.###.#
#MMM....#MTT#FFF#F#M#.....#...#
#.#######T#F###F#F#T#######.###
#.#MMMMMMT#FFF#FFF#TTT#MMM#...#
###M#########F#######T#M#M#.#.#
#MMM#...FFFFFF#FFFFF#TMM#M#.#.#
#M###.###########F#F#####M###.#
#MMM#...FFFF#FFFFF#FFF..#MMMMM#
###M#######F#F#F#######.#####M#
#MMM#...FF#F#F#FFFFF#FF.#...#M#
#M###.#####F#######F#F###.###M#
#MMM#...FFFF#FFF#FFF#FF.#...#M#
#.#M###.#####F#F#F#####.###.#M#
#.#MMM#.FF#FFF#F#FFF#F......#M#
#.###M###F#F###F###F#######.#M#
#.#MMM#..F#FFF#FFFFF#MMM#...#M#
#.#M#####.###F#######M#M#.###M#
#.#MMMMM#...#.#TMMMMMM#M#.#MMM#
#.#####M###.#.#M#######M###M###
#.#...#M#.#.#.#MMMMMMM#MMMMM#.#
#.###.#M#.#.#.#######M#######.#
#.....#M#.......#MMM#MMM#.....#
#####.#M#########M#M###M#####.#
#.....#MMMMMMMMMMM#MMMMM......#
###############################
```

## Case 499 (final failure)

Loop gain: `-0.0265`. First loop F1 `0.1310` with 86 false positives and 166 misses. loop12 F1 `0.1046` with 105 false positives and 169 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#.......#.............#
#.###.###.###.#.#.#####.#####.#
#...#.....#.#.#.#...#.#.#.....#
###.#######.#.#####.#.#S#.#####
#...#.......#.....#...#M#.#MMM#
#.#######.#.#####F###.#M###M#M#
#.#...#...#...#F#FFF#.#M#MMM#M#
#.#.#.#.#####F#F###F#.#M#M###M#
#...#...#...#F#FFF#FFF#MMM#MMM#
###########.#F#F#F#########M###
#........FFF#FFF#FFFFFF.#.#MMM#
#.#.###################.#.###M#
#.#.#....FFF#FFF#FFFFF#F#.#MMM#
#.#.#.#####F#F#F#F#F###F#.#M###
#.#...FF.F#FFF#FFF#FFFFFF.#MMM#
#.#######F###################M#
#.#MMM#.#FFFFF#FFFFF#TTTTT#MMM#
###M#M#.#####F#####F#T###M#M#.#
#MMM#M#.....FFFFFFFF#T#.#M#M#.#
#M###M#############F#T#.#M#M###
#M#MMM#MMMMMTTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MTT#TTMMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

### loop2

```text
###############################
#.......#.......#.............#
#.###.###.###.#.#.#####.#####.#
#...#.....#.#F#F#...#.#.#.....#
###.#######.#F#####.#.#S#.#####
#...#......F#FFFF.#...#M#.#MMM#
#.#######.#F#####.###.#M###M#M#
#.#...#..F#FFF#F#...#.#M#MMM#M#
#.#.#.#.#####F#F###F#.#M#M###M#
#...#...#FFF#F#FFF#FF.#MMM#MMM#
###########F#F#F#F#########M###
#.......FFFF#FFF#FFFFF..#.#MMM#
#.#.###################.#.###M#
#.#.#..FFFFF#FFF#FFFFF#.#.#MMM#
#.#.#.#####F#F#F#F#F###.#.#M###
#.#...FFFF#FFF#FFF#FFFF...#MMM#
#.#######F###################M#
#.#MMM#F#FFFFF#FFFFF#TTMMM#MMM#
###M#M#.#####F#####F#T###M#M#.#
#MMM#M#.FFFFFFFFFFFF#T#.#M#M#.#
#M###M#############F#M#.#M#M###
#M#MMM#MMMTTMTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MMT#TTTMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

### loop4

```text
###############################
#.......#.......#.............#
#.###.###.###.#.#.#####.#####.#
#...#.....#.#F#F#...#.#.#.....#
###.#######.#F#####.#.#S#.#####
#...#......F#FFFF.#...#M#.#MMM#
#.#######.#F#####F###.#M###M#M#
#.#...#...#FFF#F#F..#.#M#MMM#M#
#.#.#.#.#####F#F###F#F#M#M###M#
#...#...#FFF#F#FFF#FFF#MMM#MMM#
###########F#F#F#F#########M###
#.......FFFF#FFF#FFFFF..#.#MMM#
#.#.###################.#.###M#
#.#.#...FFFF#FFF#FFFFF#.#.#MMM#
#.#.#.#####F#F#F#F#F###.#.#M###
#.#....FFF#FFF#FFF#FFFF...#MMM#
#.#######F###################M#
#.#MMM#F#FFFFF#FFFFF#TTMMM#MMM#
###M#M#.#####F#####F#T###M#M#.#
#MMM#M#.FFFFFFFFFFFF#T#.#M#M#.#
#M###M#############F#M#.#M#M###
#M#MMM#MMTTTTTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MMT#MTMMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

### loop6

```text
###############################
#.......#.......#.............#
#.###.###.###.#.#.#####.#####.#
#...#.....#.#F#F#...#.#.#.....#
###.#######.#F#####.#.#S#.#####
#...#.......#FFFF.#...#M#.#MMM#
#.#######.#F#####F###.#M###M#M#
#.#...#...#FFF#F#F..#.#M#MMM#M#
#.#.#.#.#####F#F###F#F#M#M###M#
#...#...#FFF#F#FFF#FFF#MMM#MMM#
###########F#F#F#F#########M###
#.......FFFF#FFF#FFFFF..#.#MMM#
#.#.###################.#.###M#
#.#.#..FFFFF#FFF#FFFFF#.#.#MMM#
#.#.#.#####F#F#F#F#F###.#.#M###
#.#.....FF#FFF#FFF#FFFF...#MMM#
#.#######F###################M#
#.#MMM#F#FFFFF#FFFFF#TTMMM#MMM#
###M#M#.#####F#####F#T###M#M#.#
#MMM#M#.FFFFFFFFFFFF#T#.#M#M#.#
#M###M#############F#M#.#M#M###
#M#MMM#MMTTTTTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MTT#MTMMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

### loop10

```text
###############################
#.......#.......#.............#
#.###.###.###.#.#.#####.#####.#
#...#.....#.#F#F#...#.#.#.....#
###.#######.#F#####.#.#S#.#####
#...#.......#FFFFF#...#M#.#MMM#
#.#######.#F#####F###.#M###M#M#
#.#...#...#FFF#F#FF.#.#M#MMM#M#
#.#.#.#.#####F#F###F#F#M#M###M#
#...#...#FFF#F#FFF#FFF#MMM#MMM#
###########F#F#F#F#########M###
#.......FFFF#FFF#FFFFF..#.#MMM#
#.#.###################.#.###M#
#.#.#...FFFF#FFF#FFFFF#.#.#MMM#
#.#.#.#####F#F#F#F#F###.#.#M###
#.#.....FF#FFF#FFF#FFFF...#MMM#
#.#######F###################M#
#.#MMM#F#FFFFF#FFFFF#TTMMM#MMM#
###M#M#.#####F#####F#T###M#M#.#
#MMM#M#.FFFFFFFFFFFF#T#.#M#M#.#
#M###M#############F#M#.#M#M###
#M#MMM#MMTTTTTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MMT#MTMMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

### loop12

```text
###############################
#.......#.......#.............#
#.###.###.###.#F#.#####.#####.#
#...#.....#.#F#F#...#.#.#.....#
###.#######.#F#####.#.#S#.#####
#...#.......#FFFFF#...#M#.#MMM#
#.#######.#F#####F###.#M###M#M#
#.#...#...#FFF#F#FF.#.#M#MMM#M#
#.#.#.#.#####F#F###F#F#M#M###M#
#...#...#FFF#F#FFF#FFF#MMM#MMM#
###########F#F#F#F#########M###
#.......FFFF#FFF#FFFFFF.#.#MMM#
#.#.###################.#.###M#
#.#.#...FFFF#FFF#FFFFF#.#.#MMM#
#.#.#.#####F#F#F#F#F###.#.#M###
#.#.....FF#FFF#FFF#FFFF...#MMM#
#.#######F###################M#
#.#MMM#.#FFFFF#FFFFF#TTMMM#MMM#
###M#M#F#####F#####F#T###M#M#.#
#MMM#M#.FFFFFFFFFFFF#T#.#M#M#.#
#M###M#############F#M#.#M#M###
#M#MMM#MMTTTTTTTTT#F#M#MMM#MMM#
#M#M###M#########T###M#M#####M#
#M#M#..MMMMM#MMT#MTMMM#M#MMMMM#
#M#M#####.#M#M#M#######M#M#####
#M#M#...#.#MMM#MMM#.#MMM#MMMMM#
#M#M#G###########M#.#M#######M#
#M#MMM#MMM#MMMMMMM#.#M#MMMMM#M#
#M#####M#M#M#######.#M#M###M#M#
#MMMMMMM#MMM#........MMM#..MMM#
###############################
```

## Case 346 (final failure)

Loop gain: `-0.0137`. First loop F1 `0.1193` with 97 false positives and 154 misses. loop12 F1 `0.1056` with 98 false positives and 156 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#...#MMS#.#.........#
#.#############M#.#.#########.#
#...#.........#M#F#...#.....#.#
###.#.#######F#M#.#####.#.###.#
#.....#MMMTM#.#T#FFF..#.#.....#
#######M###T###T#####.#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#FF.#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#.....#F#FFFFF#F#FFF#MMMMM#
#M#.#####F#F#F###F#F###F#M#####
#M#....FFF#FFF#FFF#FFF#F#M#MMM#
#M#####F#####F#F#####F#F#T#M#M#
#M....#.#FFF#F#FFFFFFF#F#M#M#M#
#M###.#.#F#F#F#########F#M#M#M#
#MMM#.#.#.#F#FFF#FFFFFF.#MMM#M#
###M#.###.#.#####F#######.###M#
#MMM#.....#..FFFFF#FF.....#MMM#
#M#########################M###
#M#..MMMMM#..MTM#TTT#MMMMM#MMM#
#M#.#M###M###M#T#T#M#M###M#.#M#
#M#.#M#..MMMMM#T#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

### loop2

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#.F.#TTS#.#.........#
#.#############T#.#.#########.#
#...#......FFF#T#.#...#.....#.#
###.#.#######F#T#.#####.#.###.#
#.....#MMTTT#F#T#FF...#.#.....#
#######M###T###T#####.#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#F..#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#..FFF#F#FFFFF#F#FF.#MMMMM#
#M#.#####F#F#F###F#F###.#M#####
#M#....FFF#FFF#FFF#FFF#F#M#MMM#
#M#####F#####F#F#####F#.#M#M#M#
#M....#F#FFF#F#FFFFFFF#.#M#M#M#
#M###.#F#F#F#F#########.#M#M#M#
#MMM#.#.#F#F#FFF#FFFFF..#MMM#M#
###M#.###F#F#####F#######.###M#
#MMM#.....#FF.FFFF#FF.....#MMM#
#M#########################M###
#M#..MMMMM#..MMM#TMT#MMMMM#MMM#
#M#.#M###M###M#M#M#M#M###M#.#M#
#M#.#M#..MMMMM#M#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

### loop4

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#.FF#TTS#.#.........#
#.#############T#F#.#########.#
#...#.......FF#T#F#...#.....#.#
###.#.#######F#T#F#####.#.###.#
#.....#MMMTT#F#T#FF...#.#.....#
#######M###T###T#####F#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#FF.#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#...FF#F#FFFFF#F#FF.#MMMMM#
#M#.#####F#F#F###F#F###.#M#####
#M#.....FF#FFF#FFF#FFF#.#M#MMM#
#M#####F#####F#F#####F#.#M#M#M#
#M....#.#FFF#F#FFFFFFF#.#M#M#M#
#M###.#.#F#F#F#########.#M#M#M#
#MMM#.#.#F#F#FFF#FFFFF..#MMM#M#
###M#.###F#F#####F#######.###M#
#MMM#....F#FFFFFFF#FF.....#MMM#
#M#########################M###
#M#..MMMMM#..MMT#MMM#MMMMM#MMM#
#M#.#M###M###M#M#M#M#M###M#.#M#
#M#.#M#..MMMMM#M#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

### loop6

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#.FF#TTS#.#.........#
#.#############T#F#.#########.#
#...#......FFF#T#.#...#.....#.#
###.#.#######F#T#F#####.#.###.#
#.....#MMMTT#F#T#FF...#.#.....#
#######M###T###T#####.#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#FF.#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#...FF#F#FFFFF#F#FF.#MMMMM#
#M#.#####F#F#F###F#F###.#M#####
#M#.....FF#FFF#FFF#FFF#.#M#MMM#
#M#####F#####F#F#####F#.#M#M#M#
#M....#F#FFF#F#FFFFFFF#.#M#M#M#
#M###.#.#F#F#F#########.#M#M#M#
#MMM#.#.#F#F#FFF#FFFFF..#MMM#M#
###M#.###F#F#####F#######.###M#
#MMM#....F#FFFFFFF#F......#MMM#
#M#########################M###
#M#..MMMMM#..MMT#MMM#MMMMM#MMM#
#M#.#M###M###M#M#M#M#M###M#.#M#
#M#.#M#..MMMMM#M#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

### loop10

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#..F#TTS#.#.........#
#.#############T#.#.#########.#
#...#.......FF#T#.#...#.....#.#
###.#.#######F#T#.#####.#.###.#
#.....#MMMTT#F#T#FF.F.#.#.....#
#######M###T###T#####.#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#F..#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#...FF#F#FFFFF#F#FF.#MMMMM#
#M#.#####F#F#F###F#F###.#M#####
#M#....FFF#FFF#FFF#FFF#.#M#MMM#
#M#####.#####F#F#####F#.#M#M#M#
#M....#F#FFF#F#FFFFFFF#.#M#M#M#
#M###.#.#F#F#F#########.#M#M#M#
#MMM#.#.#F#F#FFF#FFFFF..#MMM#M#
###M#.###F#F#####F#######.###M#
#MMM#....F#FFFFFFF#F......#MMM#
#M#########################M###
#M#..MMMMM#..MMT#MTM#MMMMM#MMM#
#M#.#M###M###M#M#M#M#M###M#.#M#
#M#.#M#..MMMMM#M#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

### loop12

```text
###############################
#.......#...#.....#.....#.....#
#.#.#####.#.#.###.#.###.#####.#
#.#.......#..F#TTS#.#.........#
#.#############T#.#.#########.#
#...#.......FF#T#.#...#.....#.#
###.#.#######F#T#.#####.#.###.#
#.....#MMMTT#F#T#FF.F.#.#.....#
#######M###T###T#####F#.#######
#MMMMMMM#F#TTTTT#F#FFF#.#.#MMM#
#M#######F#######F#F###.#.#M#M#
#M#.#...FFFFFFFFFF#F#F..#GMM#M#
#M#.#.#####F#######F#F#######M#
#M#.#...FF#F#FFFFF#F#FF.#MMMMM#
#M#.#####F#F#F###F#F###.#M#####
#M#.....FF#FFF#FFF#FFF#.#M#MMM#
#M#####F#####F#F#####F#.#M#M#M#
#M....#.#FFF#F#FFFFFFF#.#M#M#M#
#M###.#.#F#F#F#########.#M#M#M#
#MMM#.#.#F#F#FFF#FFFF...#MMM#M#
###M#.###F#F#####F#######.###M#
#MMM#....F#FFFFFFF#F......#MMM#
#M#########################M###
#M#..MMMMM#..MMM#MMM#MMMMM#MMM#
#M#.#M###M###M#M#M#M#M###M#.#M#
#M#.#M#..MMMMM#M#M#MMM#MMM#.#M#
#M#.#M#########M#M#####M#####M#
#M#.#MMM#...#MMM#M#.#MMM#MMMMM#
#M#####M#.#.#M###M#.#M###M###.#
#MMMMMMM#.#..MMMMM#..MMMMM#...#
###############################
```

## Case 355 (final failure)

Loop gain: `-0.0947`. First loop F1 `0.2090` with 75 false positives and 137 misses. loop12 F1 `0.1143` with 99 false positives and 149 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#...#.....#.#...#.#
#.###.#.#####.#F#######.#.###.#
#.....#.#.....#...#...#...#MMM#
#.#####.#.#######.#.#.#.###M#M#
#.....#.......F.#F#F#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#.#..F#FFF#F#TMM#MMM#M#
#.#####.#F#.###F###F#T#M#.#M#M#
#MMM..#.#.FFFF#FFFFF#T#T#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#..F#.FF#FFF#FFFFF#T#MMM#M#
#.#M#.###F#F#####F#####T#M###M#
#.#M#..F.F#F#FFFFF#FFF#TTT#MMM#
###M#######F#F#####F#F#####M###
#MMM..#.FF.F#FFF#TTT#FFF#.#M#.#
#M#####.#######F#T#T###F#.#M#.#
#MMMMM#...#FFFFF#T#TTT#F..#M#.#
#.###M###.###F###T###T###.#M#.#
#.#MMM..#.....#F#T#F#TMM#.#MMM#
###M###.#######F#T#.###M#.###M#
#MMM#.#.#MMMMMTT#TTM#.#M#...#M#
#M###.#.#M#####T###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

### loop2

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#FFF#.....#.#...#.#
#.###.#.#####F#F#######.#.###.#
#.....#.#..FFF#FF.#...#...#MMM#
#.#####.#.#######.#.#.#.###M#M#
#.....#..FFFFFFF#F#.#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#F#FFF#FFF#F#MMM#MMM#M#
#.#####.#F#F###F###F#M#M#.#M#M#
#MMM..#.#FFFFF#FFFFF#T#M#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#..F#FFF#FFF#FFFFF#M#MMM#M#
#.#M#.###F#F#####F#####M#M###M#
#.#M#..FFF#F#FFFFF#FFF#TMM#MMM#
###M#######F#F#####F#F#####M###
#MMM..#FFFFF#FFF#TTT#FF.#.#M#.#
#M#####F#######F#T#T###.#.#M#.#
#MMMMM#.FF#FFFFF#T#TTT#...#M#.#
#.###M###F###F###T###M###.#M#.#
#.#MMM..#.FFFF#F#T#F#MMM#.#MMM#
###M###.#######F#T#F###M#.###M#
#MMM#.#.#MMMMMMM#TMM#.#M#...#M#
#M###.#.#M#####M###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

### loop4

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#FFF#.....#.#...#.#
#.###.#.#####F#F#######.#.###.#
#.....#.#..FFF#FF.#...#...#MMM#
#.#####.#.#######F#F#.#.###M#M#
#.....#...FFFFFF#F#F#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#F#FFF#FFF#F#TMM#MMM#M#
#.#####.#F#F###F###F#T#M#.#M#M#
#MMM..#.#FFFFF#FFFFF#T#M#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#..F#FFF#FFF#FFFFF#M#MMM#M#
#.#M#.###F#F#####F#####M#M###M#
#.#M#...FF#F#FFFFF#FFF#MMM#MMM#
###M#######F#F#####F#F#####M###
#MMM..#FFFFF#FFF#TTT#FF.#.#M#.#
#M#####F#######F#T#T###.#.#M#.#
#MMMMM#.FF#FFFFF#T#TTT#...#M#.#
#.###M###F###F###T###M###.#M#.#
#.#MMM..#FFFFF#F#T#F#MMM#.#MMM#
###M###.#######F#T#F###M#.###M#
#MMM#.#.#MMMMMMT#TMM#.#M#...#M#
#M###.#.#M#####M###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

### loop6

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#FFF#.....#.#...#.#
#.###.#.#####F#F#######.#.###.#
#.....#.#...FF#FFF#...#...#MMM#
#.#####.#.#######.#.#.#.###M#M#
#.....#...FFFFFF#F#.#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#F#FFF#FFF#F#TMM#MMM#M#
#.#####.#F#F###F###F#T#M#.#M#M#
#MMM..#.#FFFFF#FFFFF#T#M#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#...#FFF#FFF#FFFFF#M#MMM#M#
#.#M#.###F#F#####F#####M#M###M#
#.#M#..FFF#F#FFFFF#FFF#MMM#MMM#
###M#######F#F#####F#F#####M###
#MMM..#FFFFF#FFF#TTT#F..#.#M#.#
#M#####.#######F#T#T###.#.#M#.#
#MMMMM#.FF#FFFFF#T#TTT#...#M#.#
#.###M###F###F###T###M###.#M#.#
#.#MMM..#FFFFF#F#T#F#MMM#.#MMM#
###M###.#######F#T#F###M#.###M#
#MMM#.#.#MMMMMTT#TMM#.#M#...#M#
#M###.#.#M#####M###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

### loop10

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#FFF#.....#.#...#.#
#.###.#.#####F#F#######.#.###.#
#.....#.#...FF#FF.#...#...#MMM#
#.#####.#.#######F#.#.#.###M#M#
#.....#...FFFFFF#F#.#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#F#FFF#FFF#F#TMM#MMM#M#
#.#####.#F#F###F###F#T#M#.#M#M#
#MMM..#.#FFFFF#FFFFF#T#M#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#...#FFF#FFF#FFFFF#M#MMM#M#
#.#M#.###F#F#####F#####M#M###M#
#.#M#..FFF#F#FFFFF#FFF#MMM#MMM#
###M#######F#F#####F#F#####M###
#MMM..#.FFFF#FFF#TTT#FF.#.#M#.#
#M#####.#######F#T#T###.#.#M#.#
#MMMMM#.FF#FFFFF#T#TTT#...#M#.#
#.###M###F###F###T###M###.#M#.#
#.#MMM..#FFFFF#F#T#F#MMM#.#MMM#
###M###.#######F#T#.###M#.###M#
#MMM#.#.#MMMMMMT#TMM#.#M#...#M#
#M###.#.#M#####M###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

### loop12

```text
###############################
#...#.....#.#.....#...#.......#
###.#.###.#.#.###.###.#.#####.#
#...#.#.#...#FFF#.....#.#...#.#
#.###.#.#####F#F#######.#.###.#
#.....#.#...FF#FF.#...#...#MMM#
#.#####.#.#######F#.#.#.###M#M#
#.....#...FFFFFF#F#.#...#MMM#M#
#####.#######F###F#F#####M###M#
#...#...#F#FFF#FFF#F#TMM#MMM#M#
#.#####.#F#F###F###F#T#M#.#M#M#
#MMM..#.#FFFFF#FFFFF#T#M#.#M#M#
#S#M###.#####F#######G#M###M#M#
#.#M#...#FFF#FFF#FFFFF#M#MMM#M#
#.#M#.###F#F#####F#####M#M###M#
#.#M#..FFF#F#FFFFF#FFF#MMM#MMM#
###M#######F#F#####F#F#####M###
#MMM..#.FFFF#FFF#TTT#F..#.#M#.#
#M#####.#######F#T#T###.#.#M#.#
#MMMMM#.FF#FFFFF#T#TTT#...#M#.#
#.###M###F###F###T###M###.#M#.#
#.#MMM..#FFFFF#F#T#F#MMM#.#MMM#
###M###.#######F#T#.###M#.###M#
#MMM#.#.#MMMMMMT#MMM#.#M#...#M#
#M###.#.#M#####M###M#.#M#.###M#
#MMM#...#MMM#.#M..#MMM#M#.#MMM#
#.#M#######M#.#M#####M#M###M###
#.#MMM#MMMMM#.#MMMMM#M#M#MMM#.#
#.###M#M#####.#####M#M#M#M###.#
#...#MMM..........#MMM#MMM....#
###############################
```

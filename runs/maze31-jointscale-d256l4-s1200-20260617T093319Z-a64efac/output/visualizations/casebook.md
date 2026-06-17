# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 349 (final failure)

Loop gain: `0.0264`. First loop F1 `0.3251` with 243 false positives and 85 misses. loop12 F1 `0.3515` with 244 false positives and 77 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMMMMMMMMMMMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#TMM#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#.#TTT#T#FFF#FFF..#MMM..#.#
#M#M#.#####S###F#####F###M#####
#M#MTTTT#F#FFF#F#F#FFF#MMM#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#M#####F###T#####F###########F#
#MTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFFFF......#
###############################
```

### loop2

```text
###############################
#MMMMMMMMMMMMMMTMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMT#TTT#FFFFF#TMM#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#F#TTT#T#FFF#FFF.F#MMM..#.#
#M#M#F#####S###F#####F###M#####
#M#TTTTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFFF.......#
###############################
```

### loop4

```text
###############################
#MMMMMMMMMMMMMMTMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#TMT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#F#TTT#T#FFF#FFF.F#MMM..#.#
#M#T#F#####S###F#####F###M#####
#M#TTTTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFF........#
###############################
```

### loop6

```text
###############################
#MMMMMMMMMMMMMMTMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#TMT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#F#TTT#T#FFF#FFF.F#MMM..#.#
#M#T#F#####S###F#####F###M#####
#M#TTTTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFF........#
###############################
```

### loop10

```text
###############################
#MMMMMMMMMMMMMMTMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMT#TTT#FFFFF#TMT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#F#TTT#T#FFF#FFF.F#MMM..#.#
#M#T#F#####S###F#####F###M#####
#M#MTTTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFF........#
###############################
```

### loop12

```text
###############################
#MMMMMMMMMMMMMMTMMMM#MMM#.....#
#M#################M#M#M#.###.#
#M#MMMMM#TTT#FFFFF#TMT#M#.#...#
#M#M###T#T#T#F###F#####M###.#.#
#M#M#F#TTT#T#FFF#FFF.F#MMM..#.#
#M#M#F#####S###F#####F###M#####
#M#TTTTT#F#FFF#F#F#FFF#TTT#...#
#M#####T#F###F#F#F#F###T###F###
#MTTTT#TTTTT#F#FFF#F#FFG#FFF#.#
#.###T#####T#F###F#F#####F###.#
#.#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#MTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#MTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#M###F#F#F#F#T#F#F###F#####F#F#
#M#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#M###F#F###F#T#F###F#F#F#F#F###
#MTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MMT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####.#
#M#MMM#T#T#FFFFFFF#FFF#F#FFFF.#
#M#M###T#T#F#####F#####F#F#####
#M#M#MMT#T#F#FFF#FFFFFFF#F#F..#
#M#M#M###T#F#F#F#########F###.#
#MMM#MMMMM..#F#FFFFFFF........#
###############################
```

## Case 470 (final failure)

Loop gain: `0.0034`. First loop F1 `0.3678` with 233 false positives and 73 misses. loop12 F1 `0.3711` with 233 false positives and 72 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#......FFFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#F..#.#
#.#####F#F#F#F#####F#F#####.#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#.#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#M#
#T###T#########F#F#F#######F#M#
#TTT#TTG#FFFFFFF#F#FFF#TMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTTT#M#MMM#.#
#######T###F#F#F#T###T#M#####.#
#MTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MMT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MMT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMMFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

### loop2

```text
###############################
#...#.......FFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#F#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFF..#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#F..#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTMM#M#MMM#.#
#######T###F#F#F#T###M#M#####.#
#TTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MTT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMMFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

### loop4

```text
###############################
#...#......FFFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#F#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFF..#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FF.#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTMM#M#MMM#.#
#######T###F#F#F#T###M#M#####.#
#TTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MTT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMTFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

### loop6

```text
###############################
#...#......FFFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#F#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#F..#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTMT#M#MMM#.#
#######T###F#F#F#T###M#M#####.#
#TTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MTT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMTFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

### loop10

```text
###############################
#...#......FFFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#FF.#.#
#.#####F#F#F#F#####F#F#####F#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#F#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#F.F#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTMT#M#MMM#.#
#######T###F#F#F#T###M#M#####.#
#TTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MTT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMTFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

### loop12

```text
###############################
#...#......FFFFFF.....#.......#
#.#.#.#####F#####F#F###.#.###.#
#.#..F#FFF#F#FFFFF#F#FFF#FF.#.#
#.#####F#F#F#F#####F#F#####F#.#
#..FFF#F#F#F#F#FFF#F#FFFFF#F#.#
#.#####F#F#F#F#F#F###F#####F#.#
#.#FFFFF#F#F#FFF#FFFFF#FFF#F#.#
###F#####F#############F#F#F#.#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#.#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTM#
#######F#####F#####F###F#####M#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFM#
#T#T#F#####F#####F###F#F###F#M#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#M#
#T#######T#####F###F#F###F###M#
#T#FFTTTTTFFFF#F#FFF#FFFF.#.#M#
#T###T#########F#F#F#######.#M#
#TTT#TTG#FFFFFFF#F#FFF#MMM#MMM#
#F#T#####F#######F#####M#M#M###
#F#TTTTT#FFF#FFF#TTTMT#M#MMM#.#
#######T###F#F#F#T###M#M#####.#
#TTTTT#T#F#F#F#F#TTT#MMM#MMMMM#
#M###T#T#F#F#F#####T#####M###M#
#MTT#TTT#FFF#TTTFF#T#..MMM#.#M#
###T#F#######T#T#F#T###M###.#M#
#MTT#F#TTTTT#T#T#F#TMMMM#MMM#M#
#M#####T###T#T#T#########M#M#M#
#MMMMMMTFF#TTT#TTTMMMMMMMM#MMM#
###############################
```

## Case 348 (final failure)

Loop gain: `0.0064`. First loop F1 `0.3673` with 236 false positives and 74 misses. loop12 F1 `0.3737` with 232 false positives and 73 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.......FFFF#F.MMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#..FFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#M#
#####F#F#F#####T#######F###F#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#F#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#FF...#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFF..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop2

```text
###############################
#...#.......FFFF#FFMMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#.FFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###.#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#.#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#.....#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#F.F..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop4

```text
###############################
#...#.......FFFF#FFMMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#..FFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###.#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#.#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#.....#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFF..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop6

```text
###############################
#...#......FFFFF#FFMMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#..FFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###.#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#.#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#.....#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFF..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop10

```text
###############################
#...#.......FFFF#FFMMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#..FFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###.#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFF.#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#.#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#MMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTM#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#.....#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#F....#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

### loop12

```text
###############################
#...#.......FFFF#FFMMMMMMMMM#S#
#.###.#.#####F###F#T#######M#M#
#.....#FFFFF#F#FFF#TTT#FFF#M#M#
#.#########F###F#####T#F###M#M#
#...#..FFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTM#.#
#.#F#F#F#F#F###F#F#####F###M###
#.#FFF#F#FFFFF#FFF#FFF#F#F#MTM#
#F#####F#####F#######F#F#F###M#
#FFFFFFF#FFFFF#TTTTTGF#F#FF.#M#
#####F#F#F#####T#######F###.#M#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#M#
#F#####F###F#T#######F#####F#M#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#M#
#F#######F#########T#F###F#.#M#
#FFF#FFF#F#FFFFFFF#T#FFF#FF.#M#
#F#F#F#F#F#F#####F#T###F#####M#
#F#FFF#F#F#FFF#FFF#T#FFF#TMM#M#
#######F#F#F#F#F###T#####M#M#M#
#FFFFFFF#F#F#F#FFF#T#TTTTT#MMM#
#F#######F###F#####T#T#######.#
#.FFFFFFFFFFFF#TTTTT#TTMMMMM#.#
###############T#####F#####M###
#MMTTT#TTTTT#TTT#FFF#.....#MMM#
#M###T#T###T#T#####F###.#####M#
#MMM#TTT#F#TTT#TTT#FFF..#MMMMM#
###M#####F#####T#T#######M###.#
#..MMMMMTTTTTTTT#TTTTMMMMM#...#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0336`. First loop F1 `0.4134` with 244 false positives and 54 misses. loop12 F1 `0.3798` with 242 false positives and 65 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.......FF#FFF#.......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#.#...#
#.#.#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#F###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTM#
#####F#F#F#T#F#######F#T#####M#
#.FFFF#FFF#T#F#TTT#FFF#TTTTM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#M#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#...#MMMMMMM#
###############################
```

### loop2

```text
###############################
#...#.......FF#FFF#F......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#F#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#.FFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TF.#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######T#M###
#FFF#FFF#F#T#FFF#FFFFF#TTT#MMM#
#####F#F#F#T#F#######F#T#####M#
#FFFFF#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MTT#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#F..#MMMMMMM#
###############################
```

### loop4

```text
###############################
#...#.......FF#FFF#F......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#F#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TF.#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######M#M###
#FFF#FFF#F#T#FFF#FFFFF#TTT#MMM#
#####F#F#F#T#F#######F#T#####M#
#FFFFF#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MTT#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMT#TTTTTTTTT#F..#MMMMMMM#
###############################
```

### loop6

```text
###############################
#...#.......FF#FFF#F......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#F#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TF.#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######M#M###
#FFF#FFF#F#T#FFF#FFFFF#TTT#MMM#
#####F#F#F#T#F#######F#T#####M#
#FFFFF#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MTT#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMT#TTTTTTTTT#F..#MMMMMMM#
###############################
```

### loop10

```text
###############################
#...#.......FF#FFF#F......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#F#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TF.#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######M#M###
#FFF#FFF#F#T#FFF#FFFFF#TTT#MMM#
#####F#F#F#T#F#######F#T#####M#
#FFFFF#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MTT#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMM#TTTTTTTTT#F..#MMMMMMM#
###############################
```

### loop12

```text
###############################
#...#.......FF#FFF#F......#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#.#FFF#F#FFF#F#F#FFF#F#...#
#.#.#.#F#F#F#####F#F###F#F###.#
#.#.FF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#..F#FFF#F#F#FFF#FFFFF#FFF#MMM#
###F#F#F#F#F#F#######F#F#F###M#
#.FF#F#F#FFF#F#TTTFF#F#F#FFF#M#
#.###F#F###F###T#T#F#F#####F#M#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#M#
#F#####F###F#T###T###########M#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTM#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TMM#
#####F#F#F#F###T###F#F###T###M#
#FFF#F#F#F#F#TTT#FFF#FFF#TF.#M#
#F###F#F#F###T###F#####F#T###M#
#F#FFFFF#F#TTT#FFF#FFFFF#T#MMM#
#F#F#####F#T###F#F#######M#M###
#FFF#FFF#F#T#FFF#FFFFF#TTT#MMM#
#####F#F#F#T#F#######F#T#####M#
#FFFFF#FFF#T#F#TTT#FFF#TTMMM#M#
#.#########T#F#T#T#F#F#####M#M#
#MMTTTTTTTTT#F#T#TGF#F#TMM#M#M#
#M###########F#T#####F#M#M#M#M#
#M#MTT#TTT#FFF#TTT#F#F#M#MMM#M#
#M#M#T#T#T#######T#F#F#M#####M#
#MMM#MMT#TTTTTTTTT#F..#MMMMMMM#
###############################
```

## Case 129 (final failure)

Loop gain: `0.0041`. First loop F1 `0.3827` with 231 false positives and 69 misses. loop12 F1 `0.3868` with 230 false positives and 68 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#....F#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#...FF#F#FFFFFFF#T#TTTTG#MMM#
#.#.#####F#######F#T#T#####T###
#.#.#FFFFF#FFF#FFF#TTT#TTT#TMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#M#
#F###F#F#F#######F#T###T#F#F#M#
#.FFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#MMM#
#.###F###F#F#####F#F###T#.#M###
#.FFFF#FFFFFFFFF#F#FFF#T#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FFF#M#MMM#M#
#.#F#F#F#############F#M#M#M#M#
#.#.#F#FFFFFFF#F#TTTF.#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#........F..FF#TTTMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#.....#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#..FFF#F#FFFFFFF#T#TTTTG#MMM#
#.#.#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#M#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#TMM#
#.###F###F#F#####F#F###T#F#M###
#.FFFF#FFFFFFFFF#F#FFF#T#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FFF#M#MMM#M#
#.#F#F#F#############.#M#M#M#M#
#.#.#F#FFFFFFF#F#TTT..#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#...........FF#TTMMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#.....#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#..FFF#F#FFFFFFF#T#TTTTG#MMM#
#.#.#####F#######F#T#T#####T###
#.#.#FFFFF#FFF#FFF#TTT#TTT#TTM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#M#
#F###F#F#F#######F#T###T#F#F#M#
#.FFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#TMM#
#.###F###F#F#####F#F###T#F#T###
#.FFFF#FFFFFFFFF#F#FFF#M#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FF.#M#MMM#M#
#.#F#F#F#############.#M#M#M#M#
#.#.#F#FFFFFFF#F#TTT..#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#............F#TTMMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#.....#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#..FFF#F#FFFFFFF#T#TTTTG#MMM#
#.#.#####F#######F#T#T#####T###
#.#.#FFFFF#FFF#FFF#TTT#TTT#TMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#M#
#.FFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#TMM#
#.###F###F#F#####F#F###T#F#T###
#.FFFF#FFFFFFFFF#F#FFF#M#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FFF#M#MMM#M#
#.#F#F#F#############F#M#M#M#M#
#.#.#F#FFFFFFF#F#TTT..#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#...........FF#TTMMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#.....#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#..FFF#F#FFFFFFF#T#TTTTG#TMM#
#.#.#####F#######F#T#T#####T###
#.#.#FFFFF#FFF#FFF#TTT#TTT#TTM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#M#
#F###F#F#F#######F#T###T#F#F#M#
#.FFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#TMM#
#.###F###F#F#####F#F###T#F#T###
#.FFFF#FFFFFFFFF#F#FFF#M#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FF.#M#MMM#M#
#.#F#F#F#############F#M#M#M#M#
#.#.#F#FFFFFFF#F#TTT..#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#...........FF#TTMMMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.....#.....#FFFFFFTMMMMMMMMMM#
#.#####.###F#######T#########M#
#.#..FFF#F#FFFFFFF#T#TTTTG#MMM#
#.#.#####F#######F#T#T#####T###
#.#.#FFFFF#FFF#FFF#TTT#TTT#TTM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
#.#####F###F#F#####T#T#T#####.#
#.#FFF#FFFFF#F#FFTTT#TTT#TTT#.#
#.###F#######F#F#T###F###T#T###
#.#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#M#
#F###F#F#F#######F#T###T#F#F#M#
#.FFFF#F#F#FFFFFSTTT#TTT#FFF#M#
#######F#F###########T###F###M#
#.FF#FFF#F#FFFFFFF#F#TTT#F#TMM#
#.###F###F#F#####F#F###T#F#T###
#.FFFF#FFFFFFFFF#F#FFF#M#.#MMM#
#.###############F#F###M#####M#
#.#FFF#FFFFFFFFFFF#FFF#M#MMM#M#
#.#F#F#F#############F#M#M#M#M#
#.#.#F#FFFFFFF#F#TTT..#M#M#MMM#
#.#.#F###F###F#F#T#T###M#M#####
#...#FFF#FFF#F#TTT#TMMMM#MMMMM#
#######F#####F#T#############M#
#...........FF#TTMMMMMMMMMMMMM#
###############################
```

## Case 226 (final failure)

Loop gain: `0.0049`. First loop F1 `0.3903` with 241 false positives and 62 misses. loop12 F1 `0.3952` with 243 false positives and 60 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#.###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MMM#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTMTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#T###T#F###F#F#F#####F###F#F#
#M#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#FSTTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#..FFFFFF.#.#.....#
###############################
```

### loop2

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#.###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#M###T#F###F#F#F#####F###F#F#
#M#MTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#.STTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#.FFFFFFFF#.#.....#
###############################
```

### loop4

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#F###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MMT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#T###T#F###F#F#F#####F###F#F#
#M#MTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#.STTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#..FFFFFF.#.#.....#
###############################
```

### loop6

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#F###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#T###T#F###F#F#F#####F###F#F#
#M#MTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#.STTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#.FFFFFFFF#.#.....#
###############################
```

### loop10

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#F###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#T###T#F###F#F#F#####F###F#F#
#M#MTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#.STTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#..FFFFFFF#.#.....#
###############################
```

### loop12

```text
###############################
#.#MMM#MMM..#MTT#F#FFFFFFF#...#
#.#M#M#M#M###T#T#F#F#F###F#F#.#
#MMM#MMM#MMTTT#T#FFF#FFF#F#F#.#
#M#############T###F###F#F###.#
#M#.......#TTT#TTT#F#F#F#FFFFF#
#M#.#####F#T#T#F#T#F#F#F#####F#
#M#.#...#F#T#T#F#T#FFF#FFF#FFF#
#M#F###G###T#T###T###F###F#F###
#MMT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#MTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#M###F#####F#T#T#F#####F#####F#
#MTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#.FF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#.#######T#############F#####F#
#MTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#M###T#T###F#F#F#####F###F###F#
#M#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#M#T###T#F###F#F#F#####F###F#F#
#M#MTTTT#FFF#FFF#F#FFFFF#FFF#F#
#M#########F#####F#F###F#F#F#F#
#MMM#.STTTFF#FFF#F#F#FFF#F#F#.#
###M#####T#####F#F###F###F###.#
#MMM#MMM#T#FFFFF#FFF#F#FFFFF#.#
#M###M#M#T#F#######F#F#F###F#.#
#M#MMM#MMT#FFF#FFFFF#F#F#FFF#.#
#M#M#######F#F#F#####F#F#F###.#
#MMM#.......#.FFFFFFFF#.#.....#
###############################
```

## Case 177 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.4016` with 235 false positives and 69 misses. loop12 F1 `0.3992` with 226 false positives and 72 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#..........FFFFFFF#FFF....#
#M#M###.#####F#######F###F###.#
#M#MMM#..F.F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#..FFF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MTMTGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#.FF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#M#F###F#####F###T#F###T#T###F#
#M#...#F#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#..F#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMTTTTTFFF...#MMM#
###############################
```

### loop2

```text
###############################
#MMM#..........FFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#....F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#...FF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MMMMGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMTM#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#..F#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#MTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#F.FFF#FFFFF#TTT#F#TTT#T#FFF#
#M#F###F#####F###T#F###T#T###F#
#M#...#.#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#...#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMTTTTTTFF....#MMM#
###############################
```

### loop4

```text
###############################
#MMM#..........FFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#....F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#...FF#F#F#FFFFFFF#FFFFF#
###M#M#######F#F#F###F###F#####
#MMM#MMMMGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMTM#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#..F#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#MTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FF.FF#FFFFF#TTT#F#TTT#T#FFF#
#M#F###F#####F###T#F###T#T###F#
#M#...#.#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#...#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMTTTTTTFF....#MMM#
###############################
```

### loop6

```text
###############################
#MMM#..........FFFFFFF#F......#
#M#M###.#####F#######F###F###.#
#M#MMM#....F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#...FF#F#F#FFFFFFF#FFFFF#
###M#M#######F#F#F###F###F#####
#MMM#MMMMGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMTM#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#..F#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#M#F###F#####F###T#F###T#T###F#
#M#...#.#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#...#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMTTTTTTFF....#MMM#
###############################
```

### loop10

```text
###############################
#MMM#..........FFFFFFF#FF.....#
#M#M###.#####F#######F###F###.#
#M#MMM#....F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#....F#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MMMMGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMTM#.#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#..F#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#.FFFF#FFFFF#TTT#F#TTT#T#FFF#
#M#F###F#####F###T#F###T#T###F#
#M#...#.#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#...#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMMTTTTTFF....#MMM#
###############################
```

### loop12

```text
###############################
#MMM#..........FFFFFFF#FF.....#
#M#M###.#####F#######F###F###.#
#M#MMM#....F#FFF#FFF#FFFFF#F#.#
#M###M#####F###F###F#######F#.#
#MMM#M#...FF#F#F#FFFFFFF#FFFF.#
###M#M#######F#F#F###F###F#####
#MMM#MMMMGFFFF#F#FFF#F#FFF#FFF#
#M#############F###F#F#F#####F#
#MMMTM#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#..F#T#FFF#T#T#F#F#FFFFFFF#F#F#
#.#F#T#####T#T###F#########F#F#
#.#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#.#############T#F#F#####F#####
#.FFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#MTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#M#T###F###F###########T#####F#
#M#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#M#######F###F#T#####T###T#F###
#M#.FFFF#FFFFF#TTT#F#TTT#T#FFF#
#M#.###F#####F###T#F###T#T###F#
#M#...#.#FFF#FFF#T#FFF#T#T#TTM#
#M#####.#F#F###F#T#F###T#T#T#M#
#M#MMM#..F#F#FFF#T#F#TTT#TTT#M#
#M#M#M#.###F#F###T#F#T#######M#
#M#M#M#...#FFFFF#TTT#TTTTTTT#M#
#M#M#M#############T#######M#M#
#MMM#MMMMMMMMMTTTTTTFF....#MMM#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0175`. First loop F1 `0.4175` with 237 false positives and 56 misses. loop12 F1 `0.4000` with 243 false positives and 60 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#TTTTTTTTT#F#FFF#MMMMM#
#.###.#M#######F#####F#F#T#####
#.....#MTT#FFFFF#FFF#F#F#TTMMM#
#########T#F#####F#F#F#F#####M#
#MMTTTTTTT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTM#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#T#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#M#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#F###F###F#F#F###F#T#######T#M#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#.#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#.#F#F#F###F#F###F#T###F#F#.#.#
#.#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#G.F#TTTTTTT#FFFFF#TTT#FFF#...#
#M###T#####T#########T#F#.###.#
#MMMMM..FF#TTTTTTTTMMM..#.....#
###############################
```

### loop2

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FFF#TTTMM#
#.###.#M#######F#####F#F#T#####
#.....#MMM#.FFFF#FFF#F#F#TTTMM#
#########M#F#####F#F#F#F#####M#
#MMMMMMMMT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#F###F###F#F#F###F#T#######T#M#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#F#F#F#F###F#F###F#T###F#F#.#.#
#F#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#FFF#...#
#T###T#####T#########T#F#.###.#
#MMMMTFFFF#TTTTTTTTMMM..#.....#
###############################
```

### loop4

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FFF#TTTMM#
#.###.#M#######F#####F#F#T#####
#.....#MMM#.FFFF#FFF#F#F#TTTMM#
#########M#F#####F#F#F#F#####M#
#MMMMMMMMT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#M#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#F#F#F#F###F#F###F#T###F#F#.#.#
#F#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#FFF#...#
#T###T#####T#########T#F#F###.#
#MMMMTFFFF#TTTTTTTTMMM..#.....#
###############################
```

### loop6

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FFF#TTTMM#
#.###.#M#######F#####F#F#T#####
#.....#MMM#.FFFF#FFF#F#F#TTTMM#
#########M#F#####F#F#F#F#####M#
#MMMMMMMMT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#F###F###F#F#F###F#T#######T#M#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#F#F#F#F###F#F###F#T###F#F#.#.#
#F#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#FFF#...#
#T###T#####T#########T#F#F###.#
#MMMMTFFFF#TTTTTTTTMMM..#.....#
###############################
```

### loop10

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FFF#TTTMM#
#.###.#M#######F#####F#F#T#####
#.....#MMM#.FFFF#FFF#F#F#TTTMM#
#########M#F#####F#F#F#F#####M#
#MMMMMMMMT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#M#
#F###F###F#F#F###F#T#######T#M#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#F#F#F#F###F#F###F#T###F#F#.#.#
#F#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#FFF#...#
#T###T#####T#########T#F#F###.#
#MMMMTFFFF#TTTTTTTTMMM..#.....#
###############################
```

### loop12

```text
###############################
#......MMM#......MMMMMMMMMMMMM#
#.#####M#M#######T#F#########M#
#.#...#M#MMMTTTTTT#F#FFF#TTTMM#
#.###.#M#######F#####F#F#T#####
#.....#MMM#.FFFF#FFF#F#F#TTTMM#
#########M#F#####F#F#F#F#####M#
#MMMMMMMMT#F#FFFFF#FFF#F#FFF#M#
#M###F#######F#########F#F###M#
#M#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#M#####F#F#F###T#######F#F#T###
#M#FFFFF#FFF#FFT#FFFFFFF#F#TTM#
#M#F###########T#F#######F###M#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#M#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#M#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTM#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFF..#
#F#F#F#F###F#F###F#T###F#F#.#.#
#F#F#F#F#FFF#F#FFF#T#FFFFF#.#.#
###F#####F###F#F###T#######.###
#GFF#TTTTTTT#FFFFF#TTT#FFF#...#
#T###T#####T#########T#F#F###.#
#MMMMTFFFF#TTTTTTTTMMM..#.....#
###############################
```

## Case 91 (final failure)

Loop gain: `0.0121`. First loop F1 `0.3936` with 241 false positives and 61 misses. loop12 F1 `0.4057` with 234 false positives and 59 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#...FFFFFFFFFF#F#FGMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#M#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FF.#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####F#####M#
#M#F#FFFFTTTTTTT#FFFFF#F#MMM#M#
#M#F#F#########T#####F#F#M#M#M#
#M#FFF#TTT#FFFFT#FFF#F#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMT#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTTMMMMMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#...FFFFFFFFFF#F#FGTMM#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######T#
#..F#FFFFF#F#F#F#FFF#FFFFFFF#T#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#.###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#FF.#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#FFF#TTT#FFFFT#FFF#.#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTTMMMMMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#...FFFFFFFFFF#F#FGTMM#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######T#
#...#FFFFF#F#F#F#FFF#FFFFFFF#T#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#.###F###F#F#####F#F#######F#T#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#F..#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#FFF#TTT#FFFFT#FFF#.#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTMMMMMMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#...FFFFFFFFFF#F#FGTMM#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######T#
#...#FFFFF#F#F#F#FFF#FFFFFFF#T#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#.###F###F#F#####F#F#######F#T#
#.FF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#F..#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#FFF#TTT#FFFFT#FFF#.#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTMMMMMMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#...FFFFFFFFFF#F#FGTMM#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#.#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######T#
#...#FFFFF#F#F#F#FFF#FFFFFFF#T#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#.###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#TTT#FFF#TTT#FFF#FFFFF#F..#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#FFF#TTT#FFFFT#FFF#.#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTTMMMMMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.....#...FFFFFFFFFF#F#FGTMM#.#
#.#.#.#.#########F#F#F#F###T#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#TTM#
###.#F#####F###F#####F#######T#
#...#FFFFF#F#F#F#FFF#FFFFFFF#T#
#.#######F#F#F#F#F#F#######F#T#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
#.###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###.#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#.#
###T###F###T#F###F###F#F###M#.#
#MTT#FFF#TTT#FFF#FFFFF#F..#MMM#
#M###F###T#######F#####.#####M#
#M#F#FFFFTTTTTTT#FFFFF#.#MMM#M#
#M#F#F#########T#####F#.#M#M#M#
#M#FFF#TTT#FFFFT#FFF#.#.#M#MMM#
#M###F#T#T#####T#F#F###.#M#####
#MMM#F#T#TTTTT#TSF#F#...#MMM#.#
#.#M###T#####T#####F#.#.###M#.#
#.#MMT#TTT#F#T#FFFFF..#...#MMM#
#.###M###T#F#T###############M#
#...#MMMMM..#TTTTMMMMMMMMMMMMM#
###############################
```

## Case 12 (final failure)

Loop gain: `-0.0053`. First loop F1 `0.4211` with 231 false positives and 55 misses. loop12 F1 `0.4158` with 222 false positives and 59 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#..MMM#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#FFFF.#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#.FF#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#F#####M#
#.#F#T#F#FFT#TTT#T#F#FFF#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#TTM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#TMMMM#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FF.#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFF.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#FF.#
#.#F#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTT..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

### loop2

```text
###############################
#.#..MMT#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#F....#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#..F#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#.#####M#
#.#F#T#F#FFT#TTT#T#F#FF.#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#MMM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#MMMMM#
#F#F#F###########T#F###T#M#####
#F#F#FFFFFFFFFFF#T#FFF#TTM#...#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFF.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#F..#
#.#.#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTM..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

### loop4

```text
###############################
#.#..MMT#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#F.F..#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#.FF#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#.#####M#
#.#F#T#F#FFT#TTT#T#F#FF.#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#MMM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#MMMMM#
#F#F#F###########T#F###T#M#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#...#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFF.F.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#F..#
#.#.#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTT..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

### loop6

```text
###############################
#.#..MMT#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#F....#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#..F#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#.#####M#
#.#F#T#F#FFT#TTT#T#F#FF.#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#MMM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#MMMMM#
#F#F#F###########T#F###T#M#####
#F#F#FFFFFFFFFFF#T#FFF#TTM#...#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFF.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#F..#
#.#F#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTT..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

### loop10

```text
###############################
#.#..MMT#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#F.F..#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#..F#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#.#####M#
#.#F#T#F#FFT#TTT#T#F#FF.#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#MMM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#MMMMM#
#F#F#F###########T#F###T#M#####
#F#F#FFFFFFFFFFF#T#FFF#TTM#...#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFF.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#F..#
#.#.#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTT..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

### loop12

```text
###############################
#.#..MMT#TTT#FFFFTTMMMMMMM#...#
#.#.#M#T#T#T#F###T#######M#.###
#.#.#T#TTT#T#F#TTT#F.F..#M#...#
#.#F#T#####T#F#T###F###.#M###.#
#..F#T#FFF#T#F#TTT#F#F#.#MMMMM#
#.###T#F#F#T#####T#F#F#.#####M#
#.#F#T#F#FFT#TTT#T#F#FF.#MMMMM#
#F#F#T#####T#T#T#T#F#F###M#####
#F#F#TTTTT#TTT#TTT#F#FFF#M#MMM#
#F#F#####T#########F###F#M#M#M#
#FFF#FFF#TTT#FFFFFGT#FFF#MMM#M#
###F#F#F###T#######T#########M#
#F#F#F#FFF#TTTTTTT#TTTTT#MMMMM#
#F#F#F###########T#F###T#M#####
#F#F#FFFFFFFFFFF#T#FFF#TTM#...#
#F#F###F#####F#F#T#F###F#####.#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFF.#
#F###F#####F###F#T###F#######.#
#.FF#F#FFF#F#FFF#TTT#FFFFF#FF.#
###F#F#F#F#F#######T#####F#####
#.FF#F#F#F#FFFFFFF#TFFFF#FFFF.#
#.###F#F#F#######F#T###F#####.#
#..F#FFF#FFFFFFFFF#T#FFFFF#F..#
#.#.#############F#T#######S###
#.#.#FFFFFFFFFFFFF#TTTTTTT#MMM#
#.#.#F###################T###M#
#.#.#.#FFFFFFF#FFFFF#TTTTT..#M#
#.#.#.#####F#F#F###F#T#######M#
#.#........F#FFF#FFF#MMMMMMMMM#
###############################
```

## Case 302 (final failure)

Loop gain: `0.0045`. First loop F1 `0.4178` with 232 false positives and 69 misses. loop12 F1 `0.4223` with 234 false positives and 67 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#M#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTM#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#.#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FF.#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#............F#F#TTTTTTTMMMMMM#
###############################
```

### loop2

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#T#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTT#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#F#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FFF#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#............F#F#TTTTTTTMMMMMM#
###############################
```

### loop4

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#T#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTT#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#F#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FF.#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#.............#F#TTTTTTTMMMMMM#
###############################
```

### loop6

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#M#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTT#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#F#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FFF#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#.............#F#TTTTTTTMMMMMM#
###############################
```

### loop10

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#T#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTT#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#F#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FFF#
#.#######T#T###T#####F#F#T###.#
#...#...#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#.............#F#TTTTTTTMMMMMM#
###############################
```

### loop12

```text
###############################
#...#MMMMMMMMMMMMMMMMMMMMM#MMM#
###G#M#######F###########M#M#M#
#MMM#MMT#FFFFF#FFF#F#TTTTM#M#M#
#M#.###T#######F#F#F#T#####M#M#
#M#.#F#TTT#FFFFF#F#F#TTTTT#M#M#
#M#F#F###S#F#####F#F#####T#T#M#
#M#FFF#FFF#F#FFF#F#FFFFF#TTT#M#
#M#F###F#F#F#F#F#F###F#######M#
#M#F#FFF#F#FFF#F#FFF#F#FFFFF#M#
#M###F#####F#######F#F#F###F#T#
#MTT#FFFFFFF#FFFFFFF#FFFFF#TTT#
###T#########F#######F#####T###
#TTTFF#FFFFF#FFFFF#FFF#FFF#TTT#
#T###F#F###F#####F###F#F#F###T#
#TTT#FFF#FFFFFFF#FFF#F#F#F#F#T#
#F#T###############F###F#F#F#T#
#F#T#FFFFF#FFFFFFF#F#FFF#F#TTT#
#F#T#####F#F#####F#F#F###F#T###
#.#TFFFF#FFF#FFFFF#FFF#FFF#TTT#
###T#F#######F#########F#####T#
#MTT#F#TTTFF#FFFFFFFFF#F#TTTTT#
#M#####T#T###########F#F#T#####
#MMTTTTT#T#TTTTTFFFF#F#F#T#FFF#
#.#######T#T###T#####F#F#T###.#
#...#..F#T#T#F#TTTTT#FFF#TTTMM#
###.#.###T#T#F#####T#########M#
#...#...#TTT#FFF#TTTFFFFFFFF#M#
#.#####.#####F#F#T###########M#
#.............#F#TTTTTTTMMMMMM#
###############################
```

## Case 494 (final failure)

Loop gain: `0.0359`. First loop F1 `0.3896` with 241 false positives and 63 misses. loop12 F1 `0.4254` with 236 false positives and 53 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#F#F#F###F###.#.#.#
#MMM#MMMMM#T#F#F#FFF#F#FFF#.#.#
#M#########T#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#F#FFFFFFF#F#FFF#F#FF.#
#.###M###F#######F###F#F#F#F#F#
#.#MMM#FFFFFFFFF#FFF#F#F#FFF#F#
###M###F###########F#F#F#####F#
#MMT#.#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#M#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#M#F#F#######F###T#T#####F###F#
#M#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#M#######T#T###T###T#F#F###F###
#M#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#M#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#M###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#MFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#M#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###.###F#T###T###T#T###T###.#
#...#...#.GM#FFTTTTT#TTTMM#...#
###############################
```

### loop2

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#.#F#F###F###.#.#.#
#MMM#MMMMM#M#F#F#FFF#F#FFF#F#.#
#M#########M#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#.#FFFFFFF#F#FFF#F#F..#
#.###M###.#######F###F#F#F#F#.#
#.#TMM#.FFFFFFFF#FFF#F#F#FFF#F#
###T###.###########F#F#F#####F#
#MMT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#.#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#T#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###F###F#T###T###T#T###T###.#
#...#...#FGT#FFTTTTT#TTTMM#...#
###############################
```

### loop4

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#.#F#F###F###.#.#.#
#MMM#MMMMM#M#F#F#FFF#F#FFF#.#.#
#M#########M#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#.#FFFFFFF#F#FFF#F#F..#
#.###M###F#######F###F#F#F#F#.#
#.#TMM#.FFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#MMT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#.#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#M#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###F###F#T###T###T#T###T###.#
#...#...#FGT#FFTTTTT#TTTMM#...#
###############################
```

### loop6

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#.#F#F###F###.#.#.#
#MMM#MMMMM#M#F#F#FFF#F#FFF#.#.#
#M#########M#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#.#FFFFFFF#F#FFF#F#F..#
#.###M###F#######F###F#F#F#F#.#
#.#TMM#.FFFFFFFF#FFF#F#F#FFF#F#
###T###.###########F#F#F#####F#
#MMT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#.#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#T#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###F###F#T###T###T#T###T###.#
#...#...#FGT#FFTTTTT#TTTMM#...#
###############################
```

### loop10

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#.#F#F###F###.#.#.#
#MMM#MMMMM#M#F#F#FFF#F#FFF#.#.#
#M#########M#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#.#FFFFFFF#F#FFF#F#F..#
#.###M###F#######F###F#F#F#F#.#
#.#TMM#.FFFFFFFF#FFF#F#F#FFF#F#
###T###.###########F#F#F#####F#
#MMT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#.#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#T#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###F###F#T###T###T#T###T###.#
#...#...#FGT#FFTTTTT#TTTMM#...#
###############################
```

### loop12

```text
###############################
#..MMM#..MMM#...#FFF......#...#
###M#M###M#M#.#F#F###F###.#.#.#
#MMM#MMMMM#M#F#F#FFF#F#FFF#.#.#
#M#########M#F#####F#F#F###F#.#
#M#MMMMM#.#S#FFFFFFF#F#FFF#F#.#
#M#M###M#.#F#######F#F###F###.#
#MMM#MMM#.#FFFFFFF#F#FFF#F#F..#
#.###M###.#######F###F#F#F#F#.#
#.#TMM#.FFFFFFFF#FFF#F#F#FFF#F#
###T###.###########F#F#F#####F#
#MMT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#M###F#F###F#F###########F#F#F#
#M#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#M#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#M#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#.#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#.#
#M#T#############F#####T###F#.#
#MMM#FFFFF#TTTTT#F#TTT#TTT#FF.#
#.###F###F#T###T###T#T###T###.#
#...#...#FGT#FFTTTTT#TTTMM#...#
###############################
```

## Case 3 (final failure)

Loop gain: `-0.0263`. First loop F1 `0.4531` with 216 false positives and 52 misses. loop12 F1 `0.4268` with 213 false positives and 61 misses. Final exact `0.0000`.

### loop1

```text
###############################
#........FFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#TMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###F#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#TMM#.#
#F#####F#F#######F#F#F#F#T###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TTMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TTM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#TMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#F#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTTMMMM#
###F#F###T###F#T#####T#T#######
#.#F#F#TTT#FFF#TTTTT#TTM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#F..#.#.#
#.###.###T###T#T#####T#F#.###.#
#.......#TTTTT#TTTTTMM..#.....#
###############################
```

### loop2

```text
###############################
#........FFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#MMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###.#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#MMM#.#
#F#####F#F#######F#F#F#F#M###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TMMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TMM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#MMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#F#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTMMMMM#
###F#F###T###F#T#####T#M#######
#.#F#F#TTT#FFF#TTTTT#TTM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#...#.#.#
#.###.###T###T#T#####T#.#.###.#
#.......#TTTTT#TTTTMMM..#.....#
###############################
```

### loop4

```text
###############################
#.......FFFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#MMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###.#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#MMM#.#
#F#####F#F#######F#F#F#F#T###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TMMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TMM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#TMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#.#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTMMMMM#
###F#F###T###F#T#####T#M#######
#.#F#F#TTT#FFF#TTTTT#TTM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#...#.#.#
#.###.###T###T#T#####T#.#.###.#
#.......#TTTTT#TTTTMMM..#.....#
###############################
```

### loop6

```text
###############################
#.......FFFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#MMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###.#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#MMM#.#
#F#####F#F#######F#F#F#F#T###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TMMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TMM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#MMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#.#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTMMMMM#
###F#F###T###F#T#####T#M#######
#.#F#F#TTT#FFF#TTTTT#TMM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#...#.#.#
#.###.###T###T#T#####T#.#.###.#
#.......#TTTTT#TTTTMMM..#.....#
###############################
```

### loop10

```text
###############################
#........FFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#MMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###.#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#MMM#.#
#F#####F#F#######F#F#F#F#T###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TMMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TMM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#MMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#.#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTMMMMM#
###F#F###T###F#T#####T#T#######
#.#F#F#TTT#FFF#TTTTT#TMM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#...#.#.#
#.###.###T###T#T#####T#.#.###.#
#.......#TTTTT#TTTTMMM..#.....#
###############################
```

### loop12

```text
###############################
#.......FFFFFFFF#F....#MMMMM#.#
#####.#########F#F###.#M###M#.#
#...FF#FFFFF#F#FFF#F#.#M#..MMM#
#.#######F#F#F#####F#.#M#####M#
#.#F#FFFFF#F#F#FFFFF#F#MMM#.#M#
#.#F#F#F###F#F#F###F#F###M#.#M#
#.#F#F#FFF#FFF#FFF#FFF#MMM#MMM#
###F#F###F#######F#####T###M###
#FFF#F#F#FFFFFFFFF#GTTTT#.#M#.#
#F#F#F#F#############F###.#M#.#
#F#FFF#FFF#FFFFFFFFF#F#F#MMM#.#
#F#####F#F#######F#F#F#F#T###.#
#FFFFF#F#FFFFFFFFF#FFF#F#TMMMM#
#####F#F###############F#####M#
#FFF#FFF#FFFFTTTTT#TTT#FFS#TMM#
#F###########T###T#T#T###T#T###
#F#FFFFFFTTTTT#TTT#T#TTTTT#MMM#
#F#F#####T#####T###T#####F###M#
#F#F#FFF#TTT#FFTTTTT#FFF#F#MMM#
#F#F###F###T#F#######F#F#.#M###
#FFF#FFFFF#T#F#TTT#FFF#FFF#MMM#
#F###F#F###T#F#T#T###########M#
#.FF#F#F#TTT#F#T#TTTTT#TTMMMMM#
###F#F###T###F#T#####T#M#######
#.#F#F#TTT#FFF#TTTTT#TMM#.....#
#.#F#F#T###F#######T#######.#.#
#.#.FF#TTT#F#TTTFF#TTT#...#.#.#
#.###.###T###T#T#####T#.#.###.#
#.......#TTTTT#TTTTMMM..#.....#
###############################
```

## Case 277 (final failure)

Loop gain: `-0.0137`. First loop F1 `0.4488` with 210 false positives and 70 misses. loop12 F1 `0.4351` with 208 false positives and 75 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#MMM......#TTTTTMMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###T#.#.#.#
#MMMFF#TTTTTFFFF#F#F#TTT#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#T###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#T#####F#F#F#######F#####F#M#M#
#T#FFFFF#F#FFFFFFF#FFFFF#F#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#M#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTMM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#TTTTT#T#F#FFF#TTT#GTTTTTTTM#M#
#######T#F#####T###F#######M#M#
#.FFFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#..F#TTTTT#TTT#T#F#FFFFF..#MMM#
#.#.#M#####T#T#T#F#######.#####
#.#..MMMTTTT#TTT#FFF..........#
###############################
```

### loop2

```text
###############################
#.#MMM......#TTTTTMMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMMF.#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#M###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#T#FFFFF#F#FFFFFFF#FFFFF#F#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#T#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#MTTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#..FFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#TTTTT#TTT#T#F#FFFF...#MMM#
#.#.#T#####T#T#T#F#######.#####
#.#..MMMTTTT#TTT#FFF..........#
###############################
```

### loop4

```text
###############################
#.#MMM.....F#TTTTTTMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMMF.#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#M###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#T#FFFFF#F#FFFFFFF#FFFFF#.#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#M#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#MTTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#..FFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#TTTTT#TTT#T#F#FFFFFF.#MMM#
#.#.#T#####T#T#T#F#######.#####
#.#..MMMMTTT#TTT#FFF..........#
###############################
```

### loop6

```text
###############################
#.#MMM.....F#TTTTTTMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMM..#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#M###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#T#FFFFF#F#FFFFFFF#FFFFF#F#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#M#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#MTTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#..FFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#TTTTT#TTT#T#F#FFFFFF.#MMM#
#.#.#T#####T#T#T#F#######.#####
#.#..MMMMTTT#TTT#FFF..........#
###############################
```

### loop10

```text
###############################
#.#MMM.....F#TTTTTTMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMMF.#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#M###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#T#FFFFF#F#FFFFFFF#FFFFF#F#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#M#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#MTTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#..FFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#TTTTT#TTT#T#F#FFFFFF.#MMM#
#.#.#T#####T#T#T#F#######.#####
#.#..MMMMTTT#TTT#FFF..........#
###############################
```

### loop12

```text
###############################
#.#MMM.....F#TTTTTTMMMMM#.....#
#.#M#M###F###T#######.#M#.###.#
#.#M#MMT#F#TTT#F#FFF#F#M#.#.#.#
#.#M###T###T###F#F#F###M#.#.#.#
#MMMF.#TTTTTFFFF#F#F#TTM#...#.#
#M###############F###S#######.#
#M#FFFFFFFFFFF#FFF#FFFFF#MMMMM#
#M#F#########F#F#######F#M###M#
#M#FFFFF#FFF#FFF#FFFFFFF#MMM#M#
#M#####F#F#F#######F#####.#M#M#
#M#FFFFF#F#FFFFFFF#FFFFF#F#M#M#
#T#F#########F###F#F###F#F#M#M#
#T#FFFFFFFFFFF#FFF#FFF#F#F#M#M#
#T#############F#####F#F###T#M#
#TTTTT#FFF#FFF#F#FFF#F#F#TTT#M#
#####T#F###F#F#F#F#F###F#T###M#
#TTT#T#F#FFF#FFF#F#FFFFF#TTM#M#
#T#T#T#F#F#######F#########M#M#
#T#TTT#F#FFF#FFFFF#FFTTTTTTM#M#
#T#####F###F#F#######T#######M#
#T#FFTTT#F#F#F#FFTTTTT#FFFFF#M#
#T###T#T#F#F#F###T#####F#####M#
#MTTTT#T#F#FFF#TTT#GTTTTTTMM#M#
#######T#F#####T###F#######M#M#
#..FFF#TTT#FFFFT#F#F#FFFF.#M#M#
#.#######T#####T#F#F#####.#M#M#
#...#TTTTT#TTT#T#F#FFFFFF.#MMM#
#.#.#T#####T#T#T#F#######.#####
#.#..MMMMTTT#TTT#FFF..........#
###############################
```

## Case 107 (final failure)

Loop gain: `-0.0044`. First loop F1 `0.4427` with 229 false positives and 58 misses. loop12 F1 `0.4384` with 227 false positives and 60 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#F#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTT#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#T#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#F#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#...FFFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#......FFFFFFF#FFF#FFFF...#
###############################
```

### loop2

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#.#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTT#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#M#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#.#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#....FFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#.......FFFFFF#FFF#FFFF...#
###############################
```

### loop4

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#F#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTT#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#M#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#.#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#...FFFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#.......FFFFFF#FFF#FFFF...#
###############################
```

### loop6

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#F#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTM#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#M#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#.#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#...FFFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#......FFFFFFF#FFF#FFFF...#
###############################
```

### loop10

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#.#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTT#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#T#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#.#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#...FFFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#.......FFFFFF#FFF#FFFF...#
###############################
```

### loop12

```text
###############################
#MMMMM#..MMMMM#.....#MMM......#
#M###M###T###S###F#.#M#M#######
#M#.#MTTTT#F#F#FFF#F#M#MMMMMMM#
#M#.#######F#F#F###F#M#######M#
#M#..FFFFFFF#FFF#FFF#MMM#MMM#M#
#M###F#F#####F#########M#M#M#M#
#MMT#F#FFFFFFF#TTTTTTM#MMM#M#M#
###T#F#########T#####T#####M#M#
#MTT#FFF#TTTTTTTFFFF#T#TTTTT#M#
#M###F###T###########T#T#####M#
#MTT#F#TTT#FFFFF#TTT#TTT#TTT#M#
#.#T#F#T###F#####T#T#####T#T#M#
#F#T#F#T#FFF#FFF#T#TTTTTTT#T#T#
#F#T###T#F###F###T#########T#T#
#F#TTT#TTT#FFF#TTT#TTTTT#F#TTT#
#F###T###T#F###T###T###T#F###F#
#F#F#TTTTT#FFF#TTTTT#TTT#FFFFF#
#F#F#########F#####F#T#F#####F#
#FFF#FFFFFFFFF#FFF#F#T#F#FFF#F#
###F#F#########F#F###T###F#F###
#.#F#F#FFFFF#FFF#FFF#TTTTG#FFF#
#.#F#F#F###F#F#####F#F#######F#
#.#F#FFF#F#FFF#FFF#F#FFFFF#FFF#
#.#.#####F#######F#F#######F#F#
#...#FFFFFFFFFFFFF#F#FFFFF#F#F#
#.###########F###F#F#F###F#F###
#.#...FFFFFFFF#F#F#F#F#F#F#FFF#
#.#.###########F#F#F#F#F#F###.#
#...#.......FFFFFF#FFF#FFFF...#
###############################
```

## Case 130 (final failure)

Loop gain: `-0.0031`. First loop F1 `0.4418` with 224 false positives and 54 misses. loop12 F1 `0.4386` with 224 false positives and 55 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#.#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#..M#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TTM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTT#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFF......#MMM#
###############################
```

### loop2

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#.#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#..M#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TMM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTMMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTT#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMMTTTTTTTTTTFF......#MMM#
###############################
```

### loop4

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#..M#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TMM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTT#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFF......#MMM#
###############################
```

### loop6

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#TMM#
###.#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#.FM#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TMM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTM#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTTMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFF......#MMM#
###############################
```

### loop10

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#F#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#..M#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TMM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTM#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTMMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFF......#MMM#
###############################
```

### loop12

```text
###############################
#...#.#.....FFFFFFFF.....MMM#S#
#.#.#.#.#################T#M#M#
#.#...#F#FFFFFFFFFFFFFFF#T#M#M#
#.###.#F###############F#T#M#M#
#...#.#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#..F#F#F#TTTTTTT#FFF#FFF#TTT#.#
#.#####F#T#####T#F###F#####T#.#
#.FFFFFF#TTT#F#T#FFFFF#TTTTT#.#
###########T#F#T#######T#####.#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#.#
#T###########F###########F#.#.#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FF.#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTM#
###F###T#F#F#####F#####F#T###M#
#FFF#TTT#F#F#FFF#F#FFFFF#T#.FM#
#F###T###F###F#F#F#####F#T###M#
#F#F#T#F#FFF#F#F#FFFFFFF#TMM#M#
#F#F#T#F#F#F#F#F#####F#####M#M#
#FFF#T#FFF#FFF#FFFFFFF#TTTMM#M#
#####T#F###############T#####M#
#MTT#T#FFF#FFFFFFFGTTTTM#...#M#
#M#T#T#####F#############.#.#M#
#M#TTT#F#FFF#FFFFFFFFFF...#.#M#
#M###F#F#F###F#F#############M#
#MMM#F#FFF#FFF#F#TTTMMMMMMMM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFF......#MMM#
###############################
```

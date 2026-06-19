# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 449 (final failure)

Loop gain: `-0.0027`. First loop F1 `0.4679` with 269 false positives and 29 misses. loop16 F1 `0.4651` with 269 false positives and 30 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###T#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

### loop2

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###M#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

### loop4

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###M#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

### loop6

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###M#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

### loop10

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###T#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

### loop16

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFFF......#
#M###M#T#T#F#F###F#F#######.#.#
#MMT#TTT#T#F#FFF#FFFFF#FFFFF#.#
###T#####T#F###F#######F#####.#
#MTT#FFF#T#F#F#FFFFFFFFF#FFF..#
#M#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FF.#
#T###F#########F#######F#####F#
#T#FFF#FFFFFFFFF#FFFFFFF#FFFFF#
#T#F###F#####F###F#########F###
#T#FFF#FFFFF#F#FFF#FFFFFFF#FFF#
#T###F#####F#F#F#######F#F#F#F#
#TTT#F#FFF#F#F#FFFFFFFFF#F#F#F#
###T#F#F#F#F#F###########F#F#F#
#F#TTT#F#FFF#F#FFFFF#FFF#F#F#F#
#F###T#F#######F#F###F#F#F#F#F#
#FFF#T#F#FFFFFFF#F#FFF#FFF#F#F#
#F###T#F#F#######F#F#########F#
#F#TTT#FFF#TTTTT#FFF#TTTFF#FFF#
#F#T#######T###T#####T#T#F#F#F#
#TTT#FFTTT#TTT#TTT#F#T#T#FFF#F#
#T###F#T#T###T###T#F#T#T#######
#T#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTM#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFTMMMM#.#
###############################
```

## Case 91 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4706` with 270 false positives and 27 misses. loop16 F1 `0.4706` with 270 false positives and 27 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

### loop16

```text
###############################
#.....#FFFFFFFFFFFFF#F#FGMMM#.#
#.#.#F#F#########F#F#F#F###M#.#
#.#F#F#FFF#FFFFFFF#FFF#FFF#TMM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#M#
#F#######F#F#F#F#F#F#######F#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#M#
#F###F###F#F#####F#F#######F#T#
#FFF#FFF#FFF#FFFFF#F#TTTTT#F#T#
###F###F#####F#####F#T###T###T#
#FFFFF#FFFFFFF#FFFFF#TTT#T#TTT#
#F#######F#####F###F###T#T#T###
#F#TTTTT#FFFFF#FFF#FFF#T#TTT#F#
#F#T###T#####F###F###F#T#####F#
#F#T#F#TTT#FFF#FFF#FFF#TTT#FFF#
#F#T#F###T###F#F###F#####T###F#
#F#T#FFF#TTT#F#FFF#F#FFF#TTT#F#
###T###F###T#F###F###F#F###T#F#
#TTT#FFF#TTT#FFF#FFFFF#FFF#TTT#
#T###F###T#######F#####F#####T#
#T#F#FFFFTTTTTTT#FFFFF#F#TTT#T#
#T#F#F#########T#####F#F#T#T#T#
#T#FFF#TTT#FFFFT#FFF#F#F#T#TTT#
#M###F#T#T#####T#F#F###F#T#####
#MTT#F#T#TTTTT#TSF#F#FFF#TTM#.#
#.#T###T#####T#####F#F#F###M#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###M###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTTMMMMMM#
###############################
```

## Case 130 (final failure)

Loop gain: `-0.0035`. First loop F1 `0.4770` with 267 false positives and 29 misses. loop16 F1 `0.4735` with 268 false positives and 30 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#.FF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMTTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

### loop2

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMTTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

### loop4

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMTTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

### loop6

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

### loop10

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMTTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

### loop16

```text
###############################
#...#.#FFFFFFFFFFFFFFFFF.MMM#S#
#.#.#F#F#################M#M#M#
#.#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#.###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#F#
#F#####F#T#####T#F###F#####T#F#
#FFFFFFF#TTT#F#T#FFFFF#TTTTT#F#
###########T#F#T#######T#####F#
#TTTTTTTTTTT#F#TTTTTTTTT#FFF#F#
#T###########F###########F#F#F#
#TTTTTTT#FFFFFFF#FFF#FFFFF#FFF#
#F#####T#F#####F#F#F#F#########
#FFFFF#T#F#FFFFF#F#FFF#F#TTTTT#
###F###T#F#F#####F#####F#T###T#
#FFF#TTT#F#F#FFF#F#FFFFF#T#FFT#
#F###T###F###F#F#F#####F#T###T#
#F#F#T#F#FFF#F#F#FFFFFFF#TTT#T#
#F#F#T#F#F#F#F#F#####F#####T#T#
#FFF#T#FFF#FFF#FFFFFFF#TTTTT#T#
#####T#F###############T#####T#
#TTT#T#FFF#FFFFFFFGTTTTT#FFF#T#
#M#T#T#####F#############F#.#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#.#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFFF..#MMM#
###############################
```

## Case 348 (final failure)

Loop gain: `-0.0008`. First loop F1 `0.4770` with 267 false positives and 29 misses. loop16 F1 `0.4762` with 268 false positives and 29 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#..FFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####M###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#.FFFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####M###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#.FFFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####M###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#.FFFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####T###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#.FFFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####M###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

### loop16

```text
###############################
#...#..FFFFFFFFF#FFTTTTTMMMM#S#
#.###F#F#####F###F#T#######M#M#
#.FFFF#FFFFF#F#FFF#TTT#FFF#T#M#
#.#########F###F#####T#F###T#M#
#.FF#FFFFF#FFF#FFFFF#T#FFFFTMM#
###F#F###F###F#####F#T#######.#
#F#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
#F#F#F#F#F#F###F#F#####F###T###
#F#FFF#F#FFFFF#FFF#FFF#F#F#TTT#
#F#####F#####F#######F#F#F###T#
#FFFFFFF#FFFFF#TTTTTGF#F#FFF#T#
#####F#F#F#####T#######F###F#T#
#FFFFF#F#FFF#TTT#FFFFF#FFFFF#T#
#F#####F###F#T#######F#####F#T#
#F#FFFFF#FFF#TTTTTTT#FFFFF#F#T#
#F#######F#########T#F###F#F#T#
#FFF#FFF#F#FFFFFFF#T#FFF#FFF#T#
#F#F#F#F#F#F#####F#T###F#####T#
#F#FFF#F#F#FFF#FFF#T#FFF#TTT#T#
#######F#F#F#F#F###T#####T#T#T#
#FFFFFFF#F#F#F#FFF#T#TTTTT#TTT#
#F#######F###F#####T#T#######F#
#FFFFFFFFFFFFF#TTTTT#TTTTTTT#F#
###############T#####F#####M###
#MTTTT#TTTTT#TTT#FFF#FFFFF#MMM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######T###.#
#..MMMMTTTTTTTTT#TTTTTTTMM#...#
###############################
```

## Case 131 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4779` with 268 false positives and 27 misses. loop16 F1 `0.4779` with 268 false positives and 27 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTTTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TMM#M#
#.###############F#F###F###M#M#
#.....FFFFFFFFFFFFFF#FFF..#MMM#
###############################
```

### loop2

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTTTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#MTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FFF..#MMM#
###############################
```

### loop4

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTTTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#MTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FFF..#MMM#
###############################
```

### loop6

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTMTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#MTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FFF..#MMM#
###############################
```

### loop10

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTTTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#MTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#.....FFFFFFFFFFFFFF#FFF..#MMM#
###############################
```

### loop16

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MTTTT#F#TTT#T#FFF#T#T#F#F#TMM#
#####T#F#####T#F###T#T#F#####M#
#MTTTT#FFF#F#T#F#TTT#T#FFFFFFM#
#T#####F#F#F#T###T###T#######M#
#TTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
###########F###T###T###T#T#T#T#
#F#FFFFFFF#F#F#TFF#TTT#TTT#TTT#
#F#F#F#####F#F#T#####T#########
#F#F#FFFFFFF#F#TTTTT#TTT#FFFFF#
#F#F#########F#####T###T#F###F#
#F#FFF#FFFFFFFFF#TTT#F#T#F#FFF#
#F###F###F#####F#T#F#F#T#F#F#F#
#FFFFF#FFF#F#FFF#T#FFF#T#F#F#F#
#F#####F###F#F###T#####T###F#F#
#F#FFFFF#FFF#FFF#TTTTT#T#FFF#F#
#F###F###F#F###F#####T#T#F###F#
#FFF#FFF#F#FFFFF#FFF#T#T#FFF#F#
###F#F#F#F#######F###T#T###F###
#F#F#F#F#FFF#FFF#F#TTT#TTT#FFF#
#F#F###F###F#F#F#F#T#F###T###F#
#F#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTF.#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#.....FFFFFFFFFFFFFF#FFF..#MMM#
###############################
```

## Case 93 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4806` with 268 false positives and 26 misses. loop16 F1 `0.4806` with 268 false positives and 26 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

### loop2

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

### loop4

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

### loop6

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

### loop10

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

### loop16

```text
###############################
#......FFFFFFFFFFFFF#TTTMM....#
#######F###########F#T###M###.#
#.FF#F#FFFFF#FFFFF#F#TTT#TTT#.#
#.#F#F#F###F#F###S#F###T###T###
#F#FFF#FFF#FFF#F#T#F#TTTFF#TTM#
#F###########F#F#T###T#######M#
#F#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
#F#F#F#F###F#F#F#####F#F#F###T#
#F#F#F#F#F#F#F#F#FFF#FFF#F#TTT#
#F#F#F#F#F#F#F#F#F#F#####F#T###
#F#F#FFF#F#F#FFF#F#F#FFF#F#TTT#
#F#F#####F#F###F#F#F#F#F#F###T#
#FFF#FFFFF#FFF#F#F#FFF#FFF#TTT#
#####F#F#F###F###F#########T###
#FFFFF#F#FFF#FFFFF#FFFFFFF#TTT#
#F#####F###F#######F#########T#
#FFF#FFF#F#F#F#TTTTTTT#TTT#TTT#
###F#F###F#F#F#T#####T#T#T#T###
#F#F#F#FFFFF#F#T#TTT#TTT#T#T#F#
#F#F#F#F#####F#T#T#T#####T#T#F#
#FFF#F#FFFFFFF#TTT#T#F#TTT#TTT#
#####F#####F#######T#F#T#F###T#
#GFF#FFFFF#F#TTTTTTTFF#T#F#TTT#
#M#F#####F#F#T#########T#F#T###
#M#FFF#FFF#F#TTTTT#F#TTT#F#MMM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####T###M#
#.#MMMMTTT#TTTTTTTFFFFFF#MMMMM#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.4830` with 265 false positives and 24 misses. loop16 F1 `0.4813` with 267 false positives and 24 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#.#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#F#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#F#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#F#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#F#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#F#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#F#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#F#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#F#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

### loop16

```text
###############################
#...#..FFFFFFF#FFF#FFFFF..#...#
###.#F#######F#F#F#F#F###.#.###
#.#F#F#FFF#F#FFF#F#F#FFF#F#F..#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FST#.#
#F#####F#F#F#F#F#F#####F###T#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTT#
###F#F#F#F#F#F#######F#F#F###T#
#FFF#F#F#FFF#F#TTTFF#F#F#FFF#T#
#F###F#F###F###T#T#F#F#####F#T#
#FFFFF#F#FFF#TTT#T#F#FFFFFFF#T#
#F#####F###F#T###T###########T#
#F#FFF#FFF#F#T#F#TTTTTTTFF#TTT#
#F#F#F###F#F#T#F#######T###T###
#FFF#F#F#F#F#TTT#FFFFF#TTT#TTT#
#####F#F#F#F###T###F#F###T###T#
#FFF#F#F#F#F#TTT#FFF#FFF#TFF#T#
#F###F#F#F###T###F#####F#T###T#
#F#FFFFF#F#TTT#FFF#FFFFF#T#TTT#
#F#F#####F#T###F#F#######T#T###
#FFF#FFF#F#T#FFF#FFFFF#TTT#TTT#
#####F#F#F#T#F#######F#T#####T#
#FFFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#M#
#MTTTTTTTTTT#F#T#TGF#F#TTT#M#M#
#M###########F#T#####F#T#T#M#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#M#T#T#######T#F#F#T#####M#
#MMM#MTT#TTTTTTTTT#FFF#TMMMMMM#
###############################
```

## Case 129 (final failure)

Loop gain: `0.0045`. First loop F1 `0.4804` with 265 false positives and 27 misses. loop16 F1 `0.4850` with 266 false positives and 25 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#M#M#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#FFFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#T#T#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#FFFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#T#M#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#FFFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#T#M#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#FFFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#T#T#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

### loop16

```text
###############################
#.....#FFFFF#FFFFFFTTTTTMMMMMM#
#.#####F###F#######T#########M#
#.#FFFFF#F#FFFFFFF#T#TTTTG#TMM#
#.#F#####F#######F#T#T#####T###
#.#F#FFFFF#FFF#FFF#TTT#TTT#TMM#
#F#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTT#
#F#####F###F#F#####T#T#T#####F#
#F#FFF#FFFFF#F#FFTTT#TTT#TTT#F#
#F###F#######F#F#T###F###T#T###
#F#F#FFFFF#FFF#F#TTT#F#TTT#TTT#
#F#F#####F#F###F###T###T#####T#
#FFF#FFF#FFF#FFFFF#TTTTTFFFF#T#
###F#F#F###################F#T#
#FFF#F#F#FFFFFFFFF#TTTTT#F#F#T#
#F###F#F#F#######F#T###T#F#F#T#
#FFFFF#F#F#FFFFFSTTT#TTT#FFF#T#
#######F#F###########T###F###T#
#FFF#FFF#F#FFFFFFF#F#TTT#F#TTT#
#F###F###F#F#####F#F###T#F#T###
#FFFFF#FFFFFFFFF#F#FFF#T#F#TTT#
#F###############F#F###T#####T#
#F#FFF#FFFFFFFFFFF#FFF#T#TTT#T#
#.#F#F#F#############F#T#T#T#T#
#.#F#F#FFFFFFF#F#TTTFF#T#T#MMM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTMMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTTMMMMMM#
###############################
```

## Case 285 (final failure)

Loop gain: `0.0009`. First loop F1 `0.4851` with 264 false positives and 29 misses. loop16 F1 `0.4859` with 263 false positives and 29 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#.FF#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

### loop2

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#..F#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

### loop4

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#..F#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

### loop6

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#..F#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

### loop10

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#..F#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

### loop16

```text
###############################
#.#....FFFFF#F#TTTTTTTTT#MMM..#
#.#.#F#F#####F#T#######T#M#M#.#
#..F#F#F#FFFFF#T#FFFFF#TTT#T#.#
#####F###F#####T###F#F#####T###
#FFFFF#FFF#FFFFT#FFF#FFFFF#TTM#
#F#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFM#
#F#F#####F###F#####F###F###F#T#
#F#F#FFFFF#F#F#FFF#F#FFF#FFF#T#
#F#F#F#####F#F#F#F#F###F#F###T#
#FFF#F#FFF#FFF#F#F#FFF#FFF#TTT#
#F###F###F#F###F#F###F#####T###
#FFFFF#FFF#FFFFF#FFFFF#FFF#T#F#
#######F#F#############F#F#T#F#
#FFFFFFF#FFFFF#TTTTTTTTT#F#TTT#
#F#F#######F###T#######T#####T#
#F#F#FFF#FFF#TTT#FFGTT#T#TTTTT#
#F#F#F#F#F###T#######T#T#T#####
#F#F#F#F#F#TTT#TTTTTTT#T#TTT#F#
###F#F###F#T###T#####F#T###T#F#
#FFF#FFFFF#T#FFT#FFFFF#TFF#TTT#
#F###F#####T###T#F#####T#F###T#
#F#FFFFF#TTT#TTT#F#FFF#T#FFF#T#
#.#######T###T###F#F#F#T###F#M#
#MMTTTTTTT#TTT#F#FFF#F#T#FF.#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###T#M#M#
#MMM#..TTTTT#FFFFFFF#TTTMM#MMM#
###############################
```

## Case 422 (final failure)

Loop gain: `0.0054`. First loop F1 `0.4831` with 264 false positives and 27 misses. loop16 F1 `0.4885` with 264 false positives and 25 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTMMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTMM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTMMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTMM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTMMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTMM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTTMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTMM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTTMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTTM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop16

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTTTMMM#.#
#M#####T#####F#T###########M#.#
#M#FFFFTTTTT#F#TTT#FFFFFFF#T#.#
#M#########T#####T#######F#T#.#
#MTTTTSF#TTT#TTTTT#FFFFFFF#TTM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
#F#F#F#####T#T#F#####T###T#F#T#
#F#F#F#FFF#T#T#FFTTTTT#TTT#F#T#
#F#F#F#F###T#T###T#####T#####T#
#FFF#F#FFF#T#TTT#TTTTT#TTTTTTT#
#####F#F#F#T###T#####T#########
#FFFFF#F#FFTTTTT#FFFFTTT#FFTTT#
#F###############F#####T###T#T#
#FFFFFFFFF#FFFFF#F#FFF#TTTTT#T#
#####F###F#F###F###F#F#######T#
#FFF#FFF#F#FFF#FFF#F#F#TTTTTTT#
#F#F###F#F###F###F#F###T#######
#F#FFF#F#FFF#F#F#FFF#F#T#FFFFF#
#F###F###F###F#F#####F#T#F#####
#FFF#F#FFF#FFF#FFFFF#TTT#FFFFF#
#F#F#F#F#F#F#######F#T###F###F#
#F#F#FFF#F#FFF#FFF#TTT#FFFFF#F#
#.#F#####F###F#F#F#T#########.#
#.#F#FFF#FFF#FFF#F#TTTTTTTTM#.#
#.#F#F#F#########F#########M#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#F###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

## Case 94 (final failure)

Loop gain: `-0.0053`. First loop F1 `0.4947` with 260 false positives and 26 misses. loop16 F1 `0.4894` with 260 false positives and 28 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFF..#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#TTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#T#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

### loop2

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFF..#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#MTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#M#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

### loop4

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFF..#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#MTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#M#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

### loop6

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFF..#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#TTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#M#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

### loop10

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFFF.#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#MTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#M#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

### loop16

```text
###############################
#MMM#MMT#FFTTT#FFFFFFFFF..#...#
#M#M#T#T###T#T#F#F#######.###.#
#M#MTT#TTTTT#T#F#FFF#FFFFFFF#.#
#M###########T#F###F#######F#.#
#M#F#TTTTTTTTT#FFF#F#FFFFF#F#.#
#M#F#T###########F#F#F###F#F#.#
#T#FFTTTTTTTTTTS#F#FFF#FFF#FF.#
#T#############F#######F#####F#
#TTT#TTT#TTTTT#F#FFFFFFF#FFF#F#
#F#T#T#T#T###T#F#F#######F#F#F#
#F#T#T#TTT#TTT#F#FFF#FFFFF#F#F#
#F#T#T#####T###F###F#F#####F#F#
#F#TTT#TTTTT#F#FFFFF#FFFFF#F#F#
#######T#####F#######F###F###F#
#TTTTTTT#FFFFF#FFFFF#F#F#FFFFF#
#T#######F###F#F#F###F#F#######
#T#FFFFF#FFF#F#F#FFFFFFFFFFFFF#
#T###F#F###F#F#F#############F#
#TTT#F#FFFFF#F#F#FFF#FFFFF#FFF#
#F#T#F#####F###F###F#F###F#####
#F#T#F#FFFFF#FFFFFFF#F#FFF#TTT#
###T#F#######F#######F#F###T#T#
#TTT#FFFFFFF#F#FFFFF#F#FGTTT#T#
#M###F#####F#F#F###F#F#######M#
#M#FFFFFFF#FFF#F#F#FFF#F#TTM#M#
#M#####F#########F#####F#T#M#M#
#MMTTT#F#TTTTT#TTTTTTT#TTT#MMM#
#.###M###T###T#T#####T#T#######
#...#MMTTTFF#TTTFFFF#TTT......#
###############################
```

## Case 312 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4902` with 263 false positives and 24 misses. loop16 F1 `0.4902` with 263 false positives and 24 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

### loop2

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

### loop4

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

### loop6

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

### loop10

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

### loop16

```text
###############################
#......FFF#FFFFFFFFF#FFTMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#.#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#.#
###F#######F#F#####T#####F#T#.#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#M#
#M#TTT#TTT#FFFFFFF#F#F#FFF#.#M#
#M#T###T###########F#F#F###.#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#.###T#T#T#T#T#####T#T###M#
#MMM..FF#TTT#TTT#TTTTTTT#MMMMM#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4911` with 263 false positives and 23 misses. loop16 F1 `0.4911` with 263 false positives and 23 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#M#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

### loop16

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#F#M#
#M#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#M#
#F###T#####T#T#F#F#########F#M#
#F#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
#F#F###T#T###T#F###########F#T#
#F#F#TTT#TTTTT#F#FFFFFFFFF#FFS#
#F#F#T#F#######F###F#F###F###F#
#FFF#T#F#FFF#F#FFF#F#F#F#F#F#F#
###F#T###F#F#F###F###F#F#F#F#F#
#FFF#T#FFF#F#FFFFFFFFF#FFF#F#F#
#####T#F###F#F#############F#F#
#TTTTT#F#F#F#FFF#FFFFFFFFF#F#F#
#T###F#F#F#F#####F#######F#F###
#T#FFF#FFF#F#FFFFF#F#FFFFF#FFF#
#T#####F###F#F#####F#F###F###F#
#TFFFFFF#FFF#F#FFTTT#F#FFF#FFF#
#T#######F###F###T#T#F###F#F###
#T#FFFFFFF#FFF#TTT#T#FFF#F#FFF#
#T#F###F###F###T###T###F#####F#
#T#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#.#
#MTT#FFFFF#FFT#FFF#T#T#FFF#.#.#
###T#######F#T###F#T#T#F###.#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#F......#
###############################
```

## Case 177 (final failure)

Loop gain: `0.0053`. First loop F1 `0.4860` with 262 false positives and 32 misses. loop16 F1 `0.4913` with 262 false positives and 30 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#M#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MTTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

### loop2

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#TTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#T#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MMTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

### loop4

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#TTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#T#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MMTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

### loop6

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#MTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#T#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MMTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

### loop10

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#TTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#T#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MMTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

### loop16

```text
###############################
#MMM#..FFFFFFFFFFFFFFF#F......#
#M#M###F#####F#######F###.###.#
#M#TTT#FFFFF#FFF#FFF#FFFFF#F#.#
#M###T#####F###F###F#######F#.#
#MTT#T#FFFFF#F#F#FFFFFFF#FFFF.#
###T#T#######F#F#F###F###F#####
#TTT#TTTTGFFFF#F#FFF#F#FFF#FFF#
#T#############F###F#F#F#####F#
#TTTTT#F#FFTTT#FFF#F#F#F#FFFFF#
#####T#F#F#T#T#F#F#F###F#F###F#
#FFF#T#FFF#T#T#F#F#FFFFFFF#F#F#
#F#F#T#####T#T###F#########F#F#
#F#F#TTTTTTT#TTT#F#FFFFFFF#FFF#
#F#############T#F#F#####F#####
#FFFFFFFFFFFFF#T#FFFFFFF#FFFFF#
#####F###F###F#T#########F###F#
#TTT#F#FFF#FFF#TTTTTTTTT#FFF#F#
#T#T###F###F###########T#####F#
#T#TTSFF#FFF#F#TTTTTTT#TTT#FFF#
#T#######F###F#T#####T###T#F###
#T#FFFFF#FFFFF#TTT#F#TTT#T#FFF#
#T#F###F#####F###T#F###T#T###F#
#T#FFF#F#FFF#FFF#T#FFF#T#T#TTT#
#M#####F#F#F###F#T#F###T#T#T#M#
#M#TTT#FFF#F#FFF#T#F#TTT#TTM#M#
#M#T#T#F###F#F###T#F#T#######M#
#M#T#T#FFF#FFFFF#TTT#TTTTTTM#M#
#M#M#M#############T#######M#M#
#MMM#MTTTTTTTTTTTTTTFFFF..#MMM#
###############################
```

## Case 417 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.4938` with 261 false positives and 24 misses. loop16 F1 `0.4929` with 262 false positives and 24 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####F###F#T#####T#T#T#T#M#M#M#
#...FF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

### loop2

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####F###F#T#####T#T#T#T#M#M#M#
#..FFF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

### loop4

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####F###F#T#####T#T#T#T#M#M#M#
#..FFF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

### loop6

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####F###F#T#####T#T#T#T#M#M#M#
#..FFF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

### loop10

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####.###F#T#####T#T#T#T#M#M#M#
#..FFF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

### loop16

```text
###############################
#......F#FFTTTTTTT#TTT#TMM#MMM#
#####F###F#T#####T#T#T#T#M#M#M#
#..FFF#FFF#TTT#FFTTT#TTT#TTT#M#
#.#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#M#
#F#F#F###F#F###T#F#F#########M#
#F#F#F#FFF#F#F#T#F#F#TTTTT#TTT#
#F#F#F#F#F#F#F#S###F#T###T#T###
#F#F#F#F#F#FFF#FFF#F#TTT#TTT#F#
#F###F###F#######F#F###T#####F#
#FFF#FFFFF#FFFFFFF#F#TTT#FFFFF#
#F#F#####F#F#######F#T###F#####
#F#FFFFFFF#F#FFF#TTTTT#FFFFFFF#
#F#########F#F#F#T###########F#
#F#FFFFFFFFF#F#F#TTT#TTT#TTTTT#
###F#########F#F###T#T#T#T###T#
#FFF#FFFFFFFFF#FFF#TTT#TTT#F#T#
#F#####F#####F###F#########F#T#
#FFFFF#FFF#F#F#FFFFFFFFF#TTTTT#
#####F#####F###F#######F#T#####
#FFF#FFFFFFF#FFF#TTTTT#F#TTTTT#
#F###########F#F#T###T#######T#
#F#FGTTTTTTTTT#F#T#F#TTTFF#TTT#
#.#F#########T###T#F###T#F#T###
#.#FFF#FFFFF#T#TTT#FFF#T#F#MMM#
#.###F#F###F#T#T#####F#T#F###M#
#..FFF#FFF#F#TTT#FFF#F#T#F#MMM#
#.#######F#F#####F#F#F#T###M###
#.....FFFF#FFFFFFF#FFF#TMMMM..#
###############################
```

## Case 470 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4938` with 262 false positives and 23 misses. loop16 F1 `0.4938` with 262 false positives and 23 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#F#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#F#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

### loop16

```text
###############################
#...#..FFFFFFFFFFFFFFF#F......#
#.#.#F#####F#####F#F###F#.###.#
#.#FFF#FFF#F#FFFFF#F#FFF#FFF#.#
#.#####F#F#F#F#####F#F#####F#.#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#.#
#F#####F#F#F#F#F#F###F#####F#.#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#TTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###M#
#MTT#TTT#FFF#TTTFF#T#FFTTT#.#M#
###T#F#######T#T#F#T###T###.#M#
#MMT#F#TTTTT#T#T#F#TTTTT#TTM#M#
#M#####T###T#T#T#########T#M#M#
#MMMMMMTFF#TTT#TTTTTTTTTMM#MMM#
###############################
```

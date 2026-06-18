# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 449 (final failure)

Loop gain: `-0.0036`. First loop F1 `0.4627` with 262 false positives and 33 misses. loop16 F1 `0.4590` with 263 false positives and 34 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###T#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
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
#M#FFF#T#TTTTT#TTT#TTT#TTT#TTT#
#M###F#T#######T###T#####T#T#M#
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

### loop2

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###T#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

### loop4

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###T#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

### loop6

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###T#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

### loop10

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###T#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

### loop16

```text
###############################
#MMMMM#TTT#FFF#FFF#FFFF.......#
#M###M#T#T#F#F###F#F#######.#.#
#MMM#TTT#T#F#FFF#FFFFF#FF...#.#
###M#####T#F###F#######F#####.#
#MMT#FFF#T#F#F#FFFFFFFFF#F....#
#M#####F#T###F###########F#####
#M#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#MTT#F#TTTTTTT#T#FFTSFFF#TTT#M#
###T#F#######T#T#########F###M#
#.#T#FFFFF#TTT#TTTTTTTTT#F#MMM#
#.#M#######T###########T###M#.#
#..MMMMTTTTT#FFFFFFFFFFMMMMM#.#
###############################
```

## Case 131 (final failure)

Loop gain: `-0.0036`. First loop F1 `0.4665` with 262 false positives and 33 misses. loop16 F1 `0.4629` with 263 false positives and 34 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#.#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
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
#.#F#FFF#FFF#F#F#F#T#FFF#TTTTT#
#.#F#F###F###F#F#F#T#########M#
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#.FFFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

### loop2

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#.#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#M#
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
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

### loop4

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#.#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#T#
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
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

### loop6

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#.#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#M#
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
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

### loop10

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#.#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#M#
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
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

### loop16

```text
###############################
#MMMMMMTTT#TTTFF#F#TTT#.#..MMG#
#M#######T#T#T#F#F#T#T#F#.#M###
#MMMTT#F#TTT#T#FFF#T#T#F#F#MMM#
#####T#F#####T#F###T#T#F#####M#
#MMTTT#FFF#F#T#F#TTT#T#FFF...M#
#M#####F#F#F#T###T###T#######M#
#MTTTTSF#FFF#TTT#TTT#TTT#TTT#M#
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
#.#FFF#F#FFF#F#F#F#TTTTTTTFF#M#
#.#####F###F#F#F#F#######T###M#
#..FFFFFFF#FFF#FFF#FFFFF#TTM#M#
#.###############F#F###F###M#M#
#......FFFFFFFFFFFFF#FF...#MMM#
###############################
```

## Case 478 (final failure)

Loop gain: `-0.0055`. First loop F1 `0.4712` with 260 false positives and 34 misses. loop16 F1 `0.4657` with 260 false positives and 36 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#MMTTTTTFFFF#TTT#TTMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMTT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#F#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#T#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

### loop2

```text
###############################
#...#MMTTTTTFFFF#TTT#TTTMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#F#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#M#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

### loop4

```text
###############################
#...#MMTTTTTFFFF#TTT#TTMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#M#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

### loop6

```text
###############################
#...#MMTTTTTFFFF#TTT#TTTMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#F#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#T#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

### loop10

```text
###############################
#...#MMTTTTTFFFF#TTT#TTTMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#M#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

### loop16

```text
###############################
#...#MMTTTTTFFFF#TTT#TTMMM....#
#.###M#####T#####T#T#T###M###.#
#MMMMT#FFF#TTTTTTT#TTTFF#MMM#.#
#M#######F#################M#.#
#MMTTTTT#FFFFFFFFFFFFF#TTTMM#.#
#######T#####F#F#####F#T#####.#
#.FFFF#TTTTT#F#F#FFF#F#T#FFF#F#
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
#FFFFF#TTTTTTT#FFF#F#FFF#F#FFF#
###F#F#T#########F#F###F#F###F#
#FFF#F#T#FFF#FFF#F#F#FFFFF#FFF#
#F#####T#F###F#F#F#F#F#####F###
#F#TTTTT#FFF#F#FFF#FFF#FFFFF#F#
#.#T#######F#F#########F#####.#
#.#T#TTTTT#F#FFFFFFFFFFF#TTT#.#
#.#T#T###T#F#############T#T#.#
#MMT#T#F#T#TTT#FFFFFFFFFFT#M#.#
#M###T#F#T#T#T###########M#M#.#
#MMMMM#FFTTT#TTTTTTTTTTMMM#MMS#
###############################
```

## Case 93 (final failure)

Loop gain: `-0.0028`. First loop F1 `0.4702` with 261 false positives and 32 misses. loop16 F1 `0.4674` with 261 false positives and 33 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTT..#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

### loop2

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFM#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTT..#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

### loop4

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFM#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

### loop6

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFM#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

### loop10

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFT#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTTF.#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

### loop16

```text
###############################
#......FFFFFFFFFFFFF#TTMMM....#
#######F###########F#T###M###.#
#...#F#FFFFF#FFFFF#F#TTT#MMM#.#
#.#F#F#F###F#F###S#F###T###M###
#.#FFF#FFF#FFF#F#T#F#TTTFF#MMM#
#.###########F#F#T###T#######M#
#.#FFF#FFFFF#F#F#TTTTT#F#FFFFM#
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
#M#FFF#FFF#F#TTTTT#F#TTT#F#TTM#
#M###F#F#########T#F#T#######M#
#MMT#FFF#TTTFFFF#T#F#TTTTT..#M#
#.#M#####T#T#####T#F#####M###M#
#.#MMMMTTT#TTTTTTTFFFFF.#MMMMM#
###############################
```

## Case 72 (final failure)

Loop gain: `-0.0055`. First loop F1 `0.4740` with 260 false positives and 33 misses. loop16 F1 `0.4685` with 260 false positives and 35 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTT#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MTT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

### loop2

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTT#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MMT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

### loop4

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MMT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

### loop6

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MMT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

### loop10

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MMT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

### loop16

```text
###############################
#MMMMMMTTTTTTT#TTT#TTTTMMMMM#.#
#M###########T#T#T#T#######M#.#
#MMM#FFF#FFFFTTT#TTT#F#TTMMM#.#
###T#F#F#F###########F#T#####.#
#MMT#F#FFF#F#FFFFFFF#F#TTTMM#.#
#M###F#####F#F#####F#F#####M#.#
#M#F#F#FFFFF#F#FFF#FFF#TTT#TTM#
#T#F#F###F#F#F#F#F#####T#T###T#
#T#FFFFF#F#F#FFF#FFF#TTT#TTTTT#
#T#####F#F#########F#T#########
#TTTTT#FFF#TGFFF#FFF#TTT#FFFFF#
#####T###F#T#F###F#####T#F#F###
#FFF#TTT#F#T#FFFFF#FFF#T#F#FFF#
#F#####T###T#######F#F#T#####F#
#F#TTT#TTTTT#TTTTTTT#F#T#FFF#F#
#F#T#T#######T#####T###T#F#F#F#
#TTT#TTTTTTTTT#F#TTT#TTT#F#FFF#
#T#############F#T###T###F###F#
#T#FFF#FFFFFFFFF#TTTTT#FFF#FFF#
#T###F#F#F#############F###F###
#TTT#FFF#FFFFFFFFFFFFF#FFF#FFF#
###T#####F#######F#F###F#F###F#
#F#T#FFF#FFFFF#FFF#F#FFF#F#FFF#
#.#T#F#F#######F#####F###F#####
#.#T#F#FFFFFFF#F#FFFFFFF#FFFF.#
#.#T#F#######F#F#F#######F###.#
#MMT#FFFFF#F#F#FFF#FFFFF#F#...#
#M#######F#F#F#####F###F###.#.#
#MMMMMSFFF#FFFFFFFFF#FF.....#.#
###############################
```

## Case 130 (final failure)

Loop gain: `0.0025`. First loop F1 `0.4668` with 263 false positives and 34 misses. loop16 F1 `0.4693` with 260 false positives and 34 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#.FF#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop2

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTT#.#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop4

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#FFF#F#F#TTTTTTT#FFF#FFF#TTT#.#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop6

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTT#.#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop10

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTT#.#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

### loop16

```text
###############################
#...#.#FFFFFFFFFFFFFFFF..MMM#S#
#.#.#.#F#################M#M#M#
#.#..F#F#FFFFFFFFFFFFFFF#M#M#M#
#.###F#F###############F#T#M#M#
#..F#F#FFFFFFFFFFFFFFFFF#T#MMM#
###F#F#F###############F#T###.#
#.FF#F#F#TTTTTTT#FFF#FFF#TTT#.#
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
#M#T#T#####F#############F#F#M#
#M#TTT#F#FFF#FFFFFFFFFFFFF#F#M#
#M###F#F#F###F#F#############M#
#MMT#F#FFF#FFF#F#TTTTTTTTTTM#M#
#.#M#######F#####T#########M#M#
#.#MMMMTTTTTTTTTTTFFFFF...#MMM#
###############################
```

## Case 285 (final failure)

Loop gain: `0.0036`. First loop F1 `0.4676` with 259 false positives and 37 misses. loop16 F1 `0.4712` with 258 false positives and 36 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TMT#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

### loop2

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#FFT#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

### loop4

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#FFT#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

### loop6

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#FFT#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

### loop10

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
#F#FFFFF#FFFFF#TTS#FFFFF#F#FFT#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TMM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

### loop16

```text
###############################
#.#....FFFFF#F#TTTTTTTTM#MMM..#
#.#.#.#F#####F#T#######T#M#M#.#
#...#F#F#FFFFF#T#FFFFF#TMM#M#.#
#####F###F#####T###F#F#####M###
#..FFF#FFF#FFFFT#FFF#FFFFF#MMM#
#.#####F#######T#F#######F###M#
#.#FFFFF#FFFFF#TTS#FFFFF#F#FFT#
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
#MTTTTTTTT#TTT#F#FFF#F#T#FFF#M#
#M#########T###F###F###T#####M#
#M#TTTTT#FFT#F#FFFFF#TTT#TTM#M#
#M#M###T###T#F#F#####T###M#M#M#
#MMM#..TTTTT#FFFFFFF#TTMMM#MMM#
###############################
```

## Case 312 (final failure)

Loop gain: `-0.0028`. First loop F1 `0.4757` with 261 false positives and 30 misses. loop16 F1 `0.4729` with 261 false positives and 31 misses. Final exact `0.0000`.

### loop1

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#.#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#T#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

### loop2

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#F#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

### loop4

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#F#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

### loop6

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#F#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

### loop10

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#F#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

### loop16

```text
###############################
#......FFF#FFFFFFFFF#FFMMMMM#.#
#.#####F#F###F#####F#F#T###M#.#
#.#.FFFF#FFFFF#F#FFF#F#T#.#M#.#
#.#F###########F#F#####T#F#M#.#
#.#F#FFFFFFF#FFFFF#TTTTT#F#M#.#
###F#######F#F#####T#####F#M#.#
#.FF#FFFFF#FFF#FFFFTTTTT#F#TTS#
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
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#M#
#M#T###T###########F#F#F###F#M#
#M#T#F#TTT#TTT#TTT#FFF#TTT#..M#
#M#M#F###T#T#T#T#T#####T#M###M#
#MMM...F#TTT#TTT#TTTTTTM#MMMMM#
###############################
```

## Case 348 (final failure)

Loop gain: `0.0000`. First loop F1 `0.4748` with 260 false positives and 32 misses. loop16 F1 `0.4748` with 260 false positives and 32 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MTT#TTT#F#TTT#TTT#FFFFF#TMMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#.#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#.#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

### loop16

```text
###############################
#...#..FFFFFFFFF#FFTTTTMMMMM#S#
#.###.#F#####F###F#T#######M#M#
#...FF#FFFFF#F#FFF#TTT#FF.#M#M#
#.#########F###F#####T#F###M#M#
#..F#FFFFF#FFF#FFFFF#T#FFF.MMM#
###F#F###F###F#####F#T#######.#
#.#F#F#F#F#F#FFF#FFF#TTTTTTT#F#
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
#MTTTT#TTTTT#TTT#FFF#FFFFF#TTM#
#M###T#T###T#T#####F###F#####M#
#MMT#TTT#F#TTT#TTT#FFFFF#TTMMM#
###M#####F#####T#T#######M###.#
#..MMMMTTTTTTTTT#TTTTTTMMM#...#
###############################
```

## Case 123 (final failure)

Loop gain: `0.0063`. First loop F1 `0.4712` with 261 false positives and 33 misses. loop16 F1 `0.4776` with 260 false positives and 31 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#M#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#F#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#.#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#.#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTTMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#.#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTMMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#.#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

### loop16

```text
###############################
#MMMMMMT#FFTTT#FFFFFFTTTMM#MMM#
#M#####T#F#T#T#######T###M#M#M#
#M#.FF#T#F#T#TTTTTTTTT#F#M#M#M#
#M#F#F#T###T###########F#T#M#M#
#M#F#FFTTTTT#FFFFF#TTTTT#TMM#M#
#M#############F###T###T#####M#
#M#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
#T#F#########F#F#F###T#####T#T#
#TFF#FFF#F#FFF#F#FFF#TTTTT#TTT#
#T###F#F#F#F###F###F#####T#####
#TTTTT#FFF#FFF#FFF#FFF#F#T#TTT#
#####T#####F#F#F###F#F#F#T#T#T#
#F#TTT#FFF#F#F#F#FFF#F#F#T#T#T#
#F#T###F#F###F###F###F#F#T#T#T#
#TTT#F#F#F#FFF#FFF#FFFFF#TTT#T#
#T###F#F#F#F###F#############T#
#T#TTT#F#FFF#FFFFFFF#FFFFFFSTT#
#T#T#T#F#####F#####F#F#########
#TTT#T#F#FFF#FFFFF#F#FFF#FFFFF#
#####T#F#F#F#####F#F###F#####F#
#TTTTT#F#F#FFFFF#F#F#FFFFFFF#F#
#T#####F#F#####F#F#F#F#####F#F#
#T#FFF#FFF#F#FFF#F#FFF#F#F#F#F#
#M###F#####G#F#########F#F#F#.#
#MTT#TTTTT#T#FFF#FFFFFFF#F#F#.#
###T#T###T#T###F#F#######F#F#.#
#.#TTT#F#TTTFF#F#F#FFF#FFF#.#.#
#.#####F#####F#F#F#F#F#F#.#.#.#
#......FFFFFFF#FFF#F#FF.#.....#
###############################
```

## Case 422 (final failure)

Loop gain: `0.0027`. First loop F1 `0.4757` with 260 false positives and 31 misses. loop16 F1 `0.4784` with 260 false positives and 30 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
#.#####F#T###T#####F#########M#
#F#F#FFF#TTT#T#FFFFF#TTTTTFF#M#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FF.#.GMMM#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
#.#####F#T###T#####F#########M#
#.#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
#.#####F#T###T#####F#########M#
#.#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FF.#.GMMM#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FF.#.GMMM#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
#.#####F#T###T#####F#########M#
#.#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FF.#.GMMM#
###############################
```

### loop16

```text
###############################
#MMMMMMT#FFFFFFTTTTTTTTMMMMM#.#
#M#####T#####F#T###########M#.#
#M#.FFFTTTTT#F#TTT#FFFFFF.#M#.#
#M#########T#####T#######F#M#.#
#MMTTTSF#TTT#TTTTT#FFFFFFF#MMM#
#.#####F#T###T#####F#########M#
#.#F#FFF#TTT#T#FFFFF#TTTTTFF#T#
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
#.#F#FFF#FFF#FFF#F#TTTTTTTTT#.#
#.#F#F#F#########F#########T#.#
#.#FFF#FFF#FFFFF#FFF#F#FFF#MMM#
#.#######F#F###F###F#F#F#.###M#
#......F#FFFFF#FFFFF#FFF#.GMMM#
###############################
```

## Case 129 (final failure)

Loop gain: `0.0009`. First loop F1 `0.4794` with 263 false positives and 28 misses. loop16 F1 `0.4803` with 262 false positives and 28 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTT#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#.FF#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTT#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTT#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#FFFFF#F#FFF#FFF#FFTTT#T#TTTTM#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

### loop16

```text
###############################
#.....#FFFFF#FFFFFFTTTTMMMMMMM#
#.#####F###F#######T#########M#
#.#.FFFF#F#FFFFFFF#T#TTTTG#MMM#
#.#F#####F#######F#T#T#####M###
#.#F#FFFFF#FFF#FFF#TTT#TTT#MMM#
#.#F#F###F#F###F#######T#T###M#
#.FFFF#F#FFF#FFF#FFTTT#T#TTTTM#
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
#.#F#F#FFFFFFF#F#TTTFF#T#T#TTM#
#.#F#F###F###F#F#T#T###T#T#####
#..F#FFF#FFF#F#TTT#TTTTT#TTTMM#
#######F#####F#T#############M#
#......FFFFFFF#TTTTTTTTMMMMMMM#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0064`. First loop F1 `0.4882` with 259 false positives and 24 misses. loop16 F1 `0.4819` with 260 false positives and 26 misses. Final exact `0.0000`.

### loop1

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#.#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TTM#
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
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop2

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#.#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#FFF#FFF#F#F#FFF#FFFFF#FFF#TTM#
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
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TMM#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop4

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#F#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TTM#
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
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TMM#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop6

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#F#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TTT#
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
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TTM#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop10

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#F#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
#.FF#FFF#F#F#FFF#FFFFF#FFF#TTT#
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
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TMM#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

### loop16

```text
###############################
#...#..FFFFFFF#FFF#FFFF...#...#
###.#.#######F#F#F#F#F###.#.###
#.#.#F#FFF#F#FFF#F#F#FFF#.#...#
#.#F#F#F#F#F#####F#F###F#F###.#
#.#FFF#F#F#FFF#F#F#FFF#F#FSM#.#
#.#####F#F#F#F#F#F#####F###M#.#
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
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#M#
#M###########F#T#####F#T#T#T#M#
#M#TTT#TTT#FFF#TTT#F#F#T#TMM#M#
#M#M#T#T#T#######T#F#F#T#####M#
#MMM#MMT#TTTTTTTTT#FFF#MMMMMMM#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0064`. First loop F1 `0.4891` with 258 false positives and 24 misses. loop16 F1 `0.4828` with 259 false positives and 26 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#F..#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#TMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

### loop2

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#F..#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
#FFFFF#FFF#F#FFF#F#F#FFFFFFF#T#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#TMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

### loop4

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#F.F#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

### loop6

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#F..#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#M#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#TMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

### loop10

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#FFF#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
#.FFFF#FFF#F#FFF#F#F#FFFFFFF#T#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

### loop16

```text
###############################
#.....#FFFFFFFFFFFFF#F#.GMMM#.#
#.#.#.#F#########F#F#F#F###M#.#
#.#.#F#FFF#FFFFFFF#FFF#F..#MMM#
###F#F#####F###F#####F#######M#
#..F#FFFFF#F#F#F#FFF#FFFFF..#M#
#.#######F#F#F#F#F#F#######.#M#
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
#MTT#F#T#TTTTT#TSF#F#FFF#TTT#.#
#.#T###T#####T#####F#F#F###T#.#
#.#TTT#TTT#F#T#FFFFFFF#FFF#MMM#
#.###T###T#F#T###############M#
#...#MMTTTFF#TTTTTTTTTTMMMMMMM#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0045`. First loop F1 `0.4801` with 260 false positives and 28 misses. loop16 F1 `0.4846` with 258 false positives and 27 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTM#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#M#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########F#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
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
#M#FFF#F#FFF#TTT#F#T#G#FFFFF#F#
#M###F###F###T###F#T#T###F#F#F#
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

### loop2

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTM#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#M#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
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
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

### loop4

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTT#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#M#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#M#
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
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

### loop6

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTM#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
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
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

### loop10

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTM#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#T#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
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
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

### loop16

```text
###############################
#MMMMMMT#FFFFTTTTTTTTTTM#MMMMM#
#M#####T#####T#########T#M###M#
#M#MTT#T#TTT#T#FFFFF#TTT#M#.#M#
#M#T#T#T#T#T#T#F###F#T###T#.#M#
#MMT#T#TTT#T#T#FFF#F#TTTTT#.#M#
#.###T#####T#T#F#F#########.#M#
#.#F#TTT#TTT#T#F#FFFFFFFFFFF#T#
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
#MTT#FFFFF#FFT#FFF#T#T#FFF#F#.#
###T#######F#T###F#T#T#F###F#.#
#.#TTTTT#FFF#TTT#F#TTT#F#F..#.#
#.#####T#######T#F###F#F#####.#
#......TTTTTTTTT#FFFFF#.......#
###############################
```

## Case 74 (final failure)

Loop gain: `-0.0054`. First loop F1 `0.4902` with 259 false positives and 26 misses. loop16 F1 `0.4847` with 259 false positives and 28 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTMMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#T#M#
#MTT#F#TTT#F#T#T#F#TTTTT#T#T#T#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

### loop2

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTMMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#M#M#
#TTT#F#TTT#F#T#T#F#TTTTT#T#T#M#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

### loop4

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTMMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#M#M#
#MTT#F#TTT#F#T#T#F#TTTTT#T#T#M#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

### loop6

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTTMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#M#M#
#MTT#F#TTT#F#T#T#F#TTTTT#T#T#M#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

### loop10

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTMMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#M#M#
#MTT#F#TTT#F#T#T#F#TTTTT#T#T#M#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

### loop16

```text
###############################
#MMMMMMTFF#FFF#TTTTTTTTMMMMMMM#
#S#####T#F###F#T#############M#
#...FF#T#FFFFF#TTT#FFF#TTMMM#M#
#.#####T#########T###F#T###M#M#
#.#TTTTT#TTTTT#TTTFFFF#TTT#M#M#
###T#F###T###T#T#########T#M#M#
#MTT#F#TTT#F#T#T#F#TTTTT#T#T#M#
#T#####T###F#T#T#F#T###T#T#T#T#
#TTTTTTT#FFF#T#T#F#T#TTT#T#TTT#
#F#########F#T#T#F#T#T###T###F#
#F#FFFFFFFFF#T#T#F#T#TTTTT#FFF#
#F#######F#F#T#T#F#T#######F###
#FFF#FFFFF#F#TTT#F#TTTTT#FFF#F#
###F#F###########F#####T###F#F#
#FFF#FFFFFFFFFFF#FFFFF#TTT#F#F#
#F###F#####F###F#####F###T#F#F#
#F#FFF#FFFFF#F#F#FFF#F#TTT#FFF#
#F#F###F#####F#F#F#F#F#T#######
#F#FFF#F#F#FFFFFFF#FFF#T#TTTTT#
#F#####F#F#F###########T#T###T#
#F#FFF#F#F#F#FFF#FFFFF#T#T#F#T#
#F#F#F#F#F#F#F#F#F###F#T#T#F#T#
#FFF#FFF#FFF#F#F#F#F#F#TTTFF#T#
#########F###F###F#F#F###F###M#
#.FFFFFF#F#FFF#FFF#F#FFF#F#TTM#
#.#F#####F###F#####F###F###G###
#.#F#FFF#FFF#FFFFFFFFF#FFF#...#
#.###F#F###F#F#######F###.###.#
#.....#FFFFFFF#FFFFFFFF.#.....#
###############################
```

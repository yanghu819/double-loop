# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 319 (final failure)

Loop gain: `-0.0196`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5051` with 279 false positives and 11 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#FFFFFFFFF#FFF#FFFFFFF#FFF#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FFF#
#F#F#F#F#F#F#####F#F###F#F###F#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#T#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#T#T#T#T#T#######T#F#F#T#####T#
#TTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop2

```text
###############################
#FFF#FFFFFFFFF#FFF#FFFFFFF#FFF#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FFF#
#F#F#F#F#F#F#####F#F###F#F###F#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#T#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#T#T#T#T#T#######T#F#F#T#####T#
#TTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop4

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#...#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FFF#
#F#F#F#F#F#F#####F#F###F#F###F#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#T#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####M#
#MMT#TTT#TTTTTTTTT#FFF#TTTMMMM#
###############################
```

### loop6

```text
###############################
#...#FFFFFFFFF#FFF#FFFFFFF#...#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FF.#
#F#F#F#F#F#F#####F#F###F#F###F#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#T#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#M#
#M#T#T#T#T#######T#F#F#T#####M#
#MMT#TTT#TTTTTTTTT#FFF#TTTMMMM#
###############################
```

### loop10

```text
###############################
#...#FFFFFFFFF#FFF#FFFFFFF#...#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FF.#
#F#F#F#F#F#F#####F#F###F#F###.#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#T#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#M#
#M#T#T#T#T#######T#F#F#T#####M#
#MMM#TTT#TTTTTTTTT#FFF#TTTMMMM#
###############################
```

### loop12

```text
###############################
#...#FFFFFFFFF#FFF#FFFFFF.#...#
###F#F#######F#F#F#F#F###F#F###
#F#F#F#FFF#F#FFF#F#F#FFF#F#FF.#
#F#F#F#F#F#F#####F#F###F#F###.#
#F#FFF#F#F#FFF#F#F#FFF#F#FST#F#
#F#####F#F#F#F#F#F#####F###T#F#
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
#F#########T#F#T#T#F#F#####T#T#
#TTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#T###########F#T#####F#T#T#T#M#
#T#TTT#TTT#FFF#TTT#F#F#T#TTT#M#
#M#T#T#T#T#######T#F#F#T#####M#
#MMM#TTT#TTTTTTTTT#FFF#TTTMMMM#
###############################
```

## Case 131 (final failure)

Loop gain: `-0.0261`. First loop F1 `0.5320` with 285 false positives and 0 misses. loop12 F1 `0.5059` with 278 false positives and 13 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTTTT#TTTFF#F#TTT#F#FFTTG#
#T#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTT#
#####T#F#####T#F###T#T#F#####T#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#T#
#F###############F#F###F###T#T#
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTT#
###############################
```

### loop2

```text
###############################
#TTTTTTTTT#TTTFF#F#TTT#F#FFTTG#
#T#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTT#
#####T#F#####T#F###T#T#F#####T#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#T#
#F###############F#F###F###T#T#
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTT#
###############################
```

### loop4

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#F.MMG#
#T#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTT#
#####T#F#####T#F###T#T#F#####T#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#T#
#.###############F#F###F###T#M#
#..FFFFFFFFFFFFFFFFF#FFFFF#MMM#
###############################
```

### loop6

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#F.MMG#
#M#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTM#
#####T#F#####T#F###T#T#F#####T#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#T#
#.###############F#F###F###T#M#
#..FFFFFFFFFFFFFFFFF#FFFFF#MMM#
###############################
```

### loop10

```text
###############################
#MMMTTTTTT#TTTFF#F#TTT#F#F.MMG#
#M#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTM#
#####T#F#####T#F###T#T#F#####M#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#M#
#.###############F#F###F###T#M#
#...FFFFFFFFFFFFFFFF#FFFFF#MMM#
###############################
```

### loop12

```text
###############################
#MMMTTTTTT#TTTFF#F#TTT#F#..MMG#
#M#######T#T#T#F#F#T#T#F#F#T###
#TTTTT#F#TTT#T#FFF#T#T#F#F#TTM#
#####T#F#####T#F###T#T#F#####M#
#TTTTT#FFF#F#T#F#TTT#T#FFFFFFT#
#T#####F#F#F#T###T###T#######T#
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
#F#F#F###F###F#F#F#T#########T#
#F#FFF#F#FFF#F#F#F#TTTTTTTFF#T#
#F#####F###F#F#F#F#######T###T#
#FFFFFFFFF#FFF#FFF#FFFFF#TTT#M#
#.###############F#F###F###T#M#
#...FFFFFFFFFFFFFFFF#FFFF.#MMM#
###############################
```

## Case 312 (final failure)

Loop gain: `-0.0252`. First loop F1 `0.5320` with 285 false positives and 0 misses. loop12 F1 `0.5068` with 277 false positives and 13 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#T#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#T#T#F###T#T#T#T#T#####T#T###T#
#TTTFFFF#TTT#TTT#TTTTTTT#TTTTT#
###############################
```

### loop2

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#T#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#T#T#F###T#T#T#T#T#####T#T###T#
#TTTFFFF#TTT#TTT#TTTTTTT#TTTTT#
###############################
```

### loop4

```text
###############################
#..FFFFFFF#FFFFFFFFF#FFTTTMM#.#
#F#####F#F###F#####F#F#T###T#.#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#T#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###M#
#MMTFFFF#TTT#TTT#TTTTTTT#TMMMM#
###############################
```

### loop6

```text
###############################
#..FFFFFFF#FFFFFFFFF#FFTTTMM#.#
#.#####F#F###F#####F#F#T###T#.#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#T#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###M#
#MMTFFFF#TTT#TTT#TTTTTTT#TMMMM#
###############################
```

### loop10

```text
###############################
#...FFFFFF#FFFFFFFFF#FFTTTMM#.#
#.#####F#F###F#####F#F#T###T#.#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#F#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#M#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFM#
#M#T#F###T#T#T#T#T#####T#T###M#
#MMMFFFF#TTT#TTT#TTTTTTT#TMMMM#
###############################
```

### loop12

```text
###############################
#...FFFFFF#FFFFFFFFF#FFTTTMM#.#
#.#####F#F###F#####F#F#T###T#.#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#.#
#F#F###########F#F#####T#F#T#.#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
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
#T###T###T#F#F#####F#F#####F#T#
#T#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#T#T###T###########F#F#F###F#M#
#T#T#F#TTT#TTT#TTT#FFF#TTT#FFM#
#M#T#F###T#T#T#T#T#####T#T###M#
#MMMFFFF#TTT#TTT#TTTTTTT#TMMMM#
###############################
```

## Case 129 (final failure)

Loop gain: `-0.0235`. First loop F1 `0.5320` with 285 false positives and 0 misses. loop12 F1 `0.5085` with 278 false positives and 12 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTTT#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTT#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTT#
#######F#####F#T#############T#
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTT#
###############################
```

### loop2

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTTT#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTT#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTT#
#######F#####F#T#############T#
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTT#
###############################
```

### loop4

```text
###############################
#..FFF#FFFFF#FFFFFFTTTTTTTMMMM#
#F#####F###F#######T#########M#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTT#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTT#
#######F#####F#T#############M#
#..FFFFFFFFFFF#TTTTTTTTTTTMMMM#
###############################
```

### loop6

```text
###############################
#..FFF#FFFFF#FFFFFFTTTTTTTMMMM#
#.#####F###F#######T#########M#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTM#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTT#
#######F#####F#T#############M#
#..FFFFFFFFFFF#TTTTTTTTTTTMMMM#
###############################
```

### loop10

```text
###############################
#...FF#FFFFF#FFFFFFTTTTTTTMMMM#
#.#####F###F#######T#########M#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTM#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTM#
#######F#####F#T#############M#
#...FFFFFFFFFF#TTTTTTTTTTTMMMM#
###############################
```

### loop12

```text
###############################
#...FF#FFFFF#FFFFFFTTTTTTTMMMM#
#.#####F###F#######T#########M#
#F#FFFFF#F#FFFFFFF#T#TTTTG#TTM#
#F#F#####F#######F#T#T#####T###
#F#F#FFFFF#FFF#FFF#TTT#TTT#TTT#
#F#F#F###F#F###F#######T#T###T#
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
#F#F#F#F#############F#T#T#T#T#
#F#F#F#FFFFFFF#F#TTTFF#T#T#TTT#
#F#F#F###F###F#F#T#T###T#T#####
#FFF#FFF#FFF#F#TTT#TTTTT#TTTTM#
#######F#####F#T#############M#
#...FFFFFFFFFF#TTTTTTTTTTTMMMM#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0154`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5094` with 277 false positives and 10 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFF#FFFFFFFFFFFFF#F#FGTTT#F#
#F#F#F#F#########F#F#F#F###T#F#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTT#
###F#F#####F###F#####F#######T#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#F#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTT#
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
###############################
```

### loop2

```text
###############################
#FFFFF#FFFFFFFFFFFFF#F#FGTTT#F#
#F#F#F#F#########F#F#F#F###T#F#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTT#
###F#F#####F###F#####F#######T#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#F#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTT#
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
###############################
```

### loop4

```text
###############################
#..FFF#FFFFFFFFFFFFF#F#FGTMM#.#
#F#F#F#F#########F#F#F#F###T#.#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTT#
###F#F#####F###F#####F#######T#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#F#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTT#
#.###T###T#F#T###############M#
#..F#TTTTTFF#TTTTTTTTTTTTTMMMM#
###############################
```

### loop6

```text
###############################
#..FFF#FFFFFFFFFFFFF#F#FGTMM#.#
#.#F#F#F#########F#F#F#F###T#.#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTM#
###F#F#####F###F#####F#######T#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#F#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTT#
#.###T###T#F#T###############M#
#..F#TTTTTFF#TTTTTTTTTTTTTMMMM#
###############################
```

### loop10

```text
###############################
#...FF#FFFFFFFFFFFFF#F#FGTMM#.#
#.#F#F#F#########F#F#F#F###T#.#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#F#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTM#
#.###T###T#F#T###############M#
#...#TTTTTFF#TTTTTTTTTTTTTMMMM#
###############################
```

### loop12

```text
###############################
#...FF#FFFFFFFFFFFFF#F#FGTMM#.#
#.#F#F#F#########F#F#F#F###T#.#
#F#F#F#FFF#FFFFFFF#FFF#FFF#TTM#
###F#F#####F###F#####F#######M#
#FFF#FFFFF#F#F#F#FFF#FFFFFFF#T#
#F#######F#F#F#F#F#F#######F#T#
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
#T###F#T#T#####T#F#F###F#T#####
#TTT#F#T#TTTTT#TSF#F#FFF#TTT#F#
#F#T###T#####T#####F#F#F###T#.#
#F#TTT#TTT#F#T#FFFFFFF#FFF#TTM#
#.###T###T#F#T###############M#
#...#TTTTTFF#TTTTTTTTTTTTTMMMM#
###############################
```

## Case 130 (final failure)

Loop gain: `-0.0267`. First loop F1 `0.5368` with 283 false positives and 0 misses. loop12 F1 `0.5101` with 277 false positives and 13 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#F#FFFFFFFFFFFFFFFFFFTTT#S#
#F#F#F#F#################T#T#T#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#T#
#F###F#F###############F#T#T#T#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############T#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#T#
#F#T#######F#####T#########T#T#
#F#TTTTTTTTTTTTTTTFFFFFFFF#TTT#
###############################
```

### loop2

```text
###############################
#FFF#F#FFFFFFFFFFFFFFFFFFTTT#S#
#F#F#F#F#################T#T#T#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#T#
#F###F#F###############F#T#T#T#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############T#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#T#
#F#T#######F#####T#########T#T#
#F#TTTTTTTTTTTTTTTFFFFFFFF#TTT#
###############################
```

### loop4

```text
###############################
#..F#F#FFFFFFFFFFFFFFFFFFTMM#S#
#F#F#F#F#################T#T#M#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#T#
#F###F#F###############F#T#T#T#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############T#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#T#
#.#T#######F#####T#########T#M#
#.#TTTTTTTTTTTTTTTFFFFFFFF#MMM#
###############################
```

### loop6

```text
###############################
#...#F#FFFFFFFFFFFFFFFFFFTMM#S#
#.#F#F#F#################T#T#M#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#F###F#F###############F#T#T#T#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############T#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#T#
#.#T#######F#####T#########T#M#
#.#TTTTTTTTTTTTTTTFFFFFFFF#MMM#
###############################
```

### loop10

```text
###############################
#...#F#FFFFFFFFFFFFFFFFFFTMM#S#
#.#F#F#F#################T#T#M#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#F###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############M#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#M#
#.#T#######F#####T#########T#M#
#.#MTTTTTTTTTTTTTTFFFFFFFF#MMM#
###############################
```

### loop12

```text
###############################
#...#F#FFFFFFFFFFFFFFFFFFMMM#S#
#.#F#F#F#################T#T#M#
#F#FFF#F#FFFFFFFFFFFFFFF#T#T#M#
#F###F#F###############F#T#T#M#
#FFF#F#FFFFFFFFFFFFFFFFF#T#TTT#
###F#F#F###############F#T###F#
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
#T#T#T#####F#############F#F#T#
#T#TTT#F#FFF#FFFFFFFFFFFFF#F#T#
#T###F#F#F###F#F#############M#
#TTT#F#FFF#FFF#F#TTTTTTTTTTT#M#
#.#T#######F#####T#########T#M#
#.#MTTTTTTTTTTTTTTFFFFFFFF#MMM#
###############################
```

## Case 146 (final failure)

Loop gain: `-0.0177`. First loop F1 `0.5296` with 286 false positives and 0 misses. loop12 F1 `0.5119` with 275 false positives and 11 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTT#FFFFTTTTTTTTTTT#TTTTT#
#T#####T#####T#########T#T###T#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#T#
#T#T#T#T#T#T#T#F###F#T###T#F#T#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#F#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#F#
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

### loop2

```text
###############################
#TTTTTTT#FFFFTTTTTTTTTTT#TTTTT#
#T#####T#####T#########T#T###T#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#T#
#T#T#T#T#T#T#T#F###F#T###T#F#T#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#F#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#F#
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

### loop4

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TMMMM#
#T#####T#####T#########T#T###M#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#T#
#T#T#T#T#T#T#T#F###F#T###T#F#T#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#F#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#F#
#.#####T#######T#F###F#F#####.#
#..FFFFTTTTTTTTT#FFFFF#FFF....#
###############################
```

### loop6

```text
###############################
#MMMTTTT#FFFFTTTTTTTTTTT#TMMMM#
#M#####T#####T#########T#T###M#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#T#
#T#T#T#T#T#T#T#F###F#T###T#F#T#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#F#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#F#
#.#####T#######T#F###F#F#####.#
#..FFFFTTTTTTTTT#FFFFF#FFF....#
###############################
```

### loop10

```text
###############################
#MMMTTTT#FFFFTTTTTTTTTTT#TMMMM#
#M#####T#####T#########T#T###M#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#M#
#T#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#.#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#.#
#.#####T#######T#F###F#F#####.#
#...FFFTTTTTTTTT#FFFFF#FFF....#
###############################
```

### loop12

```text
###############################
#MMMTTTT#FFFFTTTTTTTTTTT#TMMMM#
#M#####T#####T#########T#T###M#
#T#TTT#T#TTT#T#FFFFF#TTT#T#F#M#
#T#T#T#T#T#T#T#F###F#T###T#F#M#
#TTT#T#TTT#T#T#FFF#F#TTTTT#F#T#
#F###T#####T#T#F#F#########F#T#
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
#T###F###F###T###F#T#T###F#F#F#
#TTT#FFFFF#FFT#FFF#T#T#FFF#F#F#
###T#######F#T###F#T#T#F###F#.#
#F#TTTTT#FFF#TTT#F#TTT#F#FFF#.#
#.#####T#######T#F###F#F#####.#
#...FFFTTTTTTTTT#FFFFF#FFF....#
###############################
```

## Case 417 (final failure)

Loop gain: `-0.0217`. First loop F1 `0.5344` with 284 false positives and 0 misses. loop12 F1 `0.5127` with 275 false positives and 12 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFF#FFTTTTTTT#TTT#TTT#TTT#
#####F###F#T#####T#T#T#T#T#T#T#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#T#
#F#F###F#####T###############T#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###T#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTT#
#F#######F#F#####F#F#F#T###T###
#FFFFFFFFF#FFFFFFF#FFF#TTTTTFF#
###############################
```

### loop2

```text
###############################
#FFFFFFF#FFTTTTTTT#TTT#TTT#TTT#
#####F###F#T#####T#T#T#T#T#T#T#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#T#
#F#F###F#####T###############T#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###T#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTT#
#F#######F#F#####F#F#F#T###T###
#FFFFFFFFF#FFFFFFF#FFF#TTTTTFF#
###############################
```

### loop4

```text
###############################
#..FFFFF#FFTTTTTTT#TTT#TTT#MMM#
#####F###F#T#####T#T#T#T#T#T#M#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#T#
#F#F###F#####T###############T#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###T#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTT#
#.#######F#F#####F#F#F#T###T###
#..FFFFFFF#FFFFFFF#FFF#TTTMM..#
###############################
```

### loop6

```text
###############################
#..FFFFF#FFTTTTTTT#TTT#TTT#MMM#
#####F###F#T#####T#T#T#T#T#T#M#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#M#
#F#F###F#####T###############T#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###T#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTT#
#.#######F#F#####F#F#F#T###T###
#..FFFFFFF#FFFFFFF#FFF#TTTMM..#
###############################
```

### loop10

```text
###############################
#...FFFF#FFTTTTTTT#TTT#TTT#MMM#
#####F###F#T#####T#T#T#T#T#T#M#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#M#
#F#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###T#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTM#
#.#######F#F#####F#F#F#T###T###
#...FFFFFF#FFFFFFF#FFF#TTTMM..#
###############################
```

### loop12

```text
###############################
#...FFFF#FFTTTTTTT#TTT#TTT#MMM#
#####F###F#T#####T#T#T#T#T#T#M#
#FFFFF#FFF#TTT#FFTTT#TTT#TTT#M#
#F#F###F#####T###############M#
#F#F#FFF#FFF#TTT#FFFFFFFFFFF#T#
#F#F#F###F#F###T#F#F#########T#
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
#F#F#########T###T#F###T#F#T###
#F#FFF#FFFFF#T#TTT#FFF#T#F#TTT#
#F###F#F###F#T#T#####F#T#F###M#
#FFFFF#FFF#F#TTT#FFF#F#T#F#TTM#
#.#######F#F#####F#F#F#T###M###
#...FFFFFF#FFFFFFF#FFF#TTMMM..#
###############################
```

## Case 261 (final failure)

Loop gain: `-0.0144`. First loop F1 `0.5272` with 287 false positives and 0 misses. loop12 F1 `0.5128` with 275 false positives and 10 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTTTTTT#TTTTTTTTTTTTTTTTT#
#T#########T#T###############T#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTT#
#T###F###F#T#####T#F#T#######F#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#F#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#F#
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FFF#
###############################
```

### loop2

```text
###############################
#TTTTTTTTTTT#TTTTTTTTTTTTTTTTT#
#T#########T#T###############T#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTT#
#T###F###F#T#####T#F#T#######F#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#F#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#F#
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FFF#
###############################
```

### loop4

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTMMMM#
#T#########T#T###############M#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTT#
#T###F###F#T#####T#F#T#######F#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#F#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#F#
#.#########F#F#F#####F#F#F###.#
#..FFFFFFF#FFFFFFFFF#FFFFF#...#
###############################
```

### loop6

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTMMMM#
#M#########T#T###############M#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTM#
#T###F###F#T#####T#F#T#######F#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#F#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#.#
#.#########F#F#F#####F#F#F###.#
#..FFFFFFF#FFFFFFFFF#FFFFF#...#
###############################
```

### loop10

```text
###############################
#MMMTTTTTTTT#TTTTTTTTTTTTTMMMM#
#M#########T#T###############M#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTM#
#T###F###F#T#####T#F#T#######.#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#F#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#.#
#.#########F#F#F#####F#F#F###.#
#...FFFFFF#FFFFFFFFF#FFFFF#...#
###############################
```

### loop12

```text
###############################
#MMMTTTTTTTT#TTTTTTTTTTTTTMMMM#
#M#########T#T###############M#
#T#FFFFFFF#T#TTTTT#F#TTTTTTTTM#
#T###F###F#T#####T#F#T#######.#
#TTT#FFF#F#TFF#TTT#F#T#FFFFFFF#
###T###F#F#T###T###F#T#F#######
#F#TTT#F#F#TTTTTFFFF#T#FFFFF#F#
#F###T#F#F#########F#T#####F#F#
#TTTTT#F#F#TTTTTTT#F#TTTTT#F#F#
#T#######F#T#####T#######T#F#F#
#TTT#FFF#F#TTT#F#TTTTT#TTT#FFF#
###T#F#F#F###T#F#####T#T#####F#
#TTT#F#FFFFF#T#FFFFF#T#T#FFFFF#
#T###F###F###T#F###F#T#T#####F#
#TTTFF#FFF#TTT#F#FFF#T#T#FFF#F#
###T#####F#T#####F###T#T#F#F#F#
#F#T#FFF#F#TTTTT#FFF#T#T#F#F#F#
#F#T#F#F#####F#T#F#F#T#T#F#F###
#TTT#F#F#FFF#F#T#F#FFTTT#F#FFF#
#T###F#F#F#F###T#F#######F###F#
#T#F#F#FFF#FFFFS#F#FFFFF#FFF#F#
#T#F#F###########F#F###F#####F#
#TGF#FFFFF#FFFFF#F#F#F#FFFFFFF#
#F#######F#F###F###F#F#########
#F#FFFFFFF#FFF#FFF#FFFFF#FFFFF#
#F#F#########F###F#####F###F#.#
#F#FFFFFFFFF#F#F#FFFFF#F#FFF#.#
#.#########F#F#F#####F#F#F###.#
#...FFFFFF#FFFFFFFFF#FFFF.#...#
###############################
```

## Case 498 (final failure)

Loop gain: `-0.0208`. First loop F1 `0.5344` with 284 false positives and 0 misses. loop12 F1 `0.5136` with 274 false positives and 12 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTT#TTTTT#TTT#FFTTTTTTTTT#
#T#####T#T#F#T#T#T#F#T#######T#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#T#
#T###F#T#T#F#####T#F#T#T###T#T#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###F#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#F#
#F###F###F###T###T###F#S#F#F#F#
#FFFFFFF#FFFFTTTTT#FFFFFFF#FFF#
###############################
```

### loop2

```text
###############################
#TTTTTTT#TTTTT#TTT#FFTTTTTTTTT#
#T#####T#T#F#T#T#T#F#T#######T#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#T#
#T###F#T#T#F#####T#F#T#T###T#T#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###F#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#F#
#F###F###F###T###T###F#S#F#F#F#
#FFFFFFF#FFFFTTTTT#FFFFFFF#FFF#
###############################
```

### loop4

```text
###############################
#MMTTTTT#TTTTT#TTT#FFTTTTTMMMM#
#T#####T#T#F#T#T#T#F#T#######M#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#T#
#T###F#T#T#F#####T#F#T#T###T#T#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###F#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#F#
#.###F###F###T###T###F#S#F#F#.#
#..FFFFF#FFFFTTTTT#FFFFFFF#...#
###############################
```

### loop6

```text
###############################
#MMTTTTT#TTTTT#TTT#FFTTTTTMMMM#
#M#####T#T#F#T#T#T#F#T#######M#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#M#
#T###F#T#T#F#####T#F#T#T###T#T#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###F#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#F#
#.###F###F###T###T###F#S#F#F#.#
#..FFFFF#FFFFTTTTT#FFFFFFF#...#
###############################
```

### loop10

```text
###############################
#MMMTTTT#TTTTT#TTT#FFTTTTTMMMM#
#M#####T#T#F#T#T#T#F#T#######M#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#M#
#T###F#T#T#F#####T#F#T#T###T#M#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###F#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#.#
#.###F###F###T###T###F#S#F#F#.#
#...FFFF#FFFFTTTTT#FFFFFFF#...#
###############################
```

### loop12

```text
###############################
#MMMMTTT#TTTTT#TTT#FFTTTTTMMMM#
#M#####T#T#F#T#T#T#F#T#######M#
#T#FFF#T#T#F#TTT#T#F#T#TTTTT#M#
#T###F#T#T#F#####T#F#T#T###T#M#
#T#FFF#TTT#F#FFF#T#F#T#TTT#TTT#
#T#F#######F#F#F#T###T###T#####
#T#FFTTT#FFF#F#FFTTTTT#FFTTT#F#
#T###T#T#F#################T#F#
#TTTTT#T#FFFFF#FFF#FFFFFFF#TGF#
#######T#####F#F#F#F#####F###F#
#TTTTT#TTT#F#F#F#FFF#FFF#FFF#F#
#T###T###T#F#F#F#####F#F###F###
#T#TTTFF#T#F#FFF#FFFFF#FFF#FFF#
#T#T#####T#F#######F#####F###F#
#T#TTTTTTT#FFFFFFF#FFF#FFF#F#F#
#T#########F#####F#####F###F#F#
#TTT#FFFFFFF#FFFFF#FFF#FFF#F#F#
###T###F#F#######F#F#F###F#F#F#
#F#TTT#F#FFF#FFF#FFF#FFFFF#F#F#
#F###T#####F#F#F#F#########F#F#
#FFF#TTTTT#FFF#F#F#FFFFFFF#FFF#
#F#F#####T#####F###F#####F#####
#F#FFFFF#TTT#FFF#FFFFFFF#FFF#F#
#F#F###F###T#F###F#######F#F#F#
#F#F#FFFFF#T#FFFFF#TTTTT#F#FFF#
###F#######T###F###T###T#F###.#
#FFF#FFFFF#TTT#F#TTT#F#T#F#F#.#
#.###F###F###T###T###F#S#F#F#.#
#...FFFF#FFFFTTTTT#FFFFFFF#...#
###############################
```

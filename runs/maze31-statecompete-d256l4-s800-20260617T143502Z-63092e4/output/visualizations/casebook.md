# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 319 (final failure)

Loop gain: `0.0025`. First loop F1 `0.5134` with 284 false positives and 6 misses. loop12 F1 `0.5159` with 284 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop2

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop4

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#M#T#T#T#T#######T#F#F#T#####T#
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop6

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#M#T#T#T#T#######T#F#F#T#####T#
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop10

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop12

```text
###############################
#..F#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#M#T#T#T#T#######T#F#F#T#####T#
#MMT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

## Case 131 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5167` with 283 false positives and 7 misses. loop12 F1 `0.5167` with 283 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop2

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop4

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop6

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop10

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop12

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTMG#
#M#######T#T#T#F#F#T#T#F#F#T###
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

## Case 312 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5209` with 281 false positives and 6 misses. loop12 F1 `0.5209` with 281 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

### loop2

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#M#T#F###T#T#T#T#T#####T#T###T#
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

### loop4

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#M#T#F###T#T#T#T#T#####T#T###T#
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

### loop6

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

### loop10

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

### loop12

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#.#
#.#####F#F###F#####F#F#T###T#.#
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
#MMTFFFF#TTT#TTT#TTTTTTT#TTTMM#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5251` with 282 false positives and 2 misses. loop12 F1 `0.5217` with 283 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#F#
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
#.###T###T#F#T###############T#
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

### loop2

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#F#
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
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

### loop4

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#F#
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
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

### loop6

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#F#
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
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

### loop10

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#F#
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
#.###T###T#F#T###############T#
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

### loop12

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
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
#.###T###T#F#T###############M#
#..F#TTTTTFF#TTTTTTTTTTTTTTTMM#
###############################
```

## Case 229 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5226` with 282 false positives and 3 misses. loop12 F1 `0.5217` with 283 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#.#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

### loop2

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#F#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

### loop4

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#F#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

### loop6

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#F#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

### loop10

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#F#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

### loop12

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFF..#
#F#T#T#F#T#T#F#F#F#F#F###F###F#
#F#T#T#F#T#T#F#F#F#F#FFFFF#FFF#
#F#T#T###T#T#F#F###F#######F#F#
#F#T#T#TTT#T#F#FFFFF#FFFFF#F#F#
#F#T#T#T###T#########F###F#F#F#
#F#T#TTT#F#TTTTTTTTT#F#FFFFF#F#
###T#####F#########T#F#########
#TTT#FFFFFFFFFFFFF#T#FFFFFFFFF#
#T###F#######F#####T#########F#
#T#FFFFF#FFFFF#TTTTT#TTTTTTT#F#
#T#####F#######T#####T#####T#F#
#TTTTT#FFFFFFF#TTTTTTT#FFF#T#F#
#####T#######F###########F#T#F#
#FFF#TTT#TTT#F#FFF#FFFFF#F#TTT#
#F#####T#T#T#F#F#F#F#F#F#F###T#
#FFFFF#TTT#T#FFF#FFF#F#FFFFF#T#
#F###F#####T#########S#F#####T#
#F#FFF#FFF#TTTTTGF#TTT#F#TTTTT#
###F#F#F#F#########T###F#T#####
#FFF#FFF#FFF#FFF#FFT#FFF#TTTTT#
#F#F#######F###F#F#T#########T#
#F#F#FFF#FFF#FFF#F#T#TTTTT#TTT#
#F###F#F###F#F###F#T#T###T#T###
#FFFFF#FFF#FFF#FFF#TTT#TTT#TTT#
#F#######F###F#F#######T#####T#
#FFFFF#F#F#FFF#F#FFFFF#TTTTT#T#
#####F#F#F#####F#F###F#####T#M#
#..FFFFF#FFFFFFFFF#FFFFFFF#TMM#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0033`. First loop F1 `0.5184` with 282 false positives and 6 misses. loop12 F1 `0.5217` with 281 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop2

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop4

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop6

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#..FFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop10

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#..FFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop12

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.#####T#######T#F###F#F#####F#
#..FFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

## Case 230 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5217` with 281 false positives and 5 misses. loop12 F1 `0.5217` with 281 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop2

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop4

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop6

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop10

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop12

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTMM#
#.#####T#T#######T#F#########M#
#F#FFF#T#TTTTTTTTT#F#FFF#TTTTT#
#F###F#T#######F#####F#F#T#####
#FFFFF#TTT#FFFFF#FFF#F#F#TTTTT#
#########T#F#####F#F#F#F#####T#
#TTTTTTTTT#F#FFFFF#FFF#F#FFF#T#
#T###F#######F#########F#F###T#
#T#FFF#FFF#FFF#TTSFFFF#F#F#TTT#
#T#####F#F#F###T#######F#F#T###
#T#FFFFF#FFF#FFT#FFFFFFF#F#TTT#
#T#F###########T#F#######F###T#
#T#FFFFF#TTTTTTT#F#F#FFFFFFF#T#
#T#####F#T#F#####F#F#F###F#F#T#
#TTTTTTTTT#F#FFF#F#FFF#FFF#F#T#
#############F#F#F#####F#####T#
#FFFFFFF#FFFFF#FFF#FFFFF#TTT#T#
#F#F###F#F#########F#####T#T#T#
#F#FFF#F#F#FFF#FFF#TTTTTTT#T#T#
#F###F###F#F#F###F#T#######T#T#
#FFF#F#FFF#F#FFF#F#TTT#FFF#TTT#
#F###F#F###F###F#F###T#F#######
#F#FFF#F#FFF#F#FFF#TTT#F#FFFFF#
#F#F#F#F###F#F###F#T###F#F#F#F#
#F#F#F#F#FFF#F#FFF#T#FFFFF#F#F#
###F#####F###F#F###T#######F###
#GFF#TTTTTTT#FFFFF#TTT#FFF#FFF#
#M###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

## Case 470 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5217` with 280 false positives and 6 misses. loop12 F1 `0.5217` with 280 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

### loop2

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

### loop4

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

### loop6

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

### loop10

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

### loop12

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFF..#
#.#F#F#####F#####F#F###F#F###.#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
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
#T###T#T#F#F#F#####T#####T###T#
#TTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#TTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#M#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

## Case 12 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5226` with 282 false positives and 3 misses. loop12 F1 `0.5226` with 282 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

### loop2

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

### loop4

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

### loop6

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

### loop10

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

### loop12

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#.#F#T#T#T#T#F###T#######T#F###
#F#F#T#TTT#T#F#TTT#FFFFF#T#FFF#
#F#F#T#####T#F#T###F###F#T###F#
#FFF#T#FFF#T#F#TTT#F#F#F#TTTTT#
#F###T#F#F#T#####T#F#F#F#####T#
#F#F#T#F#FFT#TTT#T#F#FFF#TTTTT#
#F#F#T#####T#T#T#T#F#F###T#####
#F#F#TTTTT#TTT#TTT#F#FFF#T#TTT#
#F#F#####T#########F###F#T#T#T#
#FFF#FFF#TTT#FFFFFGT#FFF#TTT#T#
###F#F#F###T#######T#########T#
#F#F#F#FFF#TTTTTTT#TTTTT#TTTTT#
#F#F#F###########T#F###T#T#####
#F#F#FFFFFFFFFFF#T#FFF#TTT#FFF#
#F#F###F#####F#F#T#F###F#####F#
#F#FFF#FFF#FFF#F#T#F#F#FFFFFFF#
#F###F#####F###F#T###F#######F#
#FFF#F#FFF#F#FFF#TTT#FFFFF#FFF#
###F#F#F#F#F#######T#####F#####
#FFF#F#F#F#FFFFFFF#TFFFF#FFFFF#
#F###F#F#F#######F#T###F#####F#
#FFF#FFF#FFFFFFFFF#T#FFFFF#FFF#
#F#F#############F#T#######S###
#F#F#FFFFFFFFFFFFF#TTTTTTT#TTT#
#F#F#F###################T###T#
#F#F#F#FFFFFFF#FFFFF#TTTTTFF#T#
#.#F#F#####F#F#F###F#T#######M#
#.#FFFFFFFFF#FFF#FFF#TTTTTTTMM#
###############################
```

## Case 204 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5235` with 281 false positives and 3 misses. loop12 F1 `0.5235` with 281 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

### loop2

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

### loop4

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

### loop6

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

### loop10

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

### loop12

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFF..#
#.###F#F###F###F#F#######F###F#
#F#FFF#F#FFF#F#F#FFF#FFFFF#FFF#
#F#F###F#F###F#F###F#F#####F###
#F#FFF#F#F#FFF#FFFFF#F#FFFFFFF#
#####F###F#F#F#######F#######F#
#TTT#FFF#FFF#FFFFF#FFF#FFF#FFF#
#T#T###F#F#######F#F###F#F###F#
#T#TTT#F#F#FFFFF#F#F#FFF#FFF#F#
#T###T#F#F#F###F###F#######F#F#
#TTT#T#FFF#F#FFFFFFFFFFFFF#F#F#
#F#T#T###F#F#######F#######F#F#
#F#T#TTT#F#F#TTT#FFF#TGFFFFF#F#
#F#T###S###F#T#T#####T#######F#
#F#TTT#F#FFF#T#TTTTTTT#F#FFFFF#
#F###T#F#F#F#T#########F#F#####
#F#TTT#F#F#F#TTTTTTTTT#FFF#TTT#
#F#T###F#F###########T#####T#T#
#F#TTT#FFF#TTTFFFF#F#TTTTTTT#T#
#F###T#####T#T###F#F#########T#
#FFF#TTTTT#T#TTT#FFF#TTTFF#F#T#
###F#####T#T###T#####T#T#F#F#T#
#FFF#FFF#T#T#TTT#TTTTT#T#F#F#T#
#F#####F#T#T#T###T#####T#F#F#T#
#FFF#FFF#T#T#TTT#T#F#TTT#FFF#T#
###F#F###T#T###T#T#F#T#####F#T#
#FFF#F#TTT#T#F#TTT#F#T#FFFFF#T#
#.###F#T###T#F#####F#T#######M#
#..FFF#TTTTTFFFFFFFF#TTTTTTTMM#
###############################
```

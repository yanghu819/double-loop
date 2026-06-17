# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 261 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5150` with 287 false positives and 5 misses. loop12 F1 `0.5158` with 286 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop2

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop4

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop6

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop10

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop12

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
#M#########T#T###############M#
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
#F#########F#F#F#####F#F#F###.#
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0025`. First loop F1 `0.5207` with 287 false positives and 2 misses. loop12 F1 `0.5183` with 287 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#TTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop2

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#TTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop4

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop6

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop10

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop12

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#F..#
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
#T#T#T#T#T#######T#F#F#T#####M#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5174` with 286 false positives and 5 misses. loop12 F1 `0.5183` with 285 false positives and 5 misses. Final exact `0.0000`.

### loop1

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
#F#####T#######T#F###F#F#####.#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
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
#F#####T#######T#F###F#F#####.#
#.FFFFFTTTTTTTTT#FFFFF#FFFFFF.#
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
#F#####T#######T#F###F#F#####.#
#.FFFFFTTTTTTTTT#FFFFF#FFFFFF.#
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
#F#####T#######T#F###F#F#####.#
#.FFFFFTTTTTTTTT#FFFFF#FFFFFF.#
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
#F#####T#######T#F###F#F#####.#
#.FFFFFTTTTTTTTT#FFFFF#FFFFFF.#
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
#F#####T#######T#F###F#F#####.#
#.FFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

## Case 131 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5223` with 285 false positives and 4 misses. loop12 F1 `0.5199` with 285 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#T#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop2

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#M#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop4

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#M#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop6

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#M#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop10

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#M#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop12

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTMG#
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
#F###############F#F###F###T#M#
#.FFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

## Case 129 (final failure)

Loop gain: `-0.0049`. First loop F1 `0.5256` with 284 false positives and 3 misses. loop12 F1 `0.5207` with 284 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTTM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop2

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop4

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop6

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop10

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop12

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#.#####F###F#######T#########M#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

## Case 255 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5216` with 285 false positives and 3 misses. loop12 F1 `0.5216` with 285 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop2

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop4

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop6

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop10

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop12

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#.#F#F#####F###F###T#########M#
#F#F#F#FFFFF#FFF#F#TTT#TTTTTTT#
#F#F#F#F#####F###F###T#T#######
#FFF#F#F#FFFFFFFFF#TTT#T#FFFFF#
#F###F###F#########T###T#F#F#F#
#F#F#F#FFFFFFF#TTTTT#TTT#F#F#F#
#F#F#F#F#######T#####T#####F###
#FFF#FFF#TTTTTTT#FFF#TTTTTTT#F#
#########T#########F#######T#F#
#TTTTTTTTT#FFFFFFFFFFFFFFF#T#F#
#T#########F###########F###T#F#
#T#FFFFFFTTT#TTTTTTTTT#F#TTT#F#
#T#######T#T#T#######T#F#T###F#
#TTTTTTT#T#TTT#FFF#TTT#F#TTT#F#
#####F#T#T#######F#T#######T#F#
#TTT#F#T#T#FFF#FFF#TTTTTFF#T#F#
#T#T###T#T#F#F#F#######T###T#F#
#T#TTT#T#TTT#FFF#FFFFF#TTTTT#F#
#T###T#T###T#####F###F#######F#
#T#F#TTT#TTT#FFFFF#F#FFFFFFFFF#
#T#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#T#########F###F#####F###F#F#F#
#TFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#T#####F#F###F#F#####F#F#F#F#F#
#TSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

## Case 505 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5216` with 285 false positives and 3 misses. loop12 F1 `0.5216` with 285 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###.#
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop2

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###.#
#.FFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop4

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###F#
#.FFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop6

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###F#
#.FFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop10

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###F#
#.FFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop12

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
#.#F#########T#T###T#########M#
#F#FFF#FFFFF#T#TTTTT#FFFFF#F#T#
#F###F###F###T#######F#F#F#F#T#
#FFF#FFF#F#TTT#FFFFFFF#F#FFF#T#
###F###F#F#T#########F#F#####T#
#FFF#F#FFF#T#FFTTTTT#F#FFF#TTT#
#F#F#F###F#T###T###T#####F#T###
#F#FFFFF#F#T#TTT#FFTTTTTTT#TTT#
#F#######F#T#T###########T###T#
#FFFFFFFFF#T#TTTTTTTTT#TTT#TTT#
#F#####F###T#F#######T#T###T###
#F#FFF#FFF#T#FFFFF#TTT#T#F#T#F#
#F#F#F#####T#######T###S#F#T#F#
#F#F#FFFFF#T#TTTTT#T#FFFFF#T#F#
#F#F#####F#T#T###T#T#######T#F#
#F#F#FFF#F#T#TTT#T#TTT#TTTTT#F#
###F###F###T###T#T#F#T#T#####F#
#FFF#TTTTT#T#TTT#T#F#TTT#FFF#F#
#F###T###T#T#T###T#F#####F#F#F#
#FFGTT#F#T#TTT#TTT#F#FFFFF#F#F#
#######F#T#####T###F#F###F#F#F#
#FFFFFFF#TTTTTTT#FFF#F#FFF#FFF#
#F#####F#########F###F#F#####F#
#F#FFF#FFFFFFFFFFF#FFF#F#FFFFF#
#F#F###################F#F#####
#F#FFFFFFFFFFF#FFF#FFFFF#F#FFF#
#F#########F#F#F#F#F#####F###F#
#.FFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0025`. First loop F1 `0.5240` with 284 false positives and 3 misses. loop12 F1 `0.5216` with 284 false positives and 4 misses. Final exact `0.0000`.

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
#T###T#####T#########T#F#F###.#
#TTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#T###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#T###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#T###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#T###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#T###T#####T#########T#F#F###.#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5258` with 284 false positives and 1 misses. loop12 F1 `0.5225` with 285 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop2

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#F###T###T#F#T###############M#
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop4

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#F###T###T#F#T###############M#
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop6

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#F###T###T#F#T###############M#
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop10

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#F###T###T#F#T###############M#
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop12

```text
###############################
#.FFFF#FFFFFFFFFFFFF#F#FGTTT#.#
#.#F#F#F#########F#F#F#F###T#.#
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
#F###T###T#F#T###############M#
#.FF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

## Case 27 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5225` with 284 false positives and 3 misses. loop12 F1 `0.5225` with 284 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop2

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop4

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop6

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop10

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop12

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#.#F###F###F###F#F#T###T#####M#
#F#FFF#FFFFF#FFF#F#TTT#TTT#F#T#
#F###F#F#####F###F###T###T#F#T#
#FFF#F#FFFFF#F#F#F#TTTFF#TFF#T#
#F#F#F#######F#F#F#T#####T###T#
#F#F#FFF#FFFFF#F#F#TTTTT#T#TTT#
###F###F#F#####F#F#####T#T#T###
#FFF#F#F#F#F#FFFFF#TTT#TTT#TTT#
#F###F#F#F#F#F#####T#T#######T#
#F#FFFFF#F#F#F#TTTTT#TTTTTTTTT#
###F#####F#F#F#T#############F#
#F#F#FFFFF#S#F#TTTTT#FFFFFFF#F#
#F#F#F#####T#F#F###T#######F#F#
#F#F#F#TTTTTFF#FFF#T#TTTTT#F#F#
#F#F#F#T#########F#T#T###T#F#F#
#F#FFF#T#FFF#FFFFF#TTTFF#T#F#F#
#F#####T#F#F#F###########T#F#F#
#FFF#TTT#F#FFFFF#TTT#TTTTT#F#F#
#F#F#T#######F###T#T#T#####F#F#
#F#F#TTTTTTT#F#TTT#TTT#FFF#FFF#
#F#########T###T#######F###F###
#F#FFFFFFF#TTT#TTTTTTT#FFF#FFF#
#F#F#F###F###T#F#####T###F###F#
#F#F#F#FFF#TTT#F#TTTTT#FFF#FFF#
#F#F#F#F###T#####T#####F###F###
#F#F#F#FFF#TTTTT#T#FFF#FFF#FFF#
#F#F#F#########T#T#F###F#F###.#
#.FF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

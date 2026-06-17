# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 261 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5223` with 287 false positives and 2 misses. loop12 F1 `0.5223` with 287 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
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
#MTTTTTTTTTT#TTTTTTTTTTTTTTTMM#
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
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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

### loop6

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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

### loop10

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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

### loop12

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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

## Case 62 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5240` with 286 false positives and 1 misses. loop12 F1 `0.5232` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFF..#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop2

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFF..#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop4

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop6

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop10

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop12

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######F#
#T#T#F#TTT#F#TTT#FFF#TTTTTTT#F#
#T#T###T###F###S#F#########T#F#
#T#TTTTT#FFF#FFF#F#TTTTTTTTT#F#
#T#######F###F###F#T#########F#
#T#FFFFFFF#FFF#FFF#T#TTTTTTT#F#
#T#F#######F#######T#T#####T###
#T#F#FFFFF#FFFFFFF#TTT#FFF#TTT#
#T#F#F###F#######F#F###F#####T#
#T#FFFFF#FFFFF#FFF#FFFFF#FFF#T#
#T#######F###F#F#########F#F#T#
#TTTTTTT#F#FFF#FFFFF#FFFFF#F#T#
#######T#F#F#F#####F#F#######T#
#FFFFF#T#F#F#FFFFF#FFF#FFFFFFT#
#####F#T###F###F#############T#
#FFFFF#TTT#F#FFF#FFFFTTTTTTTTT#
#F###F###T#F#####F###T#######F#
#FFF#FFF#T#F#FFF#FFF#TTT#F#FFF#
#F#F#####T#F#F#F###F###T#F#F###
#F#F#TTTTT#F#F#FFFFF#TTT#F#F#F#
#F#F#T#####F#F#######T###F#F#F#
#F#F#TTT#FFFFF#FFF#TTT#FFF#FFF#
#F#F###T#F#######F#T###F#####F#
#F#FFF#T#FFFFFFFFF#T#F#FFFFFFF#
#F###F#T###########T#F#F#######
#FFF#F#T#TTTTT#TTT#T#F#F#FFFFF#
###F#F#T#T###T#T#T#T#F#F###F#F#
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

## Case 146 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 286 false positives and 2 misses. loop12 F1 `0.5248` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#TTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#TTTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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

### loop6

```text
###############################
#TTTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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

### loop10

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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

### loop12

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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

## Case 27 (final failure)

Loop gain: `0.0024`. First loop F1 `0.5232` with 286 false positives and 2 misses. loop12 F1 `0.5256` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop2

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTMM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop6

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop10

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop12

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####T#
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
#F#F#F#########T#T#F###F#F###F#
#FFF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

## Case 255 (final failure)

Loop gain: `0.0033`. First loop F1 `0.5223` with 287 false positives and 2 misses. loop12 F1 `0.5256` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop2

```text
###############################
#F#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop4

```text
###############################
#F#FFFFFFFFFFF#FFFFTTTTTTTTTTM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop6

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTTM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop10

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTTM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop12

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTTM#
#F#F#F#####F###F###T#########T#
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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

## Case 390 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5248` with 287 false positives and 1 misses. loop12 F1 `0.5256` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop2

```text
###############################
#.FFFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop4

```text
###############################
#.FFFTTTTT#TTT#FFTTT#TTT#TTTMM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop6

```text
###############################
#.FFFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop10

```text
###############################
#.FFFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop12

```text
###############################
#.FFFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###T#
#F#F#T#FFTTT#T#F#T#TTT#T#TTT#T#
#F#F#T#######T###T#####T###T#T#
#FFF#TTT#TTT#TTT#TTT#FFTTTTT#S#
###F###T#T#T###T#F#T#########F#
#F#FFF#TTT#T#FFT#F#TTT#FFFFFFF#
#F###F#####T###T#####T#F#######
#FFFFF#TTT#TTT#TTTTTTT#F#FFFFF#
#######T#T#F#T#########F#####F#
#TTTTTTT#T#F#TTT#F#FFF#FFF#FFF#
#T#######T#F###T#F#F#F###F#F#F#
#T#FFFFF#T#F#TTT#FFF#F#FFF#F#F#
#T#####F#T###T###F###F#F###F#F#
#TTTTT#F#TTTTT#F#F#F#FFF#FFF#F#
#F###T#F#######F#F#F#####F###F#
#FFF#T#F#FFFFF#F#F#FFFFFFF#FFF#
#F###T#F#F#F#F#F#F#######F#F###
#F#TTT#FFF#F#FFF#F#FFFFFFF#FFF#
#F#T#######F#####F#F#F#########
#F#TTT#FFFFF#FFF#F#F#F#FFFFFFF#
#F###T#F#####F#F#F#F###F#####F#
#F#F#T#FFF#FFF#F#FFF#FFFFF#FFF#
#F#F#T###F#F###F###F#F#####F#F#
#F#F#TTT#FFFFF#FFFFF#FFF#FFF#F#
#F#F###T###############F#F#####
#F#FFF#T#TTT#FFFFFFFFFFF#FFFFF#
#F#F#F#T#T#T#G###############F#
#FFF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

## Case 449 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5281` with 286 false positives and 0 misses. loop12 F1 `0.5256` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

### loop2

```text
###############################
#TTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

### loop4

```text
###############################
#TTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

### loop6

```text
###############################
#MTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

### loop10

```text
###############################
#MTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

### loop12

```text
###############################
#MTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#F#
#TTT#TTT#T#F#FFF#FFFFF#FFFFF#F#
###T#####T#F###F#######F#####F#
#TTT#FFF#T#F#F#FFFFFFFFF#FFFFF#
#T#####F#T###F###########F#####
#T#FFFFF#GFFFFFFFFFFFF#FFF#FFF#
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
#T###F#T#######T###T#####T#T#T#
#TTT#F#TTTTTTT#T#FFTSFFF#TTT#T#
###T#F#######T#T#########F###T#
#F#T#FFFFF#TTT#TTTTTTTTT#F#TTT#
#F#T#######T###########T###T#F#
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#F#
###############################
```

## Case 505 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5256` with 286 false positives and 1 misses. loop12 F1 `0.5256` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

### loop6

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

### loop12

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########T#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFFF#
###############################
```

## Case 91 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5256` with 287 false positives and 0 misses. loop12 F1 `0.5265` with 286 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFF#FFFFFFFFFFFFF#F#FGTTT#.#
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
#FFFFF#FFFFFFFFFFFFF#F#FGTTT#.#
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
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
###############################
```

### loop6

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
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
###############################
```

### loop10

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
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
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
#F###T###T#F#T###############T#
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTT#
###############################
```

## Case 110 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5265` with 286 false positives and 0 misses. loop12 F1 `0.5265` with 286 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

### loop6

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

### loop12

```text
###############################
#.#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
#F#T#T#F#F#F#F#F#F#F#F#####F#F#
#TTT#T#FFF#FFF#FFF#F#F#F#FFF#F#
#T###T#####F#######F#F#F#F#####
#T#F#T#TTT#FFFFFFF#FFFFF#FFFFF#
#T#F#T#T#T###F###############F#
#TTT#TTT#T#F#F#FFF#FFFFFFF#FFF#
###T#####T#F#F#F#F#F#####F#F###
#F#T#F#TTT#FFF#F#FFFFF#F#F#FFF#
#F#T#F#T#####F#F#######F#F###F#
#F#T#F#T#TTT#F#FFF#F#FFF#FFF#F#
#F#T#F#T#T#T#####F#F#F#####F#F#
#F#T#F#TTT#TTTTTTG#F#FFFFFFF#F#
#F#T#F#############F#########F#
#F#TTTTT#TTT#TTT#FFFFFFFFF#FFF#
#F#####T#T#T#T#T#######F#F###F#
#FFF#TTT#T#TTT#TTTTTTT#F#FFFFF#
#F#F#T###T###########T#F#####F#
#F#F#T#FFT#FFF#FFFFF#T#F#FFFFF#
#F#F#T#F#T#F#F#F#F###T#F#######
#F#F#T#F#T#F#FFF#F#TTT#FFFFFFF#
###F#T###T#######F#T#########F#
#FFF#TTT#T#TTTTT#F#TTTTTTT#SFF#
#F#F###T#T#T###T#F#######T#T###
#F#FFF#T#TTT#TTT#FFFFF#TTT#TTT#
#F#####T#F###T###F#####T#####T#
#F#FFF#T#F#TTT#FFF#FFF#TTTTTTT#
#F#F#F#T###T###F#F#F#F#######F#
#FFF#FFTTTTT#FFF#FFF#FFFFFFFFF#
###############################
```

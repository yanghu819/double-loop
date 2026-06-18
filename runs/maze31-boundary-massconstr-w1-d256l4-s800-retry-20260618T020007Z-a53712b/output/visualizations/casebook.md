# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 261 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5239` with 288 false positives and 1 misses. loop12 F1 `0.5215` with 288 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTT#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
###############################
```

### loop2

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTT#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FFF#
###############################
```

## Case 62 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5215` with 289 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

### loop2

```text
###############################
#MTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#F#
###############################
```

## Case 226 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5215` with 289 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop2

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FF.#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop4

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FF.#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop6

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FF.#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop10

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FF.#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop12

```text
###############################
#.#TTT#TTTFF#TTT#F#FFFFFFF#FF.#
#F#T#T#T#T###T#T#F#F#F###F#F#F#
#TTT#TTT#TTTTT#T#FFF#FFF#F#F#F#
#T#############T###F###F#F###F#
#T#FFFFFFF#TTT#TTT#F#F#F#FFFFF#
#T#F#####F#T#T#F#T#F#F#F#####F#
#T#F#FFF#F#T#T#F#T#FFF#FFF#FFF#
#T#F###G###T#T###T###F###F#F###
#TTT#F#T#TTT#T#TTT#FFF#F#F#FFF#
###T#F#T#T###T#T###F###F#F#####
#TTT#F#TTT#F#T#T#F#FFFFF#FFFFF#
#T###F#####F#T#T#F#####F#####F#
#TTTTT#TTT#TTT#T#FFF#FFF#FFF#F#
#####T#T#T#T###T#F#F#F###F#F#F#
#FFF#TTT#T#TTTTT#F#FFFFFFF#F#F#
#F#######T#############F#####F#
#TTTTT#TTT#FFF#FFFFFFF#F#FFFFF#
#T###T#T###F#F#F#####F###F###F#
#T#TTT#T#FFF#F#F#FFFFF#FFF#F#F#
#T#T###T#F###F#F#F#####F###F#F#
#T#TTTTT#FFF#FFF#F#FFFFF#FFF#F#
#T#########F#####F#F###F#F#F#F#
#TTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#TTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#T###T#T#T#F#######F#F#F###F#F#
#T#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#T#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0016`. First loop F1 `0.5239` with 289 false positives and 0 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#FFF#
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
#.FF#FFFFFFFFF#FFF#FFFFFFF#FFF#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop4

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#FF.#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop6

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#FF.#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop10

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#FF.#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

### loop12

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#FF.#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
###############################
```

## Case 393 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5215` with 289 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FFF#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FFF#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FF.#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop6

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FF.#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FF.#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop12

```text
###############################
#.FFFFFFFFFFFF#FFFFFFFFFFF#FF.#
#F#######F#####F#F#######F#F#F#
#F#FFFFF#FFFFFFF#FFF#FFF#FFF#F#
#F#F###F###########F#F#######F#
#F#F#F#FFFFF#FFF#SFF#FFFFFFF#F#
#F#F#F#####F#F#F#T#####F###F#F#
#F#F#FFFFF#FFF#F#TTTTT#FFF#FFF#
#F#F#F#F#####F#######T#######F#
#FFF#F#FFF#FFF#TTTTT#TTT#TTT#F#
#F#####F#F#F###T###T###T#T#T#F#
#F#TTT#F#F#FFF#T#F#TTT#T#T#T#F#
###T#T#F#F#F###T#F###T#T#T#T#F#
#GTT#T#F#F#F#TTT#F#TTT#TTT#T#F#
#####T###F#F#T###F#T#####F#T#F#
#FFF#TTT#F#F#T#TTTTT#TTT#F#T#F#
#F#F###T#F#F#T#T#####T#T###T###
#F#FFF#T#FFF#T#T#FFTTT#TTT#TTT#
#F#####T#####T#T###T#####T###T#
#TTTTTTT#TTTTT#TTT#T#FFF#TTTTT#
#T#######T#######T#T#F#F#####F#
#T#FFTTTTT#FFFFF#TTT#F#F#FFF#F#
#T###T#####F###F#####F###F#F#F#
#TTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#TTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#T###F###T#F#####F#F#F#F#F#F#F#
#T#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#T#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

## Case 146 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5263` with 287 false positives and 1 misses. loop12 F1 `0.5239` with 287 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTTT#
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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

### loop2

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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

### loop4

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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

### loop6

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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
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
#.FFFFFTTTTTTTTT#FFFFF#FFFFFFF#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0048`. First loop F1 `0.5287` with 287 false positives and 0 misses. loop12 F1 `0.5239` with 287 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTT#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#TTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

### loop2

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTT#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

### loop4

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTM#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

### loop6

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTM#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

### loop10

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTM#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

### loop12

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTTTM#
#F#####T#T#######T#F#########T#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFFFF#
###############################
```

## Case 27 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5272` with 287 false positives and 0 misses. loop12 F1 `0.5248` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTT#
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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop2

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTT#
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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

### loop4

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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
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
#.FF#FFFFFFFFFFTTT#FFFFF#FFFFF#
###############################
```

## Case 255 (final failure)

Loop gain: `-0.0016`. First loop F1 `0.5263` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTTTT#
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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop4

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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

## Case 390 (final failure)

Loop gain: `-0.0016`. First loop F1 `0.5263` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTTT#
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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

### loop4

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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
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
#.FF#F#TTT#TTTFF#FFFFFFFFFFFFF#
###############################
```

# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 390 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5199` with 287 false positives and 3 misses. loop12 F1 `0.5199` with 287 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTMM#
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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop2

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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop4

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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop6

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTMM#
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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop10

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTMM#
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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop12

```text
###############################
#FFFFTTTTT#TTT#FFTTT#TTT#TTTMM#
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
#FFF#F#TMT#TTTFF#FFFFFFFFFFFF.#
###############################
```

## Case 261 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5239` with 288 false positives and 1 misses. loop12 F1 `0.5215` with 288 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop2

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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop4

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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop6

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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop10

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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop12

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
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

## Case 65 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5223` with 288 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

### loop2

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

### loop4

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFF..#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

### loop6

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFF..#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

### loop10

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFF..#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

### loop12

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
#F#F#F#F#F#F#F#F#F#F#F#####F###
#FFF#F#F#F#F#F#F#F#F#F#TTT#FFF#
#F###F#F#F#F#F#F#F#F###T#T###F#
#FFF#FFF#FFF#F#FFF#FFF#T#TTTTT#
#F#####F#####F#F#####F#T#####T#
#F#TTT#FFFFF#F#F#FFF#G#T#TTTTT#
###T#T#####F#F###F#F#T#T#T#####
#TTT#TTTTT#F#FFFFF#F#TTT#TTTTT#
#T#######T#F#######F#########T#
#T#FFF#TTT#F#FFF#FFF#FFFFFFF#T#
#T#F###T###F###F#F#F#F###F###T#
#T#FFF#TTT#FFFFF#F#F#FFF#F#TTT#
#T#F#F###T###F###F#F#F#F###T###
#T#F#STT#T#FFF#FFF#F#F#FFF#T#F#
#T#F###T#T#F###F###F#####F#T#F#
#T#FFF#TTT#F#FFF#F#FFFFFFF#T#F#
#T###F#######F###F#######F#T#F#
#T#FFF#FFFFF#F#FFF#FFFFF#F#T#F#
#T#F#####F#F#F#F###F###F#F#T#F#
#T#F#FFFFF#FFF#FFFFFFF#F#F#T#F#
#T#F#F#####F#########F#F#F#T#F#
#T#FFF#FFFFF#TTTTTTT#F#FFF#T#F#
#T#########F#T#####T#F#####T#F#
#TTTTTTTTT#F#T#TTT#T#FFFFF#T#F#
#F#######T#F#T#T#T#T#######T#F#
#FFFFFFF#T#F#T#T#TTT#TTTTT#TTT#
#######F#T###T#T#####T###T###T#
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

## Case 91 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5223` with 288 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop4

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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop6

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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop10

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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

### loop12

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
#FFF#TTTTTFF#TTTTTTTTTTTTTTTTM#
###############################
```

## Case 143 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5223` with 288 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

### loop2

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

### loop4

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

### loop6

```text
###############################
#.FF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

### loop10

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

### loop12

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#F#
#F#F#TFF#T#FFFFFFFFFFFFF#F#FFF#
#F###T###T#F#############F###F#
#F#TTT#TTT#F#TTTTT#FFFFFFF#FFF#
#F#T###T#F###T###T###F#F###F###
#TTTFF#T#F#TTT#F#TTT#F#FFFFFFF#
#T#####T#F#T###F###T#F#########
#T#F#TTT#F#TTTTT#TTT#F#TTTTTTT#
#T#F#T###F#####T#T###F#T#####T#
#T#F#T#FFF#TTTTT#TTT#F#T#TTTTT#
#T#F#T#F###T#######T###M#T###F#
#T#F#T#FFF#T#FFF#TTT#TTT#TTT#F#
#T#F#T#####T#F#F#T###T#####T#F#
#T#F#TTTTTTT#F#F#TTT#T#FFF#T#F#
#T#F###########F#F#T#T###F#T#F#
#TTTTTTT#F#FFFFF#F#TTT#FFF#T#F#
#######T#F#F#####F#####F###T###
#FFFFF#TTT#F#FFF#FFF#FFFFF#TTT#
#F#######T#F#F#F###F#####F###T#
#FFGTTTTTT#FFF#FFF#F#FFFFF#F#T#
#F###############F#F#F###F#F#T#
#FFFFF#FFFFFFFFF#F#F#F#FFFFF#T#
#F#####F#######F#F#F#F#####F#T#
#F#FFFFF#FFFFF#F#F#F#FFF#FFF#T#
###F#########F#F#F#F###F#####T#
#FFF#FFFFFFFFF#FFF#FFFFF#STTTT#
#F#######F###F###########F###F#
#FFFFFFFFF#FFFFFFFFFFFFFFF#FF.#
###############################
```

## Case 129 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5256` with 284 false positives and 3 misses. loop12 F1 `0.5223` with 285 false positives and 4 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTTTM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop2

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop4

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop6

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop10

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTTM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

### loop12

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTMM#
#F#####F###F#######T#########T#
#F#FFFFF#F#FFFFFFF#T#TMTTG#TTT#
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
#FFFFFFFFFFFFF#TTTTTTTTTTTTTTM#
###############################
```

## Case 12 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5240` with 286 false positives and 1 misses. loop12 F1 `0.5232` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop2

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop4

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#F..#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop6

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop10

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop12

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
#F#F#T#T#T#T#F###T#######T#F###
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
#F#F#F#####F#F#F###F#T#######T#
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

## Case 196 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5240` with 285 false positives and 2 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFF..#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

### loop2

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFFF.#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

### loop4

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFFF.#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

### loop6

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFF..#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

### loop10

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFFF.#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

### loop12

```text
###############################
#FFFFFFF#TTTTTTTTTFFFFFF#FFFF.#
#######F#T#######T#####F###F#F#
#FFFFFFF#TTTFF#TTT#FFFFFFFFF#F#
#F#########T###T#F#############
#TTT#TTT#TTT#TTT#F#F#FFF#FFFFF#
#T#M#T#T#T###T#####F#F#F#F###F#
#T#T#T#T#TTT#TTTTT#FFF#FFF#FFF#
#T#T#T#T###T#####T#########F###
#T#T#T#TFF#M#F#TTT#FFFFFFF#FFF#
#T#T#T#T###T#F#T#F#F#########F#
#T#TTT#TTTTTFF#T#FFF#FFFFF#FFF#
#T###########F#T#####F###F#F###
#T#TTTTTTTTT#F#TTT#FFF#F#FFF#F#
#T#T#######T#####T#F###F#####F#
#T#T#FFFFF#TTTTTTT#FFFFFFFFF#F#
#T#T#####F#######F#########F#F#
#T#TTTTT#FFFFFFF#F#FFF#FFF#F#F#
#T#####T#######F#F#F#F#F#F#F#F#
#TTTTT#TTTTTTT#FFF#F#F#F#FFF#F#
#####T#######T###F#F#F#F#####F#
#FFF#TTT#FGT#TTT#F#F#F#F#FFF#F#
#F#F###T###T###T#F#F#F#F#F#F#F#
#F#F#FFTTTTT#TTT#F#F#FFF#F#FFF#
#F#F#########T###F#F#####F###F#
#F#FFFFSTT#TTT#FFF#F#FFFFF#FFF#
#F#######T#T#####F#F###F#####F#
#FFF#FFF#TTT#FFF#F#FFF#FFFFF#F#
#F#F#F#F#####F#F#####F#####F#F#
#F#FFF#FFFFFFF#FFFFFFFFFFFFF#.#
###############################
```

## Case 255 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5265` with 285 false positives and 1 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop2

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop4

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop6

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop10

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop12

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
#FFFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

## Case 505 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5265` with 285 false positives and 1 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop2

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop4

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop6

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop10

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

### loop12

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTTMM#
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
#FFFFFFFFFFF#FFF#FFF#FFFFFFFF.#
###############################
```

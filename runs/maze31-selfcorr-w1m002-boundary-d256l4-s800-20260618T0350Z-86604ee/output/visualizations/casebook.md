# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 261 (final failure)

Loop gain: `-0.0040`. First loop F1 `0.5206` with 289 false positives and 2 misses. loop12 F1 `0.5166` with 288 false positives and 4 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTTTT#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FFF#
###############################
```

### loop2

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop4

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop6

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop10

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop12

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTTTM#
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
#F#########F#F#F#####F#F#F###F#
#FFFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

## Case 146 (final failure)

Loop gain: `-0.0064`. First loop F1 `0.5255` with 288 false positives and 1 misses. loop12 F1 `0.5190` with 287 false positives and 4 misses. Final exact `0.0000`.

### loop1

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
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

### loop4

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

### loop6

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

### loop10

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

### loop12

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTTTM#
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
#F#####T#######T#F###F#F#####F#
#FFFFFFTTTTTTTTT#FFFFF#FFFFFF.#
###############################
```

## Case 62 (final failure)

Loop gain: `-0.0007`. First loop F1 `0.5223` with 288 false positives and 1 misses. loop12 F1 `0.5216` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

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

### loop2

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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

### loop4

```text
###############################
#MMTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######.#
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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

### loop6

```text
###############################
#MMTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######.#
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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

### loop10

```text
###############################
#MMTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######.#
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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

### loop12

```text
###############################
#MMTFFFF#TTTTT#FFFFFFTTTGFFFF.#
#T#T#F###T###T###F###T#######.#
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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

## Case 12 (final failure)

Loop gain: `0.0017`. First loop F1 `0.5206` with 290 false positives and 1 misses. loop12 F1 `0.5223` with 288 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFTTT#TTT#FFFFTTTTTTTTT#FFF#
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
#.#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
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
#.#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
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
#.#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
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
#.#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
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
#.#FFTTT#TTT#FFFFTTTTTTTTT#FF.#
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

## Case 65 (final failure)

Loop gain: `-0.0016`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5232` with 287 false positives and 1 misses. Final exact `0.0000`.

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
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTT#
###############################
```

### loop2

```text
###############################
#.#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
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
#.#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
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
#.#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
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
#.#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
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
#.#FFF#FFF#FFFFF#F#FFF#FFFFFF.#
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

## Case 27 (final failure)

Loop gain: `0.0017`. First loop F1 `0.5215` with 288 false positives and 2 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#.FF#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#..F#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#FFF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop6

```text
###############################
#..F#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#FFF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop10

```text
###############################
#..F#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#FFF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

### loop12

```text
###############################
#..F#FFFFF#FFFFFFF#TGF#TTTTTTM#
#F#F###F###F###F#F#T###T#####M#
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
#FFF#FFFFFFFFFFTTT#FFFFF#FFFF.#
###############################
```

## Case 390 (final failure)

Loop gain: `0.0002`. First loop F1 `0.5230` with 289 false positives and 1 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

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
#..FFTTTTT#TTT#FFTTT#TTT#TTTTM#
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
#..FFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###M#
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
#FFF#F#TTT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop6

```text
###############################
#..FFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###M#
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
#FFF#F#TTT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop10

```text
###############################
#..FFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###M#
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
#FFF#F#TTT#TTTFF#FFFFFFFFFFFF.#
###############################
```

### loop12

```text
###############################
#..FFTTTTT#TTT#FFTTT#TTT#TTTTM#
#F###T###T#T#T#F#T#T#T#T#T###M#
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
#FFF#F#TTT#TTTFF#FFFFFFFFFFFF.#
###############################
```

## Case 449 (final failure)

Loop gain: `-0.0007`. First loop F1 `0.5239` with 288 false positives and 1 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

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

### loop2

```text
###############################
#MTTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#.#
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
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#.#
###############################
```

### loop4

```text
###############################
#MMTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#.#
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
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#.#
###############################
```

### loop6

```text
###############################
#MMTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#.#
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
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#.#
###############################
```

### loop10

```text
###############################
#MMTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#.#
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
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#.#
###############################
```

### loop12

```text
###############################
#MMTTT#TTT#FFF#FFF#FFFFFFFFFF.#
#T###T#T#T#F#F###F#F#######F#.#
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
#FFTTTTTTTTT#FFFFFFFFFFTTTTT#.#
###############################
```

## Case 505 (final failure)

Loop gain: `-0.0040`. First loop F1 `0.5272` with 287 false positives and 0 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTT#
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
#F#F#########T#T###T#########M#
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
#F#F#########T#T###T#########M#
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
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########M#
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
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########M#
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
#.#FFFFFFFFFFTTT#FFTTTTTTTTTTM#
#F#F#########T#T###T#########M#
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

## Case 131 (final failure)

Loop gain: `-0.0048`. First loop F1 `0.5287` with 286 false positives and 1 misses. loop12 F1 `0.5239` with 286 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#MMTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop4

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop6

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop10

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

### loop12

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFTTG#
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
#FFFFFFFFFFFFFFFFFFF#FFFFF#TTM#
###############################
```

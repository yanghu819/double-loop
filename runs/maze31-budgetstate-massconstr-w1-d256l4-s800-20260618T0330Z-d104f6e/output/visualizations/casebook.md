# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 261 (final failure)

Loop gain: `-0.0074`. First loop F1 `0.5183` with 286 false positives and 4 misses. loop12 F1 `0.5109` with 286 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

### loop2

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

### loop4

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#F#########F#F#F#####F#F#F###F#
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

### loop6

```text
###############################
#MTTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#F#########F#F#F#####F#F#F###F#
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

### loop10

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#F#########F#F#F#####F#F#F###F#
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

### loop12

```text
###############################
#MMTTTTTTTTT#TTTTTTTTTTTTTTMMM#
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
#F#########F#F#F#####F#F#F###F#
#.FFFFFFFF#FFFFFFFFF#FFFFF#F..#
###############################
```

## Case 146 (final failure)

Loop gain: `-0.0074`. First loop F1 `0.5207` with 285 false positives and 4 misses. loop12 F1 `0.5133` with 285 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop2

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTTMM#
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
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop4

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTMMM#
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
#F#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop6

```text
###############################
#MTTTTTT#FFFFTTTTTTTTTTT#TTMMM#
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
#F#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop10

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTMMM#
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
#F#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

### loop12

```text
###############################
#MMTTTTT#FFFFTTTTTTTTTTT#TTMMM#
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
#F#####T#######T#F###F#F#####F#
#.FFFFFTTTTTTTTT#FFFFF#FFFFF..#
###############################
```

## Case 131 (final failure)

Loop gain: `-0.0058`. First loop F1 `0.5207` with 284 false positives and 5 misses. loop12 F1 `0.5150` with 285 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#..FFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop2

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop4

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop6

```text
###############################
#MTTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop10

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

### loop12

```text
###############################
#MMTTTTTTT#TTTFF#F#TTT#F#FFMMG#
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
#.FFFFFFFFFFFFFFFFFF#FFFFF#TMM#
###############################
```

## Case 129 (final failure)

Loop gain: `-0.0023`. First loop F1 `0.5215` with 286 false positives and 4 misses. loop12 F1 `0.5191` with 283 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFF#FFFFF#FFFFFFTTTTTTTTTMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTMMT#
###############################
```

### loop2

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTMMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTMM#
###############################
```

### loop4

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTMMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTMM#
###############################
```

### loop6

```text
###############################
#.FFFF#FFFFF#FFFFFFTTTTTTTTMMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTMM#
###############################
```

### loop10

```text
###############################
#..FFF#FFFFF#FFFFFFTTTTTTTTMMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTMM#
###############################
```

### loop12

```text
###############################
#..FFF#FFFFF#FFFFFFTTTTTTTTMMM#
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
#.FFFFFFFFFFFF#TTTTTTTTTTTTTMM#
###############################
```

## Case 319 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5191` with 286 false positives and 3 misses. loop12 F1 `0.5200` with 285 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#...#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop2

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#...#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop4

```text
###############################
#.FF#FFFFFFFFF#FFF#FFFFFFF#...#
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
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop6

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
#T#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop10

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
#T#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

### loop12

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
#T#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

## Case 505 (final failure)

Loop gain: `0.0017`. First loop F1 `0.5183` with 286 false positives and 4 misses. loop12 F1 `0.5200` with 284 false positives and 4 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

### loop2

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

### loop4

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

### loop6

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

### loop10

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

### loop12

```text
###############################
#.#FFFFFFFFFFTTT#FFTTTTTTTTMMM#
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
#.FFFFFFFFFF#FFF#FFF#FFFFFFF..#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0065`. First loop F1 `0.5265` with 284 false positives and 2 misses. loop12 F1 `0.5200` with 283 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTTMM#
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
#TTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop2

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTMMM#
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
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
###############################
```

### loop4

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTMMM#
#.#####T#T#######T#F#########T#
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
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop6

```text
###############################
#.FFFFFTTT#FFFFFFTTTTTTTTTTMMM#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop10

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTMMM#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

### loop12

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTMMM#
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
#T###T#####T#########T#F#F###F#
#MTTTTFFFF#TTTTTTTTTTTFF#FFF..#
###############################
```

## Case 123 (final failure)

Loop gain: `-0.0104`. First loop F1 `0.5311` with 284 false positives and 2 misses. loop12 F1 `0.5207` with 282 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTT#FFTTT#FFFFFFTTTTT#TMM#
#T#####T#F#T#T#######T###T#T#T#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#FFFFFFFFFFFFF#FFF#F#FFF#FFFF.#
###############################
```

### loop2

```text
###############################
#MTTTTTT#FFTTT#FFFFFFTTTTT#MMM#
#T#####T#F#T#T#######T###T#T#T#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#.FFFFFFFFFFFF#FFF#F#FFF#FFFF.#
###############################
```

### loop4

```text
###############################
#MTTTTTT#FFTTT#FFFFFFTTTTT#MMM#
#M#####T#F#T#T#######T###T#T#T#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#.FFFFFFFFFFFF#FFF#F#FFF#FFF..#
###############################
```

### loop6

```text
###############################
#MTTTTTT#FFTTT#FFFFFFTTTTT#MMM#
#M#####T#F#T#T#######T###T#T#M#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#.FFFFFFFFFFFF#FFF#F#FFF#FFF..#
###############################
```

### loop10

```text
###############################
#MMTTTTT#FFTTT#FFFFFFTTTTT#MMM#
#M#####T#F#T#T#######T###T#T#M#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#.FFFFFFFFFFFF#FFF#F#FFF#FFF..#
###############################
```

### loop12

```text
###############################
#MMTTTTT#FFTTT#FFFFFFTTTTT#MMM#
#M#####T#F#T#T#######T###T#T#M#
#T#FFF#T#F#T#TTTTTTTTT#F#T#T#T#
#T#F#F#T###T###########F#T#T#T#
#T#F#FFTTTTT#FFFFF#TTTTT#TTT#T#
#T#############F###T###T#####T#
#T#FFFFFFFFFFF#FFF#TTT#TTTTT#T#
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
#T###F#####G#F#########F#F#F#F#
#TTT#TTTTT#T#FFF#FFFFFFF#F#F#F#
###T#T###T#T###F#F#######F#F#F#
#F#TTT#F#TTTFF#F#F#FFF#FFF#F#F#
#F#####F#####F#F#F#F#F#F#F#F#F#
#.FFFFFFFFFFFF#FFF#F#FFF#FFF..#
###############################
```

## Case 255 (final failure)

Loop gain: `-0.0007`. First loop F1 `0.5216` with 285 false positives and 3 misses. loop12 F1 `0.5209` with 283 false positives and 4 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTMMM#
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
#.#FFFFFFFFFFF#FFFFTTTTTTTTTMM#
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
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#.#
###############################
```

### loop4

```text
###############################
#.#FFFFFFFFFFF#FFFFTTTTTTTTMMM#
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
#.#FFFFFFFFFFF#FFFFTTTTTTTTMMM#
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
#.#FFFFFFFFFFF#FFFFTTTTTTTTMMM#
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
#.#FFFFFFFFFFF#FFFFTTTTTTTTMMM#
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

## Case 498 (final failure)

Loop gain: `-0.0080`. First loop F1 `0.5296` with 284 false positives and 2 misses. loop12 F1 `0.5216` with 282 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#TTTTTTT#TTTTT#TTT#FFTTTTTTTMM#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#FFF#
###############################
```

### loop2

```text
###############################
#MTTTTTT#TTTTT#TTT#FFTTTTTTMMM#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#F..#
###############################
```

### loop4

```text
###############################
#MTTTTTT#TTTTT#TTT#FFTTTTTTMMM#
#M#####T#T#F#T#T#T#F#T#######T#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#F..#
###############################
```

### loop6

```text
###############################
#MTTTTTT#TTTTT#TTT#FFTTTTTTMMM#
#M#####T#T#F#T#T#T#F#T#######M#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#F..#
###############################
```

### loop10

```text
###############################
#MTTTTTT#TTTTT#TTT#FFTTTTTTMMM#
#M#####T#T#F#T#T#T#F#T#######M#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#F..#
###############################
```

### loop12

```text
###############################
#MTTTTTT#TTTTT#TTT#FFTTTTTTMMM#
#M#####T#T#F#T#T#T#F#T#######M#
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
#.FFFFFF#FFFFTTTTT#FFFFFFF#F..#
###############################
```

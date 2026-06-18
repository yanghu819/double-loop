# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 0 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

### loop2

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

### loop6

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

### loop10

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

### loop12

```text
###############################
#FFF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#F#F#F#####F#F#T###T#F#####F###
#F#FFF#FFFFF#F#T#TTT#FFFFF#FFF#
#F#########F#F#T#T#######F#F#F#
#F#TTTTTTT#FFF#T#T#FFFFFFF#F#F#
#F#T#####T#####T#T#####F#####F#
#F#T#FFF#T#TTT#T#TTTTT#FFFFF#F#
#F#T#F#F#T#T#T#T#####T#####F#F#
#TTT#F#F#TTT#T#T#FFF#TTTTT#FFF#
#T###F#F#####T#T#F#F#####T###F#
#TTT#F#FFF#TTT#T#F#FFFFF#T#FFF#
###T###F#F#T###T#######F#T#F###
#F#TTT#F#F#TTTTTFFFFFFFF#T#FFF#
#F###T#F#######F#######F#T###F#
#TTTTTFF#TTTTT#F#TTT#FFF#T#FFF#
#T#######T###T###T#T#F###T#F###
#T#F#FFFST#TTT#TTT#T#FFF#T#FFF#
#T#F#F#####T###T###T#####T###F#
#T#F#FFF#F#T#FFT#F#TTTTT#TTT#F#
#T#F###F#F#T###T#F#####T###T###
#T#FFF#F#F#TTTTT#F#FFF#TTT#TTT#
#T#F#F#F#F#######F#F#F###T###T#
#TGF#F#F#FFF#FFFFF#F#FFF#T#TTT#
###F###F#F#F#F#F###F#F###T#T#F#
#FFF#FFF#F#FFF#F#FFF#FFF#TTT#F#
#####F###F#######F#####F#####F#
#FFFFF#FFF#FFFFF#F#FFF#FFFFF#F#
#######F#F#F###F#F#F#F###F###F#
#FFFFFFF#FFFFF#FFFFF#FFF#FFFFF#
###############################
```

## Case 65 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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

### loop4

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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

### loop6

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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

### loop10

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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

### loop12

```text
###############################
#F#FFF#FFF#FFFFF#F#FFF#FFFFFFF#
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

## Case 110 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFFF#
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

## Case 143 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

### loop2

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

### loop4

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

### loop6

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

### loop10

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

### loop12

```text
###############################
#FFF#TTTTTFFFF#FFFFFFFFFFFFF#F#
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
#T#F#T#F###T#######T###T#T###F#
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
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
###############################
```

## Case 162 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

### loop2

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

### loop4

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

### loop6

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

### loop10

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

### loop12

```text
###############################
#FFFFFFF#FFFFFFF#FFTTTTTTT#FFF#
#F#F#####F###F#F#F#T#####T###F#
#F#F#FFFFF#FFF#F#F#T#FFF#TTT#F#
#F###F#####F#####F#T#F#####T#F#
#FFF#F#TTT#FFFFFFF#T#TTTTT#T#F#
#F#F#F#T#T###F#####T#T###T#T#F#
#F#FFF#T#TTT#F#FFF#T#TTT#TTT#F#
#######T###T#F###F#T###T#####F#
#TTT#TTT#TTT#FFF#F#TTT#T#FFGFF#
#T#T#T###T#####F#F###T#T#F#T#F#
#T#TTT#F#TTTTT#FFFFF#T#T#F#T#F#
#T#####F#####T###F###T#T###T#F#
#TFF#FFF#TTTTT#FFF#TTT#TTT#T#F#
#T#F#F#F#T#########T#####T#T###
#T#F#F#F#TTTTT#TTTTT#FFF#T#TTT#
#T#F#F#######T#T#####F#F#T###T#
#T#F#FFFFF#TTT#T#FFFFF#TTT#TTT#
#T#F#F###F#T###T#######T###T#F#
#S#F#FFF#FFTTT#T#FFFFF#TTTTT#F#
###F#F#######T#T#F###F#######F#
#FFF#F#FFF#F#T#T#FFF#FFFFFFF#F#
#F#####F#F#F#T#T###F#######F###
#F#FFFFF#F#FFT#TFF#F#F#FFF#FFF#
#F#F#####F###T#T#F#F#F#F#F###F#
#FFF#FFF#FFF#TTT#F#F#F#F#FFF#F#
#F#####F###F#####F#F#F#F#F###F#
#F#FFFFF#FFF#FFF#F#F#FFF#FFF#F#
#F#F###F#F###F#F#####F#####F#F#
#FFF#FFF#FFFFF#FFFFFFFFFFF#FFF#
###############################
```

## Case 229 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

### loop2

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

### loop4

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

### loop6

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

### loop10

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

### loop12

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFFF#
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
#####F#F#F#####F#F###F#####T#T#
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTT#
###############################
```

## Case 319 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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

### loop6

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

### loop10

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

### loop12

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

## Case 393 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFFFFFFFF#FFFFFFFFFFF#FFF#
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
#TTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

## Case 395 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop2

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop4

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop6

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop10

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop12

```text
###############################
#FFFFFFF#FFFFFFF#FFFFF#FFFFFFF#
#######F#F###F#F#F#####F###F#F#
#TTTTT#FFFFF#F#FFF#FFFFFFF#F#F#
#T###T#####F#F#####F#######F#F#
#T#FSTFFFF#F#FFFFFFF#FFFFFFF#F#
#T#######F###########F#######F#
#T#TTTTT#FFF#FFFFFFFFF#F#FFF#F#
#T#T###T###F#F###F#####F#F#F#F#
#T#T#F#TTT#F#F#FFF#FFF#F#F#FFF#
#T#T#F###T#F#F#####F#F#F#F#####
#T#TTT#TTT#FFF#FFFFF#F#FFF#FFF#
#T###T#T#######F#####F#F#####F#
#TTT#T#T#TTT#FFF#FFF#F#FFFFFFF#
#F#T#T#T#T#T#F#####F#F#########
#F#TTT#TTT#T#F#FFF#F#FFFFFFFFF#
#########F#T#F#F#F#G#########F#
#TTT#TTT#F#T#FFF#FFT#FFF#FFF#F#
#T#T#T#T###T#######T###F#F#F#F#
#T#TTT#T#TTTFF#FFF#TTTFF#F#F#F#
#T#####T#T###F#F#####T###F#F#F#
#TTTTT#TTT#F#F#FFFFF#T#FFF#FFF#
#####T#####F#F#####F#T#########
#TTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#T#T#T#F#################T#F###
#T#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#T#####F#######T###T#####T###F#
#TTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#F###T###T###T#T#F###T#T###F#F#
#FFF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

## Case 420 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

### loop2

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

### loop4

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

### loop6

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

### loop10

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

### loop12

```text
###############################
#FFF#FFFFF#FFFFFFFFFFF#F#FFFFF#
#F###F#F###F#####F###F#F#F#F#F#
#FFFFF#F#FFF#FFF#F#FFFFF#F#F#F#
#F#####F#F#####F#F#######F#F#F#
#FFF#F#FFFFFFFFF#FFF#FFFFF#F#F#
###F#F#############F#F#####F#F#
#FFF#F#FFFFF#TTTTT#FFF#FFF#F#F#
#F###F#F###F#T###T#F#####F#F###
#FFFFF#FFF#F#T#F#T#FFFFF#F#FFF#
#########F#F#T#F#T#####F#F###F#
#FFFFFFFFF#F#T#FFTTT#FFFFF#FFF#
#F#########F#T#####T#######F###
#F#FFFFFFF#F#TTT#F#T#TTTTT#FFF#
#F###F###F#F###T#F#T#T###T###F#
#FFF#F#FFF#F#TTT#F#T#TTT#TTTTT#
###F#F#####F#T###F#T###T#####T#
#FFF#FFG#FFF#TTT#F#T#TTT#TTTTT#
#F#####T#F###F#T#F#T#T###T#####
#FFF#TTT#F#FFF#T#TTT#TTT#SFFFF#
#F#F#T###F#####T#T#####T#####F#
#F#F#TTTFF#TTTTT#TTTTT#T#FFF#F#
###F###T###T#########T#T#F#F#F#
#F#FFF#TTT#TTT#F#TTTTT#T#F#FFF#
#F#F#F###T###T#F#T#####T###F###
#F#F#F#F#T#TTT#TTT#TTT#TTT#FFF#
#F###F#F#T#T###T###T#T###T###F#
#FFFFF#TTT#T#F#T#FFT#T#TTT#F#F#
#######T###T#F#T###T#T#T###F#F#
#FFFFFFTTTTT#FFTTTTT#TTT#FFFFF#
###############################
```

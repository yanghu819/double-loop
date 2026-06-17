# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 0 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFF#F#TTTTTFF#FFFFFFF#
#.#F#F#####F#F#T###T#F#####F###
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

## Case 12 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

### loop2

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

### loop4

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

### loop6

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

### loop10

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

### loop12

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
#F#FFFFFFFFF#FFF#FFF#TTTTTTTTT#
###############################
```

## Case 62 (final failure)

Loop gain: `0.0016`. First loop F1 `0.5215` with 289 false positives and 1 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

### loop2

```text
###############################
#TTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#TTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#TTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#TTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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
#TTTFFFF#TTTTT#FFFFFFTTTGFFFFF#
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

## Case 65 (final failure)

Loop gain: `0.0024`. First loop F1 `0.5206` with 290 false positives and 1 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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
#FFFFFFF#TTTTT#TTTTTTTFF#TTTTM#
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

## Case 71 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFF#FFFFFFFFFFFFFFF#FF.#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFFFF#FFFFFFFFFFFFFFF#FFF#
#######F#F#F###########F#F#F###
#TTTTT#F#FFF#FFF#FFF#FFF#F#FFF#
#T###T#######F#F#F#F#F###F###F#
#TTT#TTTTTGFFF#F#F#FFF#FFF#FFF#
###T###########F#F#######F#F#F#
#TTT#F#FFFFF#FFF#FFFFFFF#F#F#F#
#T###F#F###F#####F#####F#F#F#F#
#T#FFFFF#FFFFF#FFF#F#FFF#FFF#F#
#T#######F#F###F###F#F#########
#T#FFFFFFF#F#FFF#FFF#FFFFFFFFF#
#T#F#########F#####F#########F#
#T#FFF#FFFFFFF#FFFFF#TTTTTTTTT#
#T###F#F#######F#F###T#######T#
#TTT#F#F#FFFFF#F#F#TTT#FFTTT#T#
###T#F#F#F###F###F#T#####T#T#T#
#F#T#FFFFF#FFFFFFF#TTTTT#T#T#T#
#F#T#F#########F#######T#T#T#T#
#F#T#F#TTTTT#FFF#TTTTT#T#T#TTT#
#F#T###T###T#####T###T#T#T#####
#F#TTT#T#F#TTT#TTT#F#TTT#TTT#F#
#F###T#T#F###T#T###F#######T#F#
#FFF#TTT#FFF#T#TTT#FFFFFFF#T#F#
#F#######F###T#F#T#######F#T#F#
#F#TTTTT#TTTTT#F#T#FFFFFFF#T#F#
#F#T###T#T#####F#T#F#####F#S#F#
#F#TTT#TTT#FFFFF#T#F#FFF#F#FFF#
#F###T###########T#F#F#F#####F#
#FFFFTTTTTTTTTTTTT#FFF#FFFFFFF#
###############################
```

## Case 77 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFTTTTTTT#FFFFFFFFFFFFFFFF.#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop2

```text
###############################
#FFFFTTTTTTT#FFFFFFFFFFFFFFFFF#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFFFTTTTTTT#FFFFFFFFFFFFFFFFF#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop6

```text
###############################
#FFFFTTTTTTT#FFFFFFFFFFFFFFFFF#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop10

```text
###############################
#FFFFTTTTTTT#FFFFFFFFFFFFFFFFF#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop12

```text
###############################
#FFFFTTTTTTT#FFFFFFFFFFFFFFFFF#
#####T#####T#####F###########F#
#TTTTT#STTTT#FFF#F#FFFFF#FFFFF#
#T#####F#####F#F###F###F#F###F#
#T#FFF#FFFFFFF#F#FFF#F#FFF#FFF#
#T#F#####F#####F#F###F#####F###
#T#FFFFFFF#TTT#FFF#FFF#FFF#F#F#
#T#F#######T#T#####F#F#F###F#F#
#T#FFF#TTTTT#TTT#FFF#F#FFF#F#F#
#T#####T#######T#F#F#F###F#F#F#
#TTTTTTT#FFFFF#T#F#F#F#FFF#F#F#
#F###########F#T#F#F#F#F#F#F#F#
#F#FFF#FFFFF#F#T#F#F#FFF#F#F#F#
#F#F#F#F###F#F#T###F#######F#F#
#F#F#FFF#FFFFF#TTT#FFFFF#FFF#F#
#F#F#F#####F#####T#####F#F###F#
#FFF#F#TTT#F#TTT#TTTTT#FFFFF#F#
#####F#T#T###T#T#####T#####F#F#
#FFFFF#T#T#TTT#T#TTG#TTT#F#F#F#
#F#####T#T#T###T#T#F#F#T#F#F#F#
#F#TTTTT#TTT#TTT#T#F#F#T#FFF#F#
#F#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#TTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#T#######F###########T#######F#
#TTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#FFFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

## Case 91 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

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

### loop6

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

### loop10

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

### loop12

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

## Case 110 (final failure)

Loop gain: `-0.0017`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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
#F#TTT#F#FFF#FFF#FFF#FFFFFFFF.#
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

Loop gain: `-0.0026`. First loop F1 `0.5256` with 287 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#TTTTTFFFF#FFFFFFFFFFFFF#.#
#F#F#T###T###F#########F###F#.#
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

## Case 155 (final failure)

Loop gain: `-0.0035`. First loop F1 `0.5265` with 286 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFFFFFFFFFF#TTTTT#TTTF.#
#.#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#.FFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFFFFFFFFFFFF#TTTTT#TTTF.#
#F#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#FFFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFFFFFFFFFFFF#TTTTT#TTTFF#
#F#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#FFFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFFFFFFFFFFFF#TTTTT#TTTFF#
#F#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#FFFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFFFFFFFFFFFF#TTTTT#TTTFF#
#F#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#FFFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFFFFFFFFFFFF#TTTTT#TTTFF#
#F#F#############F#T###T#T#T###
#F#FFFFF#FFFFFFFFF#T#TTT#T#TTT#
#F#####F###########T#T#F#T###T#
#FFF#FFF#TTTTT#TTTTT#T#F#T#TTT#
###F#F#F#T###T#T#####T###T#T#F#
#FFF#F#F#T#F#T#T#TTT#TTTTT#T#F#
#F###F###G#F#T#T#T#T#####F#T#F#
#FFF#FFFFF#F#T#T#T#TTTTT#F#T#F#
###F#######F#T#T#T#####T#F#T###
#FFFFFFFFFFF#TTT#TTT#F#T#F#TTT#
###############F###T#F#T#####T#
#FFFFFFFFF#FFF#F#TTT#F#TTTTTTT#
#F#######F#F#F###T###F#######F#
#FFF#FFF#F#F#FFF#T#FFFFFFFFFFF#
#F#F#F#F#F#F###F#T#######F#####
#F#F#F#FFFFF#FFF#TTTTTTT#FFFFF#
#F#F#########F###F#####T#####F#
#F#FFFFF#F#FFF#F#FFF#TTT#FFF#F#
#F#####F#F#F###F###F#T###F###F#
#F#FFFFF#F#FFF#FFFFF#T#FFFFFFF#
###F#####F###F#######T#F#######
#FFF#F#FFFFFFF#TTT#TTT#FFF#FFF#
#F###F#F#######T#T#T#####F#F#F#
#F#FFF#F#TTTTTTT#T#TTTTT#FFF#F#
#F#F#F#F#T#F#####T#####T#####F#
#F#F#FSTTT#F#FFF#T#TTT#T#F#FFF#
#F#############F#T#T#T#T#F#F###
#FFFFFFFFFFFFFFF#TTT#TTT#FFFFF#
###############################
```

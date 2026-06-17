# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 0 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

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

## Case 91 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

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

## Case 143 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

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

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

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

### loop2

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

## Case 158 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFF#F#FFF#FFFFF#FFFFFFFFF#
#F#F###F#F#F#F#F#F#F#F#######F#
#F#FFF#FFF#F#FFF#F#FFF#FFF#FFF#
#####F#####F#####F#####F#F#####
#TTT#FFF#FFF#FFF#FFF#FFF#FFF#F#
#T#G###F#F###F#F###F#F#####F#F#
#T#FFF#F#FFF#F#FFF#FFF#TTT#FFF#
#T###F#F###F###F#F#####T#T#####
#TTT#FFFFF#FFF#F#FFTTTTT#TTTTT#
#F#T#########F###F#T#########T#
#F#T#TTT#TTT#FFF#F#TTT#FFF#TTT#
#F#T#T#T#T#T###F#F###T#F###S###
#F#TTT#TTT#T#FFF#FFF#T#FFFFFFF#
#F#########T#F#####F#T#######F#
#F#TTTTTTTTT#FFF#F#F#TTTTT#FFF#
###T###F#######F#F#F#####T#F###
#TTT#FFF#FFFFFFF#FFFFFFF#T#FFF#
#T#######F###############T#####
#TTTTTTT#FFFFFFFFFFFFFFF#TTTTT#
#F#####T###############F#####T#
#FFF#F#TFFFF#FFFFFFFFF#FFFFF#T#
###F#F#T###F#####F###F#####F#T#
#F#FFF#TTT#F#FFFFF#FFF#TTT#FFT#
#F###F###T#F#F#F#######T#T###T#
#FFFFFFF#T#FFF#F#TTTTTTT#TTT#T#
#F#######T#####F#T#########T#T#
#FFF#F#TTT#FFFFF#TTTTTTTTT#TTT#
###F#F#T#################T#####
#FFFFF#TTTTTTTTTTTTTTTTTTTFFFF#
###############################
```

## Case 162 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

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

## Case 181 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5256` with 287 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFTTT#FFTTTTT#TTTTT#FFFFF#
#F#####T#T###T###T#T###T#F#F###
#F#TTTTT#TTT#TTT#TTT#F#T#F#FFF#
###T#######T#F#T#####F#S#####F#
#TTT#FFF#TTT#F#TTT#FFF#FFFFF#F#
#T#####F#T#######T#F#F#####F#F#
#T#FFFFF#TTTTTTTTT#F#FFF#FFF#F#
#T#F#F#############F###F#F###F#
#T#F#F#FFF#FFFFFFFFFFF#FFFFFFF#
#T###F#F#F#F#################F#
#TTT#F#F#F#FFF#TTTTTTTTT#FFFFF#
#F#T#F#F#F###F#T#######T#####F#
#F#T#FFF#FFFFF#T#FFFFF#T#TTT#F#
###T#F#########T#F###F#T#T#T#F#
#TTT#FFFFF#TTT#T#F#FFF#TTT#T#F#
#T#########T#T#T#F#F#F#####T#F#
#T#TTTTTTTTT#TTT#F#F#F#TTTTT#F#
#T#T###############F###T#####F#
#TTT#FFFFF#FFFFF#FFF#TTT#FFFFF#
#F#####F#F#F###F#F#F#T#########
#F#FFFFF#F#F#FFF#F#F#TTTTTTTTT#
#F#F#####F#F###F#F#F#########T#
#F#FFF#FFF#FFF#FFF#F#F#TTTTT#T#
#F###F#F###F#F#######F#T###T#T#
#F#FFF#F#FFF#FFFFFFF#F#TTT#TTT#
#F#F###F#######F###F#F###T###F#
#F#F#F#F#FFFFF#F#FFF#F#FGT#F#F#
#F#F#F#F#F###F###F#F#F#F###F#F#
#FFF#FFFFF#FFFFFFF#F#FFF#FFFFF#
###############################
```

## Case 204 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

### loop2

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

### loop4

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

### loop6

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

### loop10

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

### loop12

```text
###############################
#FFFFFFFFF#FFFFF#FFFFFFFFFFFFF#
#F###F#F###F###F#F#######F###F#
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
#F###F#T###T#F#####F#T#######T#
#FFFFF#TTTTTFFFFFFFF#TTTTTTTTT#
###############################
```

## Case 216 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

### loop12

```text
###############################
#FFFFFFFFFFFFF#FSTFFFFFFFF#FFF#
#F#######F#####F#T#######F#F#F#
#F#FFFFF#FFFFFFF#T#TTT#FFFFF#F#
#F#F#F###########T#T#T#######F#
#F#F#FFFFFFF#TTTTT#T#TTT#FFF#F#
#F#######F#F#T#####T###T#F###F#
#F#FFF#FFF#F#TTTTTTTFF#T#FFFFF#
#F#F#F#F###F###########T#F#####
#FFF#FFF#F#FFFFF#FFF#TTT#F#FFF#
#########F#####F#F###T###F###F#
#FFFFFFF#FFFFF#FFF#TTT#FFF#FFF#
#F#######F#F#####F#T###F###F#F#
#FFFFFFFFF#FFFFFFF#T#FFFFFFF#F#
#F#######F#########T###########
#FFF#TGF#FFFFFFFFF#TTT#TTTTTTT#
#F###T#F#########F###T#T#####T#
#F#TTT#F#TTTTT#FFFFF#TTTFF#TTT#
#F#T#####T###T#############T#F#
#F#TTTTT#T#F#TTT#TTT#TTTTT#T#F#
#F#####T#T#F###T#T#T#T###T#T#F#
#F#FFF#T#T#FFF#T#T#TTT#F#T#T#F#
#F###F#T#T###F#T#T#####F#T#T###
#FFF#F#TTT#FFF#T#TTTTT#TTT#TTT#
###F#F#####F#F#T#F###T#T#####T#
#F#FFF#FFFFF#F#T#F#TTT#T#TTTTT#
#F###F#####F#F#T###T###T#T###F#
#F#FFF#FFFFF#F#TTT#T#F#TTT#F#F#
#F#F###F#####F###T#T#F#####F#F#
#FFFFFFF#FFFFFFF#TTT#FFFFFFFFF#
###############################
```

## Case 226 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5248` with 288 false positives and 0 misses. loop12 F1 `0.5248` with 288 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop2

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop4

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop6

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop10

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop12

```text
###############################
#F#TTT#TTTFF#TTT#F#FFFFFFF#FFF#
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
#TTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 62 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

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

## Case 71 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

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

## Case 155 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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

## Case 204 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

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

## Case 406 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

### loop2

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

### loop4

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

### loop6

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

### loop10

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

### loop12

```text
###############################
#FFFFF#F#FFFFFFFFF#FFF#FFF#FFF#
#F###F#F#F#######F#F#F#F#F#F#F#
#TTT#FFF#FFF#FFF#F#F#FFF#FFF#F#
#T#T###F###F###F#F###F#######F#
#T#TTT#FFF#FFF#F#FFF#F#FFFFF#F#
#T###T#######F#F###F###F###F#F#
#T#F#TTTTTFF#F#FFF#F#FFF#F#F#F#
#T#F#####T#F#F###F#F#F###F#F#F#
#T#TTTTG#T#F#FFF#FFFFF#F#F#FFF#
#T#T#####T#####F###F###F#F###F#
#T#T#TTT#TTTFF#FFF#F#FFF#FFFFF#
#T#T#T#T###T#####F#F#F###F#####
#T#T#T#TTT#TTTTT#F#F#F#FFF#FFF#
#T#T#T###T#####T#F###F#F###F#F#
#T#TTT#TTT#F#TTT#FFFFF#FFF#F#F#
#T#####T#F#F#T###########F###F#
#TTTTT#T#FFF#TTTFFFFFFFF#F#FFF#
#####T#T#F#####T#####F###F#F#F#
#FFF#TTT#F#TTTTT#FFFFF#FFF#F#F#
#F#####F###T#####F#####F###F###
#FFFFFFF#TTT#F#FFFFF#FFF#F#STT#
#F#######T###F#F#####F###F#F#T#
#FFFFFFF#TTT#FFF#FFFFF#FFF#F#T#
###########T#F###F#######F#F#T#
#TTTTTTTTTTT#F#F#F#FFTTT#FFF#T#
#T###########F#F#F#F#T#T#####T#
#TTTTTTT#FFFFFFF#F#F#T#T#TTTTT#
#F#####T#########F###T#T#T###F#
#FFFFF#TTTTTTTTTTTTTTT#TTT#FFF#
###############################
```

## Case 428 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5230` with 290 false positives and 0 misses. loop12 F1 `0.5230` with 290 false positives and 0 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

### loop2

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

### loop4

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

### loop6

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

### loop10

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

### loop12

```text
###############################
#FFFFFFTTTTT#FFFFTTT#SFFFFFFFF#
#F#####T###T#####T#T#T#######F#
#FFFFF#T#TTT#TTTTT#TTT#FFF#FFF#
#####F#T#T###T#########F#F#F###
#F#FFF#T#T#FFT#F#FFFFF#F#F#F#F#
#F#F###T#T###T#F#F###F#F#F#F#F#
#FFF#TTT#TTTTT#FFFFF#F#F#F#FFF#
#F###T#F###########F#F#F#F###F#
#F#TTT#F#TTTTTTT#F#F#FFF#FFFFF#
#F#T###F#T#####T#F#F###########
#F#T#FFF#TFFFF#T#F#F#FFFFFFFFF#
###T#####T#####T#F#F#F#####F#F#
#TTT#TTTTT#TTT#T#FFF#FFF#F#F#F#
#T###T#####T#T#T#F#####F#F#F#F#
#TTTTTFFFF#T#TTT#F#FFF#FFF#F#F#
#F#######F#T#####F#F#F###F#F#F#
#F#F#FFFFF#TTTTT#FFF#FFF#F#F#F#
#F#F#F#########T#######F#F#F###
#F#G#FFFFFFFFF#T#FFF#FFF#F#FFF#
#F#T#########F#T###F#F###F###F#
#F#TTTTTTTTT#F#TTT#FFF#FFF#FFF#
#F#########T#F###T###F###F#F###
#FFF#F#TTT#T#FFF#TTT#FFF#F#FFF#
###F#F#T#T#T#######T###F#####F#
#F#FFF#T#TTT#TTTTT#TTT#F#FFFFF#
#F###F#T#####T###T###T#F#F###F#
#FFF#F#T#FFTTT#F#TTT#T#F#FFF#F#
#F###F#T###T###F###T#T#F###F#F#
#FFFFF#TTTTT#FFFFFFTTT#FFFFF#F#
###############################
```

## Case 0 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5239` with 289 false positives and 0 misses. loop12 F1 `0.5239` with 289 false positives and 0 misses. Final exact `0.0000`.

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

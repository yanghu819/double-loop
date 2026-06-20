# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 393 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5117` with 286 false positives and 6 misses. loop16 F1 `0.5084` with 287 false positives and 7 misses. Final exact `0.0000`.

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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFF.F.#
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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop6

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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

### loop10

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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MTTTTTTTT#FFF#FFFFF#FFFFFFFF.#
###############################
```

### loop16

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
#MTT#TTT#FFF#FFF#FFFFF#FFF#FFF#
###T#F#T#####F#F#F#####F#######
#MTT#F#TTT#FFF#FFF#FFF#F#FFF#F#
#M###F###T#F#####F#F#F#F#F#F#F#
#M#FFFFF#T#F#FFF#F#F#F#FFF#F#F#
#M#######T#F#F#F###F#F#####F#F#
#MMTTTTTTT#FFF#FFFFF#FFFFFFFFF#
###############################
```

## Case 226 (final failure)

Loop gain: `-0.0042`. First loop F1 `0.5133` with 287 false positives and 5 misses. loop16 F1 `0.5092` with 289 false positives and 6 misses. Final exact `0.0000`.

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
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
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
#MTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop6

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
#MTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
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
#MTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

### loop16

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
#MTT#FSTTTFF#FFF#F#F#FFF#F#F#F#
###T#####T#####F#F###F###F###F#
#MTT#TTT#T#FFFFF#FFF#F#FFFFF#F#
#M###T#T#T#F#######F#F#F###F#F#
#M#TTT#TTT#FFF#FFFFF#F#F#FFF#F#
#M#T#######F#F#F#####F#F#F###F#
#MTT#FFFFFFF#FFFFFFFFF#F#FFFFF#
###############################
```

## Case 319 (final failure)

Loop gain: `-0.0016`. First loop F1 `0.5109` with 287 false positives and 6 misses. loop16 F1 `0.5092` with 286 false positives and 7 misses. Final exact `0.0000`.

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
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTM#
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
#.FFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMT#
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
#.FFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTTT#
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
#.FFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMT#
###############################
```

### loop10

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
#.FFFF#FFF#T#F#TTT#FFF#TTTTT#T#
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMT#
###############################
```

### loop16

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
#.#########T#F#T#T#F#F#####T#T#
#MTTTTTTTTTT#F#T#TGF#F#TTT#T#T#
#M###########F#T#####F#T#T#T#T#
#M#TTT#TTT#FFF#TTT#F#F#T#TTT#T#
#M#T#T#T#T#######T#F#F#T#####T#
#MTT#TTT#TTTTTTTTT#FFF#TTTTTMM#
###############################
```

## Case 470 (final failure)

Loop gain: `-0.0058`. First loop F1 `0.5158` with 284 false positives and 7 misses. loop16 F1 `0.5100` with 285 false positives and 9 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MTTTTTTFF#TTT#TTTTTTTTTTT#TTM#
###############################
```

### loop2

```text
###############################
#FFF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#T###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MTTTTTTFF#TTT#TTTTTTTTTTT#TTT#
###############################
```

### loop4

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MTTTTTTFF#TTT#TTTTTTTTTTT#TTM#
###############################
```

### loop6

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#T###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MTTTTTTFF#TTT#TTTTTTTTTTT#TTM#
###############################
```

### loop10

```text
###############################
#.FF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#T###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MTTTTTTFF#TTT#TTTTTTTTTTT#TMT#
###############################
```

### loop16

```text
###############################
#FFF#FFFFFFFFFFFFFFFFF#FFFFFFF#
#F#F#F#####F#####F#F###F#F###F#
#F#FFF#FFF#F#FFFFF#F#FFF#FFF#F#
#F#####F#F#F#F#####F#F#####F#F#
#FFFFF#F#F#F#F#FFF#F#FFFFF#F#F#
#F#####F#F#F#F#F#F###F#####F#F#
#F#FFFFF#F#F#FFF#FFFFF#FFF#F#F#
###F#####F#############F#F#F#F#
#FFF#FFFFF#FFFFF#FFF#FFF#FFF#F#
#F#######F#F###F#F#F#F#########
#FFFFFFF#FFF#F#FFF#F#FFFSTTTTT#
#######F#####F#####F###F#####T#
#TTT#F#FFFFF#FFFFF#FFF#FFF#FFT#
#T#T#F#####F#####F###F#F###F#T#
#T#TTTTTTT#FFFFF#FFF#F#F#FFF#T#
#T#######T#####F###F#F###F###T#
#T#FFTTTTTFFFF#F#FFF#FFFFF#F#T#
#T###T#########F#F#F#######F#T#
#TTT#TTG#FFFFFFF#F#FFF#TTT#TTT#
#F#T#####F#######F#####T#T#T###
#F#TTTTT#FFF#FFF#TTTTT#T#TTT#F#
#######T###F#F#F#T###T#T#####F#
#MTTTT#T#F#F#F#F#TTT#TTT#TTTTT#
#M###T#T#F#F#F#####T#####T###T#
#MTT#TTT#FFF#TTTFF#T#FFTTT#F#T#
###T#F#######T#T#F#T###T###F#T#
#MTT#F#TTTTT#T#T#F#TTTTT#TTT#T#
#M#####T###T#T#T#########T#T#T#
#MMTTTTTFF#TTT#TTTTTTTTTTT#TMM#
###############################
```

## Case 255 (final failure)

Loop gain: `-0.0065`. First loop F1 `0.5199` with 287 false positives and 3 misses. loop16 F1 `0.5133` with 286 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#F#FFFFFFFFFFF#FFFFTTTTTTTTTTT#
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
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop2

```text
###############################
#F#FFFFFFFFFFF#FFFFTTTTTTTTTTT#
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
#M#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#M#########F###F#####F###F#F#F#
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop4

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
#M#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#M#########F###F#####F###F#F#F#
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop6

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
#M#F#####T###F#####F###########
#T#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#M#########F###F#####F###F#F#F#
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop10

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
#M#F#####T###F#####F###########
#M#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#M#########F###F#####F###F#F#F#
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

### loop16

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
#M#F#####T###F#####F###########
#M#FFFFFFG#FFF#FFFFFFFFFFF#FFF#
#M#########F###F#####F###F#F#F#
#MFFFFFF#FFF#F#FFFFF#F#F#F#F#F#
#M#####F#F###F#F#####F#F#F#F#F#
#MSF#F#F#FFF#FFF#FFFFF#F#F#F#F#
###F#F#F###F#####F#####F#F#F#F#
#.FFFF#FFFFFFFFFFF#FFFFFFFFF#F#
###############################
```

## Case 375 (final failure)

Loop gain: `-0.0009`. First loop F1 `0.5150` with 286 false positives and 6 misses. loop16 F1 `0.5141` with 287 false positives and 6 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#F#T#F#T#T#F#F#F#F#######T###F#
#TTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFFFF#
###############################
```

### loop2

```text
###############################
#FFF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#F#T#F#T#T#F#F#F#F#######T###F#
#MTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#.#T#F#T#T#F#F#F#F#######T###F#
#TTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFF.F#
###############################
```

### loop6

```text
###############################
#FFF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#F#T#F#T#T#F#F#F#F#######T###F#
#MTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFFFF#
###############################
```

### loop10

```text
###############################
#.FF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#F#T#F#T#T#F#F#F#F#######T###F#
#TTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFFF.#
###############################
```

### loop16

```text
###############################
#FFF#FFFFF#FFFFFFFFF#FFFFFFF#F#
#F#F#####F#F#F#####F#F#####F#F#
#F#FFFFF#F#F#F#FFF#FFF#TTT#FFF#
#F#####F#F#F#F#F#F#####T#T###F#
#FFF#FFF#FFF#FFF#F#TTTTT#T#F#F#
#F#F#F###F#######F#T#####T#F#F#
#F#F#F#FFF#FFFFFFF#T#TTTTT#F#F#
###F#F###F#F#######T#T#####F#F#
#FFF#FFF#F#F#TTT#TTT#TTTTTFF#F#
#F#####F#F#F#T#T#T#F#####T###F#
#F#FFF#F#F#F#T#TTT#F#FFF#T#FFF#
###F#F#F###F#T#######F#F#T#F###
#TTT#F#FFFFF#TFFFFFF#F#F#T#FFF#
#T#T#F#######T#####F#F#F#T###F#
#T#T#FFFFF#TTT#TTT#FFF#F#T#FFF#
#T#T#######T###T#T#####F#T#F###
#T#TTTTTTT#TTTTT#T#FFFFF#T#F#F#
#T#######T#######T#F#####T#F#F#
#TTT#FFF#TTTTTTTTT#F#TTTTT#F#F#
###T#F#############F#T#####F#F#
#F#T#FFTTT#F#FFF#FFF#TTTTT#FFF#
#F#T#F#T#T#F#F#F#F#######T###F#
#TTT#F#T#TTT#F#FFF#FFFST#TFF#F#
#M#####T###G#F#####F###T#T###F#
#M#TTT#TTT#F#FFF#FFFFF#TTT#FFF#
#M#T#T###T#F###F###########F###
#M#T#TTT#T#F#F#FFF#FFFFFFF#FFF#
#M#T###T#T#F#F###F#F#####F###F#
#MTT#FFTTT#FFFFF#FFFFFFF#FFFFF#
###############################
```

## Case 312 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5174` with 285 false positives and 6 misses. loop16 F1 `0.5141` with 286 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTT#
###############################
```

### loop2

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#MTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTT#
###############################
```

### loop4

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#MTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTM#
###############################
```

### loop6

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTT#
###############################
```

### loop10

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#MTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTM#
###############################
```

### loop16

```text
###############################
#FFFFFFFFF#FFFFFFFFF#FFTTTTT#F#
#F#####F#F###F#####F#F#T###T#F#
#F#FFFFF#FFFFF#F#FFF#F#T#F#T#F#
#F#F###########F#F#####T#F#T#F#
#F#F#FFFFFFF#FFFFF#TTTTT#F#T#F#
###F#######F#F#####T#####F#T#F#
#FFF#FFFFF#FFF#FFFFTTTTT#F#TTS#
#F###F###F#F###F#######T#F###F#
#FFFFF#FFF#FFF#F#TTT#TTT#FFFFF#
#F#####F#####F###T#T#T###F#####
#FFFFF#F#FFF#F#TTT#T#T#F#FFFFF#
#F#####F#F#F#F#T###T#T#F#####F#
#F#FFF#F#F#F#F#TTT#TTT#FFF#FFF#
###F#F#F###F#F#F#T#####F###F###
#FFF#FFF#FFF#F#F#T#FFFFF#FFFFF#
#########F###F#F#T#####F#F#####
#FFFFF#FFFFFFF#F#TTTTT#FFF#TTT#
#####F#F#######F#####T###F#T#T#
#TTTGF#FFFFFFF#FFF#F#TTT#F#T#T#
#T###F#######F###F#F###T###T#T#
#TTT#F#TTTFF#FFF#FFFFF#TTTTT#T#
###T###T#T#F###F#############T#
#TTT#TTT#T#F#F#FFFFF#FFFFFFF#T#
#M###T###T#F#F#####F#F#####F#T#
#M#TTT#TTT#FFFFFFF#F#F#FFF#F#T#
#M#T###T###########F#F#F###F#T#
#M#T#F#TTT#TTT#TTT#FFF#TTT#FFT#
#M#T#F###T#T#T#T#T#####T#T###T#
#MTTFFFF#TTT#TTT#TTTTTTT#TTTTM#
###############################
```

## Case 88 (final failure)

Loop gain: `-0.0049`. First loop F1 `0.5199` with 285 false positives and 5 misses. loop16 F1 `0.5150` with 285 false positives and 7 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#T#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MTTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

### loop2

```text
###############################
#.FFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#T#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MTTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#T#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MMTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

### loop6

```text
###############################
#.FFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#T#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MMTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

### loop10

```text
###############################
#.FFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#M#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MMTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

### loop16

```text
###############################
#.FFFFFF#TTTTTTT#F#FFFFF#FFFFF#
###F###F#T#####T#F#F#F#F#####F#
#FFF#F#F#T#F#FST#FFF#F#FFFFFFF#
#F###F#F#T#F#########F#######F#
#F#F#FFF#T#FFF#FFFFFFF#FFFFF#F#
#F#F#F###T#F#F#F#F#####F#####F#
#F#F#FFF#T#F#F#F#F#FFF#FFFFFFF#
#F#F#F###T#F###F#F#F#F#F#######
#F#F#F#TTT#FFFFF#F#F#FFF#FFFFF#
#F#F#F#T###F#####F#######F###F#
#F#FFF#T#FFF#TTT#FFFFFFFFF#FFF#
#F#####T#####T#T###F#######F#F#
#FFF#TTT#TTT#T#TTT#FFF#FFF#F#F#
###F#T###T#T#T###T###F#F#F#F#F#
#FFF#TTT#T#T#T#F#T#FFF#F#FFF#F#
#F#####T#T#T#T#F#T#F###F#######
#F#TTT#TTT#T#T#F#T#FFF#FFFFFFF#
#F#T#T#####T#T#F#T###F#######F#
#F#T#TTTTT#TTT#F#T#FFFFF#FFF#F#
#F#T#####T#####F#T#######F#F#F#
#TTT#FFF#TTTFFFF#TTTTT#FFF#F#F#
#T###F#####T#F#######T#F###F#F#
#M#FFGTT#TTT#F#TTTTT#T#F#FFF#F#
#M#####T#T#####T###T#T#F#F###F#
#MTT#TTT#TTTTT#T#TTT#T#F#FFF#F#
###T#T#######T#T#T###T#F###F#F#
#MTT#TTTFF#F#TTT#TTT#T#F#F#FFF#
#M#####T#F#F#######T#T#F#F###F#
#MMTTTTT#FFFFFFFFF#TTTFF#FFFFF#
###############################
```

## Case 395 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5158` with 287 false positives and 4 misses. loop16 F1 `0.5158` with 287 false positives and 4 misses. Final exact `0.0000`.

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
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
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
#MTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
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
#MTT#T#FFFFFFFFFFFFF#TTTTT#FFF#
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
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
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
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
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

### loop16

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
#M#T#T#F#################T#F###
#M#TTT#FFFFFFF#TTTTTFFFF#T#FFF#
#M#####F#######T###T#####T###F#
#MTTTT#F#TTTTT#T#F#TTT#TTT#FFF#
#.###T###T###T#T#F###T#T###F#F#
#.FF#TTTTTFF#TTTFFFF#TTTFFFF#F#
###############################
```

## Case 349 (final failure)

Loop gain: `-0.0025`. First loop F1 `0.5199` with 283 false positives and 7 misses. loop16 F1 `0.5174` with 283 false positives and 8 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#TTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MTT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

### loop2

```text
###############################
#TTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MMT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

### loop4

```text
###############################
#TTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MMT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

### loop6

```text
###############################
#MTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MMT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

### loop10

```text
###############################
#TTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MMT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

### loop16

```text
###############################
#TTTTTTTTTTTTTTTTTTT#TTT#FFFFF#
#T#################T#T#T#F###F#
#T#TTTTT#TTT#FFFFF#TTT#T#F#FFF#
#T#T###T#T#T#F###F#####T###F#F#
#T#T#F#TTT#T#FFF#FFFFF#TTTFF#F#
#T#T#F#####S###F#####F###T#####
#T#TTTTT#F#FFF#F#F#FFF#TTT#FFF#
#T#####T#F###F#F#F#F###T###F###
#TTTTT#TTTTT#F#FFF#F#FFG#FFF#F#
#F###T#####T#F###F#F#####F###F#
#F#TTT#F#TTT#FFFFF#FFFFFFF#FFF#
###T#F#F#T###################F#
#TTT#FFF#TTT#FFFFFFFFFFFFFFFFF#
#T#####F###T#####F###########F#
#TTT#F#FFF#TTT#FFF#FFF#FFFFF#F#
###T#F###F###T#F###F#F#F#####F#
#TTT#FFF#F#F#T#F#FFF#F#FFFFF#F#
#T###F#F#F#F#T#F#F###F#####F#F#
#T#FFF#F#FFF#T#FFF#F#F#FFF#FFF#
#T###F#F###F#T#F###F#F#F#F#F###
#TTT#F#FFFFF#T#FFF#F#FFF#F#F#F#
###T#########T###F#F#####F#F#F#
#MTT#TTT#TTTTT#FFFFFFF#FFF#FFF#
#M###T#T#T###########F#F#####F#
#M#TTT#T#T#FFFFFFF#FFF#F#FFFFF#
#M#T###T#T#F#####F#####F#F#####
#M#T#TTT#T#F#FFF#FFFFFFF#F#FFF#
#M#T#T###T#F#F#F#########F###F#
#MMT#TTTTTFF#F#FFFFFFFFFFFFFFF#
###############################
```

## Case 494 (final failure)

Loop gain: `-0.0042`. First loop F1 `0.5225` with 284 false positives and 3 misses. loop16 F1 `0.5183` with 286 false positives and 4 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#T#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FF.#
###############################
```

### loop2

```text
###############################
#.FTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#M#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FFF#
###############################
```

### loop4

```text
###############################
#FFTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#M#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FFF#
###############################
```

### loop6

```text
###############################
#FFTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#M#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FFF#
###############################
```

### loop10

```text
###############################
#.FTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#M#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FFF#
###############################
```

### loop16

```text
###############################
#FFTTT#FFTTT#FFF#FFFFFFFFF#FFF#
###T#T###T#T#F#F#F###F###F#F#F#
#TTT#TTTTT#T#F#F#FFF#F#FFF#F#F#
#T#########T#F#####F#F#F###F#F#
#T#TTTTT#F#S#FFFFFFF#F#FFF#F#F#
#T#T###T#F#F#######F#F###F###F#
#TTT#TTT#F#FFFFFFF#F#FFF#F#FFF#
#F###T###F#######F###F#F#F#F#F#
#F#TTT#FFFFFFFFF#FFF#F#F#FFF#F#
###T###F###########F#F#F#####F#
#TTT#F#FFFFF#FFFFFFF#F#FFF#F#F#
#T###F#F###F#F###########F#F#F#
#T#FFF#F#F#FFF#FFFFFFF#FFF#F#F#
#T#F###F#F#####F#####F#F###F#F#
#T#F#FFFFFFFFFFF#TTT#FFF#FFF#F#
#T#F#F#######F###T#T#####F###F#
#T#FFFFF#TTT#F#TTT#T#FFFFF#FFF#
#T#######T#T###T###T#F#F###F###
#T#TTTTT#T#TTTTT#TTT#F#FFFFF#F#
#T#T###T#T#######T#F#####F###F#
#T#TTT#TTT#FFFFF#T#F#TTT#FFF#F#
#T###T#####F#F#F#T###T#T###F#F#
#T#TTT#FFFFF#F#F#TTT#T#TTT#F#F#
#M#T#########F#F###T#T###T#F#F#
#M#TFFFFFFFFFF#FFF#TTT#TTT#F#F#
#M#T#############F#####T###F#F#
#MTT#FFFFF#TTTTT#F#TTT#TTT#FFF#
#.###F###F#T###T###T#T###T###F#
#.FF#FFF#FGT#FFTTTTT#TTTTT#FFF#
###############################
```

## Case 373 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5216` with 284 false positives and 4 misses. loop16 F1 `0.5183` with 285 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#TTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFF.F#
###############################
```

### loop2

```text
###############################
#FFFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#TTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#TTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFFFF#
###############################
```

### loop6

```text
###############################
#FFFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#MTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFF.F#
###############################
```

### loop10

```text
###############################
#FFFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#MTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFFF.#
###############################
```

### loop16

```text
###############################
#.FFFFFFFFFFFF#TTTTT#FFFFF#FFF#
#F#####F#######T###T#F###F#F#F#
#FFFFF#F#TTTTTTT#TTT#FFF#FFF#F#
#####F###G#######T###########F#
#FFFFF#FFF#FFFFF#TTTTT#TTTTTTT#
#F###F#F###F#F#######T#T#####T#
#FFF#F#F#FFF#FFFFFFF#TTT#FFF#T#
#F#F###F#F#######F#F#####F###T#
#F#F#F#F#F#FFFFFFF#FFFFF#FFF#T#
#F#F#F#F#F#F###########F###F#T#
#F#F#FFF#F#F#FFFFFFF#F#FFF#F#T#
###F#######F#F#####F#F###F#F#T#
#FFFFFFFFFFF#FFF#F#FFFFF#FFF#T#
#F#############F#F###F###F###T#
#TTT#TTTTT#FFFFF#FFFFF#FFF#TTT#
#T#T#T###T#F#####F#####F###T###
#T#TTTFF#T#F#F#FFF#F#FFF#F#T#F#
#T#######T#F#F#F###F#F###F#T#F#
#TTT#F#TTT#F#FFF#FFF#FFF#F#T#F#
###T#F#T###F###F#F#####F#F#T#F#
#F#TFF#T#F#FFF#FFF#FFFFFFF#T#F#
#F#T###T#F#F#F#####F#######T#F#
#MTT#TTT#FFF#FFF#FFF#TTTTT#T#F#
#M###T#########F#F###T###T#T#F#
#M#F#T#TTT#TTT#F#F#TTT#F#T#T#F#
#M#F#T#T#T#T#T#F#F#T###F#T#T#F#
#MTT#TTT#TTT#T#FFF#TTTTT#TTT#F#
###T#########T#########T#####F#
#.FTTTTTSFFF#TTTTTTTTTTTFFFFFF#
###############################
```

## Case 32 (final failure)

Loop gain: `-0.0041`. First loop F1 `0.5225` with 284 false positives and 3 misses. loop16 F1 `0.5184` with 283 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#T#####F#F###F#F#T###T#####T###
#TTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFFFF#
###############################
```

### loop2

```text
###############################
#FFFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#M#####F#F###F#F#T###T#####T###
#TTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFFFF#
###############################
```

### loop4

```text
###############################
#FFFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#M#####F#F###F#F#T###T#####T###
#TTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFF.F#
###############################
```

### loop6

```text
###############################
#FFFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#M#####F#F###F#F#T###T#####T###
#MTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFFFF#
###############################
```

### loop10

```text
###############################
#FFFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#M#####F#F###F#F#T###T#####T###
#MTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFF..#
###############################
```

### loop16

```text
###############################
#.FFFFFF#FFFFFFFFTTTTTTTTT#FFF#
#F#F#####F#######T#######T#F###
#F#F#FFFFFFFFFFF#T#TTTTTTT#FFF#
###F#F#######F###T#T#########F#
#FFF#F#F#TTT#F#TTT#TSF#FFFFFFF#
#F###F#F#T#T###T#####F#F#####F#
#F#FFFFF#T#T#TTT#FFF#F#FFF#FFF#
#F#F#####T#T#T###F###F###F#F###
#F#F#TTTTT#TTT#F#FFF#FFFFF#FFF#
#F#F#T#########F#F#F#F#########
#F#F#TTT#FFFFFFF#F#FFF#TTGFFFF#
#F#F###T#F#F#####F#####T#####F#
#FFF#TTT#F#F#FFF#F#FFF#TTTFF#F#
#####T###F###F#F#F#F#####T###F#
#TTTTT#F#FFFFF#F#F#FFFFF#T#FFF#
#T#####F#F#####F#F#F#F#F#T#F###
#TTT#FFF#FFF#F#FFF#F#F#F#T#FFF#
#F#T#F#####F#F#######F#F#T###F#
#F#T#FFFFF#F#FFFFFFFFF#F#T#F#F#
###T#F###F#F###F#######F#T#F#F#
#TTT#FFF#F#FFF#F#TTTTT#F#TTT#F#
#M#####F#F###F#F#T###T#####T###
#MTT#FFF#FFFFF#F#TTT#TTTTT#T#F#
###T#########F#F###T#####T#T#F#
#MTT#TTTTTTT#FFF#TTT#FFF#T#T#F#
#M###T#####T#####T#####F#T#T#F#
#MTT#T#F#TTT#TTTTT#FFFFF#TTT#F#
#.#T#T#F#T###T#####F###F#####F#
#.#TTTFF#TTTTTFFFFFF#FFFFFFF.F#
###############################
```

## Case 77 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5183` with 287 false positives and 3 misses. loop16 F1 `0.5191` with 286 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFTTTTTTT#FFFFFFFFFFFFFFFFF#
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
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
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
#.#T#########T###T#F###T#F###F#
#.#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
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
#.#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
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
#.#T#########T###T#F###T#F###F#
#.#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
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
#.#T#########T###T#F###T#F###F#
#F#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

### loop16

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
#.#T#########T###T#F###T#F###F#
#.#TTT#FFF#TTT#TTT#FFF#T#F#FFF#
#####T#F###T###T#######T#F###F#
#MTTTT#FFF#TTTTT#FFFFTTT#FFFFF#
#M#######F###########T#######F#
#MTTTTTTTT#FFFFTTTTTTT#FFFFF#F#
#########T#####T#######F#F###F#
#.FFFFFFFTTTTTTT#FFFFFFF#FFFFF#
###############################
```

## Case 441 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5191` with 284 false positives and 5 misses. loop16 F1 `0.5191` with 284 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#T#####T###T#T###T#T#T#####F#F#
#T#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#.FFFFFTTTTTTT#F#FFTTTTTTT#TMG#
###############################
```

### loop2

```text
###############################
#F#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#M#####T###T#T###T#T#T#####F#F#
#M#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#.FFFFFTTTTTTT#F#FFTTTTTTT#TTG#
###############################
```

### loop4

```text
###############################
#.#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#M#####T###T#T###T#T#T#####F#F#
#M#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#.FFFFFTTTTTTT#F#FFTTTTTTT#TTG#
###############################
```

### loop6

```text
###############################
#F#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#M#####T###T#T###T#T#T#####F#F#
#M#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#.FFFFFTTTTTTT#F#FFTTTTTTT#TTG#
###############################
```

### loop10

```text
###############################
#F#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#M#####T###T#T###T#T#T#####F#F#
#M#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#..FFFFTTTTTTT#F#FFTTTTTTT#TTG#
###############################
```

### loop16

```text
###############################
#.#FFFFF#FFFFFFF#FFFFFFFFF#FFF#
#F#F###F#F#####F#F#####F#F#F#F#
#F#F#F#FFF#TTT#F#FFF#FFF#FFF#F#
#F#F#F#####T#T#F###F#F#########
#F#FFF#STTTT#T#FFFFF#FFFFFFFFF#
#F###F###F###T###############F#
#F#FFF#FFF#F#TTT#FFFFFFFFF#FFF#
#F#F###F###F###T###F#####F#F###
#F#F#FFF#FFFFF#TTT#F#FFF#F#FFF#
#F#F#F###F#F#F###T#F#F###F###F#
#F#FFF#FFF#F#F#TTTFF#F#FFFFF#F#
#F#########F#F#T#####F#######F#
#FFFFFFFFFFF#F#T#F#FFF#FFF#FFF#
#F#F#########F#T#F#F###F#F#F###
#F#F#F#FFFFFFF#T#F#FFF#F#F#FFF#
###F#F#F#####F#T#F###F#F#F###F#
#FFF#FFF#TTT#F#TTTTT#FFF#FFF#F#
#F#####F#T#T#######T#######F#F#
#F#TTT#F#T#TTTTTFF#TTT#FFFFF#F#
#F#T#T###T#####T#####T#F#####F#
#TTT#TTT#TTT#TTT#TTT#T#FFFFF#F#
#M#####T###T#T###T#T#T#####F#F#
#T#FFF#TTTTT#TTTTT#TTT#TTT#FFF#
#M#F#F#################T#T#####
#M#F#FFFFTTTTTTTTTTT#FFT#TTTTT#
#M#######T#########T#F#T#####T#
#MTTTTTT#TTTTT#FFF#T#F#TTT#TTT#
#######T#####T#F#F#T#####T#T###
#.FFFFFTTTTTTT#F#FFTTTTTTT#TTG#
###############################
```

## Case 83 (final failure)

Loop gain: `0.0025`. First loop F1 `0.5174` with 285 false positives and 6 misses. loop16 F1 `0.5199` with 285 false positives and 5 misses. Final exact `0.0000`.

### loop1

```text
###############################
#MTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#T###########T###########F#F###
#M#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#M#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

### loop2

```text
###############################
#TTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#M###########T###########F#F###
#T#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#T#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

### loop4

```text
###############################
#TTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#M###########T###########F#F###
#T#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#M#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

### loop6

```text
###############################
#TTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#M###########T###########F#F###
#T#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#M#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

### loop10

```text
###############################
#TTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#M###########T###########F#F###
#M#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#M#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

### loop16

```text
###############################
#TTTTT#TTT#TTT#FFFFFFFFFFFFFFF#
#T###T#T#T#T#G#F###F#########F#
#TTT#TTT#TTT#F#FFF#F#FFFFFFF#F#
###T#########F#F###F#F#####F#F#
#F#T#FFFFFFF#F#F#FFF#FFF#FFF#F#
#F#T#F#####F#F###F#####F#F###F#
#TTT#FFFFF#F#F#FFF#FFFFF#FFFFF#
#T#####F#F#F#F#F#########F#####
#TTTTT#F#F#F#F#FFFFS#TTT#F#FFF#
#####T###F#F###F###T#T#T#F#F#F#
#FFF#T#FFF#FFF#FFF#TTT#T#FFF#F#
#F#F#T#F#########F#####T#####F#
#F#F#T#FFFFFFFFFFF#TTT#T#FFF#F#
#F#F#T#F###########T#T#T#F#F#F#
#F#F#T#FFFFFFF#F#TTT#TTT#F#FFF#
#F#F#T#######F#F#T#######F#####
#F#F#TTT#TTT#FFF#TTT#FFF#F#FFF#
###F###T#T#T###F###T#F###F###F#
#FFFFF#TTT#T#FFFFF#T#FFF#FFFFF#
#F#########T#######T#F#F#####F#
#TTTTTTTTTTT#TTTTTTT#F#FFF#FFF#
#M###########T###########F#F###
#T#TTTTTTT#FFTTTTT#FFF#FFF#F#F#
#M#T#####T#F#####T#F#F#F#F#F#F#
#M#TTT#FFT#F#TTTTT#F#FFF#F#FFF#
#M###T###T#F#T#####F#####F###F#
#MTT#TTT#T#F#T#FFFFF#FFF#F#FFF#
###T###T#T###T#F###F#F#F###F###
#.FTTTTT#TTTTT#FFF#FFF#FFFFFFF#
###############################
```

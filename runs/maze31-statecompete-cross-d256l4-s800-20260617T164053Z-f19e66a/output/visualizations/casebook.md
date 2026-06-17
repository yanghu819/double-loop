# Maze 31x31 Loop Trajectory Casebook

Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, `F` false-positive path, `M` missed true path, `.` open non-path cell.

Cases are selected in this order: final failures, hard low-F1 cases, final over-prediction cases, then largest loop-gain solved cases.

## Case 319 (final failure)

Loop gain: `-0.0025`. First loop F1 `0.5199` with 288 false positives and 2 misses. loop12 F1 `0.5174` with 288 false positives and 3 misses. Final exact `0.0000`.

### loop1

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
#TMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

### loop2

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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
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
#MMT#TTT#TTTTTTTTT#FFF#TTTTTTM#
###############################
```

## Case 226 (final failure)

Loop gain: `-0.0024`. First loop F1 `0.5232` with 287 false positives and 1 misses. loop12 F1 `0.5207` with 287 false positives and 2 misses. Final exact `0.0000`.

### loop1

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
#MTT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MTT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MTT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MTT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MMT#FFFFFFF#FFFFFFFFF#F#FFFF.#
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
#MMT#FFFFFFF#FFFFFFFFF#F#FFFF.#
###############################
```

## Case 393 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5207` with 287 false positives and 2 misses. loop12 F1 `0.5207` with 287 false positives and 2 misses. Final exact `0.0000`.

### loop1

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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
###############################
```

### loop2

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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
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
#MMTTTTTTT#FFF#FFFFF#FFFFFFFF.#
###############################
```

## Case 230 (final failure)

Loop gain: `-0.0033`. First loop F1 `0.5256` with 285 false positives and 2 misses. loop12 F1 `0.5223` with 286 false positives and 3 misses. Final exact `0.0000`.

### loop1

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTTTM#
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

### loop2

```text
###############################
#..FFFFTTT#FFFFFFTTTTTTTTTTTTM#
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
#MTTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#MMTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#MMTTTFFFF#TTTTTTTTTTTFF#FFFF.#
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
#MMTTTFFFF#TTTTTTTTTTTFF#FFFF.#
###############################
```

## Case 12 (final failure)

Loop gain: `0.0000`. First loop F1 `0.5232` with 287 false positives and 1 misses. loop12 F1 `0.5232` with 287 false positives and 1 misses. Final exact `0.0000`.

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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

### loop2

```text
###############################
#.#FFTTT#TTT#FFFFTTTTTTTTT#F..#
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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
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
#.#FFFFFFFFF#FFF#FFF#TTTTTTTTM#
###############################
```

## Case 229 (final failure)

Loop gain: `0.0017`. First loop F1 `0.5215` with 289 false positives and 1 misses. loop12 F1 `0.5232` with 287 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#FFTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#FFFFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

### loop2

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#.FFFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

### loop4

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#.FFFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

### loop6

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#.FFFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

### loop10

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#..FFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

### loop12

```text
###############################
#.FTTT#FFTTT#FFFFF#FFFFF#FFFF.#
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
#.FFFFFF#FFFFFFFFF#FFFFFFF#TTM#
###############################
```

## Case 261 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5223` with 287 false positives and 2 misses. loop12 F1 `0.5232` with 286 false positives and 2 misses. Final exact `0.0000`.

### loop1

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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

### loop2

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
#..FFFFFFF#FFFFFFFFF#FFFFF#FF.#
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
#.FFFFFFFF#FFFFFFFFF#FFFFF#FF.#
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
#..FFFFFFF#FFFFFFFFF#FFFFF#FF.#
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
#..FFFFFFF#FFFFFFFFF#FFFFF#FF.#
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
#..FFFFFFF#FFFFFFFFF#FFFFF#FF.#
###############################
```

## Case 65 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5232` with 287 false positives and 1 misses. loop12 F1 `0.5240` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

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
#.FFFFFF#TTTTT#TTTTTTTFF#TTTTM#
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
#.FFFFFF#TTTTT#TTTTTTTFF#TTTTM#
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
#..FFFFF#TTTTT#TTTTTTTFF#TTTTM#
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
#..FFFFF#TTTTT#TTTTTTTFF#TTTTM#
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
#..FFFFF#TTTTT#TTTTTTTFF#TTTTM#
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
#..FFFFF#TTTTT#TTTTTTTFF#TTTTM#
###############################
```

## Case 204 (final failure)

Loop gain: `0.0009`. First loop F1 `0.5232` with 287 false positives and 1 misses. loop12 F1 `0.5240` with 286 false positives and 1 misses. Final exact `0.0000`.

### loop1

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#F.FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

### loop2

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#..FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

### loop4

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#..FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

### loop6

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#..FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

### loop10

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#..FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

### loop12

```text
###############################
#.FFFFFFFF#FFFFF#FFFFFFFFFFFF.#
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
#..FFF#TTTTTFFFFFFFF#TTTTTTTTM#
###############################
```

## Case 62 (final failure)

Loop gain: `0.0017`. First loop F1 `0.5232` with 287 false positives and 1 misses. loop12 F1 `0.5249` with 285 false positives and 1 misses. Final exact `0.0000`.

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
#FFF#FFTTT#FFTTT#TTT#FFFFFFF#.#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#.#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#.#
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
#.FF#FFTTT#FFTTT#TTT#FFFFFFF#.#
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
#..F#FFTTT#FFTTT#TTT#FFFFFFF#.#
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
#..F#FFTTT#FFTTT#TTT#FFFFFFF#.#
###############################
```

# Native Directional-MQAR Swap Topology

## Scope

This is a zero-parameter, prediction-only audit of the frozen deterministic
P-REPRO-001 replay-B endpoint. It does not change logits, run another model or
select a candidate. The source is the preserved `cases.json` containing 1,000
L1024 examples, four associations per example, two future and two past.

## Endpoint

- balanced/future/past/joint accuracy: `.494/.454/.534/.041`;
- total query errors: `2024/4000`;
- valid-value wrong-key swaps: `1546` (`76.3834%` of errors);
- each example's four target values are distinct, so owner attribution is
  unambiguous.

## Topology

Wrong-owner direction counts are:

| true owner -> predicted owner | events |
| --- | ---: |
| future -> future | 847 |
| past -> past | 692 |
| past -> future | 4 |
| future -> past | 3 |

Thus `1539/1546 = 99.5472%` of swaps stay in the correct direction class.
There are `173` reciprocal owner pairs, accounting for `346` swapped events or
`22.38%` of all valid-value swaps. `99.6119%` choose the adjacent owner by
write rank; `99.5472%` choose the nearest write because each direction has only
two associations.

Absolute write-position displacement averages `86.38` tokens: `46.64%` are at
most 64 and `75.55%` are at most 128. Signs are nearly balanced (`800` negative,
`746` positive), rejecting a single preferred lag. Absolute query-position
displacement averages `88.75` tokens: `42.88%` are at most 64 and `74.58%` are
at most 128.

Future accuracy decreases with longer query distance (`.5359`, `.4879`,
`.4103`, `.3967` over the four registered distance bins), while past accuracy
increases (`.4542`, `.5160`, `.5513`, `.6174`). Direction is therefore not the
only difficulty, but almost every valid-value error has already selected the
right direction-specific value set.

## Decision

The dominant failure is same-direction instance ownership: the model retrieves
the right coarse class and a valid payload, then assigns it to the other nearby
binding. This is not global random collision, missing payload content, a fixed
one-token phase error or FutureSeed overwrite. The diagnostic closes attempts
to infer success from lower conditional swap counts when absolute retrieval
falls. A successor must preserve the frozen native retrieval path and improve
generic occurrence ownership without a Sudoku rule, semantic selector, cache,
key split, fixed lag or discrete side bank.

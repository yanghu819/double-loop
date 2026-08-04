# P-CAUSAL-004: Zoology MQAR no-FutureSeed carrier gate

- Status: stopped at the preregistered invalid-carrier gate
- Source SHA: `b0a924762a2be99f8ef682fe709d1f69b3808eab`
- GPU: AIStation task-mode GPU1, one visible A100-SXM4-80GB
- Condition: strict official FLA GDN2, no FutureSeed
- Endpoint: step 500 of an allowed 1,000 steps

The ordinary write-before-query half failed to open: loop4 past accuracy was
`0.000244` against the required `0.50`. Future accuracy was `0.000488`, query
exact was `0`, and query CE was `8.358737`. Loop1-to-loop4 changes were
`+0.000244` past and `0` future. Therefore the FutureSeed arm was intentionally
not started.

This is an invalid benchmark carrier, not a negative FutureSeed result. The
next decision is exact upstream Zoology model/trainer reproduction; no rescue
tuning or second seed is allowed for this shell.

Artifacts include the full config, score, log, abort record, source patch,
source HEAD, machine-readable output, and loop1-4 hard-case visualization.
The large source snapshot remains on the persistent remote path recorded in
`source_snapshot.ref` and is intentionally excluded from Git.

# gdn-transition-d224l12-tail56-s9000-resume7000-20260702T093823Z-4a0330a

Early-stopped GPU1 continuation of the D224/L12 native-FutureSeed GDN hard-tail
scale run.

Primary result: step8000 holes60 loop5 exact `0.18359375`, blank accuracy
`0.63958377`. This did not improve over the step7000 holes60 readout
`0.1875/0.6624`, so the run was stopped before burning the remaining short
GPU1 lease.

Conclusion: loop computation is still useful on hard holes, but same-shape
56-64 tail training no longer has good information gain. Next work should
change recurrent state capacity/formulation, not continue this exact tail.

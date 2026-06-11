# Aborted h120 Loop6 All-Loop Launch

Run: `h120loopall-12x12-d256-l12-loop6-s7400-20260611T1615Z-8ccc4a9`

Source SHA: `8ccc4a9afb73b927ca8d3976837abf372ddd2e04`

This was the first launch of the h120 loop6 all-loop scale test. AIStation GPU1 halted while the job was running, so the run is marked aborted by `abort.json`. It reached step100 with CE `1.5641`, which shows the config was viable and the interruption was platform-side, not an OOM or model failure.

The same experiment was relaunched as `h120loopall-retry-12x12-d256-l12-loop6-s7400-20260611T1640Z-8ccc4a9` and completed.

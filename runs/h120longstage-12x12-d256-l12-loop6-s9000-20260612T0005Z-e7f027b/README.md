# Aborted h120 Long-Stage Run

- Run: `h120longstage-12x12-d256-l12-loop6-s9000-20260612T0005Z-e7f027b`
- Source SHA: `e7f027b193812cb6a9f7dd4bedb9af41f7cee868`
- Abort time: 2026-06-12T04:35:56Z
- Exact PID stopped: `290`

## Reason

This run used the same h120 long-stage curriculum but only produced a final eval. The user asked for more data and a direct 1k/2k/3k training-length comparison. Continuing this run would have burned GPU time without answering that question.

The run was stopped at stage `84-96` after step3900. GPU memory did not release cleanly after killing the bash PID, so GPU1 was restarted through AIStation/Kimi WebBridge. The replacement run is `h120curve-12x12-d256-l12-loop6-s9000-ckpt1k2k3k-20260612T0455Z-2e95bd8`, which adds in-run fixed checkpoint evals.

## Artifacts

- Abort record: `abort.json`
- Logs: `logs/run.log`, `logs/driver.log`

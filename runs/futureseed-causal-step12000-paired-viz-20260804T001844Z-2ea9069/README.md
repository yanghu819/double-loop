# Aborted step12000 paired visualization launch

This launch stopped before model construction because the pinned FLA package
path was omitted from `PYTHONPATH`. The error was `No module named 'fla'`.
No model forward or CUDA scientific evaluation ran, so this is a configuration
failure rather than a scientific result. The corrected launch is archived as
`futureseed-causal-step12000-paired-viz-20260804T002500Z-2ea9069`.

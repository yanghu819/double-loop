# 16x16 High-Hole Probe Abort

Purpose: test whether 16x16 can jump directly to a h96/h112-style hard-hole regime after 12x12 succeeded.

Source SHA on GPU1: `f67e6a2e909dc39ce5ba48f622d93b1ec43f5ab3`

Run: `frontier-16x16-d192-h112-20260604T1006Z-f67e6a2`

Abort UTC: `2026-06-04T10:17:49.758716+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- Curriculum: `24-64:300,64-96:400,96-112:200`
- Batch: `48`
- Kernel: `RWKV_KERNEL=cuda`

## Abort Reason

The high-hole schedule did not establish a foothold:

| step | stage | ce | loop1 loss | loop last loss |
| ---: | --- | ---: | ---: | ---: |
| 100 | 24-64 | 1.7717 | 1.7730 | 1.7717 |
| 200 | 24-64 | 1.5689 | 1.5713 | 1.5689 |
| 300 | 24-64 | 1.2512 | 1.2588 | 1.2512 |
| 400 | 64-96 | 1.5498 | 1.5799 | 1.5498 |
| 500 | 64-96 | 1.4809 | 1.5230 | 1.4809 |
| 600 | 64-96 | 1.4201 | 1.4607 | 1.4201 |

I stopped it by exact PID tree after observing high loss and low information gain, then relaunched the gentler `frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2` run.

## Decision

This abort does not prove 16x16 impossible. It proves the h96/h112 jump is low ROI before a h32/h48 foothold exists.

## Provenance Note

`abort.json` records the exact PID and child PIDs stopped. The source snapshot tarball is retained on GPU1/local disk and represented in GitHub by `source_snapshot.tar.gz.sha256` and `source_snapshot.ls.txt` because it exceeds GitHub's `100MB` file limit.

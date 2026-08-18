# Contrast fine-tuning (Option 3)

Trains the 4-class {ð, z, d, v} + blank CTC judge for the /ð/ drill, starting
from the charsiu CTC base. The laptop with the RTX 4060 runs the training;
this Mac exports data and later runs the eval exam.

## Layout

- `export.py` (openschwa_training.export) - builds the labeled segment
  dataset from L2-ARCTIC's TRAIN split. The eval harness's held-out split is
  never exported. Output: `data/l2arctic-dh/` (audio/, labels.csv,
  manifest.json), gitignored.
- `train.py` (openschwa_training.train) - fine-tunes: frozen base + fresh
  4-class CTC head for `--freeze-epochs`, then full fine-tune at low LR.
  Checkpoints resume (`--out/last.pt`); the best model is assembled into
  `--out/model/` in the exact layout the engine registry expects.
- `setup.sh` - one-shot laptop setup (uv, ml extra, CUDA check, smoke test).

## Laptop workflow

```bash
# 1. get the code (git) and two local folders:
#    - training/data/l2arctic-dh/          (66 MB, exported on the Mac)
#    - .models/charsiu-en-w2v2-ctc/        (the 0.38 GB base, from the Mac)
./setup.sh

uv run python -m openschwa_training.train \
    --data data/l2arctic-dh \
    --base-model ../.models/charsiu-en-w2v2-ctc \
    --out runs/v1 --epochs 8 --freeze-epochs 4 --batch-size 32
# resumable: re-run the same command after any interruption
# watch the per-epoch val_f1: the frozen phase (epochs 0-3) should climb;
# if the unfrozen phase collapses it, drop --epochs to 4 (head-only model)
# overnight reruns: raise epochs / batch-size as VRAM allows (4060: 8 GB)
```

Then copy `runs/v1/model/` back to the Mac as `.models/dh-contrast-v1/` and
run the eval exam (`cd eval && just eval -- --contrast 'ð:z,d,v' --model
dh-contrast-v1 --l2arctic data/l2arctic --speechocean762 data/speechocean762`).
The exam split was never exported, so the result is a real held-out number.

## Known data notes

- The v class has no /ð/->/v/ errors in L2-ARCTIC; it trains on correct /v/
  segments instead (export.py gathers all four phones).
- SpecAugment is disabled: its fixed mask length exceeds our 60-300 ms
  segments. The frozen-base phase and per-class weights regularize instead.
- Determinism: every RNG is seeded (--seed, default 42).

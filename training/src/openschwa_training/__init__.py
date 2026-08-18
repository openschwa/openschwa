"""Contrast fine-tuning (Option 3): data export and training scripts.

The export only ever touches the harness's TRAIN split; the held-out split
is the exam and must stay blind. Exported audio is corpus-derived and never
enters git (training/data/ is ignored).
"""

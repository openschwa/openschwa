"""GOP + closed-set contrast scoring (M1).

Contract: per-phone goodness-of-pronunciation from aligned frame posteriors;
for each focus phone, renormalize posterior mass over {target} ∪ confusion_set
→ margin → Platt-calibrated confidence → verdict (on_target / substituted /
uncertain). Operating thresholds live in scoring/calibration.yaml, produced
ONLY by the eval harness (eval/) — never hand-tuned. Precision ≥ 0.90 on
held-out L2 data is the shipping bar (`eval/README.md`).
"""

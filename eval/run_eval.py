"""Run the offline evaluation for one contrast. M1.

Usage (once implemented):
    uv run --project ../engine python run_eval.py \
        --contrast "ð:z,d" \
        --speechocean762 /path/to/corpus --l2arctic /path/to/corpus \
        --out reports/

Procedure: see README.md. Output: PR curve + chosen operating point +
per-L1 breakdown, written as a committed markdown + JSON report.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", help="target:confusion[,confusion...] in canonical IPA")
    parser.add_argument("--speechocean762", help="local corpus root")
    parser.add_argument("--l2arctic", help="local corpus root")
    parser.add_argument("--out", default="reports/")
    parser.parse_args()
    raise NotImplementedError("M1")


if __name__ == "__main__":
    main()

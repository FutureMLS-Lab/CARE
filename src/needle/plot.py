#!/usr/bin/env python3
"""Plot a heatmap from CARE needle evaluation JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CARE needle evaluation results")
    parser.add_argument("--results-json", required=True, help="Path to needle_results.json")
    parser.add_argument("--metric", choices=["exact_match", "token_recall"], default="token_recall")
    parser.add_argument("--output", default=None, help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_json)
    rows = json.loads(results_path.read_text())
    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        values=args.metric,
        index="depth_percent",
        columns="context_length",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="viridis", vmin=0.0, vmax=1.0, annot=True, fmt=".2f")
    plt.title(f"Needle Evaluation ({args.metric})")
    plt.xlabel("Context Length")
    plt.ylabel("Needle Depth Percent")
    plt.tight_layout()

    output_path = Path(args.output) if args.output else results_path.with_suffix(f".{args.metric}.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot Niranjan TF–RBP event counts and exploratory pairwise exact tests."""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'graphs/niranjan'


def main():
    subprocess.run(['Rscript', str(ROOT / 'super_exact_overlap.R')], cwd=ROOT, check=True)
    out = pd.read_csv(BASE / 'tables/tf_rbp_count_exact_test_exploratory.csv')
    counts = out.pivot(index='TF', columns='RBP', values='shared_events')
    # Order by mean count, independent of Jaccard similarity.
    rows = counts.mean(axis=1).sort_values(ascending=False).index
    cols = counts.mean(axis=0).sort_values(ascending=False).index
    counts = counts.loc[rows, cols]
    significant = out.pivot(index='TF', columns='RBP', values='exploratory_fdr_below_0_05').loc[rows, cols]
    fig, ax = plt.subplots(figsize=(24, 10))
    im = ax.imshow(counts.to_numpy(), cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks(range(len(cols)), cols, rotation=55, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    threshold = counts.to_numpy().max() * .55
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = counts.iloc[i, j]
            ax.text(j, i, f'{val}{"*" if significant.iloc[i, j] else ""}',
                    ha='center', va='center', fontsize=6,
                    color='white' if val > threshold else 'black')
    ax.set(xlabel='RNA-binding protein / RNA-associated regulator', ylabel='Transcription factor',
           title='Shared exact events; * exploratory BH FDR < 0.05 (observed-event background)')
    fig.colorbar(im, ax=ax, label='Shared exact events', fraction=.025, pad=.02)
    fig.tight_layout()
    (BASE / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE / 'figures/tf_rbp_shared_event_count_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'{len(out)} pairs, {out.exploratory_fdr_below_0_05.sum()} exploratory FDR < 0.05')


if __name__ == '__main__':
    main()

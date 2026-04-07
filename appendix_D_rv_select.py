#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix D: RV Curve Selection and Comparison

Usage:
    python appendix_D_rv_select.py

Assumes 'rv_multimethod_noHalpha.csv' exists in the current directory.
Edit the SELECTED_LINE and SELECTED_METHOD variables below after reviewing rankings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# CONFIGURATION – EDIT THESE AFTER REVIEWING RANKINGS
# ===============================
INPUT_CSV = 'rv_multimethod_noHalpha.csv'
LIT_RV = 46.1                                   # km/s
EXCLUDE_LINES = ['Hgamma']
EXCLUDE_METHODS = []

OUTPUT_CSV = 'final_rv_selected.csv'
OUTPUT_PLOT = 'final_rv_selected.png'
OUTPUT_COMBINED_PLOT = 'all_rv_curves_combined.png'

# Change these after reviewing the rankings!
SELECTED_LINE = 'Hdelta'
SELECTED_METHOD = 'lorentzian_weighted'

LINE_COLORS = {
    'Halpha': 'red',
    'Hbeta': 'blue',
    'Hgamma': 'green',
    'Hdelta': 'purple',
    'Hepsilon': 'orange'
}

METHOD_MARKERS = {
    'centroid': 'o',
    'gaussian': 's',
    'gaussian_weighted': '^',
    'lorentzian': 'D',
    'lorentzian_weighted': 'v'
}

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    data = pd.read_csv(INPUT_CSV)
    required = ['MJD', 'RV_corrected_kmps', 'RV_error_kmps', 'line', 'method']
    if not all(col in data.columns for col in required):
        print("CSV missing required columns.")
        return

    # Apply exclusions
    if EXCLUDE_LINES:
        data = data[~data['line'].isin(EXCLUDE_LINES)]
    if EXCLUDE_METHODS:
        data = data[~data['method'].isin(EXCLUDE_METHODS)]

    # Compute statistics per (line, method)
    stats = []
    for (line, method), group in data.groupby(['line', 'method']):
        rv_vals = group['RV_corrected_kmps'].dropna()
        if len(rv_vals) < 5:
            continue
        mean_rv = np.mean(rv_vals)
        std_rv = np.std(rv_vals)
        n = len(rv_vals)
        diff_lit = abs(mean_rv - LIT_RV)

        stats.append({
            'line': line,
            'method': method,
            'mean_rv': mean_rv,
            'std': std_rv,
            'n': n,
            'diff_lit': diff_lit
        })

    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        print("No valid data after filtering.")
        return

    # Rankings
    print("\n" + "="*60)
    print(" RANKING BY SCATTER (std) – lowest first")
    print("="*60)
    print(stats_df.sort_values('std')[['line', 'method', 'mean_rv', 'std', 'n']].to_string(index=False))

    print("\n" + "="*60)
    print(f" RANKING BY CLOSENESS TO LITERATURE (|mean - {LIT_RV:.1f}|) – closest first")
    print("="*60)
    print(stats_df.sort_values('diff_lit')[['line', 'method', 'mean_rv', 'diff_lit', 'std']].to_string(index=False))

    # Combined plot
    print("\nGenerating combined plot with all RV curves...")
    plt.figure(figsize=(14, 8))

    for _, row in stats_df.iterrows():
        line = row['line']
        method = row['method']
        subset = data[(data['line'] == line) & (data['method'] == method)].dropna(subset=['RV_corrected_kmps']).sort_values('MJD')
        t = subset['MJD'].values
        rv = subset['RV_corrected_kmps'].values

        color = LINE_COLORS.get(line, 'gray')
        marker = METHOD_MARKERS.get(method, 'o')
        label = f"{line} ({method})"

        plt.scatter(t, rv, s=25, alpha=0.6, color=color, marker=marker, label=label, zorder=5)

    plt.axhline(LIT_RV, color='black', linestyle='-', linewidth=1.5, alpha=0.7,
                label=f'Literature RV = {LIT_RV} km/s')

    plt.xlabel('MJD', fontsize=12)
    plt.ylabel('RV (km/s)', fontsize=12)
    plt.title('All RV Curves – Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_COMBINED_PLOT, dpi=150)
    plt.show()
    print(f"Combined plot saved to {OUTPUT_COMBINED_PLOT}")

    # Individual plots
    print("\nGenerating individual scatter plots for all candidates...")
    for _, row in stats_df.iterrows():
        line = row['line']
        method = row['method']
        subset = data[(data['line'] == line) & (data['method'] == method)].dropna(subset=['RV_corrected_kmps']).sort_values('MJD')
        t = subset['MJD'].values
        rv = subset['RV_corrected_kmps'].values
        mean_rv = row['mean_rv']
        std = row['std']

        plt.figure(figsize=(10, 3))
        plt.scatter(t, rv, s=20, alpha=0.7, color='blue')
        plt.axhline(mean_rv, color='r', linestyle='--', alpha=0.5, label=f'Mean = {mean_rv:.2f}')
        plt.xlabel('MJD')
        plt.ylabel('RV (km/s)')
        plt.title(f'{line} – {method} (std = {std:.3f} km/s)')
        plt.ylim(mean_rv - 3*std, mean_rv + 3*std)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'temp_{line}_{method}.png', dpi=120)
        plt.close()
        print(f"  Saved scatter plot: temp_{line}_{method}.png")

    print(f"\nCurrently selected: {SELECTED_LINE} – {SELECTED_METHOD}")

    # Extract the chosen final curve
    final_mask = (data['line'] == SELECTED_LINE) & (data['method'] == SELECTED_METHOD)
    final_data = data[final_mask].dropna(subset=['RV_corrected_kmps', 'RV_error_kmps']).copy()
    if len(final_data) == 0:
        print(f"No data for selected combination {SELECTED_LINE} – {SELECTED_METHOD}. Check spelling.")
        return

    final_data = final_data[['MJD', 'RV_corrected_kmps', 'RV_error_kmps']].sort_values('MJD')
    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal RV curve saved to {OUTPUT_CSV} ({len(final_data)} points)")

    # Plot final curve
    plt.figure(figsize=(10, 5))
    plt.scatter(final_data['MJD'], final_data['RV_corrected_kmps'],
                s=30, alpha=0.7, color='blue', label='RV measurements')
    plt.xlabel('MJD')
    plt.ylabel('RV (km/s)')
    plt.title(f'Final RV curve: {SELECTED_LINE} – {SELECTED_METHOD}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.show()
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()

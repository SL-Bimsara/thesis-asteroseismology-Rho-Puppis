#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix C: ρ Puppis Post-Processing Temperature Plots

Usage:
    python appendix_C_plots.py

Assumes 'rho_pup_balmer_results.csv' exists in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ===============================
# CONFIGURATION
# ===============================
INPUT_CSV = "rho_pup_balmer_results.csv"
USE_PHASE = False
TIME_BINS = 20
Y_MIN = 2800
Y_MAX = 3200
MARKERSIZE_INDIV = 8
MARKERSIZE_BIN = 10
ALPHA_INDIV = 0.8

OUTPUT_BINNED = "rho_pup_texc_vs_time_binned.png"
OUTPUT_RAW = "rho_pup_texc_vs_time_raw.png"

# ===============================
# Binning function
# ===============================
def bin_data(x, y, nbins, x_range=None):
    if x_range is None:
        x_min, x_max = x.min(), x.max()
    else:
        x_min, x_max = x_range
    bin_edges = np.linspace(x_min, x_max, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    means = []
    stds = []
    for i in range(nbins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        y_bin = y[mask]
        if len(y_bin) > 0:
            means.append(np.nanmean(y_bin))
            stds.append(np.nanstd(y_bin))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return bin_centers, np.array(means), np.array(stds)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found. Please run the pipeline first.")

    df = pd.read_csv(INPUT_CSV)
    df_valid = df.dropna(subset=['T_ab', 'T_ag'], how='all')
    print(f"Loaded {len(df_valid)} valid spectra out of {len(df)} total.")

    if USE_PHASE:
        if 'phase' not in df_valid.columns:
            raise KeyError("Column 'phase' not found. Set USE_PHASE=False to plot vs. MJD.")
        x_vals = df_valid['phase'].values
        xlabel = 'Pulsation Phase'
        x_range = (0, 1)
    else:
        if 'MJD' not in df_valid.columns:
            raise KeyError("Column 'MJD' not found.")
        x_vals = df_valid['MJD'].values
        xlabel = 'MJD'
        x_range = (x_vals.min(), x_vals.max())

    # Prepare data for both ratios
    y_ab = df_valid['T_ab'].values
    mask_ab = ~np.isnan(y_ab)
    x_ab = x_vals[mask_ab]
    y_ab = y_ab[mask_ab]

    y_ag = df_valid['T_ag'].values
    mask_ag = ~np.isnan(y_ag)
    x_ag = x_vals[mask_ag]
    y_ag = y_ag[mask_ag]

    # Bin the data
    bin_centers_ab, means_ab, stds_ab = bin_data(x_ab, y_ab, TIME_BINS, x_range)
    bin_centers_ag, means_ag, stds_ag = bin_data(x_ag, y_ag, TIME_BINS, x_range)

    # Increase font sizes
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'legend.fontsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

    # Figure 1: Binned data
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.errorbar(bin_centers_ab, means_ab, yerr=stds_ab, fmt='o', color='#1f77b4',
                 ecolor='gray', capsize=4, markersize=MARKERSIZE_BIN, label='Hα/Hβ binned')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Excitation Temperature (K)')
    ax1.set_title('ρ Puppis – T$_{\\mathrm{exc}}$ from Hα/Hβ (binned)')
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    ax2.errorbar(bin_centers_ag, means_ag, yerr=stds_ag, fmt='o', color='#d62728',
                 ecolor='gray', capsize=4, markersize=MARKERSIZE_BIN, label='Hα/Hγ binned')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Excitation Temperature (K)')
    ax2.set_title('ρ Puppis – T$_{\\mathrm{exc}}$ from Hα/Hγ (binned)')
    ax2.set_ylim(Y_MIN, Y_MAX)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_BINNED, dpi=150)
    plt.show()
    print(f"Binned plot saved to {OUTPUT_BINNED}")

    # Figure 2: Raw data
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(x_ab, y_ab, color='#1f77b4', s=MARKERSIZE_INDIV, alpha=ALPHA_INDIV,
                label='Hα/Hβ raw')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Excitation Temperature (K)')
    ax1.set_title('ρ Puppis – T$_{\\mathrm{exc}}$ from Hα/Hβ (all points)')
    ax1.set_ylim(Y_MIN, Y_MAX)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    ax2.scatter(x_ag, y_ag, color='#d62728', s=MARKERSIZE_INDIV, alpha=ALPHA_INDIV,
                label='Hα/Hγ raw')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Excitation Temperature (K)')
    ax2.set_title('ρ Puppis – T$_{\\mathrm{exc}}$ from Hα/Hγ (all points)')
    ax2.set_ylim(Y_MIN, Y_MAX)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_RAW, dpi=150)
    plt.show()
    print(f"Raw data plot saved to {OUTPUT_RAW}")

if __name__ == '__main__':
    main()

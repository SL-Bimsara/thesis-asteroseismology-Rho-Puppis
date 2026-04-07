#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix B: Line-Depth Ratio Temperature Analysis for ρ Puppis

Usage:
    python appendix_B_ldr.py

Assumes 'ldr_results.csv' exists in the current directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def main():
    # Load data
    df = pd.read_csv('ldr_results.csv')
    mjd = df['MJD'].values

    # Use only the three good ratios
    ratios = {
        '4462/5573': df['ratio_4462_5573'].values,
        '5250/5365': df['ratio_5250_5365'].values,
        '5397/5411': df['ratio_5397_5411'].values
    }

    # Literature values
    T_mean_lit = 6550.0          # K
    S = 1500.0                   # K per unit ratio

    # Convert each ratio to temperature
    for name, r in ratios.items():
        r_mean = np.nanmean(r)
        df[f'T_{name}'] = T_mean_lit + S * (r - r_mean)

    # Average the three temperature estimates
    df['T_avg'] = np.nanmean([df[f'T_{name}'] for name in ratios], axis=0)
    df['T_std'] = np.nanstd([df[f'T_{name}'] for name in ratios], axis=0)

    # Remove any NaN rows
    df_clean = df.dropna(subset=['T_avg', 'T_std']).copy()
    mjd_clean = df_clean['MJD'].values
    T_avg = df_clean['T_avg'].values
    T_err = df_clean['T_std'].values

    # Known period (days)
    P_days = 0.14
    phase = (mjd_clean - mjd_clean.min()) / P_days
    phase = phase - np.floor(phase)

    # ---- Sine fit ----
    def sine_func(phase, A, phi, C):
        return A * np.sin(2 * np.pi * phase + phi) + C

    A_guess = (T_avg.max() - T_avg.min()) / 2
    C_guess = np.mean(T_avg)
    phi_guess = 0.0

    try:
        popt, pcov = curve_fit(sine_func, phase, T_avg,
                                p0=[A_guess, phi_guess, C_guess],
                                sigma=T_err, absolute_sigma=True)
        A_fit, phi_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))

        phase_sorted = np.linspace(0, 1, 200)
        T_fit = sine_func(phase_sorted, *popt)

        print("Sine fit results:")
        print(f"  Amplitude A = {A_fit:.1f} ± {perr[0]:.1f} K")
        print(f"  Phase φ    = {phi_fit:.3f} ± {perr[1]:.3f} rad")
        print(f"  Mean C     = {C_fit:.1f} ± {perr[2]:.1f} K")
        print(f"  Peak-to-peak variation = {2*A_fit:.1f} K")

    except Exception as e:
        print("Curve fit failed:", e)
        A_fit = phi_fit = C_fit = None

    # ---- Plotting ----
    plt.figure(figsize=(10, 6))
    plt.errorbar(phase, T_avg, yerr=T_err,
                 fmt='o', capsize=3, alpha=0.7, label='Data')

    if A_fit is not None:
        plt.plot(phase_sorted, T_fit, 'r-', lw=2,
                 label=f'Sine fit: A={A_fit:.1f} K, C={C_fit:.1f} K')

    plt.xlabel('Phase (P = 0.14 d)')
    plt.ylabel('Effective temperature (K)')
    plt.title('ρ Puppis – Temperature from line‑depth ratios with sine fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rho_pup_teff_phase_fit.png', dpi=150)
    plt.show()

    # Raw ratios plot
    plt.figure(figsize=(10,5))
    for name, r in ratios.items():
        plt.scatter(phase, r, label=name, alpha=0.7, s=50)
    plt.xlabel('Phase (P = 0.14 d)')
    plt.ylabel('Depth ratio')
    plt.title('ρ Puppis – Line‑depth ratios folded at 0.14 d')
    plt.legend()
    plt.grid(True)
    plt.savefig('ldr_folded_0.14d.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()

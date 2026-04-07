#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix F: Weighted Sine Fit for RV Curve

Usage:
    python appendix_F_sine_fit.py

Assumes 'rv_multimethod_noHalpha.csv' exists.
Edit SELECTED_LINE and SELECTED_METHOD below to match your chosen combination.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ===============================
# CONFIGURATION – EDIT THESE
# ===============================
INPUT_CSV = 'rv_multimethod_noHalpha.csv'
SELECTED_LINE = 'Hdelta'
SELECTED_METHOD = 'lorentzian_weighted'

KNOWN_PERIOD_HOURS = 3.38
KNOWN_PERIOD_DAYS = KNOWN_PERIOD_HOURS / 24.0

OUTPUT_PLOT = 'rv_sine_fit_selected.png'
OUTPUT_PARAMS = 'sine_fit_parameters_selected.txt'

def sine_model(t, gamma, K, t0):
    return gamma + K * np.sin(2 * np.pi * (t - t0) / KNOWN_PERIOD_DAYS)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    data = pd.read_csv(INPUT_CSV)
    required = ['MJD', 'RV_corrected_kmps', 'RV_error_kmps', 'line', 'method']
    if not all(col in data.columns for col in required):
        print("CSV missing required columns.")
        return

    mask = (data['line'] == SELECTED_LINE) & (data['method'] == SELECTED_METHOD)
    subset = data[mask].dropna(subset=['RV_corrected_kmps', 'RV_error_kmps']).copy()
    if len(subset) == 0:
        print(f"No valid data for {SELECTED_LINE} – {SELECTED_METHOD}")
        return

    t = subset['MJD'].values
    rv = subset['RV_corrected_kmps'].values
    rv_err = subset['RV_error_kmps'].values

    sort_idx = np.argsort(t)
    t = t[sort_idx]
    rv = rv[sort_idx]
    rv_err = rv_err[sort_idx]

    print(f"\nSelected {len(t)} measurements for {SELECTED_LINE} – {SELECTED_METHOD}")
    print(f"Mean error = {np.mean(rv_err):.3f} km/s")

    # Identify segments
    dt = np.diff(t)
    gap_idx = np.where(dt > 0.04)[0]
    if len(gap_idx) == 1:
        split = gap_idx[0] + 1
        t1, rv1, err1 = t[:split], rv[:split], rv_err[:split]
        t2, rv2, err2 = t[split:], rv[split:], rv_err[split:]
        print(f"Segment 1: {len(t1)} points, MJD {t1[0]:.5f} – {t1[-1]:.5f}")
        print(f"Segment 2: {len(t2)} points, MJD {t2[0]:.5f} – {t2[-1]:.5f}")
    else:
        t1, rv1, err1 = t, rv, rv_err
        print("No clear gap – treating as one segment.")

    # Initial fit
    gamma0 = np.mean(rv)
    K0 = 0.5 * (np.max(rv) - np.min(rv))
    t0_guess = t[0]

    try:
        popt0, pcov0 = curve_fit(
            sine_model, t, rv,
            sigma=rv_err, absolute_sigma=True,
            p0=[gamma0, K0, t0_guess],
            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        gamma0_fit, K0_fit, t0_0_fit = popt0
        perr0 = np.sqrt(np.diag(pcov0)) if pcov0 is not None else [0,0,0]

        rv_fit0 = sine_model(t, gamma0_fit, K0_fit, t0_0_fit)
        chi2_0 = np.sum(((rv - rv_fit0) / rv_err)**2)
        dof = len(t) - 3
        red_chi2_0 = chi2_0 / dof

        print("\n=== Initial weighted sine fit (original errors) ===")
        print(f"gamma = {gamma0_fit:.3f} ± {perr0[0]:.3f} km/s")
        print(f"K = {K0_fit:.3f} ± {perr0[1]:.3f} km/s")
        print(f"t0 = {t0_0_fit:.5f} ± {perr0[2]:.5f} MJD")
        print(f"Reduced chi^2 = {red_chi2_0:.2f}")

        scale_factor = np.sqrt(red_chi2_0)
        print(f"\nScaling errors by factor {scale_factor:.3f} to obtain chi^2/nu = 1")

        rv_err_scaled = rv_err * scale_factor

        popt, pcov = curve_fit(
            sine_model, t, rv,
            sigma=rv_err_scaled, absolute_sigma=True,
            p0=[gamma0_fit, K0_fit, t0_0_fit],
            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        gamma_fit, K_fit, t0_fit = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0,0,0]

        rv_fit = sine_model(t, gamma_fit, K_fit, t0_fit)
        chi2 = np.sum(((rv - rv_fit) / rv_err_scaled)**2)
        red_chi2 = chi2 / dof

        print("\n=== Final weighted sine fit (errors scaled) ===")
        print(f"gamma (systemic velocity) = {gamma_fit:.3f} ± {perr[0]:.3f} km/s")
        print(f"K (semi‑amplitude)    = {K_fit:.3f} ± {perr[1]:.3f} km/s")
        print(f"t0 (epoch of max RV)  = {t0_fit:.5f} ± {perr[2]:.5f} MJD")
        print(f"Reduced chi^2 (after scaling) = {red_chi2:.2f}")

        with open(OUTPUT_PARAMS, 'w', encoding='utf-8') as f:
            f.write(f"gamma = {gamma_fit:.3f} +/- {perr[0]:.3f} km/s\n")
            f.write(f"K     = {K_fit:.3f} +/- {perr[1]:.3f} km/s\n")
            f.write(f"t0    = {t0_fit:.5f} +/- {perr[2]:.5f} MJD\n")
            f.write(f"Period (fixed) = {KNOWN_PERIOD_DAYS:.6f} days ({KNOWN_PERIOD_HOURS} h)\n")
            f.write(f"Reduced chi^2 (after scaling) = {red_chi2:.2f}\n")
            f.write(f"Note: Original errors were scaled by {scale_factor:.3f} to achieve chi^2/nu = 1.\n")
        print(f"\nParameters saved to {OUTPUT_PARAMS}")

        # Plotting
        phase = ((t - t0_fit) / KNOWN_PERIOD_DAYS) % 1.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.axvspan(t1[0], t1[-1], alpha=0.2, color='blue', label='Window 1')
        if len(t2) > 0:
            ax1.axvspan(t2[0], t2[-1], alpha=0.2, color='green', label='Window 2')
        ax1.errorbar(t, rv, yerr=rv_err_scaled, fmt='ko', markersize=4, capsize=2, label='Data (scaled errors)')
        t_plot = np.linspace(t.min(), t.max(), 500)
        rv_model = sine_model(t_plot, gamma_fit, K_fit, t0_fit)
        ax1.plot(t_plot, rv_model, 'r-', linewidth=2, label='Weighted sine fit')
        ax1.set_xlabel('MJD')
        ax1.set_ylabel('RV (km/s)')
        ax1.set_title(f'ρ Puppis: {SELECTED_LINE} – {SELECTED_METHOD} (P = {KNOWN_PERIOD_HOURS} h fixed)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        phase_sorted = np.sort(phase)
        rv_phase_sorted = rv[np.argsort(phase)]
        err_phase_sorted = rv_err_scaled[np.argsort(phase)]
        ax2.errorbar(phase, rv, yerr=rv_err_scaled, fmt='bo', markersize=4, capsize=2, alpha=0.7, label='Data')
        phase_ext = np.concatenate([phase_sorted, phase_sorted+1])
        rv_ext = np.concatenate([rv_phase_sorted, rv_phase_sorted])
        ax2.plot(phase_ext, rv_ext, 'b-', alpha=0.3)
        phase_model = np.linspace(0, 2, 500)
        rv_model_phase = sine_model(t0_fit + phase_model * KNOWN_PERIOD_DAYS, gamma_fit, K_fit, t0_fit)
        ax2.plot(phase_model, rv_model_phase, 'r-', linewidth=2, label='Sine model')
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=1, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('RV (km/s)')
        ax2.set_title(f'Phased data (P = {KNOWN_PERIOD_HOURS} h)')
        ax2.set_xlim(0, 2)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150)
        print(f"Plot saved to {OUTPUT_PLOT}")
        plt.show()

    except Exception as e:
        print(f"Sine fitting failed: {e}")

if __name__ == '__main__':
    main()

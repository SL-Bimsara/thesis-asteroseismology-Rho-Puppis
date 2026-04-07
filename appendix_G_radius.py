#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix G: Radius Variation by Block

Usage:
    python appendix_G_radius.py

Assumes 'final_rv_selected.csv' exists in the current directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ===============================
# CONFIGURATION
# ===============================
INPUT_CSV = 'final_rv_selected.csv'
OUTPUT_CSV = 'radius_by_block.csv'
OUTPUT_PLOT = 'radius_by_block.png'

PROJECTION_FACTOR = 1.36
PROJECTION_FACTOR_ERR = 0.02
KNOWN_PERIOD_DAYS = 3.38 / 24.0

R_SUN_KM = 695700.0
SECONDS_PER_DAY = 86400.0
GAP_THRESHOLD = 0.01

N_MC = 1000

def sine_model(t, gamma, A, t0, P):
    return gamma + A * np.cos(2 * np.pi * (t - t0) / P)

def integrate_block(t_block, rv_block, gamma, p, r0_offset=0):
    v_puls = rv_block - gamma
    dt = np.diff(t_block) * SECONDS_PER_DAY
    v_avg = 0.5 * (v_puls[:-1] + v_puls[1:])
    delta_R_km = -p * np.cumsum(v_avg * dt)
    delta_R_km = np.concatenate(([0], delta_R_km))
    return (delta_R_km / R_SUN_KM) + r0_offset

def fit_block_offset(t_block, rv_block, rv_err_block, gamma, p):
    dR_zero = integrate_block(t_block, rv_block, gamma, p, r0_offset=0)
    offset = -np.mean(dR_zero)

    mc_offsets = []
    for i in range(N_MC):
        rv_pert = rv_block + np.random.normal(0, rv_err_block)
        p_pert = np.random.normal(p, PROJECTION_FACTOR_ERR)
        dR_pert = integrate_block(t_block, rv_pert, gamma, p_pert, r0_offset=0)
        mc_offsets.append(-np.mean(dR_pert))

    offset_err = np.std(mc_offsets)
    return offset, offset_err

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run appendix_D_rv_select.py first.")
        return

    data = pd.read_csv(INPUT_CSV)
    t = data['MJD'].values
    rv = data['RV_corrected_kmps'].values
    rv_err = data['RV_error_kmps'].values

    sort_idx = np.argsort(t)
    t = t[sort_idx]
    rv = rv[sort_idx]
    rv_err = rv_err[sort_idx]

    print(f"Loaded {len(t)} measurements.")

    # Split into blocks
    block_starts = [0]
    for i in range(1, len(t)):
        if t[i] - t[i-1] > GAP_THRESHOLD:
            block_starts.append(i)
    block_starts.append(len(t))
    blocks = [(block_starts[i], block_starts[i+1]) for i in range(len(block_starts)-1)]

    print(f"Detected {len(blocks)} block(s):")
    for i, (s, e) in enumerate(blocks):
        print(f"  Block {i+1}: {t[s]:.5f} – {t[e-1]:.5f}  ({e-s} points)")

    # Global gamma fit
    print("\nFitting global sine to estimate systemic velocity...")
    gamma_guess = np.median(rv)
    A_guess = 0.5 * (rv.max() - rv.min())
    t0_guess = t[np.argmax(rv)]
    P_guess = KNOWN_PERIOD_DAYS

    try:
        popt_global, pcov_global = curve_fit(
            sine_model, t, rv, sigma=rv_err, absolute_sigma=True,
            p0=[gamma_guess, A_guess, t0_guess, P_guess],
            bounds=([-100, 0, t.min()-0.1, P_guess*0.9],
                    [100, 100, t.max()+0.1, P_guess*1.1])
        )
        gamma_global, A_global, t0_global, P_global = popt_global
        gamma_err = np.sqrt(pcov_global[0,0])

        print(f"  gamma = {gamma_global:.3f} ± {gamma_err:.3f} km/s")
        print(f"  A = {A_global:.3f} km/s")
        print(f"  P = {P_global*24:.3f} hours")

    except Exception as e:
        print(f"  Global fit failed: {e}")
        print("  Using median RV as gamma")
        gamma_global = np.median(rv)
        gamma_err = np.std(rv) / np.sqrt(len(rv))

    # Process each block
    all_t = []
    all_dR = []
    all_err = []
    all_block = []
    block_offsets = []

    for block_idx, (start, end) in enumerate(blocks):
        t_block = t[start:end]
        rv_block = rv[start:end]
        rv_err_block = rv_err[start:end]
        npts = len(t_block)

        print(f"\nProcessing Block {block_idx+1}:")

        offset, offset_err = fit_block_offset(t_block, rv_block, rv_err_block,
                                              gamma_global, PROJECTION_FACTOR)
        block_offsets.append(offset)
        print(f"  Fitted offset = {offset:.5f} ± {offset_err:.5f} R☉")

        mc_dR = np.zeros((N_MC, npts))

        for i in range(N_MC):
            rv_pert = rv_block + np.random.normal(0, rv_err_block)
            p_pert = np.random.normal(PROJECTION_FACTOR, PROJECTION_FACTOR_ERR)
            gamma_pert = np.random.normal(gamma_global, gamma_err)
            offset_pert = np.random.normal(offset, offset_err)
            mc_dR[i, :] = integrate_block(t_block, rv_pert, gamma_pert, p_pert, offset_pert)

        dR_med = np.median(mc_dR, axis=0)
        dR_low = np.percentile(mc_dR, 16, axis=0)
        dR_high = np.percentile(mc_dR, 84, axis=0)
        dR_err = 0.5 * (dR_high - dR_low)

        all_t.extend(t_block)
        all_dR.extend(dR_med)
        all_err.extend(dR_err)
        all_block.extend([block_idx+1] * npts)

        print(f"  ΔR range: [{dR_med.min():.5f}, {dR_med.max():.5f}] R☉")
        print(f"  Amplitude: {dR_med.max()-dR_med.min():.5f} R☉")

    # Save
    out_df = pd.DataFrame({
        'MJD': all_t,
        'delta_R_Rsun': all_dR,
        'delta_R_error': all_err,
        'block': all_block,
        'gamma_used': gamma_global,
        'gamma_error': gamma_err
    })
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nData saved to {OUTPUT_CSV}")

    # Fit sine to each block
    def sine_model_radius(t, C, A, t0, P):
        return C + A * np.cos(2 * np.pi * (t - t0) / P)

    print("\n" + "="*50)
    print("Fitting radius variations per block:")

    for block_idx in range(1, len(blocks)+1):
        mask = out_df['block'] == block_idx
        t_blk = out_df.loc[mask, 'MJD'].values
        dR_blk = out_df.loc[mask, 'delta_R_Rsun'].values
        err_blk = out_df.loc[mask, 'delta_R_error'].values

        if len(t_blk) < 4:
            print(f"\nBlock {block_idx}: too few points ({len(t_blk)}), skipping fit")
            continue

        C0 = np.median(dR_blk)
        A0 = 0.5 * (dR_blk.max() - dR_blk.min())
        t0_guess = t_blk[np.argmax(dR_blk)]

        def model_fixedP(t, C, A, t0):
            return C + A * np.cos(2 * np.pi * (t - t0) / KNOWN_PERIOD_DAYS)

        try:
            popt, pcov = curve_fit(model_fixedP, t_blk, dR_blk,
                                   sigma=err_blk, absolute_sigma=True,
                                   p0=[C0, A0, t0_guess], maxfev=5000)
            C_fit, A_fit, t0_fit = popt
            perr = np.sqrt(np.diag(pcov))

            print(f"\nBlock {block_idx} (fixed P = {KNOWN_PERIOD_DAYS*24:.2f} h):")
            print(f"  Mean radius offset: {C_fit:.5f} ± {perr[0]:.5f} R☉")
            print(f"  Semi-amplitude:     {A_fit:.5f} ± {perr[1]:.5f} R☉")
            print(f"  Phase zero (MJD):   {t0_fit:.6f} ± {perr[2]:.6f}")

        except Exception as e:
            print(f"\nBlock {block_idx}: fixed-period fit failed: {e}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.errorbar(t, rv, yerr=rv_err, fmt='o', capsize=2, alpha=0.5, label='RV data')
    t_model = np.linspace(t.min(), t.max(), 1000)
    ax1.plot(t_model, sine_model(t_model, gamma_global, A_global, t0_global, P_global),
             'r-', label=f'Global fit: γ={gamma_global:.2f} km/s')
    ax1.axhline(y=gamma_global, color='k', linestyle='--', alpha=0.5, label=f'γ = {gamma_global:.2f} km/s')
    ax1.set_ylabel('RV (km/s)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(blocks)))
    for block_idx in range(1, len(blocks)+1):
        mask = out_df['block'] == block_idx
        ax2.errorbar(out_df.loc[mask, 'MJD'], out_df.loc[mask, 'delta_R_Rsun'],
                     yerr=out_df.loc[mask, 'delta_R_error'], fmt='o', capsize=2,
                     color=colors[block_idx-1], label=f'Block {block_idx}', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('MJD')
    ax2.set_ylabel('ΔR (R☉)')
    ax2.set_title('Relative radius variation (with global systemic velocity correction)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PLOT}")
    plt.show()

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Systemic velocity (γ): {gamma_global:.3f} ± {gamma_err:.3f} km/s")
    print(f"  Pulsation period:      {P_global*24:.3f} hours")
    print(f"  Number of blocks:      {len(blocks)}")
    print(f"  Block offsets (R☉):    {', '.join([f'{o:.5f}' for o in block_offsets])}")

if __name__ == '__main__':
    main()

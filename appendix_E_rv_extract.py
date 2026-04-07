#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix E: Radial Velocity Analysis of ρ Puppis

Usage:
    python appendix_E_rv_extract.py /path/to/data_folder

Example:
    python appendix_E_rv_extract.py ./spectra

Dependencies:
    numpy, astropy, scipy, matplotlib
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

# ===============================
# CONFIGURATION
# ===============================
C = 299792.458  # km/s
FORCE_EXTRACT = True
PLOT_FITS = False
INTERACTIVE = False
PLOT_DIR = 'rv_fit_plots'

# Air rest wavelengths (from NIST)
H_LINES = {
    'Hbeta':  4861.363,
    'Hgamma': 4340.477,
    'Hdelta': 4101.750
}

FIT_WIDTHS = {
    'Hbeta':  20.0,
    'Hgamma': 20.0,
    'Hdelta': 20.0
}

def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

def lorentzian(x, a, x0, gamma, c):
    return a * (gamma**2 / ((x - x0)**2 + gamma**2)) + c

METHODS = [
    {'name': 'gaussian_unweighted', 'func': gaussian, 'weighted': False, 'profile': 'gaussian'},
    {'name': 'gaussian_weighted',   'func': gaussian, 'weighted': True,  'profile': 'gaussian'},
    {'name': 'lorentzian_unweighted','func': lorentzian, 'weighted': False, 'profile': 'lorentzian'},
    {'name': 'lorentzian_weighted',  'func': lorentzian, 'weighted': True,  'profile': 'lorentzian'},
]

# ===============================
# HARPS reader with BERV
# ===============================
def read_harps_spectrum(filename):
    with fits.open(filename) as hdul:
        header0 = hdul[0].header
        mjd = header0.get('MJD-OBS', None)
        if mjd is None:
            date_obs = header0.get('DATE-OBS', None)
            if date_obs:
                mjd = Time(date_obs, format='fits').mjd
            else:
                mjd = 0.0

        berv = header0.get('HIERARCH ESO DRS BERV', 0.0)
        if berv == 0.0:
            berv = header0.get('BERV', 0.0)

        spec_hdu = hdul[1]
        data = spec_hdu.data
        wave = data['WAVE'][0]
        flux = data['FLUX'][0]
    return wave, flux, mjd, berv, header0

# ===============================
# Build initial parameters and bounds
# ===============================
def build_p0_bounds(x, y, line_center, half_width, profile_type):
    y_min, y_max = np.min(y), np.max(y)
    if profile_type == 'gaussian':
        p0 = [y_max - y_min, line_center, 0.5, y_min]
        bounds = ([-np.inf, line_center - half_width, 0, -np.inf],
                  [np.inf, line_center + half_width, np.inf, np.inf])
    elif profile_type == 'lorentzian':
        p0 = [y_max - y_min, line_center, 0.5, y_min]
        bounds = ([-np.inf, line_center - half_width, 0, -np.inf],
                  [np.inf, line_center + half_width, np.inf, np.inf])
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")
    return p0, bounds

# ===============================
# Measure RV for one method
# ===============================
def measure_rv_method(wave, flux, line_center, fit_width, method):
    half = fit_width / 2.0
    mask = (wave >= line_center - half) & (wave <= line_center + half)
    if np.sum(mask) < 10:
        return np.nan, np.nan
    x = wave[mask]
    y = flux[mask]

    p0, bounds = build_p0_bounds(x, y, line_center, half, method['profile'])
    func = method['func']

    try:
        if method['weighted']:
            weights = 1.0 / np.sqrt(y + 1e-6)
            popt, pcov = curve_fit(func, x, y, sigma=weights, absolute_sigma=False,
                                   p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds)

        x0_fit = popt[1]
        rv = C * (x0_fit - line_center) / line_center

        if pcov is not None:
            x0_err = np.sqrt(pcov[1,1])
            rv_err = C * x0_err / line_center
        else:
            rv_err = np.nan

        return rv, rv_err
    except Exception:
        return np.nan, np.nan

# ===============================
# Plotting function (optional)
# ===============================
def plot_fit(wave, flux, line_name, rest, fname, mjd, method_name, outdir, interactive=False):
    method = next((m for m in METHODS if m['name'] == method_name), METHODS[0])
    half = FIT_WIDTHS[line_name] / 2.0
    mask = (wave >= rest - half) & (wave <= rest + half)
    if np.sum(mask) < 10:
        return

    x = wave[mask]
    y = flux[mask]

    p0, bounds = build_p0_bounds(x, y, rest, half, method['profile'])
    func = method['func']

    try:
        if method['weighted']:
            weights = 1.0 / np.sqrt(y + 1e-6)
            popt, _ = curve_fit(func, x, y, sigma=weights, absolute_sigma=False,
                                p0=p0, bounds=bounds)
        else:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds)

        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = func(x_fit, *popt)
        rv = C * (popt[1] - rest) / rest

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'{fname} – {line_name} (MJD={mjd:.5f})\nMethod: {method_name}\nRV = {rv:.2f} km/s')

        ax1.plot(x, y, 'k-', lw=1, label='Data')
        ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'{method["profile"].capitalize()} fit')
        ax1.axhline(popt[-1], color='b', linestyle='--', lw=1, label='Continuum')
        ax1.set_ylabel('Flux')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        y_fit_data = func(x, *popt)
        residuals = y - y_fit_data
        ax2.plot(x, residuals, 'k-', lw=1)
        ax2.axhline(0, color='r', linestyle='--', lw=1)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            os.makedirs(outdir, exist_ok=True)
            plot_fname = f"{os.path.splitext(fname)[0]}_{line_name}_fit_{method_name}.png"
            plot_path = os.path.join(outdir, plot_fname)
            plt.savefig(plot_path, dpi=120)
            plt.close()
    except Exception:
        pass

# ===============================
# Main extraction
# ===============================
def extract_all_rvs(data_path, output_csv):
    files = [f for f in os.listdir(data_path) if f.lower().endswith(('.fits', '.fit', '.fits.gz'))]
    if not files:
        print(f"No FITS files found in {data_path}")
        return False
    files.sort()
    print(f"Found {len(files)} spectra. Extracting RVs...")

    csv_rows = []

    for i, fname in enumerate(files):
        fullpath = os.path.join(data_path, fname)
        print(f"\n[{i+1}/{len(files)}] {fname}")
        try:
            wave, flux, mjd, berv, header = read_harps_spectrum(fullpath)
            print(f"  MJD = {mjd:.5f}   BERV = {berv:.3f} km/s")

            for line_name, rest in H_LINES.items():
                if rest < wave[0] or rest > wave[-1]:
                    print(f"    {line_name}: rest {rest} Å outside range – skip")
                    continue

                for method in METHODS:
                    rv_raw, rv_err = measure_rv_method(wave, flux, rest, FIT_WIDTHS[line_name], method)
                    rv_corr = rv_raw + berv if not np.isnan(rv_raw) else np.nan
                    csv_rows.append([fname, mjd, line_name, method['name'],
                                     rv_raw, rv_corr, berv, rv_err])

                    if not np.isnan(rv_raw):
                        print(f"    {line_name} {method['name']}: RV = {rv_raw:.2f} ± {rv_err:.3f} km/s (raw), {rv_corr:.2f} (corr)")
                    else:
                        print(f"    {line_name} {method['name']}: fit failed")

                if PLOT_FITS:
                    plot_fit(wave, flux, line_name, rest, fname, mjd,
                             METHODS[0]['name'], PLOT_DIR, interactive=INTERACTIVE)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not csv_rows:
        print("No data collected.")
        return False

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'MJD', 'line', 'method', 'RV_raw_kmps', 'RV_corrected_kmps', 'BERV_kmps', 'RV_error_kmps'])
        writer.writerows(csv_rows)
    print(f"\nSaved results to {output_csv}")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python appendix_E_rv_extract.py <data_path>")
        print("Example: python appendix_E_rv_extract.py ./spectra")
        sys.exit(1)

    data_path = sys.argv[1]
    output_csv = 'rv_multimethod_noHalpha.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    success = extract_all_rvs(data_path, output_csv)
    if not success:
        return

    print("\nExtraction complete. Run appendix_D_rv_select.py to analyze results.")

if __name__ == '__main__':
    main()

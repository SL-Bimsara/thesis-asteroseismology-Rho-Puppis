#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix A: Curve of Growth Data Extraction for Fe I Lines

Usage:
    python appendix_A_cog.py /path/to/data_folder filename.fits

Example:
    python appendix_A_cog.py ./data 05.19.48.fits

Dependencies:
    numpy, astropy, specutils, matplotlib
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import equivalent_width

# ===============================
# Fe I line list (wavelength, EP, loggf)
# ===============================
FE_LINES = [
    (4383.55, 4.17, 0.200), (4404.75, 4.37, -0.142), (4415.12, 4.41, -0.615),
    (4442.34, 2.20, -1.255), (4447.72, 2.22, -1.342), (4459.12, 2.18, -1.279),
    (4461.65, 0.09, -3.210), (4494.57, 2.20, -1.136), (4528.62, 2.18, -0.822),
    (4531.15, 3.21, -2.155), (4602.94, 1.48, -3.154), (4630.12, 2.28, -2.486),
    (4736.77, 3.21, -1.165), (4871.32, 2.87, -0.363), (4890.76, 2.87, -0.394),
    (4891.50, 2.85, -0.112), (4918.99, 2.85, -0.342), (4920.50, 2.83, 0.068),
    (4957.30, 2.85, -0.410), (4994.13, 0.91, -3.080), (5006.12, 2.83, -0.638),
    (5012.07, 0.86, -2.642), (5049.82, 2.28, -1.355), (5051.64, 0.91, -2.795),
    (5068.77, 2.94, -1.041), (5074.75, 4.22, -0.200), (5083.34, 0.96, -2.958),
    (5090.77, 4.26, -0.400), (5125.12, 4.22, -0.140), (5131.47, 2.22, -2.515),
    (5133.69, 4.18, 0.140), (5141.74, 2.42, -2.238), (5142.93, 0.96, -3.087),
    (5150.84, 1.01, -3.022), (5151.91, 1.01, -3.322), (5162.27, 4.18, 0.020),
    (5166.28, 0.00, -4.195), (5171.60, 1.48, -1.793), (5191.46, 3.04, -0.551),
    (5194.94, 1.55, -2.090), (5198.71, 2.22, -2.135), (5215.18, 3.27, -0.871),
    (5216.27, 1.61, -2.150), (5225.53, 0.11, -4.789), (5232.94, 2.94, -0.058),
    (5242.49, 3.63, -0.967), (5250.21, 0.12, -4.938), (5250.65, 2.20, -2.181),
    (5266.56, 3.00, -0.385), (5269.54, 0.86, -1.321), (5281.79, 3.04, -0.834),
    (5283.62, 3.24, -0.432), (5302.30, 3.28, -0.720), (5307.36, 1.61, -2.987),
    (5324.18, 3.21, -0.103), (5328.04, 0.91, -1.466), (5332.90, 1.55, -2.777),
    (5364.87, 4.45, 0.228), (5367.47, 4.41, 0.443), (5369.96, 4.37, 0.536),
    (5371.49, 0.96, -1.645), (5383.37, 4.31, 0.645), (5393.17, 3.24, -0.715),
    (5397.13, 0.91, -1.993), (5405.78, 0.99, -1.844), (5410.91, 4.47, 0.398),
    (5415.20, 4.39, 0.642), (5424.07, 4.32, 0.520), (5429.70, 0.96, -1.879),
    (5434.52, 1.01, -2.122), (5445.04, 4.39, 0.040), (5446.92, 0.99, -1.914),
    (5455.61, 1.01, -2.091), (5497.52, 1.01, -2.849), (5506.78, 0.99, -2.797),
    (5522.45, 4.21, -0.360), (5525.55, 4.23, -0.540), (5560.21, 4.43, -1.090),
    (5563.60, 4.19, -0.720), (5572.84, 3.40, -0.310), (5586.76, 3.37, -0.120),
    (5615.64, 3.33, 0.050), (5624.54, 3.42, -0.650), (5633.95, 4.99, -0.270),
    (5635.82, 4.26, -1.890), (5638.26, 4.22, -0.720), (5641.43, 4.26, -1.000),
    (5650.71, 5.09, -0.930), (5651.47, 4.47, -1.900), (5652.32, 4.26, -1.800),
    (5661.35, 4.28, -1.756), (5662.52, 4.28, -0.573), (5678.42, 4.28, -0.920),
    (5691.50, 4.28, -1.420), (5701.55, 2.56, -2.160), (5705.47, 4.30, -1.355),
    (5717.83, 4.28, -0.980)
]

# Balmer lines (for continuum fitting)
H_LINES = {
    'Halpha': 6564.614,
    'Hbeta':  4862.721,
    'Hgamma': 4341.691,
    'Hdelta': 4102.899
}

# Parameters
EXCLUDE_WIDTH = 20.0
METAL_LINE_WIDTH = 5.0
EW_MIN = 0.001
EW_MAX = 1.2
K_BOLTZ = 8.617e-5
POLY_ORDER = 12
REJECT_SIGMA = 2.2
MAX_ITER = 30

# ===============================
# Functions
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
        spec_hdu = hdul[1]
        data = spec_hdu.data
        wave = data['WAVE'][0]
        flux = data['FLUX'][0]
    return wave, flux, mjd

def fit_continuum_polynomial_iterative(wave, flux,
                                       poly_order=POLY_ORDER,
                                       exclude_balmer=True,
                                       balmer_width=EXCLUDE_WIDTH,
                                       reject_sigma=REJECT_SIGMA,
                                       max_iter=MAX_ITER):
    w_min, w_max = wave.min(), wave.max()
    w_scaled = (wave - w_min) / (w_max - w_min)
    mask = np.ones_like(wave, dtype=bool)
    if exclude_balmer:
        for lam in H_LINES.values():
            mask &= (wave < lam - balmer_width) | (wave > lam + balmer_width)
    for iteration in range(max_iter):
        w_scaled_masked = w_scaled[mask]
        flux_masked = flux[mask]
        coeffs = np.polyfit(w_scaled_masked, flux_masked, poly_order)
        continuum = np.polyval(coeffs, w_scaled)
        residuals = flux - continuum
        std_resid = np.std(residuals[mask])
        new_mask = mask & (residuals > -reject_sigma * std_resid)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    continuum = np.maximum(continuum, 0.1 * np.median(flux))
    return continuum, mask

def measure_ew_safe(wave, flux_norm, line_center, width):
    half = width / 2.0
    if line_center - half < wave[0] or line_center + half > wave[-1]:
        return np.nan
    center = line_center * u.AA
    spectrum = Spectrum1D(spectral_axis=wave * u.AA,
                          flux=flux_norm * u.dimensionless_unscaled)
    region = SpectralRegion(center - half * u.AA, center + half * u.AA)
    try:
        ew = equivalent_width(spectrum, regions=region)
        return ew.value
    except:
        return np.nan

def excitation_temperature(ew_dict, ew_min=EW_MIN, ew_max=EW_MAX):
    x, y = [], []
    for wave, (ew, ep, loggf) in ew_dict.items():
        if ew_min < ew < ew_max:
            gf = 10**loggf
            y_val = np.log(ew) - 2*np.log(wave) - np.log(gf)
            x.append(ep)
            y.append(y_val)
    nlines = len(x)
    if nlines < 5:
        return None, None, nlines, None
    x = np.array(x)
    y = np.array(y)
    try:
        popt, pcov = np.polyfit(x, y, 1, cov=True)
        b, a = popt
        perr = np.sqrt(np.diag(pcov))
        b_err = perr[0]
    except:
        return None, None, nlines, None
    if b >= 0:
        return None, None, nlines, None
    Tex = -1.0 / (b * K_BOLTZ)
    Tex_err = abs(K_BOLTZ * Tex**2 * b_err)
    y_fit = a + b * x
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return Tex, Tex_err, nlines, r2

# ===============================
# Main
# ===============================
def main():
    if len(sys.argv) != 3:
        print("Usage: python appendix_A_cog.py <data_path> <filename>")
        print("Example: python appendix_A_cog.py ./data 05.19.48.fits")
        sys.exit(1)
    
    data_path = sys.argv[1]
    filename = sys.argv[2]
    fullpath = os.path.join(data_path, filename)
    
    if not os.path.exists(fullpath):
        print(f"File not found: {fullpath}")
        return

    print(f"Processing {filename} ...")
    wave, flux, mjd = read_harps_spectrum(fullpath)
    print(f"  MJD = {mjd:.5f}, λ range: {wave[0]:.1f} – {wave[-1]:.1f} Å")

    # Continuum normalisation
    continuum, _ = fit_continuum_polynomial_iterative(wave, flux)
    flux_norm = flux / continuum
    flux_norm = np.clip(flux_norm, 0, 2)

    # Measure Fe I lines
    ew_dict = {}
    for w, ep, loggf in FE_LINES:
        if w < wave[0] or w > wave[-1]:
            continue
        ew = measure_ew_safe(wave, flux_norm, w, METAL_LINE_WIDTH)
        if np.isnan(ew):
            continue
        ew_dict[w] = (ew, ep, loggf)

    print(f"  Measured {len(ew_dict)} Fe I lines.")

    # Compute excitation temperature
    Tex, Tex_err, nlines, r2 = excitation_temperature(ew_dict, EW_MIN, EW_MAX)
    if Tex is not None:
        print(f"  Tex = {Tex:.0f} ± {Tex_err:.0f} K ({nlines} lines, R² = {r2:.3f})")
    else:
        print("  Could not determine Tex, using default 6500 K.")
        Tex = 6500.0

    theta = 5040.0 / Tex

    # Prepare data for printing
    data = []
    for w, (ew, ep, loggf) in ew_dict.items():
        gf = 10**loggf
        x = gf * w * 10**(-theta * ep)
        logx = np.log10(x)
        logy = np.log10(ew / w)
        data.append((w, ew, ep, loggf, logx, logy))

    data.sort(key=lambda d: d[4])

    print("\n" + "="*100)
    print("Wavelength (Å)   EW (Å)      EP (eV)   loggf     logX        log(EW/λ)")
    print("="*100)

    for w, ew, ep, loggf, logx, logy in data:
        print(f"{w:10.2f}     {ew:8.3f}   {ep:6.2f}   {loggf:6.3f}   {logx:8.4f}   {logy:8.4f}")

    print("="*100)
    print(f"Total lines printed: {len(data)}")

if __name__ == '__main__':
    main()

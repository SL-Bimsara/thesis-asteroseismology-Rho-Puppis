# Thesis Code: Asteroseismic Analysis of ρ Puppis

Code for my undergraduate thesis on ρ Puppis: curve of growth, LDR temperature, RV analysis, and radius variations.

## Contents

| File | Description |
|------|-------------|
| `appendix_A_cog.py` | Curve of growth data extraction for Fe I lines |
| `appendix_B_ldr.py` | Line-depth ratio temperature analysis |
| `appendix_C_plots.py` | Post-processing temperature plots |
| `appendix_D_rv_select.py` | RV curve selection and comparison |
| `appendix_E_rv_extract.py` | Radial velocity analysis (Balmer lines) |
| `appendix_F_sine_fit.py` | Weighted sine fit for RV curve |
| `appendix_G_radius.py` | Radius variation by block |

## Requirements

```bash
pip install numpy scipy matplotlib pandas astropy specutils

"""
Calculate physical parameters for final/merged H I sources from ROHSA Gaussian
parameter CSV files.

Main adopted formulae
---------------------
1. Gaussian moment-0 per pixel:
       W_HI = A * sigma_v * sqrt(2*pi)        [K km/s]

2. H I column density per pixel:
       N_HI = 1.823e18 * W_HI                 [cm^-2]

   The table reports the maximum column density in each source/component group
   in units of 1e19 cm^-2, not log N_HI.

3. Column-density uncertainty:
       sigma_W,rms = sigma_T * dv * sqrt(N_line)
       N_line      = FWHM_peak / dv
       sigma_W     = sqrt(sigma_W,rms^2 + (0.10 * W_peak)^2)
       sigma_NHI   = 1.823e18 * sigma_W

   This includes the same 10% systematic term adopted for the integrated flux.

4. Integrated flux:
       F_int = sum(W_HI) * K_to_Jy / N_pix_per_beam   [Jy km/s]

5. Flux uncertainty:
       err_Fint = sqrt(N_rms * rms_bin^2 + (0.10 * F_int)^2)

   Here N_rms is the number of velocity bins in the adopted RMS/noise region.
   For dimensional consistency, rms_bin is computed as the source-integrated
   flux uncertainty per velocity bin in Jy km/s:
       rms_bin = rms(source-integrated spectrum in Jy over noise channels) * dv

6. H I mass at 50 kpc:
       M_HI = 2.356e5 * (0.050)^2 * F_int  [Msun]

   The table reports M_HI / 1e3, with the header written as
       10^3 (d/50 kpc)^2 Msun

7. Mass uncertainty:
       sigma_MHI = M_HI * err_Fint / F_int

Notes
-----
- Mean FWHM is calculated with amplitude weighting, following the user's
  original definition:
      weights = df["Amplitude_K"].values
      mean_fwhm = np.average(df["FWHM_kms"].values, weights=weights)
- Output filenames and tab-separated TXT format are kept the same as before.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

from spectral_cube import SpectralCube
import astropy.units as u
from radio_beam import Beam


# ============================================================
# 1. User settings
# ============================================================
SNR_LIST = [1.5, 2, 2.5, 3]
MERGE_LIST = [0.7, 1]

FITS_FILE = "./data/processed/CRAFTS_cutout_baseline_K.fits"

# Velocity interval used to define the RMS/noise channels.
# The default keeps the original logic: channels OUTSIDE [-310, -195] km/s
# are used as noise channels. If you want to use the interval itself as the
# noise region, set NOISE_MODE = "inside".
RMS_V1 = -310.0
RMS_V2 = -190.0
NOISE_MODE = "outside"     # allowed values: "outside" or "inside"

# Distance normalization for the table: 10^3 (d/50 kpc)^2 Msun
D0_KPC = 50.0
D0_MPC = D0_KPC / 1000.0
MASS_COEFF = 2.356e5

# Systematic terms
FLUX_SYS_FRAC = 0.10
NHI_SYS_FRAC = 0.10

# Telescope / cube setup
BEAM_FWHM_ARCSEC = 240.0
PIXEL_SCALE_DEG = 0.025
REST_FREQ = 1420.40575177 * u.MHz

# If you want to force a fixed FWHM error for Tk, set e.g. 0.2.
# If None, the script uses one velocity channel width.
FWHM_ERR_KMS = None

# H I column density coefficient for optically thin emission
NHI_COEFF = 1.823e18


# ============================================================
# 2. Beam conversion and cube loading
# ============================================================
beam = Beam(
    major=BEAM_FWHM_ARCSEC * u.arcsec,
    minor=BEAM_FWHM_ARCSEC * u.arcsec,
    pa=0.0 * u.deg,
)

K_to_Jy = (1.0 * u.K).to(u.Jy / u.beam, beam.jtok_equiv(REST_FREQ)).value

pixel_scale_arcsec = PIXEL_SCALE_DEG * 3600.0
beam_area_pix_per_beam = (
    np.pi * BEAM_FWHM_ARCSEC**2
    / (4.0 * np.log(2.0))
    / pixel_scale_arcsec**2
)

print(f"K_to_Jy = {K_to_Jy:.6e} Jy/beam/K")
print(f"beam_area = {beam_area_pix_per_beam:.4f} pixel/beam")

print(f"Reading cube: {FITS_FILE}")
try:
    cube = SpectralCube.read(FITS_FILE).with_spectral_unit(
        u.km / u.s,
        velocity_convention="radio",
    )
except Exception:
    cube = SpectralCube.read(FITS_FILE).with_spectral_unit(u.km / u.s)

VEL_AXIS = cube.spectral_axis.to_value(u.km / u.s)
DV_KMS = np.nanmedian(np.abs(np.diff(VEL_AXIS)))

# This cube is modest for the CRAFTS cutout. For very large cubes, use a
# chunked strategy instead of loading everything at once.
cube_data = cube.unmasked_data[:].value

print("Cube shape:", cube_data.shape)
print("Velocity range:", np.nanmin(VEL_AXIS), np.nanmax(VEL_AXIS), "km/s")
print("dv =", DV_KMS, "km/s")


# ============================================================
# 3. Helper functions
# ============================================================
def fmt_num(x):
    """Format numbers for paths: 2.0 -> 2, 0.7 -> 0.7."""
    if float(x).is_integer():
        return str(int(x))
    return str(x)


def get_noise_mask():
    """Return the velocity-channel mask used for RMS estimation."""
    v1, v2 = sorted([RMS_V1, RMS_V2])

    if NOISE_MODE.lower() == "inside":
        mask = (VEL_AXIS >= v1) & (VEL_AXIS <= v2)
    elif NOISE_MODE.lower() == "outside":
        mask = (VEL_AXIS < v1) | (VEL_AXIS > v2)
    else:
        raise ValueError("NOISE_MODE must be 'inside' or 'outside'.")

    if np.count_nonzero(mask) == 0:
        raise ValueError("No velocity channels selected for RMS estimation.")

    return mask


NOISE_CHAN = get_noise_mask()
N_RMS_BINS = int(np.count_nonzero(NOISE_CHAN))
print(f"N_RMS_BINS = {N_RMS_BINS} channels, NOISE_MODE = {NOISE_MODE}")


def standardize_columns(df):
    """Standardize different CSV column names."""
    rename_dict = {}

    for col in df.columns:
        low = col.lower().strip()

        if low == "final_source_id":
            rename_dict[col] = "Source"
        elif low == "component":
            rename_dict[col] = "component"
        elif low in [
            "original_component_source_id",
            "l_component_source_id",
            "component_source_id",
            "original_componnet_source_id",
        ]:
            rename_dict[col] = "original_component_source_id"
        elif low == "x_pixel":
            rename_dict[col] = "x_pixel"
        elif low == "y_pixel":
            rename_dict[col] = "y_pixel"
        elif low == "amplitude_k":
            rename_dict[col] = "Amplitude_K"
        elif low == "velocity_kms":
            rename_dict[col] = "Velocity_kms"
        elif low == "sigma_kms":
            rename_dict[col] = "Sigma_kms"
        elif low == "fwhm_kms":
            rename_dict[col] = "FWHM_kms"
        elif low in ["moment0_k_kms", "moment0_k_kms"]:
            rename_dict[col] = "Moment0_K_kms"

    return df.rename(columns=rename_dict)


def clean_pixel_list(pixel_df):
    """Return unique valid integer pixel coordinates inside the cube."""
    n_v, n_y, n_x = cube_data.shape
    pix = []

    for x, y in pixel_df[["x_pixel", "y_pixel"]].drop_duplicates().values:
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < n_x and 0 <= y < n_y:
            pix.append((x, y))

    return pix


def compute_rms_K(pixel_list):
    """
    Per-pixel brightness-temperature RMS in K, using all selected noise channels
    and all spatial pixels of the source group.
    """
    if len(pixel_list) == 0:
        return np.nan

    values = []
    for x, y in pixel_list:
        spec_noise = cube_data[NOISE_CHAN, y, x]
        spec_noise = spec_noise[np.isfinite(spec_noise)]
        if spec_noise.size > 0:
            values.append(spec_noise)

    if len(values) == 0:
        return np.nan

    values = np.concatenate(values)
    return np.nanstd(values - np.nanmean(values))


def compute_source_integrated_noise(pixel_list):
    """
    Compute source-integrated RMS for the flux uncertainty formula.

    Returns
    -------
    rms_spec_Jy : float
        RMS of the source-integrated spectrum over the noise channels, in Jy.
    rms_bin_Jy_kms : float
        Per-channel integrated-flux RMS, in Jy km/s, equal to rms_spec_Jy * dv.
    """
    if len(pixel_list) == 0:
        return np.nan, np.nan

    xs = np.array([p[0] for p in pixel_list], dtype=int)
    ys = np.array([p[1] for p in pixel_list], dtype=int)

    # Sum over the spatial pixels of the source for every velocity channel.
    # cube_data[:, ys, xs] has shape (nchan, npix).
    source_spec_K_pix = np.nansum(cube_data[:, ys, xs], axis=1)

    # Convert summed K pixels to Jy using pixel/beam normalization.
    source_spec_Jy = source_spec_K_pix * K_to_Jy / beam_area_pix_per_beam

    noise_spec = source_spec_Jy[NOISE_CHAN]
    noise_spec = noise_spec[np.isfinite(noise_spec)]

    if noise_spec.size == 0:
        return np.nan, np.nan

    rms_spec_Jy = np.nanstd(noise_spec - np.nanmean(noise_spec))
    rms_bin_Jy_kms = rms_spec_Jy * DV_KMS

    return rms_spec_Jy, rms_bin_Jy_kms


def weighted_mean_and_error(values, weights):
    """Weighted mean and an approximate standard error of the weighted mean."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(valid) == 0:
        return np.nan, np.nan

    v = values[valid]
    w = weights[valid]

    mean = np.average(v, weights=w)

    if len(v) <= 1:
        return mean, np.nan

    var = np.average((v - mean) ** 2, weights=w)
    # Effective number for weighted samples.
    n_eff = (np.sum(w) ** 2) / np.sum(w**2)
    err = np.sqrt(var) / np.sqrt(max(n_eff, 1.0))

    return mean, err


def parse_source_id(csv_file):
    """Parse source ID from source_XXX_physical_parameters.csv."""
    basename = os.path.basename(csv_file)
    match = re.search(r"source_(\d+)_", basename)
    if match is None:
        raise ValueError(f"Cannot parse source ID from filename: {basename}")
    return int(match.group(1))


# ============================================================
# 4. Core calculation
# ============================================================
def calculate_one_group(df_group, source_id, component, original_source_id):
    """Calculate physical parameters for one component group in one source."""

    required_cols = [
        "x_pixel",
        "y_pixel",
        "Amplitude_K",
        "Velocity_kms",
        "Sigma_kms",
        "FWHM_kms",
    ]

    for col in required_cols:
        if col not in df_group.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df_group.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    df = df[df["Amplitude_K"] > 0]
    df = df[df["Sigma_kms"] > 0]
    df = df[df["FWHM_kms"] > 0]

    if len(df) == 0:
        return None

    # Unique valid spatial pixels
    pixel_list = clean_pixel_list(df[["x_pixel", "y_pixel"]])
    n_pixels = len(pixel_list)
    if n_pixels == 0:
        return None

    # Moment-0 per Gaussian pixel: W_HI [K km/s]
    df["Moment0_K_kms"] = (
        df["Amplitude_K"]
        * df["Sigma_kms"]
        * np.sqrt(2.0 * np.pi)
    )
    total_m0 = df["Moment0_K_kms"].sum()

    # Integrated flux per pixel and total flux [Jy km/s]
    df["flux_pix_Jy_kms"] = (
        df["Moment0_K_kms"]
        * K_to_Jy
        / beam_area_pix_per_beam
    )
    total_flux = df["flux_pix_Jy_kms"].sum()

    # RMS estimates
    rms_K = compute_rms_K(pixel_list)
    rms_spec_Jy, rms_bin_Jy_kms = compute_source_integrated_noise(pixel_list)

    # Flux uncertainty:
    # err_Fint = sqrt(N_RMS_BINS * rms_bin^2 + (0.10 Fint)^2)
    if np.isfinite(rms_bin_Jy_kms) and np.isfinite(total_flux):
        flux_err = np.sqrt(
            400 * (rms_bin_Jy_kms*0.2)**2
            + (FLUX_SYS_FRAC * total_flux) ** 2
        )
    else:
        flux_err = np.nan

    # H I mass at 50 kpc
    mass_msun = total_flux * MASS_COEFF * D0_MPC**2
    mass_1e3_d50 = mass_msun / 1e3

    # Mass error: sigma_M = M * sigma_F / F
    if total_flux > 0 and np.isfinite(flux_err):
        mass_err_msun = mass_msun * flux_err / total_flux
        mass_err_1e3_d50 = mass_err_msun / 1e3
    else:
        mass_err_msun = np.nan
        mass_err_1e3_d50 = np.nan

    # Amplitude-weighted mean velocity and FWHM.
    # This keeps the same definition as your previous script.
    weights = df["Amplitude_K"].values
    mean_velocity, mean_velocity_err = weighted_mean_and_error(
        df["Velocity_kms"].values,
        weights,
    )
    mean_fwhm, mean_fwhm_err_from_scatter = weighted_mean_and_error(
        df["FWHM_kms"].values,
        weights,
    )

    # Kinetic temperature upper limit
    fwhm_err_for_Tk = FWHM_ERR_KMS if FWHM_ERR_KMS is not None else DV_KMS
    Tk = 21.86 * mean_fwhm**2
    Tk_err = 21.86 * 2.0 * mean_fwhm * fwhm_err_for_Tk

    # Peak brightness temperature
    peak_TB = df["Amplitude_K"].max()

    # Column density from Gaussian moment-0
    df["N_HI"] = NHI_COEFF * df["Moment0_K_kms"]

    # Take the maximum column density within this source/component group
    idx_max_nhi = df["N_HI"].idxmax()
    max_NHI = df.loc[idx_max_nhi, "N_HI"]
    W_peak = df.loc[idx_max_nhi, "Moment0_K_kms"]
    fwhm_peak = df.loc[idx_max_nhi, "FWHM_kms"]

    # Number of velocity channels across the peak line
    if np.isfinite(fwhm_peak) and np.isfinite(DV_KMS) and DV_KMS > 0:
        N_line = max(1.0, fwhm_peak / DV_KMS)
    else:
        N_line = np.nan

    # Column density uncertainty:
    # sigma_W = sqrt((rms_K * dv * sqrt(N_line))^2 + (0.10 * W_peak)^2)
    if (
        np.isfinite(rms_K)
        and np.isfinite(W_peak)
        and np.isfinite(N_line)
        and max_NHI > 0
    ):
        sigma_W_rms = rms_K * DV_KMS * np.sqrt(N_line)
        sigma_W_sys = NHI_SYS_FRAC * W_peak
        sigma_W_total = np.sqrt(sigma_W_rms**2 + sigma_W_sys**2)
        NHI_err = NHI_COEFF * sigma_W_total
    else:
        sigma_W_rms = np.nan
        sigma_W_sys = np.nan
        sigma_W_total = np.nan
        NHI_err = np.nan

    # Output in units of 10^19 cm^-2
    max_NHI_1e19 = max_NHI / 1e19
    NHI_err_1e19 = NHI_err / 1e19

    return {
        "Source": source_id,
        "component": component,
        "original_component_source_id": original_source_id,
        "N_pixels": n_pixels,
        "N_rms_bins": N_RMS_BINS,

        "rms_K": rms_K,
        "rms_source_spec_Jy": rms_spec_Jy,
        "rms_bin_Jy_kms": rms_bin_Jy_kms,

        "total_Moment0_K_kms": total_m0,
        "total_flux_Jy_kms": total_flux,
        "flux_error_Jy_kms": flux_err,

        "Mass_Msun_D50": mass_msun,
        "Mass_err_Msun_D50": mass_err_msun,
        "Mass_1e3_D50": mass_1e3_d50,
        "Mass_err_1e3_D50": mass_err_1e3_d50,

        "Mean_velocity_kms": mean_velocity,
        "Mean_velocity_err_kms": mean_velocity_err,
        "Mean_FWHM_kms": mean_fwhm,
        "Mean_FWHM_err_scatter_kms": mean_fwhm_err_from_scatter,
        "FWHM_err_for_Tk_kms": fwhm_err_for_Tk,

        "Tk_K": Tk,
        "Tk_err_K": Tk_err,

        "Peak_TB_K": peak_TB,

        "Max_NHI_cm2": max_NHI,
        "NHI_err_cm2": NHI_err,
        "Max_NHI_1e19_cm2": max_NHI_1e19,
        "NHI_err_1e19_cm2": NHI_err_1e19,
        "NHI_peak_W_K_kms": W_peak,
        "NHI_peak_FWHM_kms": fwhm_peak,
        "N_line_NHI": N_line,
        "sigma_W_rms_K_kms": sigma_W_rms,
        "sigma_W_sys_K_kms": sigma_W_sys,
        "sigma_W_total_K_kms": sigma_W_total,
    }


def calculate_one_csv(csv_file):
    """Process one source_XXX_physical_parameters.csv file."""
    df = pd.read_csv(csv_file)
    df = standardize_columns(df)

    source_id = parse_source_id(csv_file)

    need_cols = [
        "component",
        "original_component_source_id",
        "x_pixel",
        "y_pixel",
        "Amplitude_K",
        "Velocity_kms",
        "Sigma_kms",
        "FWHM_kms",
    ]

    for col in need_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in {csv_file}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=need_cols)

    df["component"] = df["component"].astype(int)
    df["original_component_source_id"] = df["original_component_source_id"].astype(int)

    rows = []
    for (component, original_source_id), df_group in df.groupby(
        ["component", "original_component_source_id"]
    ):
        row = calculate_one_group(
            df_group,
            source_id,
            component,
            original_source_id,
        )
        if row is not None:
            rows.append(row)

    return rows


# ============================================================
# 5. LaTeX output
# ============================================================
def write_latex_table(summary_df, out_latex):
    """Write a compact AASTeX deluxetable* file."""

    lines = []
    lines.append(r"\begin{deluxetable*}{lcccccccc}")
    lines.append(r"\tabletypesize{\small}")
    lines.append(r"\tablewidth{0pt}")
    lines.append(r"\tablecaption{HVC Cloud Parameters \label{tab:cloud_main}}")
    lines.append(r"\tablehead{")
    lines.append(r"\colhead{Source} &")
    lines.append(r"\colhead{$M_{\rm H\,{\sc i}}$} &")
    lines.append(r"\colhead{$V_{\rm LSR}$} &")
    lines.append(r"\colhead{FWHM} &")
    lines.append(r"\colhead{$T_K$} &")
    lines.append(r"\colhead{$F_{\rm int}$} &")
    lines.append(r"\colhead{Peak $T_B$} &")
    lines.append(r"\colhead{rms} &")
    lines.append(r"\colhead{$N_{\rm H\,{\sc i},max}$} \\")
    lines.append(r"\colhead{} &")
    lines.append(r"\colhead{[$10^{3}(d/50\,{\rm kpc})^2M_\odot$]} &")
    lines.append(r"\colhead{(km s$^{-1}$)} &")
    lines.append(r"\colhead{(km s$^{-1}$)} &")
    lines.append(r"\colhead{($10^{3}$ K)} &")
    lines.append(r"\colhead{(Jy km s$^{-1}$)} &")
    lines.append(r"\colhead{(K)} &")
    lines.append(r"\colhead{(K)} &")
    lines.append(r"\colhead{($10^{19}$ cm$^{-2}$)}")
    lines.append(r"}")
    lines.append(r"\startdata")

    previous_source = None

    for _, row in summary_df.iterrows():
        source_id = int(row["Source"])
        source_label = "" if source_id == previous_source else rf"Source\_{source_id}"

        mass_1e3 = row["Mass_1e3_D50"]
        mass_err_1e3 = row["Mass_err_1e3_D50"]

        tk_1e3 = row["Tk_K"] / 1e3
        tk_err_1e3 = row["Tk_err_K"] / 1e3

        line = (
            f"{source_label} & "
            f"${mass_1e3:.1f} \\pm {mass_err_1e3:.1f}$ & "
            f"${row['Mean_velocity_kms']:.1f}$ & "
            f"${row['Mean_FWHM_kms']:.1f}$ & "
            f"${tk_1e3:.1f} \\pm {tk_err_1e3:.1f}$ & "
            f"${row['total_flux_Jy_kms']:.1f} \\pm {row['flux_error_Jy_kms']:.1f}$ & "
            f"${row['Peak_TB_K']:.1f}$ & "
            f"${row['rms_K']:.1f}$ & "
            f"${row['Max_NHI_1e19_cm2']:.1f} \\pm {row['NHI_err_1e19_cm2']:.1f}$ \\\\"
        )

        lines.append(line)
        previous_source = source_id

    lines.append(r"\enddata")
    lines.append(r"\tablecomments{")
    lines.append(
        r"Rows are calculated separately for each unique "
        r"$(\mathrm{component},\mathrm{original\ component\ source})$ group "
        r"within a final merged source. "
        r"The H\,{\sc i} mass is calculated as "
        r"$M_{\rm H\,{\sc i}} = 2.356\times10^{5}D^2F_{\rm int}\,M_\odot$ "
        r"and is reported in units of $10^3(d/50\,{\rm kpc})^2M_\odot$. "
        r"The adopted flux uncertainty is "
        r"$\sigma_F=\sqrt{N_{\rm rms}\sigma_{\rm rms}^2+(0.10F_{\rm int})^2}$, "
        r"where $N_{\rm rms}$ is the number of velocity bins in the adopted RMS region "
        r"and $\sigma_{\rm rms}$ is the source-integrated per-bin flux RMS in Jy km s$^{-1}$. "
        r"The mass uncertainty is $\sigma_M=M_{\rm H\,{\sc i}}\sigma_F/F_{\rm int}$. "
        r"The listed H\,{\sc i} column density is the maximum value within each source/component group "
        r"and is reported in units of $10^{19}\,{\rm cm}^{-2}$. "
        r"It is calculated from the Gaussian moment-0 as "
        r"$N_{\rm H\,{\sc i}}=1.823\times10^{18}A\sigma_v\sqrt{2\pi}$. "
        r"Its uncertainty is computed from "
        r"$\sigma_N=1.823\times10^{18}\sigma_W$, with "
        r"$\sigma_W=\sqrt{(\sigma_T\Delta v\sqrt{N_{\rm line}})^2+(0.10W_{\rm H\,{\sc i}})^2}$ "
        r"and $N_{\rm line}\simeq{\rm FWHM}/\Delta v$. "
        r"The kinetic temperature is estimated from $T_K=21.86\,{\rm FWHM}^2$."
    )
    lines.append(r"}")
    lines.append(r"\end{deluxetable*}")

    with open(out_latex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. Main loop
# ============================================================
def main():
    all_rows = []

    for snr in SNR_LIST:
        for merge_factor in MERGE_LIST:
            snr_str = fmt_num(snr)
            merge_str = fmt_num(merge_factor)

            output_dir = (
                f"./baseline/SNR={snr_str}/"
                f"output_individual_source_10/"
                f"output_merged_source_{merge_str}"
            )

            phys_csv_dir = os.path.join(output_dir, "source_physical_parameters")

            out_txt = os.path.join(
                output_dir,
                "final_source_physical_summary_by_component.txt",
            )
            out_latex = os.path.join(
                output_dir,
                "final_source_physical_summary_by_component_latex.txt",
            )

            print("\n" + "=" * 80)
            print(f"Processing SNR={snr_str}, merge={merge_str}")
            print(output_dir)
            print("=" * 80)

            if not os.path.exists(phys_csv_dir):
                print(f"Skip: path not found: {phys_csv_dir}")
                continue

            csv_files = sorted(
                glob.glob(os.path.join(phys_csv_dir, "source_*_physical_parameters.csv"))
            )

            if len(csv_files) == 0:
                print(f"Skip: no csv files found in {phys_csv_dir}")
                continue

            summary_rows = []
            for csv_file in csv_files:
                rows = calculate_one_csv(csv_file)
                for row in rows:
                    row["SNR"] = snr
                    row["merge_factor"] = merge_factor
                summary_rows.extend(rows)
                all_rows.extend(rows)

            if len(summary_rows) == 0:
                print("No valid rows for this parameter set.")
                continue

            summary_df = pd.DataFrame(summary_rows)
            summary_df = summary_df.sort_values(
                ["Source", "component", "original_component_source_id"]
            ).reset_index(drop=True)

            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("# Final merged source physical summary by component\n")
                f.write(f"# SNR = {snr_str}\n")
                f.write(f"# merge_factor = {merge_str}\n")
                f.write(f"# K_to_Jy = {K_to_Jy:.8e} Jy/beam/K\n")
                f.write(f"# beam_area_pix_per_beam = {beam_area_pix_per_beam:.8f}\n")
                f.write(f"# dv_kms = {DV_KMS:.8f}\n")
                f.write(f"# RMS_V1 = {RMS_V1:.3f} km/s\n")
                f.write(f"# RMS_V2 = {RMS_V2:.3f} km/s\n")
                f.write(f"# NOISE_MODE = {NOISE_MODE}\n")
                f.write(f"# N_RMS_BINS = {N_RMS_BINS}\n")
                f.write("# Moment0 = Amplitude_K * Sigma_kms * sqrt(2*pi)\n")
                f.write("# Flux = Moment0 * K_to_Jy / beam_area_pix_per_beam\n")
                f.write("# Flux error = sqrt(N_RMS_BINS * rms_bin_Jy_kms^2 + (0.10*Flux)^2)\n")
                f.write("# Mass_Msun_D50 = Flux * 2.356e5 * (0.050)^2\n")
                f.write("# Mass error = Mass * Flux_error / Flux\n")
                f.write("# NHI = 1.823e18 * Moment0\n")
                f.write("# NHI error includes rms-based moment-0 error and 10% systematic error\n")
                f.write("\n")

                summary_df.to_csv(
                    f,
                    sep="\t",
                    index=False,
                    float_format="%.6e",
                )

            write_latex_table(summary_df, out_latex)

            print(f"Saved TXT:   {out_txt}")
            print(f"Saved LaTeX: {out_latex}")

    if len(all_rows) > 0:
        all_df = pd.DataFrame(all_rows)
        all_df = all_df.sort_values(
            ["SNR", "merge_factor", "Source", "component", "original_component_source_id"]
        ).reset_index(drop=True)

        all_out = "./baseline/final_source_physical_summary_all_SNR_all_merge.txt"
        all_df.to_csv(
            all_out,
            sep="\t",
            index=False,
            float_format="%.6e",
        )

        print("\n" + "=" * 80)
        print("All results saved to:")
        print(all_out)
        print("=" * 80)
    else:
        print("No valid results found.")


if __name__ == "__main__":
    main()

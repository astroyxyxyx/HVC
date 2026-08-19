import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.io import fits
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator
# =========================================================
# Global font: Times New Roman
# =========================================================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

# =========================================================
# Use a shared figure size so every PDF frame is identical.
# =========================================================
FIG_W = 13
FIG_H = 9

# =========================================================
# Global font size, increased by 4 points from the original settings.
# =========================================================
FONTSIZE_TITLE = 20
FONTSIZE_SUPTITLE = 20
FONTSIZE_AXIS = 18
FONTSIZE_TOPAXIS = 18
FONTSIZE_TICK = 16
FONTSIZE_CBAR = 18
FONTSIZE_CBAR_TICK = 16

# =========================================================
# 0. Parameter settings
# =========================================================
fitsname = "./data/processed/CRAFTS_cutout_baseline_K.fits"

SNR_LIST = ["2"]
SOURCE_IDS = list(range(1, 18))   # Process source_001 through source_017 in the original numbering order.  # Skip source_006; source_007 becomes source_006, and so on.

BASE_INPUT = "./baseline"
BASE_OUT = "./baseline/source_summary_figures_2x2_fixedsize"

ZOOM_PAD_PIX = 5   # Number of pixels to pad around each source in the zoom-in view.

os.makedirs(BASE_OUT, exist_ok=True)

# =========================================================
# 1. Helper function: read one source CSV file.
# =========================================================
def load_source_csv(csv_file):
    if not os.path.exists(csv_file):
        return None

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[Skip] Failed to read: {csv_file}")
        print(f"       Reason: {e}")
        return None

    required_cols = [
        "x_pixel",
        "y_pixel",
        "amplitude_K",
        "velocity_kms",
        "sigma_kms"
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"[Skip] Missing column '{col}' in {csv_file}")
            return None

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    df = df[(df["amplitude_K"] > 0) & (df["sigma_kms"] > 0)].copy()

    if len(df) == 0:
        print(f"[Skip] Empty valid dataframe: {csv_file}")
        return None

    df["x_pixel"] = df["x_pixel"].astype(int)
    df["y_pixel"] = df["y_pixel"].astype(int)

    return df


# =========================================================
# 2. Coordinate and WCS helper functions
# =========================================================
def unwrap_ra_deg(ra_deg):
    ra_rad = np.deg2rad(ra_deg)
    ra_unwrapped = np.rad2deg(np.unwrap(ra_rad))
    return ra_unwrapped


def centers_to_edges(arr):
    arr = np.asarray(arr)
    d = np.diff(arr)

    edges = np.empty(len(arr) + 1, dtype=float)
    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = arr[0] - 0.5 * d[0]
    edges[-1] = arr[-1] + 0.5 * d[-1]

    return edges


def ra_hour_formatter_from_deg(x, pos):
    """Bottom RA axis: keep hour-minute style, e.g. 23^h43^m."""
    ra_hour = (x / 15.0) % 24.0

    h = int(np.floor(ra_hour + 1e-10))
    m = int(np.round((ra_hour - h) * 60.0))

    if m == 60:
        h = (h + 1) % 24
        m = 0

    return rf"${h}^{{\mathrm{{h}}}}{m:02d}^{{\mathrm{{m}}}}$"


def ra_deg_formatter(x, pos):
    """Top RA axis: 365° format with two decimals."""
    # Convert RA in degrees to the 0-360 degree range.
    # ra_deg = x % 360.0
    return rf"{x:.2f}°"


def dec_deg_formatter(y, pos):
    """Dec axis: decimal degrees, with two decimals."""
    return rf"{y:.2f}°"


def colorbar_no_decimal_formatter(x, pos):
    """Colorbar tick labels: no decimal places."""
    return rf"{x:.0f}"


def identity_forward(x):
    return x


def identity_inverse(x):
    return x


def apply_sky_axes(ax, extent, show_ylabel=True):
    ax.set_xlabel("Right Ascension (J2000)", fontsize=FONTSIZE_AXIS)
    ax.xaxis.set_major_formatter(FuncFormatter(ra_hour_formatter_from_deg))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    if show_ylabel:
        ax.set_ylabel("Declination (J2000)", fontsize=FONTSIZE_AXIS)
    else:
        ax.set_ylabel("")

    ax.yaxis.set_major_formatter(FuncFormatter(dec_deg_formatter))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICK)

    top_ax = ax.secondary_xaxis(
        "top",
        functions=(identity_forward, identity_inverse)
    )
    # top_ax.set_xlabel("Right Ascension (J2000)", fontsize=FONTSIZE_TOPAXIS, labelpad=8)
    top_ax.tick_params(axis="x", which="major", labelsize=FONTSIZE_TICK)
    top_ax.xaxis.set_major_formatter(FuncFormatter(ra_deg_formatter))
    top_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # RA decreases from left to right.
    ax.set_xlim(max(extent[0], extent[1]), min(extent[0], extent[1]))


def _clean_floor_int(x, tol=1e-6):
    """
    Avoid flooring 9.999999 to 9.
    Treat values very close to an integer as that integer.
    """
    r = np.round(x)
    if np.isclose(x, r, atol=tol, rtol=0):
        return int(r)
    return int(np.floor(x))


def _clean_ceil_int(x, tol=1e-6):
    """
    Avoid ceiling 10.000001 to 11.
    Treat values very close to an integer as that integer.
    """
    r = np.round(x)
    if np.isclose(x, r, atol=tol, rtol=0):
        return int(r)
    return int(np.ceil(x))


def _get_integer_ticks_from_image(im, small_range_limit=5):
    """
    Inspect the colorbar range automatically from the imshow data.

    If the data range is <= small_range_limit:
        Use fixed integer ticks such as 8, 9, 10.
    If the data range is > small_range_limit:
        Return None so the caller continues with MaxNLocator.
    """
    arr = np.ma.asarray(im.get_array()).filled(np.nan)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return None, None, None

    data_min = np.nanmin(finite)
    data_max = np.nanmax(finite)
    data_range = data_max - data_min

    if not np.isfinite(data_range):
        return None, None, None

    if data_range <= small_range_limit:
        # If the data are nearly constant, for example all close to 10.
        if np.isclose(data_min, data_max, atol=1e-8, rtol=0):
            center = int(np.round(data_min))
            vmin = center - 1
            vmax = center + 1
        else:
            vmin = _clean_floor_int(data_min)
            vmax = _clean_ceil_int(data_max)

            # Avoid vmin == vmax, which prevents the colorbar from rendering correctly.
            if vmin >= vmax:
                center = int(np.round(0.5 * (data_min + data_max)))
                vmin = center - 1
                vmax = center + 1

        ticks = np.arange(vmin, vmax + 1, 1)
        return vmin, vmax, ticks

    return None, None, None


def add_cbar_below(fig, ax, im, label, nbins=5):
    divider = make_axes_locatable(ax)

    # pad controls the distance between the colorbar and the main x-axis labels.
    # Larger values place the colorbar farther from the axis.
    cax = divider.append_axes("bottom", size="6%", pad=0.7)

    # =====================================================
    # Check the data range first.
    # Use fixed integer ticks when the range is <= 5.
    # =====================================================
    fixed_vmin, fixed_vmax, fixed_ticks = _get_integer_ticks_from_image(
        im,
        small_range_limit=5
    )

    if fixed_ticks is not None:
        im.set_clim(fixed_vmin, fixed_vmax)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=FONTSIZE_CBAR, labelpad=8)
    cbar.ax.tick_params(labelsize=FONTSIZE_CBAR_TICK)

    formatter = FuncFormatter(colorbar_no_decimal_formatter)

    if fixed_ticks is not None:
        # Small range: use fixed integer ticks such as 8, 9, 10.
        locator = FixedLocator(fixed_ticks)
    else:
        # Large range: choose ticks automatically while preferring integers.
        locator = MaxNLocator(nbins=nbins, integer=True)

    cbar.locator = locator
    cbar.formatter = formatter
    cbar.update_ticks()

    cbar.ax.xaxis.set_major_locator(locator)
    cbar.ax.xaxis.set_major_formatter(formatter)

    return cbar


# =========================================================
# 3. Compute Moment 0, Moment 1, and FWHM for the current source at full size.
# =========================================================
def compute_source_maps_full(df, nx_full, ny_full):
    df = df.copy()

    df["moment0_component"] = (
        df["amplitude_K"] * df["sigma_kms"] * np.sqrt(2.0 * np.pi)
    )

    moment0_map = np.full((ny_full, nx_full), np.nan, dtype=float)
    moment1_map = np.full((ny_full, nx_full), np.nan, dtype=float)
    fwhm_map    = np.full((ny_full, nx_full), np.nan, dtype=float)

    for (y, x), g in df.groupby(["y_pixel", "x_pixel"]):
        if not (0 <= x < nx_full and 0 <= y < ny_full):
            continue

        w = g["moment0_component"].values
        v = g["velocity_kms"].values
        sig = g["sigma_kms"].values

        m0 = np.nansum(w)
        if not np.isfinite(m0) or m0 <= 0:
            continue

        moment0 = m0
        moment1 = np.nansum(w * v) / m0

        m2_sq = np.nansum(w * (sig**2 + (v - moment1)**2)) / m0
        if not np.isfinite(m2_sq) or m2_sq < 0:
            continue

        dispersion = np.sqrt(m2_sq)
        fwhm = 2.35482 * dispersion

        moment0_map[y, x] = moment0
        moment1_map[y, x] = moment1
        fwhm_map[y, x] = fwhm

    return {
        "moment0": moment0_map,
        "moment1": moment1_map,
        "fwhm": fwhm_map
    }


# =========================================================
# 4. Compute the zoom region.
# =========================================================
def _expand_1d_window_around_center(center, target_width, nmax):
    """
    Expand the window to target_width around center.
    Return the integer slice [i0, i1), clipping safely at the boundaries.
    """
    target_width = int(np.ceil(target_width))
    target_width = max(1, min(target_width, nmax))

    i0 = int(np.floor(center - target_width / 2.0))
    i1 = i0 + target_width

    if i0 < 0:
        i1 -= i0
        i0 = 0

    if i1 > nmax:
        shift = i1 - nmax
        i0 -= shift
        i1 = nmax

    i0 = max(i0, 0)
    i1 = min(i1, nmax)

    return i0, i1


def get_zoom_window_match_fullfield_aspect(
    df,
    nx_full,
    ny_full,
    x_edges,
    y_edges,
    full_extent,
    pad=5,
    pad_y=None,
    min_pad_x=None,
    verbose=False,
    source_tag=""
):
    """
    Automatically determine how much to widen the x direction for each source zoom-in region.
    Match the data aspect ratio of the three zoomed moment maps to the upper-left full-field panel.

    Keep the y-direction padding fixed and expand only in x.
    This keeps the Moment 0, Moment 1, and FWHM zoom aspect ratios consistent with the first panel.
    """
    if pad_y is None:
        pad_y = pad
    if min_pad_x is None:
        min_pad_x = pad

    pix = df[["x_pixel", "y_pixel"]].drop_duplicates()
    xs = pix["x_pixel"].values.astype(int)
    ys = pix["y_pixel"].values.astype(int)

    # Determine the y range using the normal padding first.
    y0 = max(ys.min() - pad_y, 0)
    y1 = min(ys.max() + pad_y + 1, ny_full)

    # Keep at least the source x range after the default padding.
    x0_base = max(xs.min() - min_pad_x, 0)
    x1_base = min(xs.max() + min_pad_x + 1, nx_full)
    base_width_pix = x1_base - x0_base

    # Full-field data aspect ratio: width / height.
    full_width_deg = abs(full_extent[1] - full_extent[0])
    full_height_deg = abs(full_extent[3] - full_extent[2])
    target_ratio = full_width_deg / full_height_deg

    # Angular height corresponding to the current y range.
    zoom_height_deg = abs(y_edges[y1] - y_edges[y0])

    # Angular scale per x pixel, used to convert target angular width to pixels.
    dx_deg = np.nanmedian(np.abs(np.diff(x_edges)))
    target_width_deg = target_ratio * zoom_height_deg
    target_width_pix = int(np.ceil(target_width_deg / dx_deg))

    # Only expand x; do not shrink the existing x range.
    final_width_pix = max(base_width_pix, target_width_pix)

    # Center on the source x range itself rather than on the padded window center.
    x_center = 0.5 * (xs.min() + xs.max() + 1)
    x0, x1 = _expand_1d_window_around_center(
        center=x_center,
        target_width=final_width_pix,
        nmax=nx_full
    )

    actual_width_deg = abs(x_edges[x1] - x_edges[x0])
    actual_height_deg = abs(y_edges[y1] - y_edges[y0])
    actual_ratio = actual_width_deg / actual_height_deg

    extra_x_pix = max(0, final_width_pix - base_width_pix)

    if verbose:
        print(
            f"[Zoom aspect] {source_tag}: "
            f"target width/height={target_ratio:.3f}, "
            f"actual={actual_ratio:.3f}, "
            f"base_x_width={base_width_pix} pix, "
            f"final_x_width={x1 - x0} pix, "
            f"extra_x={extra_x_pix} pix"
        )

    return x0, x1, y0, y1


def get_crop_extent(x_edges, y_edges, x0, x1, y0, y1):
    return [x_edges[x0], x_edges[x1], y_edges[y0], y_edges[y1]]


# =========================================================
# 5. Plotting function: one 2x2 four-panel figure per source.
# =========================================================
def plot_source_summary_figure_2x2(
    source_tag,
    source_maps,
    full_extent,
    zoom_extent,
    x0, x1, y0, y1,
    output_pdf
):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
    fig.set_size_inches(FIG_W, FIG_H, forward=True)

    ax_full = axes[0, 0]
    ax_m0   = axes[0, 1]
    ax_m1   = axes[1, 0]
    ax_fwhm = axes[1, 1]
    special_cbar_bins = {
        "source_002": 3,
        "source_003": 2,
    }
    
    cbar_nbins = special_cbar_bins.get(source_tag, 5)
    # =====================================================
    # A. Full-field Moment 0 of current source only
    # =====================================================
    full_source_moment0 = source_maps["moment0"]

    finite_data = full_source_moment0[np.isfinite(full_source_moment0)]
    if finite_data.size > 0:
        vmin = np.nanmin(finite_data)
        vmax = np.nanmax(finite_data)
        if np.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6
    else:
        vmin, vmax = None, None

    cmap_full = plt.get_cmap("viridis").copy()
    cmap_full.set_bad("white")

    im_full = ax_full.imshow(
        full_source_moment0,
        origin="lower",
        extent=full_extent,
        cmap=cmap_full,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    add_cbar_below(
        fig,
        ax_full,
        im_full,
        r"Integrated Intensity [K km s$^{-1}$]",
        nbins=cbar_nbins
    )

    # Red dashed box marking the zoom-in region.
    zx0, zx1, zy0, zy1 = zoom_extent

    rect_x = min(zx0, zx1)
    rect_y = min(zy0, zy1)
    rect_w = abs(zx1 - zx0)
    rect_h = abs(zy1 - zy0)

    rect = Rectangle(
        (rect_x, rect_y),
        rect_w,
        rect_h,
        fill=False,
        edgecolor="red",
        linewidth=1.8,
        linestyle="--"
    )
    ax_full.add_patch(rect)

    # =====================================================
    # B. Zoom-in maps
    # =====================================================
    plot_info = [
        (
            ax_m0,
            "moment0",
            "Moment 0 (zoom-in)",
            r"Integrated intensity [K km s$^{-1}$]",
            "viridis",
            False
        ),
        (
            ax_m1,
            "moment1",
            "Moment 1 (zoom-in)",
            r"Velocity [km s$^{-1}$]",
            "RdBu_r",
            True
        ),
        (
            ax_fwhm,
            "fwhm",
            "FWHM (zoom-in)",
            r"FWHM [km s$^{-1}$]",
            "YlOrBr",
            False
        ),
    ]

    for ax, key, title, cbar_label, cmap_name, show_ylabel in plot_info:
        cropped = source_maps[key][y0:y1, x0:x1]

        finite_data = cropped[np.isfinite(cropped)]
        if finite_data.size == 0:
            vmin, vmax = None, None
        else:
            vmin = np.nanmin(finite_data)
            vmax = np.nanmax(finite_data)
            if np.isclose(vmin, vmax):
                vmin -= 1e-6
                vmax += 1e-6

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")

        im = ax.imshow(
            cropped,
            origin="lower",
            extent=zoom_extent,
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        add_cbar_below(fig, ax, im, cbar_label, nbins=cbar_nbins)

    # =====================================================
    # C. Axis formatting
    #    No longer use the earlier apply_sky_axes().
    #    Set the axes uniformly near the end of the plotting function, as in plot_all_sources_wcs().
    # =====================================================

    def _get_visible_ticks(axis_obj, which="x", reverse=False):
        """
        Get major ticks within the current visible axis range.
        Use reverse=True for the RA axis because RA usually decreases from left to right.
        """
        if which == "x":
            ticks = axis_obj.get_xticks()
            lim0, lim1 = axis_obj.get_xlim()
        else:
            ticks = axis_obj.get_yticks()
            lim0, lim1 = axis_obj.get_ylim()

        lo = min(lim0, lim1)
        hi = max(lim0, lim1)

        ticks = [t for t in ticks if lo - 1e-8 <= t <= hi + 1e-8]
        ticks = sorted(ticks, reverse=reverse)

        return ticks


    def _ra_deg_to_hour_minute(ra_deg):
        """
        RA degree -> hour/minute.
        Example: 356 deg -> 23h44m.
        """
        ra_hour = (ra_deg / 15.0) % 24.0

        h = int(np.floor(ra_hour + 1e-10))
        m = int(np.round((ra_hour - h) * 60.0))

        if m == 60:
            h = (h + 1) % 24
            m = 0

        return h, m


    def _deg_to_degree_minute(value_deg):
        """
        degree decimal -> degree/minute.
        Example: 356.50 deg -> 356 deg 30 arcmin.
        Example: -5.50 deg -> -5 deg 30 arcmin.
        """
        sign = "-" if value_deg < 0 else ""
        value_abs = abs(value_deg)

        deg = int(np.floor(value_abs + 1e-10))
        minute = int(np.round((value_abs - deg) * 60.0))

        if minute == 60:
            deg += 1
            minute = 0

        return sign, deg, minute


    def make_ra_hour_formatter(axis_obj):
        """
        Bottom RA axis:
        Show the hour label only once, then write later ticks as minutes such as 44m, 43m, 42m.
        """
        def formatter(x, pos):
            ticks = _get_visible_ticks(
                axis_obj,
                which="x",
                reverse=axis_obj.xaxis_inverted()
            )

            if len(ticks) == 0:
                h, m = _ra_deg_to_hour_minute(x)
                return rf"${h}^{{\mathrm{{h}}}}{m:02d}^{{\mathrm{{m}}}}$"

            labels = []
            prev_h = None

            for t in ticks:
                h, m = _ra_deg_to_hour_minute(t)

                if h != prev_h:
                    label = rf"${h}^{{\mathrm{{h}}}}{m:02d}^{{\mathrm{{m}}}}$"
                else:
                    label = rf"${m:02d}^{{\mathrm{{m}}}}$"

                labels.append(label)
                prev_h = h

            idx = int(np.argmin(np.abs(np.asarray(ticks) - x)))
            return labels[idx]

        return formatter


    def make_deg_min_formatter(axis_obj, which="x"):
        """
        RA degree / Dec degree axis:
        Show each degree label only once, such as 356 deg or -5 deg.
        Write later ticks using only arcminutes such as 30 arcmin, 45 arcmin.
        """
        def formatter(value, pos):
            reverse = False

            if which == "x":
                reverse = axis_obj.xaxis_inverted()

            ticks = _get_visible_ticks(
                axis_obj,
                which=which,
                reverse=reverse
            )

            if len(ticks) == 0:
                sign, deg, minute = _deg_to_degree_minute(value)
                return rf"{sign}{deg}°{minute:02d}′"

            labels = []
            prev_key = None

            for t in ticks:
                sign, deg, minute = _deg_to_degree_minute(t)
                key = (sign, deg)

                if key != prev_key:
                    label = rf"{sign}{deg}°{minute:02d}′"
                else:
                    label = rf"{minute:02d}′"

                labels.append(label)
                prev_key = key

            idx = int(np.argmin(np.abs(np.asarray(ticks) - value)))
            return labels[idx]

        return formatter


    axes_info = [
            # ax,      extent,       show_dec_axis, show_dec_label
            (ax_full, full_extent,  True, True),
            (ax_m0,   zoom_extent,  True, False),
            (ax_m1,   zoom_extent,  True, True),
            (ax_fwhm, zoom_extent,  True, False),
        ]

    for ax, extent, show_dec_axis, show_dec_label in axes_info:

        # RA decreases from left to right.
        ax.set_xlim(max(extent[0], extent[1]), min(extent[0], extent[1]))

        # y increases normally from bottom to top.
        ax.set_ylim(min(extent[2], extent[3]), max(extent[2], extent[3]))

        # -------------------------
        # Bottom RA axis: hour format
        # Keep only one 23h label; later ticks show only 44m, 43m, and so on.
        # -------------------------
        ax.set_xlabel("Right Ascension (J2000)", fontsize=FONTSIZE_AXIS)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(
            FuncFormatter(make_ra_hour_formatter(ax))
        )

        ax.tick_params(
            axis="x",
            which="major",
            labelsize=FONTSIZE_TICK,
            direction="in"
        )

        # -------------------------
        # Left Dec axis: degree-minute format
        # Example: for -5 deg 30 arcmin, later ticks in the same degree show only 45 arcmin.
        # -------------------------
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(
            FuncFormatter(make_deg_min_formatter(ax, which="y"))
        )

        if show_dec_axis:
            if show_dec_label:
                ax.set_ylabel("Declination (J2000)", fontsize=FONTSIZE_AXIS)
            else:
                ax.set_ylabel("")
        
            ax.tick_params(
                axis="y",
                which="major",
                labelsize=FONTSIZE_TICK,
                direction="in",
                labelleft=True,
                left=True
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(
                axis="y",
                which="major",
                labelleft=False,
        left=False
            )

        # -------------------------
        # Top RA axis: degree-minute format
        # Example: for 356 deg 30 arcmin, later ticks in the same degree show only 45 arcmin.
        # -------------------------
        top_ax = ax.secondary_xaxis(
            "top",
            functions=(lambda x: x, lambda x: x)
        )

        # top_ax.set_xlabel("Right Ascension (J2000)", fontsize=FONTSIZE_TOPAXIS, labelpad=8)

        top_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        top_ax.xaxis.set_major_formatter(
            FuncFormatter(make_deg_min_formatter(top_ax, which="x"))
        )

        top_ax.tick_params(
            axis="x",
            which="major",
            labelsize=FONTSIZE_TICK,
            direction="in"
        )

    # -------------------------
    # Figure title: Source_1, Source_2, ...
    # -------------------------
    try:
        source_id = int(source_tag.split("_")[-1])
        source_title = f"Source_{source_id}"
    except Exception:
        source_title = source_tag
    
    fig.suptitle(
        source_title,
        fontsize=FONTSIZE_SUPTITLE,
        y=0.98,
        x=0.52
    )

    # Use fixed margins so every figure keeps the same layout.
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.075,
        top=0.90,
        wspace=0.24,
        hspace=0.36
    )

    # Note: do not use bbox_inches='tight', otherwise each figure will have a different physical size.
    fig.savefig(output_pdf, dpi=300)
    plt.close(fig)


# =========================================================
# 6. Read the original FITS file only once.
# =========================================================
with fits.open(fitsname) as hdul:
    cube_data = hdul[0].data
    header = hdul[0].header

if cube_data.ndim != 3:
    raise ValueError(f"Expected a 3D FITS cube, but got ndim={cube_data.ndim}")

nv, ny_full, nx_full = cube_data.shape
print(f"Full canvas size from FITS: nx = {nx_full}, ny = {ny_full}")

wcs3d = WCS(header)
wcs2d = wcs3d.celestial

# =========================================================
# 7. Generate the full spatial coordinate grids.
# =========================================================
yy, xx = np.mgrid[0:ny_full, 0:nx_full]

ra_deg, dec_deg = wcs2d.pixel_to_world_values(xx, yy)
ra_deg_unwrapped = unwrap_ra_deg(ra_deg)

x_ra_deg_1d = np.nanmedian(ra_deg_unwrapped, axis=0)
y_dec_deg_1d = np.nanmedian(dec_deg, axis=1)

x_edges = centers_to_edges(x_ra_deg_1d)
y_edges = centers_to_edges(y_dec_deg_1d)

full_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

# =========================================================
# 8. Main loop: output one 2x2 figure for each source.
# =========================================================
for snr_str in SNR_LIST:
    out_dir = os.path.join(BASE_OUT, f"SNR={snr_str}")
    os.makedirs(out_dir, exist_ok=True)

    for sid in SOURCE_IDS:
        # ID remapping.
        # Do not renumber; use the original source ID directly.
        source_tag = f"source_{sid:03d}"
        
        input_csv = (
            f"{BASE_INPUT}/SNR={snr_str}/output_individual_source_10/"
            f"output_merged_source_0.7/source_physical_parameters/"
            f"source_{sid:03d}_physical_parameters.csv"
        )

        df = load_source_csv(input_csv)
        if df is None:
            continue

        df = df[
            (df["x_pixel"] >= 0) & (df["x_pixel"] < nx_full) &
            (df["y_pixel"] >= 0) & (df["y_pixel"] < ny_full)
        ].copy()

        if len(df) == 0:
            print(f"[Skip] {source_tag}: no valid pixels inside full canvas")
            continue

        source_maps = compute_source_maps_full(df, nx_full, ny_full)

        # Widen the x direction as needed so all zoomed moment maps share
        # the full-field panel's aspect ratio.
        x0, x1, y0, y1 = get_zoom_window_match_fullfield_aspect(
            df=df,
            nx_full=nx_full,
            ny_full=ny_full,
            x_edges=x_edges,
            y_edges=y_edges,
            full_extent=full_extent,
            pad=ZOOM_PAD_PIX,
            pad_y=ZOOM_PAD_PIX,
            min_pad_x=ZOOM_PAD_PIX,
            verbose=True,
            source_tag=source_tag
        )

        zoom_extent = get_crop_extent(
            x_edges,
            y_edges,
            x0,
            x1,
            y0,
            y1
        )

        output_pdf = os.path.join(
            out_dir,
            f"{source_tag}_summary_2x2_fullfield_zoom.pdf"
        )

        plot_source_summary_figure_2x2(
            source_tag=source_tag,
            source_maps=source_maps,
            full_extent=full_extent,
            zoom_extent=zoom_extent,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            output_pdf=output_pdf
        )

        print(f"[Done] {source_tag} -> {output_pdf}")

print("\nAll available sources processed.")

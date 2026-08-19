# merge_utils.py

import os
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from scipy import ndimage
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from collections import defaultdict

from astropy import units as u
from spectral_cube import SpectralCube


# ============================================================
# 0. Set the Times New Roman font.
# ============================================================
def set_times_new_roman_font():
    """
    Set Times New Roman as the global matplotlib font.

    If Times New Roman is not available, fall back to Liberation Serif / DejaVu Serif.
    This function also prints the actual font used by matplotlib.
    """

    font_loaded = False
    loaded_font_name = None

    possible_paths = [
        # Windows
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",

        # Linux / WSL common paths
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/TTF/Times_New_Roman.ttf",
        "/usr/local/share/fonts/Times_New_Roman.ttf",
    ]

    # --------------------------------------------------------
    # 1. Search explicit file paths first
    # --------------------------------------------------------
    for path in possible_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            loaded_font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = loaded_font_name
            mpl.rcParams["font.family"] = loaded_font_name
            font_loaded = True
            print(f"Loaded font file: {path}")
            print(f"Matplotlib font name: {loaded_font_name}")
            break

    # --------------------------------------------------------
    # 2. Try fc-match
    # --------------------------------------------------------
    if not font_loaded:
        try:
            font_path = subprocess.check_output(
                ["fc-match", "-f", "%{file}", "Times New Roman"],
                text=True
            ).strip()

            if font_path and os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                loaded_font_name = fm.FontProperties(fname=font_path).get_name()

                plt.rcParams["font.family"] = loaded_font_name
                mpl.rcParams["font.family"] = loaded_font_name

                font_loaded = True
                print(f"fc-match returned font file: {font_path}")
                print(f"Matplotlib font name: {loaded_font_name}")

        except Exception as e:
            print(f"fc-match failed or Times New Roman not found: {e}")

    # --------------------------------------------------------
    # 3. Fallback
    # --------------------------------------------------------
    if not font_loaded:
        print("Times New Roman not found. Falling back to serif fonts.")
        plt.rcParams["font.family"] = "serif"
        mpl.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
            "Times",
        ]
        mpl.rcParams["font.serif"] = [
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
            "Times",
        ]

    # Math font
    plt.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["mathtext.fontset"] = "stix"

    # PDF font embedding
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # Make sure minus signs display correctly
    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.unicode_minus"] = False

    # Print actual resolved font
    try:
        resolved = fm.findfont(
            fm.FontProperties(family=plt.rcParams["font.family"]),
            fallback_to_default=True,
        )
        print(f"Resolved matplotlib font file: {resolved}")
    except Exception as e:
        print(f"Could not resolve matplotlib font: {e}")


# ============================================================
# 1. Read the spatial WCS from FITS.
# ============================================================
def get_coordinate_info(fits_file):
    try:
        cube = SpectralCube.read(fits_file)
        print(f"SpectralCube shape: {cube.shape}")

        wcs_2d = cube.wcs.celestial
        n_y, n_x = cube.shape[-2:]

        print(f"Spatial shape from FITS: X={n_x}, Y={n_y}")

        return wcs_2d, n_x, n_y

    except Exception as e:
        print(f"Warning: Could not read coordinate info from FITS: {e}")
        return None, None, None


# ============================================================
# 2. Read the DAT file.
# ============================================================
def read_filtered_dat(core, dat_file):
    """
    Read the filtered ROHSA DAT file.

    Important:
    The FITS velocity-axis values are already in km/s.
    Therefore this code no longer relies on core.physical_gaussian() for velocity-unit conversion.
    Instead, it converts pixels to km/s manually using CRVAL3, CRPIX3, and CDELT3 from the header.

    position_phys:
        v = CRVAL3 + (pos_pixel + 1 - CRPIX3) * CDELT3

    dispersion_phys:
        sigma_v = abs(sigma_pixel * CDELT3)

    Final quantities:
        position_phys is in km/s.
        dispersion_phys is in km/s.
        FWHM is in km/s.
        Moment0 is in K km/s.
    """

    print("\n" + "=" * 60)
    print("READING FILTERED ROHSA DAT FILE")
    print("=" * 60)

    print("Reading Gaussian parameters in pixel units...")
    gaussian_pixel = core.read_gaussian(dat_file)

    print(f"Gaussian pixel array shape: {gaussian_pixel.shape}")
    print(f"Data type: {gaussian_pixel.dtype}")

    # --------------------------------------------------------
    # Read velocity-axis information from the header.
    # --------------------------------------------------------
    if not hasattr(core, "hdr"):
        raise ValueError("core.hdr does not exist. Please set core.hdr = hdr in the notebook.")

    hdr = core.hdr

    crval3 = float(hdr["CRVAL3"])
    crpix3 = float(hdr["CRPIX3"])
    cdelt3 = float(hdr["CDELT3"])
    cunit3 = str(hdr.get("CUNIT3", "")).strip()

    print("\nSpectral WCS from header:")
    print(f"  CRVAL3 = {crval3}")
    print(f"  CRPIX3 = {crpix3}")
    print(f"  CDELT3 = {cdelt3}")
    print(f"  CUNIT3 = {cunit3}")

    # --------------------------------------------------------
    # The data have been verified to store velocity-axis values in km/s.
    # Use km/s directly here; do not divide or multiply by 1000.
    # --------------------------------------------------------
    velocity_unit = "km/s"
    print("\nUsing spectral values directly as km/s.")
    print("No m/s <-> km/s conversion is applied.")

    n_params_times_comp, n_y, n_x = gaussian_pixel.shape
    n_components = n_params_times_comp // 3

    print(f"\nSpatial dimensions: X={n_x}, Y={n_y}")
    print(f"Number of Gaussian components: {n_components}")

    results = {}

    for comp in range(1, n_components + 1):

        amp_pixel = gaussian_pixel[3 * (comp - 1)]
        pos_pixel = gaussian_pixel[3 * (comp - 1) + 1]
        dis_pixel = gaussian_pixel[3 * (comp - 1) + 2]

        # ----------------------------------------------------
        # Amplitude is already in K.
        # ----------------------------------------------------
        amp_phys = amp_pixel

        # ----------------------------------------------------
        # Convert manually from pixel coordinates to km/s.
        # FITS WCS is 1-based, so use pos_pixel + 1.
        # ----------------------------------------------------
        pos_phys = crval3 + (pos_pixel + 1.0 - crpix3) * cdelt3

        # ----------------------------------------------------
        # Convert sigma units by multiplying by the velocity width per channel.
        # Because sigma is a width, do not add CRVAL3/CRPIX3 offsets.
        # ----------------------------------------------------
        dis_phys = np.abs(dis_pixel * cdelt3)

        results[f"comp{comp}"] = {
            "amplitude_pixel": amp_pixel,
            "position_pixel": pos_pixel,
            "dispersion_pixel": dis_pixel,

            # Use these physical quantities consistently in downstream code.
            "amplitude_phys": amp_phys,
            "position_phys": pos_phys,
            "dispersion_phys": dis_phys,

            "component_id": comp,
        }

        valid = amp_phys > 0

        print(f"\nComponent {comp}:")
        print(f"  Non-zero pixels: {np.sum(valid)}")

        if np.any(valid):
            print(
                f"  Velocity range used in code: "
                f"[{np.nanmin(pos_phys[valid]):.3f}, "
                f"{np.nanmax(pos_phys[valid]):.3f}] km/s"
            )

            print(
                f"  Dispersion range used in code: "
                f"[{np.nanmin(dis_phys[valid]):.3f}, "
                f"{np.nanmax(dis_phys[valid]):.3f}] km/s"
            )

            print(
                f"  FWHM range used in code: "
                f"[{np.nanmin(2.355 * dis_phys[valid]):.3f}, "
                f"{np.nanmax(2.355 * dis_phys[valid]):.3f}] km/s"
            )

    results["n_components"] = n_components
    results["shape"] = (n_x, n_y)
    results["original_data_pixel"] = gaussian_pixel
    results["velocity_unit"] = velocity_unit

    return results


# ============================================================
# 3. Extract sources from one component.
# ============================================================
def extract_sources_from_component(results, comp, min_pixels=5, max_gap=2):
    amplitude = results[f"comp{comp}"]["amplitude_phys"]
    position = results[f"comp{comp}"]["position_phys"]
    dispersion = results[f"comp{comp}"]["dispersion_phys"]

    binary_mask = amplitude > 0

    if not np.any(binary_mask):
        return []

    structure = ndimage.generate_binary_structure(2, 1)

    if max_gap > 1:
        structure = ndimage.iterate_structure(structure, max_gap - 1)

    labeled_mask, num_features = ndimage.label(binary_mask, structure=structure)

    sources = []

    for label in range(1, num_features + 1):
        mask = labeled_mask == label
        pixel_count = np.sum(mask)

        if pixel_count < min_pixels:
            continue

        y_indices, x_indices = np.where(mask)

        amplitudes = []
        positions = []
        dispersions = []
        pixels = []

        for y, x in zip(y_indices, x_indices):
            amp = amplitude[y, x]
            pos = position[y, x]
            dis = dispersion[y, x]

            if amp > 0 and np.isfinite(amp) and np.isfinite(pos) and np.isfinite(dis):
                amplitudes.append(amp)
                positions.append(pos)
                dispersions.append(dis)
                pixels.append((x, y))

        if len(pixels) < min_pixels:
            continue

        amplitudes = np.array(amplitudes)
        positions = np.array(positions)
        dispersions = np.array(dispersions)

        total_amp = np.sum(amplitudes)

        if total_amp > 0:
            weighted_position = np.sum(positions * amplitudes) / total_amp
            weighted_dispersion = np.sum(dispersions * amplitudes) / total_amp
            weighted_amplitude = total_amp / len(pixels)
        else:
            weighted_position = np.mean(positions)
            weighted_dispersion = np.mean(dispersions)
            weighted_amplitude = 0.0

        weighted_fwhm = 2.355 * weighted_dispersion

        centroid_x = np.mean([p[0] for p in pixels])
        centroid_y = np.mean([p[1] for p in pixels])

        source_info = {
            "source_id": label,
            "component": comp,
            "pixel_count": len(pixels),
            "pixels": pixels,
            "pixel_set": set(pixels),
            "centroid": (centroid_x, centroid_y),
            "weighted_average": {
                "amplitude": weighted_amplitude,
                "velocity": weighted_position,
                "dispersion": weighted_dispersion,
                "fwhm": weighted_fwhm,
            },
            "bbox": (
                np.min(x_indices),
                np.max(x_indices),
                np.min(y_indices),
                np.max(y_indices),
            ),
        }

        sources.append(source_info)

    return sources


# ============================================================
# 4. Merge criteria.
# ============================================================
def count_overlap_pixels(source1, source2):
    pix1 = source1["pixel_set"] if "pixel_set" in source1 else set(source1["pixels"])
    pix2 = source2["pixel_set"] if "pixel_set" in source2 else set(source2["pixels"])

    overlap_pixels = pix1 & pix2

    return len(overlap_pixels), overlap_pixels


def check_merge_condition(
    source1,
    source2,
    min_overlap_pixels=2,
    velocity_threshold_factor=0.5,
):
    comp1 = source1["component"]
    comp2 = source2["component"]

    v1 = source1["weighted_average"]["velocity"]
    v2 = source2["weighted_average"]["velocity"]
    fwhm1 = source1["weighted_average"]["fwhm"]
    fwhm2 = source2["weighted_average"]["fwhm"]

    if comp1 == comp2:
        return {
            "can_merge": False,
            "reason": "same_component",
            "overlap_count": 0,
            "overlap_pixels": set(),
            "v1": v1,
            "v2": v2,
            "fwhm1": fwhm1,
            "fwhm2": fwhm2,
            "dv": abs(v1 - v2),
            "threshold": velocity_threshold_factor * max(fwhm1, fwhm2),
        }

    overlap_count, overlap_pixels = count_overlap_pixels(source1, source2)

    dv = abs(v1 - v2)
    wider_fwhm = max(fwhm1, fwhm2)
    threshold = velocity_threshold_factor * wider_fwhm

    spatial_ok = overlap_count >= min_overlap_pixels
    velocity_ok = dv < threshold
    can_merge = spatial_ok and velocity_ok

    if can_merge:
        reason = "merge"
    elif not spatial_ok:
        reason = "insufficient_spatial_overlap"
    else:
        reason = "velocity_difference_too_large"

    return {
        "can_merge": can_merge,
        "reason": reason,
        "overlap_count": overlap_count,
        "overlap_pixels": overlap_pixels,
        "v1": v1,
        "v2": v2,
        "fwhm1": fwhm1,
        "fwhm2": fwhm2,
        "dv": dv,
        "threshold": threshold,
    }


# ============================================================
# 5. Union-Find
# ============================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a

    def groups(self):
        group_dict = defaultdict(list)

        for i in range(len(self.parent)):
            group_dict[self.find(i)].append(i)

        return list(group_dict.values())


def merge_sources(sources_to_merge):
    if len(sources_to_merge) == 0:
        return None

    all_pixels = []
    all_components = []

    for src in sources_to_merge:
        all_pixels.extend(src["pixels"])
        all_components.append(src["component"])

    all_pixels = list(set(all_pixels))

    total_amp = sum(
        src["weighted_average"]["amplitude"] * src["pixel_count"]
        for src in sources_to_merge
    )

    if total_amp > 0:
        merged_velocity = sum(
            src["weighted_average"]["velocity"]
            * src["weighted_average"]["amplitude"]
            * src["pixel_count"]
            for src in sources_to_merge
        ) / total_amp

        merged_dispersion = sum(
            src["weighted_average"]["dispersion"]
            * src["weighted_average"]["amplitude"]
            * src["pixel_count"]
            for src in sources_to_merge
        ) / total_amp

        merged_amplitude = total_amp / len(all_pixels)

    else:
        merged_velocity = np.mean(
            [src["weighted_average"]["velocity"] for src in sources_to_merge]
        )
        merged_dispersion = np.mean(
            [src["weighted_average"]["dispersion"] for src in sources_to_merge]
        )
        merged_amplitude = 0.0

    x_coords = [p[0] for p in all_pixels]
    y_coords = [p[1] for p in all_pixels]

    merged_source = {
        "merged_id": None,
        "pixel_count": len(all_pixels),
        "pixels": all_pixels,
        "pixel_set": set(all_pixels),
        "centroid": (np.mean(x_coords), np.mean(y_coords)),
        "component_sources": sources_to_merge,
        "components": sorted(list(set(all_components))),
        "weighted_average": {
            "velocity": merged_velocity,
            "dispersion": merged_dispersion,
            "fwhm": 2.355 * merged_dispersion,
            "amplitude": merged_amplitude,
        },
        "is_merged": len(sources_to_merge) > 1,
    }

    return merged_source


def make_single_source(src):
    return {
        "merged_id": None,
        "pixel_count": src["pixel_count"],
        "pixels": src["pixels"],
        "pixel_set": set(src["pixels"]),
        "centroid": src["centroid"],
        "component_sources": [src],
        "components": [src["component"]],
        "weighted_average": src["weighted_average"],
        "is_merged": False,
    }


def auto_merge_all_components(
    component_sources,
    min_overlap_pixels=2,
    velocity_threshold_factor=0.5,
):
    all_original_sources = []

    for comp in sorted(component_sources.keys()):
        for src in component_sources[comp]:
            all_original_sources.append(src)

    n_sources = len(all_original_sources)

    uf = UnionFind(n_sources)
    check_results = []

    print("\n" + "=" * 60)
    print("AUTOMATIC MERGE CHECKING")
    print("=" * 60)

    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            src1 = all_original_sources[i]
            src2 = all_original_sources[j]

            result = check_merge_condition(
                src1,
                src2,
                min_overlap_pixels=min_overlap_pixels,
                velocity_threshold_factor=velocity_threshold_factor,
            )

            check_info = {
                "src1_index": i,
                "src2_index": j,
                "comp1": src1["component"],
                "comp2": src2["component"],
                "source_id1": src1["source_id"],
                "source_id2": src2["source_id"],
                "v1": result["v1"],
                "v2": result["v2"],
                "fwhm1": result["fwhm1"],
                "fwhm2": result["fwhm2"],
                "dv": result["dv"],
                "threshold": result["threshold"],
                "overlap_count": result["overlap_count"],
                "can_merge": result["can_merge"],
                "reason": result["reason"],
            }

            check_results.append(check_info)

            if src1["component"] != src2["component"]:
                print(
                    f"Comp{src1['component']}-Src{src1['source_id']} "
                    f"vs Comp{src2['component']}-Src{src2['source_id']}: "
                    f"overlap={result['overlap_count']}, "
                    f"dv={result['dv']:.3f}, "
                    f"threshold={result['threshold']:.3f}, "
                    f"merge={result['can_merge']}, "
                    f"reason={result['reason']}"
                )

            if result["can_merge"]:
                uf.union(i, j)

    groups = uf.groups()
    all_final_sources = []

    for group in groups:
        sources_in_group = [all_original_sources[idx] for idx in group]

        if len(sources_in_group) == 1:
            final_src = make_single_source(sources_in_group[0])
        else:
            final_src = merge_sources(sources_in_group)

        all_final_sources.append(final_src)

    for i, src in enumerate(all_final_sources):
        src["merged_id"] = i + 1

    return all_final_sources, check_results


# ============================================================
# 6. Create final parameter maps.
# ============================================================
def create_final_parameter_maps(all_sources, results, shape):
    n_x, n_y = shape

    merged_amplitude = np.zeros((n_y, n_x))
    merged_position = np.zeros((n_y, n_x))
    merged_dispersion = np.zeros((n_y, n_x))
    merged_fwhm = np.zeros((n_y, n_x))
    merged_moment0 = np.zeros((n_y, n_x))
    merged_source_id = np.zeros((n_y, n_x), dtype=int)

    for source in all_sources:
        source_id = source["merged_id"]

        for x, y in source["pixels"]:
            if 0 <= y < n_y and 0 <= x < n_x:
                pixel_amplitudes = []
                pixel_positions = []
                pixel_dispersions = []

                for comp_src in source["component_sources"]:
                    comp = comp_src["component"]

                    if (x, y) in comp_src["pixel_set"]:
                        amp = results[f"comp{comp}"]["amplitude_phys"][y, x]
                        pos = results[f"comp{comp}"]["position_phys"][y, x]
                        dis = results[f"comp{comp}"]["dispersion_phys"][y, x]

                        if amp > 0 and np.isfinite(amp) and np.isfinite(pos) and np.isfinite(dis):
                            pixel_amplitudes.append(amp)
                            pixel_positions.append(pos)
                            pixel_dispersions.append(dis)

                if len(pixel_amplitudes) > 0:
                    pixel_amplitudes = np.array(pixel_amplitudes)
                    pixel_positions = np.array(pixel_positions)
                    pixel_dispersions = np.array(pixel_dispersions)

                    weights = pixel_amplitudes / np.sum(pixel_amplitudes)

                    amp_sum = np.sum(pixel_amplitudes)
                    vel_w = np.sum(pixel_positions * weights)
                    dis_w = np.sum(pixel_dispersions * weights)
                    fwhm_w = 2.355 * dis_w
                    mom0 = amp_sum * dis_w * np.sqrt(2.0 * np.pi)

                    merged_amplitude[y, x] = amp_sum
                    merged_position[y, x] = vel_w
                    merged_dispersion[y, x] = dis_w
                    merged_fwhm[y, x] = fwhm_w
                    merged_moment0[y, x] = mom0
                    merged_source_id[y, x] = source_id

    return (
        merged_amplitude,
        merged_position,
        merged_dispersion,
        merged_fwhm,
        merged_moment0,
        merged_source_id,
    )


# ============================================================
# 7. WCS plotting.
# ============================================================
def plot_all_sources_wcs(
    all_sources,
    merged_position,
    merged_fwhm,
    merged_moment0,
    merged_source_id,
    output_dir,
    wcs_2d,
):
    print("\n" + "=" * 60)
    print("PLOTTING ALL SOURCES WITH WCS")
    print("=" * 60)
    
    set_times_new_roman_font()
    
    mpl.rcParams["axes.linewidth"] = 1.3
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    os.makedirs(output_dir, exist_ok=True)

    n_sources = len(all_sources)

    fig = plt.figure(figsize=(16, 14))

    plt.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.08,
        top=0.95,
        wspace=0.15,
        hspace=0.18,
    )

    ax1 = plt.subplot(2, 2, 1, projection=wcs_2d)
    ax2 = plt.subplot(2, 2, 2, projection=wcs_2d)
    ax3 = plt.subplot(2, 2, 3, projection=wcs_2d)
    ax4 = plt.subplot(2, 2, 4, projection=wcs_2d)

    axes_list = [ax1, ax2, ax3, ax4]

       # -------------------------
    # Panel 1: Source ID map
    # -------------------------
    # Fixed color scheme.
    # colors[0] is the background.
    # colors[1] corresponds to Source 1.
    fixed_colors = [
        [0.0, 0.0, 0.0, 1.0],     # 0 background: black

        [0.18, 0.49, 0.74, 1.0],  # 1: blue
        [1.00, 0.55, 0.00, 1.0],  # 2: orange
        [0.56, 0.82, 0.49, 1.0],  # 3: light green
        [0.95, 0.58, 0.58, 1.0],  # 4: pink
        [0.60, 0.40, 0.35, 1.0],  # 5: brown
        [0.00, 0.20, 0.60, 1.0],  # 6: dark blue
        [1.00, 0.20, 0.20, 1.0],  # 7: red
        [0.55, 0.00, 0.55, 1.0],  # 8: purple
        [0.65, 0.82, 0.85, 1.0],  # 9: light cyan
        [0.35, 0.35, 0.35, 1.0],  # 10: gray
    ]

    if n_sources + 1 <= len(fixed_colors):
        colors = np.array(fixed_colors[:n_sources + 1])
    else:
        extra_n = n_sources + 1 - len(fixed_colors)
        extra_colors = plt.cm.tab20(np.linspace(0, 1, extra_n))
        colors = np.vstack([np.array(fixed_colors), extra_colors])

    cmap = ListedColormap(colors)


    im1 = ax1.imshow(
        merged_source_id,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=n_sources,
        interpolation="nearest",
        transform=ax1.get_transform("pixel"),
    )

    for source in all_sources:
        x_center, y_center = source["centroid"]
        world_coords = wcs_2d.pixel_to_world(x_center, y_center)

        ra_center = world_coords.ra.deg
        dec_center = world_coords.dec.deg

        ax1.plot(
            ra_center,
            dec_center,
            marker="+",
            color="white",
            markersize=9,
            markeredgewidth=2.2,
            linestyle="None",
            transform=ax1.get_transform("world"),
            zorder=5,
        )

        # --------------------------------------------------
        # Manually adjust numeric label positions in degrees.
        # A positive offset_ra means a larger RA value.
        # Because sky-map RA axes are usually reversed, this may appear to move left visually.
        # --------------------------------------------------
        label_offsets = {
            6: (-0.13, -0.02),
            7: ( 0.11,  0.08),
            8: (-0.02,  0.13),
        }
        
        if source["merged_id"] in label_offsets:
            offset_ra, offset_dec = label_offsets[source["merged_id"]]
        else:
            offset_ra = 0.08
            offset_dec = 0.06
        
            if source["merged_id"] % 2 == 0:
                offset_ra = -0.08
            if source["merged_id"] % 3 == 0:
                offset_dec = -0.06

        ax1.text(
            ra_center + offset_ra,
            dec_center + offset_dec,
            f"{source['merged_id']}",
            color="white",
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=ax1.get_transform("world"),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="0.35",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=6,
        )

    # --------------------------------------------------
    # Panel 1 does not display the source-ID colorbar.
    # Keep an empty colorbar slot so the top edges of ax1 and ax2 stay aligned.
    # --------------------------------------------------
    cbar1 = plt.colorbar(
        im1,
        ax=ax1,
        location="bottom",
        pad=0.16,
        shrink=1.0,
        aspect=30,
        ticks=[],
    )
    
    # Clear the colorbar contents, but do not remove the axis.
    # Removing it would also remove the reserved space.
    cbar1.ax.clear()
    cbar1.ax.set_frame_on(False)
    cbar1.ax.set_xticks([])
    cbar1.ax.set_yticks([])
    cbar1.ax.patch.set_alpha(0.0)
    
    for spine in cbar1.ax.spines.values():
        spine.set_visible(False)

    # -------------------------
    # Panel 2: Moment 0
    # -------------------------
    moment0_plot = np.where(merged_moment0 > 0, merged_moment0, np.nan)
    valid_mom0 = moment0_plot[np.isfinite(moment0_plot)]
    mom0_vmax = np.nanpercentile(valid_mom0, 98) if valid_mom0.size > 0 else 40

    im2 = ax2.imshow(
        moment0_plot,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=40,
        interpolation="nearest",
        transform=ax2.get_transform("pixel"),
    )

    cbar2 = plt.colorbar(
        im2,
        ax=ax2,
        location="bottom",
        pad=0.16,
        shrink=1.0,
        aspect=30,
    )
    cbar2.set_label(r"Integrated Intensity [K km s$^{-1}$]", fontsize=22)
    cbar2.ax.tick_params(labelsize=17, direction="in", width=1.1, length=5)

    # -------------------------
    # Panel 3: Velocity
    # -------------------------
    vel_plot = np.where(merged_position != 0, merged_position, np.nan)
    valid_v = vel_plot[np.isfinite(vel_plot)]

    im3 = ax3.imshow(
        vel_plot,
        origin="lower",
        cmap="coolwarm",
        vmin=-300,
        vmax=-200,
        interpolation="nearest",
        transform=ax3.get_transform("pixel"),
    )

    cbar3 = plt.colorbar(
        im3,
        ax=ax3,
        location="bottom",
        pad=0.16,
        shrink=1.0,
        aspect=30,
    )
    cbar3.set_label(r"Velocity [km s$^{-1}$]", fontsize=22)
    cbar3.ax.tick_params(labelsize=17, direction="in", width=1.1, length=5)

    # -------------------------
    # Panel 4: FWHM
    # -------------------------
    fwhm_plot = np.where(merged_fwhm > 0, merged_fwhm, np.nan)
    valid_fwhm = fwhm_plot[np.isfinite(fwhm_plot)]
    fwhm_vmax = np.nanpercentile(valid_fwhm, 98) if valid_fwhm.size > 0 else 50

    im4 = ax4.imshow(
        fwhm_plot,
        origin="lower",
        cmap="cubehelix",
        vmin=0,
        vmax=50,
        interpolation="nearest",
        transform=ax4.get_transform("pixel"),
    )

    cbar4 = plt.colorbar(
        im4,
        ax=ax4,
        location="bottom",
        pad=0.16,
        shrink=1.0,
        aspect=30,
    )
    cbar4.set_label(r"FWHM [km s$^{-1}$]", fontsize=22)
    cbar4.ax.tick_params(labelsize=17, direction="in", width=1.1, length=5)

    # -------------------------
    # WCS coordinate formatting.
    # -------------------------
    for i, ax in enumerate(axes_list):
        ax.coords[0].set_ticks_position("b")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel("Right Ascension (J2000)", fontsize=23)
        ax.coords[0].set_format_unit(u.hourangle)
        ax.coords[0].tick_params(labelsize=18, direction="in")

        ax.coords[1].set_format_unit(u.deg)

        if i in [0, 2]:
            ax.coords[1].set_ticks_position("l")
            ax.coords[1].set_ticklabel_position("l")
            ax.coords[1].set_axislabel("Declination (J2000)", fontsize=23)
            ax.coords[1].tick_params(labelsize=18, direction="in")
        else:
            ax.coords[1].set_ticks_position("")
            ax.coords[1].set_ticklabel_position("")
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_ticks_visible(False)
            ax.coords[1].set_axislabel("")

        overlay = ax.get_coords_overlay("fk5")

        overlay[0].set_ticks_position("t")
        overlay[0].set_ticklabel_position("t")
        overlay[0].set_axislabel(" ", fontsize=23)
        overlay[0].set_format_unit(u.deg)
        overlay[0].tick_params(labelsize=18, direction="in")

        overlay[1].set_ticks_position("")
        overlay[1].set_ticklabel_position("")
        overlay[1].set_ticklabel_visible(False)
        overlay[1].set_ticks_visible(False)
        overlay[1].set_axislabel("")

        overlay.grid(
            color="gray",
            ls="dotted",
            lw=0.6,
            alpha=0.7,
        )

    overview_file = os.path.join(output_dir, "all_sources_overview.pdf")

    plt.savefig(
        overview_file,
        dpi=300,
        bbox_inches="tight",
    )

    print("PDF exists:", os.path.exists(overview_file), overview_file)

    plt.close(fig)


# ============================================================
# 7b. Save a composition PDF for each source.
# ============================================================
def plot_individual_source_compositions_wcs(
    all_sources,
    merged_source_id,
    output_dir,
    wcs_2d,
):
    """
    Save one composition figure for each final source.

    Output file:
        source_compositions/source_001_composition.pdf
        source_compositions/source_002_composition.pdf
        ...
    """

    print("\n" + "=" * 60)
    print("PLOTTING INDIVIDUAL SOURCE COMPOSITIONS")
    print("=" * 60)
    
    set_times_new_roman_font()
    

    comp_output_dir = os.path.join(output_dir, "source_compositions")
    os.makedirs(comp_output_dir, exist_ok=True)

    comp_colors = [
        "red",
        "green",
        "blue",
        "purple",
        "orange",
        "brown",
        "cyan",
        "magenta",
    ]

    for source in all_sources:

        source_id = source["merged_id"]

        fig = plt.figure(figsize=(6.5, 5.8))
        ax = plt.subplot(1, 1, 1, projection=wcs_2d)

        source_mask = merged_source_id == source_id

        # Background: full mask of this final source.
        ax.imshow(
            source_mask,
            origin="lower",
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
            transform=ax.get_transform("pixel"),
        )

        legend_handles = []

        # Contours of the different component sources.
        for j, comp_src in enumerate(source["component_sources"]):

            comp_mask = np.zeros_like(source_mask, dtype=bool)

            for x, y in comp_src["pixels"]:
                if 0 <= y < comp_mask.shape[0] and 0 <= x < comp_mask.shape[1]:
                    comp_mask[y, x] = True

            color = comp_colors[j % len(comp_colors)]

            if np.any(comp_mask):
                ax.contour(
                    comp_mask.astype(float),
                    levels=[0.5],
                    colors=[color],
                    linewidths=2.0,
                    origin="lower",
                    transform=ax.get_transform("pixel"),
                )

                legend_handles.append(
                    Patch(
                        facecolor="none",
                        edgecolor=color,
                        linewidth=2.0,
                        label=f"Comp{comp_src['component']}-Src{comp_src['source_id']}",
                    )
                )

        # Mark the final-source centroid.
        xcen, ycen = source["centroid"]
        world_coords = wcs_2d.pixel_to_world(xcen, ycen)

        ax.plot(
            world_coords.ra.deg,
            world_coords.dec.deg,
            marker="+",
            color="black",
            markersize=12,
            markeredgewidth=2.2,
            linestyle="None",
            transform=ax.get_transform("world"),
            zorder=5,
        )

        # Information box.
        info_text = (
            f"Source {source_id}\n"
            rf"$N_{{pix}}$={source['pixel_count']}" "\n"
            rf"$V$={source['weighted_average']['velocity']:.1f} km s$^{{-1}}$" "\n"
            rf"FWHM={source['weighted_average']['fwhm']:.1f} km s$^{{-1}}$"
        )

        ax.text(
            world_coords.ra.deg + 0.05,
            world_coords.dec.deg + 0.05,
            info_text,
            fontsize=11,
            color="black",
            transform=ax.get_transform("world"),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.5",
                alpha=0.85,
            ),
            zorder=6,
        )

        # -----------------------------
        # Axis format: RA in hours at the bottom, RA in degrees at the top, Dec in degrees on the left.
        # -----------------------------
        ax.coords[0].set_ticks_position("b")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel("Right Ascension [hours]", fontsize=15)
        ax.coords[0].set_format_unit(u.hourangle)
        ax.coords[0].tick_params(labelsize=12, direction="in")

        ax.coords[1].set_format_unit(u.deg)
        ax.coords[1].set_axislabel("Declination [°]", fontsize=15)
        ax.coords[1].tick_params(labelsize=12, direction="in")

        overlay = ax.get_coords_overlay("fk5")

        overlay[0].set_ticks_position("t")
        overlay[0].set_ticklabel_position("t")
        overlay[0].set_axislabel("Right Ascension [°]", fontsize=15)
        overlay[0].set_format_unit(u.deg)
        overlay[0].tick_params(labelsize=12, direction="in")

        overlay[1].set_ticks_position("")
        overlay[1].set_ticklabel_position("")
        overlay[1].set_ticklabel_visible(False)
        overlay[1].set_ticks_visible(False)
        overlay[1].set_axislabel("")

        overlay.grid(
            color="gray",
            ls="dotted",
            lw=0.5,
            alpha=0.6,
        )

        if len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=9,
                frameon=True,
                framealpha=0.85,
                edgecolor="black",
                fancybox=False,
            )

        out_file = os.path.join(
            comp_output_dir,
            f"source_{source_id:03d}_composition.pdf",
        )

        plt.savefig(
            out_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved source {source_id} composition to: {out_file}")



# ============================================================
# 8. Save results.





# ============================================================
# Save final source masks and pixel-level parameter tables
# ============================================================
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def make_2d_celestial_header_from_fits(fits_file, shape_2d=None):
    """
    Extract the 2D celestial WCS header from the original 3D FITS cube.

    Parameters
    ----------
    fits_file : str
        Path to the original FITS cube.

    shape_2d : tuple or None
        Shape of the 2D mask, as (n_y, n_x).
        If provided, NAXIS1 and NAXIS2 are written into the header.

    Returns
    -------
    header_2d : astropy.io.fits.Header
        A 2D header containing only RA/Dec WCS.
    """

    with fits.open(fits_file) as hdul:
        hdr3d = hdul[0].header

    wcs3d = WCS(hdr3d)
    wcs2d = wcs3d.celestial

    header_2d = wcs2d.to_header()

    if shape_2d is not None:
        n_y, n_x = shape_2d
        header_2d["NAXIS"] = 2
        header_2d["NAXIS1"] = n_x
        header_2d["NAXIS2"] = n_y

    header_2d["BUNIT"] = "mask"
    header_2d["COMMENT"] = "2D source mask generated from final merged ROHSA sources."

    return header_2d


def pixel_to_world_safe(wcs_2d, x, y):
    """
    Safely convert pixel coordinates to RA/Dec.
    Return NaN if WCS is unavailable.
    """

    if wcs_2d is None:
        return np.nan, np.nan

    try:
        world = wcs_2d.pixel_to_world(x, y)
        return world.ra.deg, world.dec.deg
    except Exception:
        return np.nan, np.nan


def build_final_source_id_map(all_sources, shape):
    """
    Create the final source-ID map from all_sources.

    Parameters
    ----------
    all_sources : list
        Final source list after merging.

    shape : tuple
        (n_x, n_y),that is, results["shape"].

    Returns
    -------
    source_id_map : 2D ndarray
        shape = (n_y, n_x),background is 0 and source regions are source_id.
    """

    n_x, n_y = shape
    source_id_map = np.zeros((n_y, n_x), dtype=np.int16)

    for source in all_sources:
        source_id = source["merged_id"]

        for x, y in source["pixels"]:
            if 0 <= y < n_y and 0 <= x < n_x:
                source_id_map[y, x] = source_id

    return source_id_map


def save_source_masks_and_pixel_tables(
    all_sources,
    results,
    output_dir,
    original_dat_file=None,
    fits_file=None,
    header_nlines=27,
):
    """
    Save masks for each merged source and per-pixel parameters.

    Output directory structure:

    output_dir/
        masks/
            all_sources_source_id_map.fits
            source_001_mask.fits
            source_002_mask.fits
            ...

        source_pixel_parameters/
            source_001_pixel_parameters.dat
            source_002_pixel_parameters.dat
            ...

        source_physical_parameters/
            source_001_physical_parameters.csv
            source_002_physical_parameters.csv
            ...
            all_sources_physical_parameters.csv

    Parameters
    ----------
    all_sources : list
        Final merged source list.

    results : dict
        Result returned by read_filtered_dat().

    output_dir : str
        Top-level output directory.

    original_dat_file : str or None
        Original DAT file path. If provided, the first header_nlines lines are used as the .dat header.

    fits_file : str or None
        Original FITS cube path. If provided, RA/Dec WCS is added to mask FITS files.

    header_nlines : int
        Number of header lines in the original DAT file.
    """

    print("\n" + "=" * 60)
    print("SAVING SOURCE MASKS AND PIXEL TABLES")
    print("=" * 60)

    n_x, n_y = results["shape"]
    n_components = results["n_components"]
    shape_2d = (n_y, n_x)

    masks_dir = os.path.join(output_dir, "masks")
    pixel_param_dir = os.path.join(output_dir, "source_pixel_parameters")
    phys_param_dir = os.path.join(output_dir, "source_physical_parameters")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(pixel_param_dir, exist_ok=True)
    os.makedirs(phys_param_dir, exist_ok=True)

    # --------------------------------------------------------
    # Prepare WCS header and WCS object
    # --------------------------------------------------------
    header_2d = None
    wcs_2d = None

    if fits_file is not None and os.path.exists(fits_file):
        try:
            header_2d = make_2d_celestial_header_from_fits(
                fits_file,
                shape_2d=shape_2d,
            )
            wcs_2d = WCS(header_2d)
            print(f"Loaded 2D WCS from FITS: {fits_file}")
        except Exception as e:
            print(f"Warning: failed to create 2D WCS header: {e}")
            header_2d = fits.Header()
            header_2d["NAXIS"] = 2
            header_2d["NAXIS1"] = n_x
            header_2d["NAXIS2"] = n_y
            header_2d["BUNIT"] = "mask"
            wcs_2d = None
    else:
        print("No valid FITS file provided. Masks will be saved without WCS.")
        header_2d = fits.Header()
        header_2d["NAXIS"] = 2
        header_2d["NAXIS1"] = n_x
        header_2d["NAXIS2"] = n_y
        header_2d["BUNIT"] = "mask"

    # --------------------------------------------------------
    # Read original DAT header
    # --------------------------------------------------------
    dat_header_lines = []

    if original_dat_file is not None and os.path.exists(original_dat_file):
        with open(original_dat_file, "r") as f:
            for _ in range(header_nlines):
                dat_header_lines.append(f.readline())
    else:
        dat_header_lines = [
            "# Final merged source pixel parameters\n",
            "# Columns: y x component amplitude_pixel position_pixel dispersion_pixel\n",
        ]

    # --------------------------------------------------------
    # Save all-source ID map
    # --------------------------------------------------------
    source_id_map = build_final_source_id_map(all_sources, results["shape"])

    all_mask_file = os.path.join(masks_dir, "all_sources_source_id_map.fits")

    fits.writeto(
        all_mask_file,
        source_id_map.astype(np.int16),
        header=header_2d,
        overwrite=True,
    )

    print(f"Saved all-source ID mask: {all_mask_file}")

    # --------------------------------------------------------
    # Save per-source masks and parameter tables
    # --------------------------------------------------------
    all_phys_rows = []

    for source in all_sources:
        source_id = source["merged_id"]

        source_mask = np.zeros(shape_2d, dtype=np.uint8)

        for x, y in source["pixels"]:
            if 0 <= y < n_y and 0 <= x < n_x:
                source_mask[y, x] = 1

        # -----------------------------
        # Save individual source mask
        # -----------------------------
        source_mask_file = os.path.join(
            masks_dir,
            f"source_{source_id:03d}_mask.fits",
        )

        source_header = header_2d.copy()
        source_header["SRCID"] = int(source_id)
        source_header["NPIX"] = int(source["pixel_count"])
        source_header["COMMENT"] = f"Mask for final merged source {source_id}."

        fits.writeto(
            source_mask_file,
            source_mask.astype(np.uint8),
            header=source_header,
            overwrite=True,
        )

        # -----------------------------
        # Save pixel-parameter DAT
        # -----------------------------
        dat_file = os.path.join(
            pixel_param_dir,
            f"source_{source_id:03d}_pixel_parameters.dat",
        )

        kept_comp_pixel = set()

        for comp_src in source["component_sources"]:
            comp = comp_src["component"]
            for x, y in comp_src["pixels"]:
                kept_comp_pixel.add((comp, x, y))

        with open(dat_file, "w", encoding="utf-8") as f:
            for line in dat_header_lines:
                f.write(line)

            f.write("#\n")
            f.write("# Final merged source pixel parameters\n")
            f.write(f"# Source ID: {source_id}\n")
            f.write("# Columns: y x component amplitude_pixel position_pixel dispersion_pixel\n")
            f.write("#\n")

            for y in range(n_y):
                for x in range(n_x):
                    for comp in range(1, n_components + 1):

                        if (comp, x, y) in kept_comp_pixel:
                            amp_pix = results[f"comp{comp}"]["amplitude_pixel"][y, x]
                            pos_pix = results[f"comp{comp}"]["position_pixel"][y, x]
                            dis_pix = results[f"comp{comp}"]["dispersion_pixel"][y, x]

                            if not np.isfinite(amp_pix) or amp_pix <= 0:
                                amp_pix = 0.0
                                pos_pix = 0.0
                                dis_pix = 0.0
                        else:
                            amp_pix = 0.0
                            pos_pix = 0.0
                            dis_pix = 0.0

                        f.write(
                            f"{y:6d} {x:6d} {comp:3d} "
                            f"{amp_pix:20.10f} "
                            f"{pos_pix:20.10f} "
                            f"{dis_pix:20.10f}\n"
                        )

        # -----------------------------
        # Save physical-parameter CSV
        # -----------------------------
        phys_rows = []

        for comp_src in source["component_sources"]:
            comp = comp_src["component"]
            original_source_id = comp_src["source_id"]

            for x, y in comp_src["pixels"]:
                if not (0 <= y < n_y and 0 <= x < n_x):
                    continue

                amp_pix = results[f"comp{comp}"]["amplitude_pixel"][y, x]
                pos_pix = results[f"comp{comp}"]["position_pixel"][y, x]
                dis_pix = results[f"comp{comp}"]["dispersion_pixel"][y, x]

                amp_k = results[f"comp{comp}"]["amplitude_phys"][y, x]
                vel_kms = results[f"comp{comp}"]["position_phys"][y, x]
                sigma_kms = results[f"comp{comp}"]["dispersion_phys"][y, x]

                if not (
                    np.isfinite(amp_k)
                    and np.isfinite(vel_kms)
                    and np.isfinite(sigma_kms)
                    and amp_k > 0
                ):
                    continue

                fwhm_kms = 2.355 * sigma_kms
                moment0_k_kms = amp_k * sigma_kms * np.sqrt(2.0 * np.pi)

                ra_deg, dec_deg = pixel_to_world_safe(wcs_2d, x, y)

                row = {
                    "final_source_id": source_id,
                    "component": comp,
                    "original_component_source_id": original_source_id,

                    "x_pixel": x,
                    "y_pixel": y,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,

                    "amplitude_pixel": amp_pix,
                    "position_pixel": pos_pix,
                    "dispersion_pixel": dis_pix,

                    "amplitude_K": amp_k,
                    "velocity_kms": vel_kms,
                    "sigma_kms": sigma_kms,
                    "fwhm_kms": fwhm_kms,
                    "moment0_K_kms": moment0_k_kms,
                }

                phys_rows.append(row)
                all_phys_rows.append(row)

        csv_file = os.path.join(
            phys_param_dir,
            f"source_{source_id:03d}_physical_parameters.csv",
        )

        df_source = pd.DataFrame(phys_rows)
        df_source.to_csv(csv_file, index=False)

        print(f"Saved source {source_id:03d} mask: {source_mask_file}")
        print(f"Saved source {source_id:03d} pixel DAT: {dat_file}")
        print(f"Saved source {source_id:03d} physical CSV: {csv_file}")

    # --------------------------------------------------------
    # Save combined physical table
    # --------------------------------------------------------
    all_csv_file = os.path.join(
        phys_param_dir,
        "all_sources_physical_parameters.csv",
    )

    df_all = pd.DataFrame(all_phys_rows)
    df_all.to_csv(all_csv_file, index=False)

    print(f"Saved all-source physical CSV: {all_csv_file}")

    print("\nMask and pixel table outputs:")
    print(f"  Masks:                 {masks_dir}")
    print(f"  Pixel-parameter DATs:  {pixel_param_dir}")
    print(f"  Physical CSVs:         {phys_param_dir}")

    return {
        "masks_dir": masks_dir,
        "pixel_param_dir": pixel_param_dir,
        "phys_param_dir": phys_param_dir,
        "all_mask_file": all_mask_file,
        "all_physical_csv": all_csv_file,
    }




# ============================================================
def save_source_info(all_sources, output_dir):
    info_file = os.path.join(output_dir, "final_source_information.txt")

    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL SOURCE INFORMATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total sources: {len(all_sources)}\n\n")

        for source in all_sources:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"SOURCE {source['merged_id']}\n")
            f.write("=" * 60 + "\n\n")

            if source.get("is_merged", False):
                f.write("Type: MERGED\n")
                f.write(f"Components involved: {source['components']}\n")
            else:
                f.write("Type: SINGLE COMPONENT\n")
                f.write(f"Component: {source['components'][0]}\n")

            f.write(f"Number of pixels: {source['pixel_count']}\n")
            f.write(f"Centroid (X, Y): ({source['centroid'][0]:.2f}, {source['centroid'][1]:.2f})\n")
            f.write(f"Mean Velocity: {source['weighted_average']['velocity']:.6f} km/s\n")
            f.write(f"Mean Dispersion: {source['weighted_average']['dispersion']:.6f} km/s\n")
            f.write(f"Mean FWHM: {source['weighted_average']['fwhm']:.6f} km/s\n")
            f.write(f"Mean Amplitude: {source['weighted_average']['amplitude']:.6f} K\n\n")

    print(f"Source information saved to: {info_file}")


def save_final_dat(results, all_sources, original_dat_file, output_file, header_nlines=27):
    print("\n" + "=" * 60)
    print("SAVING FINAL RESULTS TO .DAT FORMAT")
    print("=" * 60)

    n_x, n_y = results["shape"]
    n_components = results["n_components"]

    with open(original_dat_file, "r") as f:
        header_lines = []
        for _ in range(header_nlines):
            header_lines.append(f.readline())

    kept_comp_pixel = set()

    for source in all_sources:
        for comp_src in source["component_sources"]:
            comp = comp_src["component"]
            for x, y in comp_src["pixels"]:
                kept_comp_pixel.add((comp, x, y))

    data_lines = []

    for y in range(n_y):
        for x in range(n_x):
            for comp in range(1, n_components + 1):
                if (comp, x, y) in kept_comp_pixel:
                    amp = results[f"comp{comp}"]["amplitude_pixel"][y, x]
                    pos = results[f"comp{comp}"]["position_pixel"][y, x]
                    dis = results[f"comp{comp}"]["dispersion_pixel"][y, x]

                    if not np.isfinite(amp) or amp <= 0:
                        amp = 0.0
                        pos = 0.0
                        dis = 0.0
                else:
                    amp = 0.0
                    pos = 0.0
                    dis = 0.0

                line = (
                    f"    {y:4d}    {x:4d}"
                    f"    {amp:20.16f}"
                    f"    {pos:20.16f}"
                    f"    {dis:20.16f}\n"
                )
                data_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(header_lines)
        f.writelines(data_lines)

    print(f"Final .dat file saved to: {output_file}")


def save_merge_check_results(
    check_results,
    output_dir,
    min_overlap_pixels=2,
    velocity_threshold_factor=0.5,
):
    check_file = os.path.join(output_dir, "merge_check_results.txt")

    with open(check_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AUTOMATIC MERGE CHECK RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Merge criteria:\n")
        f.write(f"  Minimum overlap pixels: {min_overlap_pixels}\n")
        f.write(
            f"  Velocity criterion: |v1 - v2| < "
            f"{velocity_threshold_factor:.3f} * max(FWHM1, FWHM2)\n\n"
        )

        for check in check_results:
            f.write("-" * 70 + "\n")
            f.write(
                f"Comp{check['comp1']}-Src{check['source_id1']} "
                f"vs Comp{check['comp2']}-Src{check['source_id2']}\n"
            )
            f.write(f"  Velocity 1: {check['v1']:.6f} km/s\n")
            f.write(f"  Velocity 2: {check['v2']:.6f} km/s\n")
            f.write(f"  FWHM 1: {check['fwhm1']:.6f} km/s\n")
            f.write(f"  FWHM 2: {check['fwhm2']:.6f} km/s\n")
            f.write(f"  Overlap pixels: {check['overlap_count']}\n")
            f.write(f"  Velocity difference: {check['dv']:.6f} km/s\n")
            f.write(f"  Velocity threshold: {check['threshold']:.6f} km/s\n")
            f.write(f"  Can merge: {check['can_merge']}\n")
            f.write(f"  Reason: {check['reason']}\n\n")

    print(f"Merge check results saved to: {check_file}")

def remove_component_sources_before_merge(component_sources, remove_dict):
    """
    Remove selected source IDs from selected components before automatic merging.

    Example:
        remove_dict = {
            1: [3, 5, 8, 9],
            2: [2, 4],
        }

    This means:
        remove Sources 3, 5, 8, and 9 from Component 1
        remove Sources 2 and 4 from Component 2

    After removal, the remaining component sources are passed to auto_merge_all_components()
    for automatic merging. Final sources are automatically numbered from 1.
    """

    new_component_sources = {}

    print("\n" + "=" * 60)
    print("REMOVING COMPONENT SOURCES BEFORE MERGING")
    print("=" * 60)

    for comp, src_list in component_sources.items():

        remove_ids = set(remove_dict.get(comp, []))
        kept_sources = []

        for src in src_list:
            src_id = src["source_id"]

            if src_id in remove_ids:
                print(f"Remove Component {comp} Source {src_id}")
                continue

            kept_sources.append(src)

        new_component_sources[comp] = kept_sources

        print(
            f"Component {comp}: "
            f"{len(src_list)} original sources -> "
            f"{len(kept_sources)} kept sources"
        )

    return new_component_sources

def main(
    core,
    dat_file,
    original_dat_file,
    output_dir,
    fits_file=None,
    min_pixels=5,
    max_gap=2,
    min_overlap_pixels=2,
    velocity_threshold_factor=0.5,
    header_nlines=27,
):
    set_times_new_roman_font()

    print("\n" + "=" * 60)
    print("AUTOMATIC SOURCE MERGING")
    print("=" * 60)
    print(f"Input DAT file: {dat_file}")
    print(f"Original DAT file: {original_dat_file}")
    print(f"Output directory: {output_dir}")
    print(f"FITS file: {fits_file}")
    print(f"Min pixels per source: {min_pixels}")
    print(f"Max spatial gap: {max_gap}")
    print(f"Min overlap pixels for merging: {min_overlap_pixels}")
    print(f"Velocity threshold factor: {velocity_threshold_factor}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nFile check:")
    print("  DAT exists:      ", os.path.exists(dat_file), dat_file)
    print("  Original exists: ", os.path.exists(original_dat_file), original_dat_file)

    wcs_2d = None

    if fits_file is not None:
        print("  FITS exists:     ", os.path.exists(fits_file), fits_file)

    if fits_file is not None and os.path.exists(fits_file):
        print(f"\nReading WCS info from FITS: {fits_file}")
        wcs_2d, fits_n_x, fits_n_y = get_coordinate_info(fits_file)

        if wcs_2d is not None:
            print("Successfully loaded WCS coordinates")
        else:
            print("Could not load WCS coordinates. Plot will be skipped.")
    else:
        print("No valid FITS file provided. WCS plot will be skipped.")

    # ========================================================
    # 1. Read ROHSA results.
    # ========================================================
    results = read_filtered_dat(core, dat_file)

    shape = results["shape"]
    n_components = results["n_components"]

    print(f"\nDetected {n_components} Gaussian components")

    # ========================================================
    # 2. Extract sources from each component.
    # ========================================================
    print("\n" + "=" * 60)
    print("EXTRACTING SOURCES FROM EACH COMPONENT")
    print("=" * 60)

    component_sources = {}

    for comp in range(1, n_components + 1):
        src_list = extract_sources_from_component(
            results,
            comp,
            min_pixels=min_pixels,
            max_gap=max_gap,
        )

        component_sources[comp] = src_list

        print(f"\nComponent {comp}: {len(src_list)} sources")

        for src in src_list:
            print(
                f"  Source {src['source_id']}: "
                f"{src['pixel_count']} pixels, "
                f"v={src['weighted_average']['velocity']:.3f} km/s, "
                f"FWHM={src['weighted_average']['fwhm']:.3f} km/s, "
                f"centroid=({src['centroid'][0]:.2f}, {src['centroid'][1]:.2f})"
            )




    
    REMOVE_COMPONENT_SOURCES = True
    REMOVE_DICT = {1: [6]}
    if REMOVE_COMPONENT_SOURCES:
        component_sources = remove_component_sources_before_merge(
            component_sources,
            remove_dict=REMOVE_DICT,
             )
    # ========================================================
    # 3. Remove selected component sources before merging.
    # ========================================================
    #  REMOVE_COMPONENT_SOURCES = True

    # REMOVE_DICT = {
    #     1: [3, 5, 8, 9],   # Remove Sources 3, 5, 8, and 9 from Component 1.
    #     2: [2, 4],         # Remove Sources 2 and 4 from Component 2.
    # }

    # if REMOVE_COMPONENT_SOURCES:
    #     component_sources = remove_component_sources_before_merge(
    #         component_sources,
    #         remove_dict=REMOVE_DICT,
    #     )

    # ========================================================
    # 4. Run automatic merging on the remaining component sources.
    # ========================================================
    all_final_sources, check_results = auto_merge_all_components(
        component_sources,
        min_overlap_pixels=min_overlap_pixels,
        velocity_threshold_factor=velocity_threshold_factor,
    )

    # ========================================================
    # 5. Print the final merged source list.
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL SOURCE LIST AFTER MERGING")
    print("=" * 60)
    print(f"Total final sources: {len(all_final_sources)}")

    for src in all_final_sources:
        if src.get("is_merged", False):
            comp_src_names = [
                f"Comp{s['component']}-Src{s['source_id']}"
                for s in src["component_sources"]
            ]

            print(
                f"  Source {src['merged_id']}: MERGED, "
                f"{comp_src_names}, "
                f"{src['pixel_count']} pixels, "
                f"v={src['weighted_average']['velocity']:.3f} km/s, "
                f"FWHM={src['weighted_average']['fwhm']:.3f} km/s"
            )

        else:
            origin = src["component_sources"][0]

            print(
                f"  Source {src['merged_id']}: SINGLE, "
                f"Comp{origin['component']}-Src{origin['source_id']}, "
                f"{src['pixel_count']} pixels, "
                f"v={src['weighted_average']['velocity']:.3f} km/s, "
                f"FWHM={src['weighted_average']['fwhm']:.3f} km/s"
            )

    # ========================================================
    # 6. Generate parameter maps from the final merged sources.
    # ========================================================
    (
        merged_amplitude,
        merged_position,
        merged_dispersion,
        merged_fwhm,
        merged_moment0,
        merged_source_id,
    ) = create_final_parameter_maps(all_final_sources, results, shape)

    # ========================================================
    # 7. Save masks and pixel-level tables.
    # ========================================================
    save_source_masks_and_pixel_tables(
        all_sources=all_final_sources,
        results=results,
        output_dir=output_dir,
        original_dat_file=original_dat_file,
        fits_file=fits_file,
        header_nlines=header_nlines,
    )

    # ========================================================
    # 8. Plot figures.
    # ========================================================
    if wcs_2d is not None:
        plot_all_sources_wcs(
            all_final_sources,
            merged_position,
            merged_fwhm,
            merged_moment0,
            merged_source_id,
            output_dir,
            wcs_2d,
        )

        plot_individual_source_compositions_wcs(
            all_final_sources,
            merged_source_id,
            output_dir,
            wcs_2d,
        )

    # ========================================================
    # 9. Save text summaries and the final DAT file.
    # ========================================================
    save_source_info(all_final_sources, output_dir)

    final_dat = os.path.join(output_dir, "final_sources.dat")

    save_final_dat(
        results,
        all_final_sources,
        original_dat_file,
        final_dat,
        header_nlines=header_nlines,
    )

    save_merge_check_results(
        check_results,
        output_dir,
        min_overlap_pixels=min_overlap_pixels,
        velocity_threshold_factor=velocity_threshold_factor,
    )

    print("\n" + "=" * 60)
    print("PROCESSING FINISHED")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - all_sources_overview.pdf")
    print("  - final_source_information.txt")
    print("  - final_sources.dat")
    print("  - merge_check_results.txt")

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from spectral_cube import SpectralCube
from pvextractor import extract_pv_slice, Path
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing
from reproject import reproject_interp


# ============================================================
# 0. 字体设置
# ============================================================
def set_times_new_roman_font():
    """优先使用 Times New Roman；找不到时使用 serif 字体。"""
    font_loaded = False

    possible_paths = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/TTF/Times_New_Roman.ttf",
        "/usr/local/share/fonts/Times_New_Roman.ttf",
        "/home/yx/.local/share/fonts/Times_New_Roman.ttf",
    ]

    for font_path in possible_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            mpl.rcParams["font.family"] = font_name
            font_loaded = True
            print(f"Loaded font: {font_path}")
            break

    if not font_loaded:
        try:
            font_path = subprocess.check_output(
                ["fc-match", "-f", "%{file}", "Times New Roman"],
                text=True,
            ).strip()
            if font_path and os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rcParams["font.family"] = font_name
                mpl.rcParams["font.family"] = font_name
                font_loaded = True
                print(f"Loaded font via fc-match: {font_path}")
        except Exception:
            pass

    if not font_loaded:
        plt.rcParams["font.family"] = "serif"
        mpl.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
            "Times",
        ]
        mpl.rcParams["font.serif"] = plt.rcParams["font.serif"]
        print("Times New Roman not found; using serif fallback.")

    plt.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.unicode_minus"] = False


set_times_new_roman_font()
plt.rcParams["font.size"] = 12


# ============================================================
# 1. 用户参数（主要改这里）
# ============================================================
CUBE_FILE = "./baseline/CRAFTS_-4.7_-350_-150_baseline.fits"

OUTPUT_DIR = (
    "./baseline/SNR=2/output_individual_source_10/"
    "output_merged_source_0.7"
)

MASK_FILE = os.path.join(
    OUTPUT_DIR,
    "masks",
    "all_sources_source_id_map.fits",
)

PHYSICAL_CSV = os.path.join(
    OUTPUT_DIR,
    "source_physical_parameters",
    "all_sources_physical_parameters.csv",
)

SOURCE_IDS_TO_SHOW = [6, 7, 8]
N_SOURCES_TOTAL = 10

X_RANGE = [356.5, 355.6] * u.deg
Y_RANGE = [-6.89, -6.0] * u.deg
VELOCITY_RANGE = (-300.0, -200.0)  # km/s

PATH_WIDTH = 60 * u.arcmin

PATH_CONFIGS = {
    "45deg": {
        "path": Path([(1, 1), (36, 36)], width=PATH_WIDTH),
        "outfile": "pv_45deg_with_pv_source_contours.pdf",
    },
    "135deg": {
        "path": Path([(1, 36), (36, 1)], width=PATH_WIDTH),
        "outfile": "pv_135deg_with_pv_source_contours.pdf",
    },
    "90deg": {
        "path": Path([(18, 0), (18, 36)], width=PATH_WIDTH),
        "outfile": "pv_90deg_with_pv_source_contours.pdf",
    },
    "0deg": {
        "path": Path([(0, 18), (36, 18)], width=PATH_WIDTH),
        "outfile": "pv_0deg_with_pv_source_contours.pdf",
    },
}

# 同一 final source、同一空间 pixel 若对应多个高斯分量：
# True  -> 用 amplitude_K 加权成一个速度点（与 merged_position 一致）
# False -> 保留每条 CSV 记录（同一 pixel 可能在 PV 中出现多个点）
ONE_POINT_PER_SPATIAL_PIXEL = False

# 左图：保留原来的源边界轮廓
DRAW_SOURCE_MASK_CONTOURS_ON_LEFT = True

# 左图：叠加 moment0 contour（由 CSV 中的 moment0_K_kms 构建）
LEFT_MOM0_CONTOUR_LEVELS = [0, 10, 20, 30, 40, 50]
LEFT_MOM0_CONTOUR_CMAP = "Purples_r"
LEFT_MOM0_CONTOUR_ALPHA = 1.0
LEFT_MOM0_CONTOUR_LINEWIDTH = 3

# 右图：注意由于右图是 PV 图，不能直接叠加“2D moment0 contour”；
# 因此这里叠加的是 PV 图本身的亮温 contour。
# contour levels 与右图 0--0.8 K 的色标范围保持一致。
RIGHT_PV_CONTOUR_LEVELS = [0.36,  0.60, 0.84]
RIGHT_PV_CONTOUR_COLOR = "white"
RIGHT_PV_CONTOUR_LINEWIDTH = 1.0

# 右图：用 CSV 中每个源在 PV 平面上的像素位置生成同色轮廓
DRAW_SOURCE_PV_CONTOURS = True
SOURCE_PV_CONTOUR_DILATION = 2
SOURCE_PV_CONTOUR_SMOOTH_SIGMA = 1.0
SOURCE_PV_CONTOUR_LEVEL = 0.35
SOURCE_PV_CONTOUR_LINEWIDTH = 2.5

# 是否同时显示源的散点、中心十字和文字标签
PLOT_ALL_SOURCE_PIXELS_ON_PV = False
PV_SCATTER_SIZE = 10
PV_SCATTER_ALPHA = 0.60
DRAW_SOURCE_CENTER_MARKER_ON_PV = False
SHOW_SOURCE_LABELS_ON_PV = True
PV_MARKER = "+"

# 是否只保留位于 PV path 宽度内的像素
REQUIRE_INSIDE_PATH_WIDTH = True

# 左图和右图的标注偏移
LABEL_OFFSETS_WORLD = {
    6: (-0.12, -0.02),
    7: (0.10, 0.06),
    8: (-0.02, 0.11),
}

LABEL_OFFSETS_PV_PIXEL = {
    6: (2.0, 8.0),
    7: (2.0, -10.0),
    8: (2.0, 10.0),
}

# 是否标记左图中的源中心和文字
SHOW_SOURCE_LABELS_ON_LEFT = False

# 是否标记右图中的源中心和文字
SHOW_SOURCE_LABELS_ON_PV = False
DRAW_SOURCE_CENTER_MARKER_ON_PV = False
# ============================================================
# 2. 工具函数
# ============================================================
def require_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} 不存在：\n{path}\n请检查脚本第 1 部分的路径设置。"
        )



def load_and_crop_mask_to_subcube(mask_path, sub_cube):
    """将最终 source-ID mask 重投影到 subcube 的空间网格。"""
    require_file(mask_path, "Source ID mask")

    mask_data, mask_header = fits.getdata(mask_path, header=True)
    mask_data = np.squeeze(mask_data)

    if mask_data.ndim != 2:
        raise ValueError(
            f"Source ID mask 必须是二维数组，当前 shape={mask_data.shape}"
        )

    n_y, n_x = sub_cube.shape[-2:]

    mask_cropped, _ = reproject_interp(
        (mask_data, WCS(mask_header).celestial),
        sub_cube.wcs.celestial,
        shape_out=(n_y, n_x),
        order=0,
    )

    return np.nan_to_num(mask_cropped, nan=0.0).astype(int)



def get_source_contours(cropped_mask, source_ids):
    """从二维 source-ID mask 中提取左图轮廓。"""
    contour_dict = {}

    for sid in source_ids:
        source_mask = cropped_mask == sid
        if not np.any(source_mask):
            print(f"Warning: Source {sid} 在裁剪后的 mask 中没有像素。")
            continue

        contours = measure.find_contours(source_mask.astype(float), 0.5)
        contour_dict[sid] = contours

        print(
            f"Source {sid}: {np.sum(source_mask)} spatial pixels, "
            f"{len(contours)} contour(s)"
        )

    return contour_dict



def get_fixed_source_colors(n_sources=10):
    """返回与 source-overview 图一致的固定颜色。"""
    fixed_colors = [
        [0.00, 0.00, 0.00, 1.0],
        [0.18, 0.49, 0.74, 1.0],
        [1.00, 0.55, 0.00, 1.0],
        [0.56, 0.82, 0.49, 1.0],
        [0.95, 0.58, 0.58, 1.0],
        [0.60, 0.40, 0.35, 1.0],
        [0.00, 0.20, 0.60, 1.0],
        [1.00, 0.20, 0.20, 1.0],
        [0.55, 0.00, 0.55, 1.0],
        [0.65, 0.82, 0.85, 1.0],
        [0.35, 0.35, 0.35, 1.0],
    ]

    if n_sources + 1 <= len(fixed_colors):
        color_array = np.asarray(fixed_colors[: n_sources + 1])
    else:
        extra_n = n_sources + 1 - len(fixed_colors)
        extra_colors = plt.cm.tab20(np.linspace(0, 1, extra_n))
        color_array = np.vstack([np.asarray(fixed_colors), extra_colors])

    return {sid: color_array[sid] for sid in range(1, n_sources + 1)}



def load_physical_pixel_table(csv_file, source_ids, one_point_per_pixel=True):
    """
    读取 save_source_masks_and_pixel_tables() 生成的总 CSV。

    若 one_point_per_pixel=True：
    - 同一 final source、同一空间像素内多个 component 的 velocity 用 amplitude_K 加权
    - moment0_K_kms 直接求和
    """
    require_file(csv_file, "ROHSA pixel physical-parameter CSV")

    df = pd.read_csv(csv_file)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = {
        "final_source_id",
        "component",
        "x_pixel",
        "y_pixel",
        "ra_deg",
        "dec_deg",
        "amplitude_K",
        "velocity_kms",
    }

    if "moment0_K_kms" not in df.columns:
        if "sigma_kms" in df.columns:
            df["moment0_K_kms"] = (
                pd.to_numeric(df["amplitude_K"], errors="coerce")
                * pd.to_numeric(df["sigma_kms"], errors="coerce")
                * np.sqrt(2.0 * np.pi)
            )
        else:
            raise ValueError(
                "CSV 中没有 moment0_K_kms 列，也没有 sigma_kms 列，"
                "无法构建 moment0 contour。"
            )

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(
            "CSV 缺少以下必要列："
            + ", ".join(missing)
            + "\n当前列名为："
            + ", ".join(df.columns)
        )

    numeric_columns = list(required_columns) + ["moment0_K_kms"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["final_source_id"].isin(source_ids)].copy()
    df = df[
        np.isfinite(df["final_source_id"])
        & np.isfinite(df["x_pixel"])
        & np.isfinite(df["y_pixel"])
        & np.isfinite(df["velocity_kms"])
        & np.isfinite(df["amplitude_K"])
        & np.isfinite(df["moment0_K_kms"])
        & (df["amplitude_K"] > 0)
        & (df["moment0_K_kms"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(f"CSV 中没有找到 Source {source_ids} 的有效 pixel 记录。")

    df["final_source_id"] = df["final_source_id"].astype(int)
    df["x_pixel"] = df["x_pixel"].astype(int)
    df["y_pixel"] = df["y_pixel"].astype(int)

    if not one_point_per_pixel:
        df["n_components_at_pixel"] = 1
        return df.reset_index(drop=True)

    df["weighted_velocity_numerator"] = df["amplitude_K"] * df["velocity_kms"]

    grouped = (
        df.groupby(
            ["final_source_id", "x_pixel", "y_pixel"],
            as_index=False,
            sort=False,
        )
        .agg(
            ra_deg=("ra_deg", "first"),
            dec_deg=("dec_deg", "first"),
            amplitude_K=("amplitude_K", "sum"),
            moment0_K_kms=("moment0_K_kms", "sum"),
            weighted_velocity_numerator=("weighted_velocity_numerator", "sum"),
            n_components_at_pixel=("component", "nunique"),
        )
    )

    grouped["velocity_kms"] = (
        grouped["weighted_velocity_numerator"] / grouped["amplitude_K"]
    )
    grouped.drop(columns=["weighted_velocity_numerator"], inplace=True)

    return grouped.reset_index(drop=True)



def add_subcube_pixel_coordinates(pixel_df, full_cube, sub_cube):
    """根据 CSV 中 RA/Dec，计算每个记录在 subcube 中的像素坐标。"""
    result = pixel_df.copy()

    ra = result["ra_deg"].to_numpy(dtype=float)
    dec = result["dec_deg"].to_numpy(dtype=float)

    bad_world = ~np.isfinite(ra) | ~np.isfinite(dec)

    if np.any(bad_world):
        x_full = result.loc[bad_world, "x_pixel"].to_numpy(dtype=float)
        y_full = result.loc[bad_world, "y_pixel"].to_numpy(dtype=float)
        fallback_ra, fallback_dec = full_cube.wcs.celestial.pixel_to_world_values(
            x_full,
            y_full,
        )
        ra[bad_world] = fallback_ra
        dec[bad_world] = fallback_dec

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x_sub, y_sub = sub_cube.wcs.celestial.world_to_pixel(coords)

    result["x_subpixel"] = np.asarray(x_sub, dtype=float)
    result["y_subpixel"] = np.asarray(y_sub, dtype=float)

    n_y, n_x = sub_cube.shape[-2:]
    inside_subcube = (
        np.isfinite(result["x_subpixel"])
        & np.isfinite(result["y_subpixel"])
        & (result["x_subpixel"] >= -0.5)
        & (result["x_subpixel"] <= n_x - 0.5)
        & (result["y_subpixel"] >= -0.5)
        & (result["y_subpixel"] <= n_y - 0.5)
    )

    result = result[inside_subcube].copy().reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "选定源的 CSV 坐标均不在当前 subcube 范围内。"
            "请检查 X_RANGE、Y_RANGE 与 CSV/FITS 是否对应同一个 cube。"
        )

    return result



def build_moment0_map_from_table(pixel_df, sub_cube_shape):
    """根据 subcube 中的像素坐标构建 2D moment0 图。"""
    n_y, n_x = sub_cube_shape[-2:]
    moment0_map = np.zeros((n_y, n_x), dtype=float)

    x_int = np.rint(pixel_df["x_subpixel"].to_numpy(dtype=float)).astype(int)
    y_int = np.rint(pixel_df["y_subpixel"].to_numpy(dtype=float)).astype(int)
    mom0 = pixel_df["moment0_K_kms"].to_numpy(dtype=float)

    good = (
        np.isfinite(mom0)
        & (mom0 > 0)
        & (x_int >= 0)
        & (x_int < n_x)
        & (y_int >= 0)
        & (y_int < n_y)
    )

    for x, y, val in zip(x_int[good], y_int[good], mom0[good]):
        moment0_map[y, x] += val

    return moment0_map



def compute_auto_contour_levels(image, nlevels=6):
    """自动生成 contour levels，适合 moment0 图。"""
    valid = image[np.isfinite(image) & (image > 0)]
    if valid.size == 0:
        return None

    vmin = np.nanpercentile(valid, 30)
    vmax = np.nanpercentile(valid, 98)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = np.nanmin(valid)
        vmax = np.nanmax(valid)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None

    return np.linspace(vmin, vmax, nlevels)


# ============================================================
# 3. Path 投影与 PV 坐标转换
# ============================================================
def celestial_pixel_scale_arcmin(celestial_wcs):
    scales_deg = np.abs(proj_plane_pixel_scales(celestial_wcs)) * u.deg
    return np.mean(scales_deg).to(u.arcmin)



def path_width_in_pixels(path, celestial_wcs):
    if path.width is None:
        return 0.0
    if isinstance(path.width, u.Quantity):
        return (path.width.to(u.arcmin) / celestial_pixel_scale_arcmin(celestial_wcs)).value
    return float(path.width)



def project_table_pixels_to_path(pixel_df, path, celestial_wcs):
    """
    将 CSV 中每个空间 pixel 投影到 PV path。

    输出新增：
        offset_pixel            : PV 图 x 方向的像素坐标
        distance_to_path_pixel  : 到 path 中心线的垂直距离（空间 pixel）
    """
    path_x, path_y = path.sample_points(spacing=1)
    width_pix = path_width_in_pixels(path, celestial_wcs)
    half_width_pix = 0.5 * width_pix

    offsets = np.empty(len(pixel_df), dtype=float)
    distances_to_path = np.empty(len(pixel_df), dtype=float)

    x_values = pixel_df["x_subpixel"].to_numpy(dtype=float)
    y_values = pixel_df["y_subpixel"].to_numpy(dtype=float)

    for i, (x, y) in enumerate(zip(x_values, y_values)):
        distances = np.hypot(path_x - x, path_y - y)
        nearest_index = int(np.argmin(distances))
        offsets[i] = nearest_index
        distances_to_path[i] = distances[nearest_index]

    result = pixel_df.copy()
    result["offset_pixel"] = offsets
    result["distance_to_path_pixel"] = distances_to_path

    if REQUIRE_INSIDE_PATH_WIDTH and width_pix > 0:
        result = result[result["distance_to_path_pixel"] <= half_width_pix].copy()

    return result.reset_index(drop=True)



def velocity_kms_to_pv_pixel(velocity_kms, pv_header):
    """将 km/s 转为 PV 图 y 方向的 0-based 像素坐标。"""
    crpix2 = float(pv_header["CRPIX2"])
    crval2 = float(pv_header["CRVAL2"])
    cdelt2 = float(pv_header["CDELT2"])
    cunit2 = str(pv_header.get("CUNIT2", "m/s")).strip()

    native_unit = u.Unit(cunit2) if cunit2 else (u.m / u.s)
    velocity_native = (np.asarray(velocity_kms) * u.km / u.s).to_value(native_unit)

    return (velocity_native - crval2) / cdelt2 + crpix2 - 1.0


def build_smoothed_pv_source_mask(source_points, pv_shape):
    """
    根据该源在 PV 图中的 (offset_pixel, velocity_pixel) 构建平滑占据图。

    轮廓表示：位于当前 60 arcmin path 宽度内、并由 ROHSA velocity_kms
    投影到 PV 平面后的源像素分布。它不是把左图 RA-Dec 轮廓直接复制到右图。
    """
    occupancy = np.zeros(pv_shape, dtype=bool)

    xpix = np.rint(
        source_points["offset_pixel"].to_numpy(dtype=float)
    ).astype(int)
    ypix = np.rint(
        source_points["velocity_pixel"].to_numpy(dtype=float)
    ).astype(int)

    good = (
        (xpix >= 0)
        & (xpix < pv_shape[1])
        & (ypix >= 0)
        & (ypix < pv_shape[0])
    )

    occupancy[ypix[good], xpix[good]] = True

    if not np.any(occupancy):
        return None

    if SOURCE_PV_CONTOUR_DILATION > 0:
        occupancy = binary_dilation(
            occupancy,
            iterations=SOURCE_PV_CONTOUR_DILATION,
        )

    # 闭运算可以连接相邻的小间隙，减少锯齿和断裂
    occupancy = binary_closing(occupancy, iterations=1)

    smooth_map = gaussian_filter(
        occupancy.astype(float),
        sigma=SOURCE_PV_CONTOUR_SMOOTH_SIGMA,
    )

    max_value = np.nanmax(smooth_map)
    if not np.isfinite(max_value) or max_value <= 0:
        return None

    return smooth_map / max_value


# ============================================================
# 4. 单个两联图绘制函数
# ============================================================
def make_one_pv_figure(
    path_name,
    path,
    outfile,
    sub_cube_slab,
    max_map,
    source_contours,
    source_pixel_table,
    moment0_map,
    colors,
):
    print("\n" + "=" * 72)
    print(f"Making figure: {path_name}")
    print("=" * 72)

    pvdiagram = extract_pv_slice(cube=sub_cube_slab, path=path, spacing=1)
    pv_data = np.nan_to_num(pvdiagram.data.copy(), nan=0.0)
    pv_wcs = WCS(pvdiagram.header)

    projected_table = project_table_pixels_to_path(
        source_pixel_table,
        path,
        sub_cube_slab.wcs.celestial,
    )

    if not projected_table.empty:
        projected_table["velocity_pixel"] = velocity_kms_to_pv_pixel(
            projected_table["velocity_kms"].to_numpy(dtype=float),
            pvdiagram.header,
        )

        valid = (
            np.isfinite(projected_table["offset_pixel"])
            & np.isfinite(projected_table["velocity_pixel"])
            & (projected_table["offset_pixel"] >= 0)
            & (projected_table["offset_pixel"] <= pv_data.shape[1] - 1)
            & (projected_table["velocity_pixel"] >= 0)
            & (projected_table["velocity_pixel"] <= pv_data.shape[0] - 1)
            & (projected_table["velocity_kms"] >= VELOCITY_RANGE[0])
            & (projected_table["velocity_kms"] <= VELOCITY_RANGE[1])
        )
        projected_table = projected_table[valid].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=1.0)

    # --------------------------------------------------------
    # 左图：保持原来的画法 + 叠加 moment0 contour
    # --------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 0], projection=sub_cube_slab.wcs.celestial)

    # moment0_map 由整个 sub_cube_slab 积分得到，包含当前范围内的所有数据，
    # 不再只显示 CSV 中 Source 6/7/8 所占据的像素。
    left_image = np.asarray(moment0_map, dtype=float)

    im_map = ax_map.imshow(
        left_image,
        vmin=0.0,
        vmax=60.0,
        origin="lower",
        cmap="viridis",
    )

    path.show_on_axis(
        ax_map,
        spacing=0.6,
        edgecolor="w",
        linestyle=":",
        linewidth=2.0,
    )

    if DRAW_SOURCE_MASK_CONTOURS_ON_LEFT:
        used_labels = set()
        for sid in SOURCE_IDS_TO_SHOW:
            if sid not in source_contours:
                continue
            for cnt in source_contours[sid]:
                label = f"Source {sid}" if sid not in used_labels else None
                ax_map.plot(
                    cnt[:, 1],
                    cnt[:, 0],
                    color=colors[sid],
                    linewidth=2.8,
                    linestyle="-",
                    transform=ax_map.get_transform("pixel"),
                    label=label,
                    zorder=12,
                )
                used_labels.add(sid)

        if len(used_labels) > 0:
            legend_left = ax_map.legend(
                loc="upper left",
                fontsize=22,
                frameon=True,
                framealpha=0.85,       # 完全不透明，避免后面的图层透出来
                facecolor="white",
                edgecolor="black",
                fancybox=False,
            )
        
            # 保证整个 legend，包括文字、线条和外框，都在所有图层最上方
            legend_left.set_zorder(1000)

        

    # 左图 moment0 contour
    contour1 = ax_map.contour(
        moment0_map,
        cmap=LEFT_MOM0_CONTOUR_CMAP,
        alpha=LEFT_MOM0_CONTOUR_ALPHA,
        linewidths=LEFT_MOM0_CONTOUR_LINEWIDTH,
        levels=LEFT_MOM0_CONTOUR_LEVELS,
        origin="lower",
        transform=ax_map.get_transform("pixel"),
        zorder=10,
    )
    print(f"  Left-panel moment0 contours: {np.array(LEFT_MOM0_CONTOUR_LEVELS)}")

    # 左图是否标记源中心位置和文字
    if SHOW_SOURCE_LABELS_ON_LEFT:
        for sid in SOURCE_IDS_TO_SHOW:
            sid_rows = source_pixel_table[
                source_pixel_table["final_source_id"] == sid
            ]
    
            if sid_rows.empty:
                continue
    
            ra_med = np.nanmedian(
                sid_rows["ra_deg"].to_numpy(dtype=float)
            )
            dec_med = np.nanmedian(
                sid_rows["dec_deg"].to_numpy(dtype=float)
            )
    
            ax_map.plot(
                ra_med,
                dec_med,
                marker="+",
                color="white",
                markersize=10,
                markeredgewidth=2.0,
                linestyle="None",
                transform=ax_map.get_transform("world"),
                zorder=14,
            )
    
            dra, ddec = LABEL_OFFSETS_WORLD.get(
                sid,
                (0.08, 0.06),
            )
    
            ax_map.text(
                ra_med + dra,
                dec_med + ddec,
                f"Source {sid}",
                color=colors[sid],
                fontsize=16,
                weight="bold",
                ha="center",
                va="center",
                transform=ax_map.get_transform("world"),
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor=colors[sid],
                    alpha=0.85,
                ),
                zorder=15,
            )

    pos_map = ax_map.get_position()
    cb1_ax = fig.add_axes([pos_map.x1 + 0.02, pos_map.y0, 0.02, pos_map.height])
    cb1 = plt.colorbar(mappable=im_map, cax=cb1_ax)
    cb1.set_ticks(np.arange(0, 51, 10))
    cb1.set_label(r"Integrated Intensity [K km s$^{-1}$]", size=32, labelpad=12)
    cb1.ax.tick_params(labelsize=30, direction="in")

    ra_coord = ax_map.coords[0]
    dec_coord = ax_map.coords[1]
    ra_coord.set_axislabel("Right Ascension (J2000)", fontsize=38, minpad=1)
    dec_coord.set_axislabel("Declination (J2000)", fontsize=38, minpad=1)
    ra_coord.set_ticklabel(size=30)
    dec_coord.set_ticklabel(size=30)
    ra_coord.set_ticks(direction="out")
    dec_coord.set_ticks(direction="out")
    ra_coord.set_format_unit(u.hourangle)
    ax_map.grid(True, linestyle=":", alpha=0.5)

    # --------------------------------------------------------
    # 右图：PV 图 + PV contour + 标记源位置
    # --------------------------------------------------------
    ax_pv = fig.add_subplot(gs[0, 1], projection=pv_wcs)

    im_pv = ax_pv.imshow(
        pv_data,
        vmin=0.0,
        vmax=0.8,
        origin="lower",
        cmap="viridis",
    )

    if path_name in ["0deg", "90deg"]:
        ax_pv.set_aspect(0.11)
    else:
        ax_pv.set_aspect(0.15)

    # 右图 contour：这里必须是 PV 亮温 contour，而不是 2D moment0 contour。
    # 使用 spring colormap 区分不同等高线，并且不在等高线上显示数值标签。
    try:
        cs_pv = ax_pv.contour(
            pv_data,
            levels=RIGHT_PV_CONTOUR_LEVELS,
            colors=RIGHT_PV_CONTOUR_COLOR,
            linewidths=RIGHT_PV_CONTOUR_LINEWIDTH,
            origin="lower",
            transform=ax_pv.get_transform("pixel"),
            zorder=8,
        )

        # 不调用 ax_pv.clabel，因此不显示 0.36 K、0.60 K、0.84 K 等文字。
        print(f"  Right-panel PV contours: {np.array(RIGHT_PV_CONTOUR_LEVELS)}")
    except Exception as e:
        print(f"  Warning: 右图 contour 绘制失败: {e}")

    for sid in SOURCE_IDS_TO_SHOW:
        source_points = projected_table[
            projected_table["final_source_id"] == sid
        ]

        if source_points.empty:
            print(f"  Source {sid}: 这条 path 宽度内没有可绘制的 pixel。")
            continue

        # ----------------------------------------------------
        # 1. 用源像素在 PV 平面中的分布绘制彩色轮廓
        # ----------------------------------------------------
        if DRAW_SOURCE_PV_CONTOURS:
            source_pv_mask = build_smoothed_pv_source_mask(
                source_points,
                pv_data.shape,
            )

            if source_pv_mask is not None:
                ax_pv.contour(
                    source_pv_mask,
                    levels=[SOURCE_PV_CONTOUR_LEVEL],
                    colors=[colors[sid]],
                    linewidths=[SOURCE_PV_CONTOUR_LINEWIDTH],
                    linestyles=["-"],
                    origin="lower",
                    transform=ax_pv.get_transform("pixel"),
                    zorder=13,
                )
            else:
                print(f"  Warning: Source {sid} 无法生成 PV source contour。")

        # ----------------------------------------------------
        # 2. 可选：显示该源所有像素散点
        # ----------------------------------------------------
        if PLOT_ALL_SOURCE_PIXELS_ON_PV:
            ax_pv.scatter(
                source_points["offset_pixel"].to_numpy(dtype=float),
                source_points["velocity_pixel"].to_numpy(dtype=float),
                s=PV_SCATTER_SIZE,
                marker="o",
                c=[colors[sid]],
                alpha=PV_SCATTER_ALPHA,
                edgecolors="none",
                transform=ax_pv.get_transform("pixel"),
                zorder=14,
            )

        # 代表位置使用源像素在 PV 平面中的中位数
        x_med = np.nanmedian(source_points["offset_pixel"])
        y_med = np.nanmedian(source_points["velocity_pixel"])

        if DRAW_SOURCE_CENTER_MARKER_ON_PV:
            ax_pv.plot(
                x_med,
                y_med,
                marker=PV_MARKER,
                color=colors[sid],
                markersize=12,
                markeredgewidth=2.2,
                linestyle="None",
                transform=ax_pv.get_transform("pixel"),
                zorder=15,
            )

        if SHOW_SOURCE_LABELS_ON_PV:
            dx, dy = LABEL_OFFSETS_PV_PIXEL.get(sid, (2.0, 8.0))
            ax_pv.text(
                x_med + dx,
                y_med + dy,
                f"Source {sid}",
                color=colors[sid],
                fontsize=16,
                fontweight="bold",
                ha="left",
                va="center",
                transform=ax_pv.get_transform("pixel"),
                zorder=16,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=colors[sid],
                    linewidth=1.0,
                    alpha=0.80,
                ),
            )

        print(
            f"  Source {sid}: {len(source_points)} pixels in path; "
            f"velocity range = [{source_points['velocity_kms'].min():.2f}, "
            f"{source_points['velocity_kms'].max():.2f}] km/s"
        )

    pv_x_coord = ax_pv.coords[0]
    pv_x_coord.set_format_unit(u.deg)
    pv_x_coord.set_ticks(values=np.array([0.0, 0.4, 0.8, 1.2]) * u.deg)
    pv_x_coord.set_major_formatter("x.x")
    pv_x_coord.tick_params(axis="both", which="both", direction="out", labelsize=30)

    pv_y_coord = ax_pv.coords[1]
    pv_y_coord.set_format_unit(u.km / u.s)
    pv_y_coord.tick_params(axis="both", which="both", direction="out", labelsize=30)

    pos_pv = ax_pv.get_position()
    cb2_ax = fig.add_axes([pos_pv.x1 + 0.02, pos_pv.y0, 0.02, pos_pv.height])
    cb2 = plt.colorbar(mappable=im_pv, cax=cb2_ax)
    cb2.set_ticks(np.arange(0.0, 0.81, 0.2))
    cb2.set_label("Brightness Temp. [K]", size=32, labelpad=12)
    cb2.ax.tick_params(labelsize=30, direction="in")

    ax_pv.set_ylabel("Velocity [km s$^{-1}$]", fontsize=38)
    ax_pv.set_xlabel("Offset [°]", fontsize=38)

    plt.tight_layout()
    out_pdf = os.path.join(OUTPUT_DIR, outfile)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    return out_pdf


# ============================================================
# 5. 主程序
# ============================================================
def main():
    require_file(CUBE_FILE, "FITS cube")
    require_file(MASK_FILE, "Source ID mask")
    require_file(PHYSICAL_CSV, "Physical-parameter CSV")

    print("\n" + "=" * 72)
    print("READING DATA")
    print("=" * 72)

    cube = SpectralCube.read(CUBE_FILE).with_spectral_unit(u.km / u.s)
    print(cube)

    sub_cube = cube.subcube(
        xlo=X_RANGE[0],
        xhi=X_RANGE[1],
        ylo=Y_RANGE[0],
        yhi=Y_RANGE[1],
    )

    sub_cube_slab = sub_cube.spectral_slab(
        VELOCITY_RANGE[0] * u.km / u.s,
        VELOCITY_RANGE[1] * u.km / u.s,
    )

    max_map = sub_cube_slab.max(axis=0).value
    print(f"sub_cube_slab shape = {sub_cube_slab.shape}")

    print("\nLoading and cropping mask ...")
    mask_cropped = load_and_crop_mask_to_subcube(MASK_FILE, sub_cube)
    source_contours = get_source_contours(mask_cropped, SOURCE_IDS_TO_SHOW)

    print("\nLoading source pixel CSV ...")
    source_pixel_table = load_physical_pixel_table(
        PHYSICAL_CSV,
        SOURCE_IDS_TO_SHOW,
        one_point_per_pixel=ONE_POINT_PER_SPATIAL_PIXEL,
    )
    source_pixel_table = add_subcube_pixel_coordinates(
        source_pixel_table,
        full_cube=cube,
        sub_cube=sub_cube,
    )

    for sid in SOURCE_IDS_TO_SHOW:
        n_rows = np.sum(source_pixel_table["final_source_id"] == sid)
        print(f"  Source {sid}: {n_rows} spatial rows after loading/grouping")

    print("\nBuilding full-field moment0 map from the complete subcube slab ...")
    # 对当前 RA-Dec 范围以及 VELOCITY_RANGE 内的整个数据立方积分。
    # 这样左图显示所有观测数据，而不是只显示被选中源的 CSV 像素。
    moment0_map = np.asarray(
        sub_cube_slab.moment(order=0).value,
        dtype=float,
    )
    moment0_map = np.squeeze(moment0_map)

    if moment0_map.shape != sub_cube.shape[-2:]:
        raise ValueError(
            f"Full moment0 map shape {moment0_map.shape} does not match "
            f"subcube spatial shape {sub_cube.shape[-2:]}"
        )

    print(
        "  Full moment0 range: "
        f"{np.nanmin(moment0_map):.3f} to {np.nanmax(moment0_map):.3f} K km/s"
    )

    source_colors_all = get_fixed_source_colors(N_SOURCES_TOTAL)
    colors = {sid: source_colors_all[sid] for sid in SOURCE_IDS_TO_SHOW}

    saved_files = {}
    for path_name, cfg in PATH_CONFIGS.items():
        out_pdf = make_one_pv_figure(
            path_name=path_name,
            path=cfg["path"],
            outfile=cfg["outfile"],
            sub_cube_slab=sub_cube_slab,
            max_map=max_map,
            source_contours=source_contours,
            source_pixel_table=source_pixel_table,
            moment0_map=moment0_map,
            colors=colors,
        )
        saved_files[path_name] = out_pdf

    print("\n" + "=" * 72)
    print("ALL FIGURES FINISHED")
    print("=" * 72)
    for k, v in saved_files.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
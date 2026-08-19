# HVC

Python utilities and exploratory notebooks for analyzing high-velocity cloud
(HVC) H I emission with ROHSA Gaussian decomposition products. The workflow
loads a FITS spectral cube and ROHSA `.dat` outputs, filters and merges
Gaussian components into final sources, creates source masks and pixel-level
tables, computes physical source parameters, and generates WCS and
position-velocity diagnostic figures.

## Repository Contents

- `merge_utils.py` - reads filtered ROHSA Gaussian parameters, extracts
  component-level sources, merges them into final sources, writes masks and
  per-pixel tables, and creates overview/composition plots.
- `calculate_hvc_physical_parameters_v2.py` - computes integrated H I
  quantities for merged sources, including moment-0, column density, integrated
  flux, H I mass, and associated uncertainties.
- `source_summary_2x2_auto_aspect_v2.py` - builds fixed-size 2x2 source
  summary PDFs with full-field and zoomed Moment 0, Moment 1, and FWHM panels.
- `pv.py` - extracts and plots PV diagrams for selected source groups and
  overlays source contours in PV space.
- `all.ipynb` - documented all-sky context, moment-map comparison, PV, peak
  detection, uncertainty, and velocity-gradient analysis notebook.
- `all-ROHSA.ipynb` - exploratory ROHSA conversion, filtering, source extraction,
  source merging, physical-parameter calculation, and Gaussian-fit inspection
  notebook. Each processing stage and function is documented in English.

## Data Availability

All observational data used by this workflow are publicly available from
official data repositories. The repository does not redistribute either the
full survey products or analysis-ready cutouts. Users must download the source
data, select the required sky and velocity ranges, perform the preparation
described below, and save the resulting FITS files under `data/processed/`.

- **CRAFTS:** The CRAFTS data used in this work are publicly available from the
  [Science Data Bank](https://www.scidb.cn/detail?dataSetId=7fe34782f6ee4284aa22fc68ab8ab6cd)
  and should be cited using the reference key `Li2024CRAFTSdata`.
- **HI4PI:** The corresponding HI4PI data used for comparison are publicly
  available from the
  [VizieR repository (J/A+A/594/A116)](https://cdsarc.cds.unistra.fr/ftp/J/A+A/594/A116/CUBES/EQ2000/CAR/)
  and should be cited using the catalog identifier
  `vizier:J/A+A/594/A116`.

Users are responsible for consulting the repository records for the current
data-release documentation, citation instructions, and applicable terms of
use.

## Required Data Preparation

The files supplied by the official archives are broader survey products rather
than the exact analysis cubes consumed by these notebooks. Prepare local
cutouts before running the workflow.

For the principal CRAFTS/HI4PI comparison, use the following analysis limits:

| Quantity | Required range or preparation |
| --- | --- |
| Right ascension (ICRS, J2000) | approximately 352.8 to 356.8 deg |
| Declination (ICRS, J2000) | approximately -6.89 to -4.69 deg |
| Spectral velocity | -350 to -150 km/s |
| Intensity unit | brightness temperature in K |
| CRAFTS preparation | spatial/spectral crop, masking as required, and baseline subtraction |
| HI4PI preparation | matching spatial/spectral crop and, for pixel comparison, reprojection to the CRAFTS analysis grid |

Preserve valid celestial and spectral WCS metadata, beam information, and
physical units in the output FITS headers. If the downloaded files use
different axis ordering or intensity units, normalize them before running the
notebooks. The code assumes that the processed spectral axis can be converted
to km/s by Astropy.

The public HI4PI column-density product used for the Magellanic-system context
map may remain an all-sky image; `all.ipynb` performs its display-region crop
through the FITS WCS.

Save the prepared files using this repository-relative layout:

```text
data/
  raw/
    HI4PI_HVC_column_density.fits
  processed/
    CRAFTS_cutout_baseline_K.fits
    HI4PI_cutout_baseline_K.fits
baseline/
  CRAFTS_cube.dat
  ROHSA_3ngauss_3D_1.1.1.0.dat
  SNR=2/
    output_file_2sigma/
    output_individual_source_10/
      output_merged_source_0.7/
outputs/
  figures/
  tables/
```

The `baseline/` tree contains derived ROHSA products, not downloaded survey
data. If a different local layout is preferred, edit the centralized path
constants in the first code cell of `all.ipynb` and the configuration blocks in
the ROHSA workflow.

## Python Dependencies

The main scripts use:

```text
numpy
pandas
matplotlib
scipy
scikit-image
astropy
spectral-cube
radio-beam
pvextractor
reproject
ROHSApy
```

A typical environment setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-image astropy spectral-cube radio-beam pvextractor reproject
```

`all-ROHSA.ipynb` additionally requires the `ROHSApy` Python interface and a
compiled ROHSA executable. Install these components following the upstream
ROHSA documentation, then update the executable path in the notebook if it is
not located at `./ROHSA/src/ROHSA`.

## Typical Workflow

1. Download CRAFTS and HI4PI from the official repositories, crop and prepare
   the analysis cubes, and place them under `data/processed/` using the names
   shown above.
2. Run `all.ipynb` for the CRAFTS/HI4PI moment-map, PV, peak-detection, and
   velocity-gradient analysis.
3. Run `all-ROHSA.ipynb` from top to bottom to generate the ROHSA input,
   execute the decomposition, apply S/N filtering, identify component-level
   sources, and invoke `merge_utils.py` for final source merging.
4. Run `calculate_hvc_physical_parameters_v2.py` to compute physical source
   parameters from the merged-source CSV outputs.
5. Run `source_summary_2x2_auto_aspect_v2.py` to generate per-source summary
   PDFs.
6. Run `pv.py` to create PV diagrams for selected sources and path
   orientations.

Each script has user-editable parameters near the top. Check SNR thresholds,
merge thresholds, source IDs, velocity ranges, and output directories before
running a full analysis.

## Notes

- Saved notebook outputs have been cleared so the notebooks are lighter and do
  not preserve stale local runtime messages.
- The code assumes the relevant FITS velocity-axis values are already in km/s in
  the ROHSA conversion workflow.
- Generated figures and tables from `all.ipynb` are written under `outputs/`;
  ROHSA masks, CSV files, and `.dat` products are written under the configured
  `baseline/` analysis directories.

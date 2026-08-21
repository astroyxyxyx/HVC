# HVC

Python notebooks and utilities for analyzing high-velocity cloud (HVC) H I emission in the Magellanic Stream using FAST/CRAFTS data, HI4PI comparison products, and ROHSA Gaussian decomposition.

The repository contains two main notebooks. `all.ipynb` performs masking, unit conversion, CRAFTS/HI4PI comparison, moment-map and position-velocity analysis, peak finding, and velocity-gradient measurements. `all-ROHSA.ipynb` performs the ROHSA decomposition and subsequent component filtering, source identification, source merging, physical-parameter calculation, PV analysis, and spectrum inspection.

## Repository Contents
- `gass-LA.par` — use for SoFiA.
- `all.ipynb` — applies the SoFiA-derived mask to the CRAFTS cube, converts Jy/beam to K, creates the HI4PI Magellanic-system context map, compares CRAFTS and HI4PI moment products, constructs PV diagnostics, detects spectral peaks, estimates uncertainties, and measures velocity gradients.
- `all-ROHSA.ipynb` — runs the ROHSA decomposition, converts fitted Gaussian parameters to physical units, filters low-S/N components, identifies component-level sources, merges related sources, launches the physical-parameter and PV scripts, and inspects selected Gaussian fits.
- `merge_utils_Copy1.py` — source-merging utilities currently imported by `all-ROHSA.ipynb`.
- `calculate_hvc_physical_parameters_v2.py` — computes integrated H I observables and associated uncertainties for the merged sources.
- `source_summary_2x2_auto_aspect_v2.py` — generates per-source summary figures.
- `pv.py` — generates position-velocity diagrams for configured source groups and path orientations.


## Data Availability

The repository does not redistribute the full survey products. Obtain the public data from the official repositories and prepare the required local FITS products before running the notebooks.

- **CRAFTS:** public data are available from the Science Data Bank: https://www.scidb.cn/detail?dataSetId=7fe34782f6ee4284aa22fc68ab8ab6cd . Follow the repository record for the current citation instructions.
- **HI4PI:** public data are available from the CDS/VizieR archive for J/A+A/594/A116: https://cdsarc.cds.unistra.fr/ftp/J/A+A/594/A116/CUBES/EQ2000/CAR/ .

## Required Data Preparation

For the principal CRAFTS analysis, prepare a baseline-subtracted cube covering approximately:

| Quantity | Analysis range |
| --- | --- |
| Right ascension (ICRS, J2000) | about 352.8–356.8 deg |
| Declination (ICRS, J2000) | about -6.89 to -4.69 deg |
| Spectral velocity | -350 to -150 km/s |
| Final analysis unit | brightness temperature, K |

Preserve the celestial and spectral WCS, physical units, beam information, and spectral metadata in all FITS headers.

### Files used by `all.ipynb`

The first section of `all.ipynb` expects a CRAFTS mask and a matching unmasked Jy/beam cube:

```text
CRAFTS_-4.7_-350_-150_baseline_Jy_mask.fits
CRAFTS_-4.7_-350_-150_baseline.fits_Jy.fits
```

It writes an intermediate masked cube:

```text
CRAFTS_-4.7_-350_-150_baseline.fits_Jy---mask.fits
```

and converts that product to brightness temperature, producing:

```text
CRAFTS_-4.7_-350_-150_baseline_Jy_mask----mask_K.fits
```

Later comparison cells currently read:

```text
CRAFTS_-4.7_-350_-150_baseline_Jy_mask----mask-K.fits
HI4PI_-4.7_-350_-150_baseline_Jy_mask----mask-K.fits
```

The optional Magellanic-system context-map section also requires:

```text
hi4pi-hvc-nhi-mag-car.fits
```

This is an all-sky HI4PI HVC column-density map and is separate from the local HI4PI comparison cube.

### Files used by `all-ROHSA.ipynb`

The ROHSA workflow begins from:

```text
CRAFTS_-4.7_-350_-150_baseline.fits
```

It writes/uses the main ROHSA products:

```text
CRAFTS_cube.dat
ROHSA_3ngauss_3D_1.1.1.0.dat
parameters.txt
```

The current S/N=2 source-filtering workflow uses paths under:

```text
baseline/SNR=2/
```

including:

```text
baseline/SNR=2/output_file_2sigma/filtered_rohsa.dat
baseline/SNR=2/output_individual_source_10/source_filtered_rohsa_pixel.dat
baseline/SNR=2/output_individual_source_10/output_merged_source_0.7/
```

Before reuse, check the S/N threshold, minimum pixel count, overlap requirement, velocity-merging factor, and all input/output paths in the notebook configuration cells.

## Python Dependencies

The notebooks use the following main packages:

```text
numpy
pandas
matplotlib
scipy
astropy
spectral-cube
radio-beam
pvextractor
reproject
ROHSApy
```

`all-ROHSA.ipynb` also requires a compiled ROHSA executable. The notebook currently calls:

```text
./ROHSA/src/ROHSA parameters.txt
```

so the executable must exist at `./ROHSA/src/ROHSA` relative to the notebook working directory.

## Typical Workflow

1. Download the CRAFTS and HI4PI survey products and prepare the required spatial/velocity cutouts locally.
2. Prepare the SoFiA-derived CRAFTS mask and the Jy/beam input cube required by the first section of `all.ipynb`.
3. Run `all.ipynb` to create the masked/converted CRAFTS product and perform CRAFTS/HI4PI moment-map, PV, peak-velocity, uncertainty, and velocity-gradient analyses.
4. Run `all-ROHSA.ipynb` from top to bottom to create the ROHSA input, run the Gaussian decomposition, filter components, identify component-level sources, and merge them into final sources.
5. The ROHSA notebook invokes `calculate_hvc_physical_parameters_v2.py`, `source_summary_2x2_auto_aspect_v2.py`, and `pv.py` for the downstream source analysis.
6. Inspect selected spectra and Gaussian residuals using the final cells of `all-ROHSA.ipynb`.

## Requirements for a Complete Run

This repository is not fully runnable immediately after cloning because the
survey data and the compiled ROHSA executable are not distributed with the
source code. Before running the complete workflow, users must:

1. Download the required CRAFTS and HI4PI data from the official repositories
   listed in the Data Availability section.
2. Prepare the spatial and velocity cutouts, masks, and intermediate FITS
   products described above.
3. Create a Python environment and install all required dependencies.
4. Compile ROHSA and make its executable available at the location configured
   in `all-ROHSA.ipynb`.
5. Review every input and output path in the notebooks and scripts. The example
   paths in this repository reflect the authors' local data layout and must be
   changed to match the location and filenames of each user's downloaded data.

The full analysis should be run only after these data, environment, executable,
and path requirements have been satisfied.

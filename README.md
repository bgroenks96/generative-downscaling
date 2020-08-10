## Generative downscaling

This repository contains experiment code for the Master's thesis, *ClimAlign: Unsupservised statistical downscaling of climate variables via normalizing flows* ([ProQuest](https://search.proquest.com/openview/716b86a33eb0f7a609be1dfa7f0bae8b/1?pq-origsite=gscholar&cbl=18750&diss=y), [Full text](https://drive.google.com/file/d/18BuGh3xHyLh8NDFyoI35WfE_aF2JcZQP/view)).

Pre-formatted datasets are not currently available from any public sources. However, the raw data for [ERA-interim](https://climatedataguide.ucar.edu/climate-data/era-interim) and [Rasmussen/WRF](https://rda.ucar.edu/datasets/ds612.0/#!) can be downloaded from NCAR's servers.

The code in this repository uses [`xarray`](http://xarray.pydata.org/en/stable/) and `dask`. The data is assumed to be in [ZARR](https://zarr.readthedocs.io/en/stable/) format. You can use `xarray` to convert NetCDF files into ZARR datasets.

### Overview

Data loaders are provided by `datasource.py`. `EraiRasDataLoader` and `NoaaLivnehDataLoader` provide functions which return file mappings that can be passed to functions such as `xarray`'s `open_zarr`. See the source code in this file for the expected ZARR naming conventions. A Google Cloud service account key file with GCS access must be copied to the repository root directory and named `gcs.secret.json`.

The `*-downscaling-*` Jupyter notebooks contain experimental code for testing the baseline and ClimAlign models. The `qualitative-analysis` and `quantitative-analysis` notebooks contain the code used to produce the figures and tables in the paper. The `experiments` module contains the experiment scripts for each model and experiment set. The `core-experiment-suite.sh` runs all experiments for the primary quantiative results. Results are stored locally using MLflow.

The implementation of the ClimAlign model (referred to in this code as Joint Flow-based Latent Variable Model, JFLVM) can be found in the `normalizing-flows` git-submodule.

Baseline implementations can be found in the `baselines` directory/module.

Necessary packages are specified by the conda `envirionment.yml` file.

Please send any inquiries to brian.groenke@colorado.edu.

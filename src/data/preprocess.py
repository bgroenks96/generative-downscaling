" Transforms a directory of raw WeatherBench data into a single training-ready array on disk "

import argparse
import logging
import os
import shutil

from typing import List

import numpy as np
import xarray as xr

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


EPSILON = .0001
NA_LAT_FINE = (-5, 85+EPSILON)  # epsilon to avoid non-inclusive right
NA_LON_FINE = (-137.5+180, -47.5+180+EPSILON)

NA_LAT_COARSE = (-10, 88+EPSILON)  # Coarse should be wider for full coverage
NA_LON_COARSE = (-145+180, -45+180+EPSILON)

KELVIN_TO_CELSIUS = -273.15


def north_america_only(weatherbench: xr.Dataset, res: str) -> xr.Dataset:
    """ Filters the passed WEATHERBENCH dataset to North American coordinates only

    Arguments:
      * weatherbench: a dataset derived from WeatherBench (but with centered longitude)
      * res: resolution of the data. Either "1406" (1.4 deg) or "5625" (5.6 deg)
    """
    if res == "1406":
        return weatherbench.sel(lat=slice(*NA_LAT_FINE), lon=slice(*NA_LON_FINE))
    elif res == "5625":
        return weatherbench.sel(lat=slice(*NA_LAT_COARSE), lon=slice(*NA_LON_COARSE))
    else:
        raise ValueError(f"Unrecognized resolution '{res}'")


def filter_and_concatenate(
    fnames: List[os.PathLike],
    metric: str,
    res: str
) -> xr.Dataset:
    """ Creates a dataset from a set of Weatherbench files.

    Filters to greater North America only (Artics through Central America),
    Aggregates to daily max temperature or daily total precipitation.

    Arguments:
      * fnames: a list of WeatherBench file paths in netcdf4 format (.nc)
      * metric: either "temp" (for temperature) or "precip" (for pressure)
      * res: resolution of the data. Either "1406" (1.4 deg) or "5625" (5.6 deg)
    """

    subsets = []
    for fname in tqdm(fnames):
        subset =  xr.open_dataset(fname)

        subset['lon'] = subset['lon'] - 180  # make longitude coordinates symmetric
        subset_na = north_america_only(subset, res=res)

        if metric == "temp":
            subset_na_daily = subset_na.groupby('time.date').max() + KELVIN_TO_CELSIUS
        elif metric == "precip":
            subset_na_daily = subset_na.groupby('time.date').sum()
        else:
            raise ValueError(f"Unrecognized metric '{metric}'")

        subsets.append(subset_na_daily)

    subset_comb = xr.concat(subsets, dim="date").sortby("date")
    subset_comb['date'] = subset_comb['date'].astype(np.datetime64)  # fixes zarr unknown type complaint

    return subset_comb


def process_raw_weatherbench(data_dir: os.PathLike) -> None:
    """ Writes out processed (filtered) WeatherBench data.

        Expects input DATA_DIR to have subdirectories /temp and /precip, each
        with further resolution subdirectories /1406 and /5625
    """
    for metric in ["temp", "precip"]:
        for res in ["1406", "5625"]:
            raw_dir = os.path.join(data_dir,"raw",metric,res)
            file_names = [os.path.join(raw_dir, fname) for fname in os.listdir(raw_dir)]

            log.info(f"processing {metric}/{res}...")
            result = filter_and_concatenate(file_names, metric=metric, res=res)

            dest_dir = os.path.join(data_dir,"processed",metric,res)
            if os.path.exists(dest_dir):
                log.info("Preprocessed data directory already exists, overwriting...")
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir)

            dest_path =  os.path.join(dest_dir, f"{metric}_{res}_processed.zarr")
            result.to_zarr(dest_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing data')
    parser.add_argument(
        '-d', '--data_dir', type=str, default = "./data", required=False,
        help='path to top level data directory. Expects ./raw/temp/1406/ etc.')

    args = parser.parse_args()
    process_raw_weatherbench(args.data_dir)

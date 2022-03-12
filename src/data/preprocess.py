" Transforms a directory of raw WeatherBench data into a single training-ready array on disk "

import argparse
import logging
import os
import shutil

from typing import List

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


EPSILON = .0001
NA_LAT = (25.3125, 64.6875+EPSILON)  # epsilon to avoid non-inclusive right
NA_LON = (-129.375, -78.75+EPSILON)  # epsilon to avoid non-inclusive right
KELVIN_TO_CELSIUS = -273.15


def north_america_only(weatherbench: xr.Dataset) -> xr.Dataset:
    """ Filters the passed WEATHERBENCH dataset to North American coordinates only """

    return weatherbench.sel(lat=slice(*NA_LAT), lon=slice(*NA_LON))


def filter_and_concatenate(
    fnames: List[os.PathLike],
    metric: str
) -> xr.Dataset:
    subsets = []
    for fname in fnames:
        subset =  xr.open_dataset(fname)

        subset['lon'] = subset['lon'] - 180  # make longitude coordinates symmetric
        subset_na = north_america_only(subset)

        if metric == 'temp':
            subset_na_daily = subset_na.groupby('time.date').max() + KELVIN_TO_CELSIUS
        else:
            subset_na_daily = subset_na.groupby('time.date').sum()

        # TODO(jwhite) does any de-meaning need to occur here?

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

            result = filter_and_concatenate(file_names, metric=metric)

            dest_dir = os.path.join(data_dir,"processed",metric,res)
            if os.path.exists(dest_dir):
                log.info("Preprocessed data directory already exists, overwriting...")
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir)

            # import pdb; pdb.set_trace()
            dest_path =  os.path.join(dest_dir, f"{metric}_{res}_processed.zarr")
            result.to_zarr(dest_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing data')
    parser.add_argument(
        '-d', '--data_dir', type=str, default = "./data", required=False,
        help='path to top level data directory. Expects ./raw/temp/1406/ etc.')

    args = parser.parse_args()
    process_raw_weatherbench(args.data_dir)

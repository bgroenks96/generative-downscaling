import numpy as np
import xarray as xr

def southeast_us(dataset, size=(8,16), scale_factor=1, lon_west=False):
    ht, wt = size
    lats, lons = dataset.lat, dataset.lon
    seus_lat_st = np.abs(lats - 28).argmin().values
    seus_lat_en = seus_lat_st + ht*scale_factor
    lon = -(360 - 267) if lon_west else 267
    seus_lon_st = np.abs(lons - lon).argmin().values
    seus_lon_en = seus_lon_st + wt*scale_factor
    dataset_seus = dataset.isel(lat=slice(seus_lat_st, seus_lat_en),
                                lon=slice(seus_lon_st, seus_lon_en))
    return dataset_seus

def pacific_nw(dataset, size=(8,16), scale_factor=1, lon_west=False):
    ht, wt = size
    lats, lons = dataset.lat, dataset.lon
    seus_lat_st = np.abs(lats - 44).argmin().values
    seus_lat_en = seus_lat_st + ht*scale_factor
    lon = -(360 - 232) if lon_west else 232
    seus_lon_st = np.abs(lons - lon).argmin().values
    seus_lon_en = seus_lon_st + wt*scale_factor
    dataset_seus = dataset.isel(lat=slice(seus_lat_st, seus_lat_en),
                                lon=slice(seus_lon_st, seus_lon_en))
    return dataset_seus

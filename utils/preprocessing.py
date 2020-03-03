import numpy as np
import xarray as xr

def remove_monthly_means(x, means=None, time_dim='Time', sparse=False):
    month_index = f'{time_dim}.month'
    monthly_means = means if means is not None else x.groupby(month_index).mean(dim=time_dim)
    for month in np.unique(x[month_index]):
        x = xr.where(x[month_index] == month, x - monthly_means.sel(month=month), x)
    return x, monthly_means

def restore_monthly_means(x, means, time_dim='Time'):
    month_index = f'{time_dim}.month'
    for month in np.unique(x[month_index]):
        x = xr.where(x[month_index] == month, x + means.sel(month=month), x)
    return x

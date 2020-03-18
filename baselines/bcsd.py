import argparse
import itertools as it
import numpy as np
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
from tqdm import tqdm

class QuantileMap:
    def __init__(self, num_quantiles=1000, interp='linear', axis=0):
        self.num_quantiles = num_quantiles
        self.interp = interp
        self.axis = axis
        self.x_quantiles = None
        self.y_quantiles = None

    def fit(self, x, y):
        """
        x : input x tensor
        y : input y tensor
        """
        x_quantiles = tfp.stats.quantiles(x, self.num_quantiles, axis=self.axis, interpolation=self.interp, keep_dims=True)
        y_quantiles = tfp.stats.quantiles(y, self.num_quantiles, axis=self.axis, interpolation=self.interp, keep_dims=True)
        # move quantiles to last axis
        dims = list(range(len(x.shape)+1))
        self.x_quantiles = tf.transpose(x_quantiles, perm=dims[1:] + dims[:1])
        self.y_quantiles = tf.transpose(y_quantiles, perm=dims[1:] + dims[:1])
        return self

    @tf.function
    def predict(self, x, batch_size=1):
        # define nested predict function to apply to each batch element
        def _predict(x):
            # compute argmin to find closest x (model) quantile
            x_inds = tf.math.argmin(tf.math.abs(x - self.x_quantiles), axis=-1)
            # increase index rank by 1
            x_inds = tf.expand_dims(x_inds, axis=-1)
            # use gather to select mapped quantiles from y (observed);
            # batch_dims is set to the rank of x - 1 to index only the quantiles axis
            x_mapped = tf.gather(self.y_quantiles, x_inds, axis=-1, batch_dims=len(x_inds.shape)-1)
            # squeeze first (batch) and last (quantile) dims to get correct output dimensions
            return tf.squeeze(x_mapped, axis=[0,-1])
        x_ = tf.expand_dims(x, axis=-1)
        res = tf.map_fn(_predict, x_, parallel_iterations=batch_size)
        assert res.shape == x.shape, f'expected {x.shape} but got: {res.shape}'
        return res
    
class BCSD:
    """
    GPU-accelerated implementation of bias-correction spatial-disaggregation (BCSD).
    This class assumes inputs of type xr.DataArray with the spatial dimensions 'lat'
    and 'lon', and a given time dimension.
    """
    def __init__(self, n_quantiles=1000, pool_size=15, verbose=False, interp='bilinear', time_dim='Time'):
        self.n_quantiles = n_quantiles
        self.pool_size = pool_size
        self.verbose = verbose
        self.interp = interp
        self.time_dim = time_dim
        self.qmaps = dict()
        self.scaling_factors = dict()
        self.hr_coords = None
        self.hr_dims = None
        
    def _log(self, msg):
        if self.verbose:
            logging.info(msg)

    def fit(self, lr: xr.DataArray, obsv: xr.DataArray, obsv_lr: xr.DataArray=None, batch_size=10):
        if obsv_lr is None:
            obsv_lr = obsv.interp(lat=lr.lat, lon=lr.lon, method=self.interp)
        self.hr_coords = obsv.coords
        self.hr_dims = obsv.dims
        # fit quantile map
        days = list(range(1,367))
        for day in tqdm(days, desc='bcsd fit'):
            day_range = (np.arange(day - self.pool_size, day + self.pool_size) % 366) + 1
            inds = np.in1d(lr[f'{self.time_dim}.dayofyear'], day_range)
            lr_sub = lr[inds]
            obsv_lr_sub = obsv_lr[inds]
            obsv_sub = obsv[inds]
            x = tf.constant(lr_sub.values)
            y = tf.constant(obsv_lr_sub.values)
            qmap = QuantileMap(num_quantiles=self.n_quantiles)
            qmap.fit(x, y)
            self.qmaps[day] = qmap
            y_interp = tf.image.resize(y, (obsv.lat.size, obsv.lon.size), method=self.interp)
            y_interp = xr.DataArray(y_interp, coords=obsv_sub.coords, dims=obsv_sub.dims)
            y_interp_curr = y_interp[y_interp[f'{self.time_dim}.dayofyear'] == day]
            y_obs_curr = obsv_sub[obsv_sub[f'{self.time_dim}.dayofyear'] == day]
            y_interp_dayavg = y_interp_curr.mean(dim=self.time_dim)
            y_obs_dayavg = y_obs_curr.mean(dim=self.time_dim)
            scaling_factor = y_obs_dayavg / xr.where(y_interp_dayavg > 0.0, y_interp_dayavg, 1.0)
            self.scaling_factors[day] = scaling_factor
        return self

    def predict(self, lr: xr.DataArray, batch_size=10):
        assert self.hr_coords is not None, 'fit not called'
        N_days = len(np.unique(lr[f'{self.time_dim}.dayofyear']))
        hr_lat, hr_lon = self.hr_coords['lat'].size, self.hr_coords['lon'].size
        y_pred_per_day = []
        for day, lr_day in tqdm(lr.groupby(f'{self.time_dim}.dayofyear'), total=N_days, desc='bcsd pred'):
            lr_mapped = self.qmaps[day].predict(lr_day.values)
            lr_mapped_interp = tf.image.resize(lr_mapped, (hr_lat, hr_lon), method=self.interp)
            lr_mapped_interp = xr.DataArray(lr_mapped_interp.numpy(),
                                            coords={self.time_dim: lr_day[self.time_dim],
                                                    'lat': self.hr_coords['lat'],
                                                    'lon': self.hr_coords['lon']},
                                            dims=self.hr_dims)
            y_pred_per_day.append(self.scaling_factors[day]*lr_mapped_interp)
        # concatenate and sort all days
        y_pred = xr.concat(y_pred_per_day, dim=self.time_dim)
        y_pred = y_pred.isel({self.time_dim: y_pred.Time.argsort().values})
        y_pred = y_pred.transpose(self.time_dim, *y_pred.dims[:-1])
        return y_pred
    
    def transform(self, x):
        """
        Calls self.predict(x); for sklearn compatibility
        """
        return self.predict(x)

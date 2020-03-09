import argparse
import itertools as it
import numpy as np
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr

class QuantileMap:
    def __init__(self, num_quantiles=1000, axis=0):
        self.num_quantiles = num_quantiles
        self.axis = axis
        self.x_quantiles = None
        self.y_quantiles = None

    def fit(self, x, y):
        """
        x : input x tensor
        y : input y tensor
        """
        x_quantiles = tfp.stats.quantiles(x, self.num_quantiles, axis=self.axis, keep_dims=True)
        y_quantiles = tfp.stats.quantiles(y, self.num_quantiles, axis=self.axis, keep_dims=True)
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
    def __init__(self, n_quantiles=1000, verbose=False, interp='bilinear', time_dim='Time'):
        self.n_quantiles = n_quantiles
        self.verbose = verbose
        self.interp = interp
        self.time_dim = time_dim
        self.qmap = None
        self.scaling_factors = dict()
        self.hr_coords = None
        self.hr_dims = None
        
    def _log(self, msg):
        if self.verbose:
            logging.info(msg)

    def fit(self, lr: xr.DataArray, obsv: xr.DataArray, obsv_lr: xr.DataArray=None):
        if obsv_lr is None:
            obsv_lr = obsv.interp(lat=lr.lat, lon=lr.lon, method=self.interp)
        self.hr_coords = obsv.coords
        self.hr_dims = obsv.dims
        # fit quantile map
        self._log('initializing tensors for low resolution datasets')
        X_model = tf.constant(lr.values)
        y_obsv_lr = tf.constant(obsv_lr.values)
        self._log('fitting quantile map')
        qmap = QuantileMap(num_quantiles=self.n_quantiles)
        qmap.fit(X_model, y_obsv_lr)
        self._log('computing scaling factors')
        obsv_interp = tf.image.resize(y_obsv_lr, (obsv.lat.size, obsv.lon.size), method=self.interp)
        obsv_interp = xr.DataArray(obsv_interp, coords=obsv.coords, dims=obsv.dims)
        X_dayavg = obsv_interp.groupby(f'{self.time_dim}.dayofyear').mean(dim=self.time_dim)
        y_dayavg = obsv.groupby(f'{self.time_dim}.dayofyear').mean(dim=self.time_dim)
        for day in range(1,367):
            x_avg = X_dayavg.sel(dayofyear=day)
            scaling_factor = y_dayavg.sel(dayofyear=day) / xr.where(x_avg > 0.0, x_avg, 1.0)
            self.scaling_factors[day] = scaling_factor.drop('dayofyear')
        self.qmap = qmap
        return self

    def predict(self, lr: xr.DataArray, batch_size=10):
        assert self.qmap is not None, 'fit not called'
        self._log('applying bias-correction quantile mapping')
        lr_mapped = self.qmap.predict(lr.values, batch_size=batch_size)
        #lr_mapped = xr.DataArray(lr_mapped.numpy(), coords=lr.coords, dims=lr.dims)
        self._log('interpolating low resolution data to target resolution')
        hr_lat, hr_lon = self.hr_coords['lat'].size, self.hr_coords['lon'].size
        lr_mapped_interp = tf.image.resize(lr_mapped, (hr_lat, hr_lon), method=self.interp)
        lr_mapped_interp = xr.DataArray(lr_mapped_interp.numpy(),
                                        coords={self.time_dim: lr[self.time_dim],
                                                'lat': self.hr_coords['lat'],
                                                'lon': self.hr_coords['lon']},
                                        dims=self.hr_dims)
        y_pred_per_day = []
        for day, val in lr_mapped_interp.groupby(f'{self.time_dim}.dayofyear'):
            y_pred_per_day.append(self.scaling_factors[day]*val)
        # concatenate and sort all days
        y_pred = xr.concat(y_pred_per_day, dim=self.time_dim)
        y_pred = y_pred.isel(Time=y_pred.Time.argsort().values)
        y_pred = y_pred.transpose(self.time_dim, *y_pred.dims[:-1])
        return y_pred
    
    def transform(self, x):
        """
        Calls self.predict(x); for sklearn compatibility
        """
        return self.predict(x)

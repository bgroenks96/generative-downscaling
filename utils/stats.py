import xarray as xr
import numpy as np

def standardize(X, mu, sigma, epsilon=1.0E-5):
    return (X - mu) / ((sigma < epsilon) + sigma)

def destandardize(X, mu, sigma):
    return X*sigma + mu

def normalize(X, min_val, max_val, epsilon=1.0E-5):
    return (X - min_val) / ((np.abs(max_val) < epsilon) + max_val)

def denormalize(X, min_val, max_val):
    return X*max_val + min_val

class StandardStats:
    def __init__(self, X, dim=None, precompute=False):
        assert isinstance(X, xr.Dataset) or isinstance(X, xr.DataArray)
        self.mu, self.sigma = X.mean(dim=dim), X.std(dim=dim)
        if precompute:
            self.mu.load()
            self.sigma.load()

    def standardize(self, X):
        return standardize(X, self.mu.values, self.sigma.values)

    def destandardize(self, X):
        return destandardize(X, self.mu.values, self.sigma.values)

class MinMaxStats:
    def __init__(self, X, dim=None, precompute=False):
        assert isinstance(X, xr.Dataset) or isinstance(X, xr.DataArray)
        self.min, self.max = X.min(dim=dim), X.max(dim=dim)
        if precompute:
            self.min.load()
            self.max.load()

    def normalize(self, X, all_dims=False):
        min_val = self.min
        max_val = self.max
        if all_dims:
            min_val = min_val.min()
            max_val = max_val.max()
        return normalize(X, min_val.values, max_val.values)

    def denormalize(self, X, all_dims=False):
        min_val = self.min
        max_val = self.max
        if all_dims:
            min_val = min_val.min()
            max_val = max_val.max()
        return denormalize(X, min_val.values, max_val.values)

class Stats:
    def __init__(self, X, dim=None, precompute=False):
        assert isinstance(X, xr.Dataset) or isinstance(X, xr.DataArray)
        self.min_max_stats = MinMaxStats(X, dim=dim, precompute=precompute)
        self.standard_stats = StandardStats(X, dim=dim, precompute=precompute)

    def standardize(self, X):
        return self.standard_stats.standardize(X)

    def destandardize(self, X):
        return self.standard_stats.destandardize(X)

    def normalize(self, X, all_dims=False):
        return self.min_max_stats.normalize(X, all_dims=all_dims)

    def denormalize(self, X, all_dims=False):
        return self.min_max_stats.denormalize(X, all_dims=all_dims)

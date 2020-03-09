import tensorflow as tf
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import utils.metrics as metrics
from utils.plot import image_map_factory
from utils.data import xr_to_tf_dataset
from experiments.common import prepare_downscaling_data

# use sparse metrics for precipitation
eval_rmse = metrics.sparse_rmse_metric(epsilon=1.0)
eval_bias = metrics.sparse_bias_metric(epsilon=1.0)
eval_corr = metrics.correlation_metric()

class PrecipitationDataFold:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        
    def to_dataset(self, data_lo, data_hi, map_fn_lo=None, map_fn_hi=None, load_batch_size=1000, **kwargs):
        ds_lo = xr_to_tf_dataset(data_lo, load_batch_size)
        ds_hi = xr_to_tf_dataset(data_hi, load_batch_size)
        if map_fn_lo is not None:
            ds_lo = ds_lo.map(map_fn_lo)
        if map_fn_hi is not None:
            ds_hi = ds_hi.map(map_fn_hi)
        return prepare_downscaling_data(ds_lo, ds_hi, **kwargs)
    
    def train_dataset(self, **kwargs):
        return self.to_dataset(*self.train, **kwargs)
    
    def test_dataset(self, **kwargs):
        return self.to_dataset(*self.test, **kwargs)
    
def preprocess_dataset(data, shape, epsilon=1.0):
    """
    Replaces all entries with values < epsilon mm of precip with uniform random values
    from 0 - epsilon
    """
    assert epsilon > 0.0, 'epsilon must be positive'
    data = np.cbrt(data)
    return xr.where(data > 1.0, data, np.random.uniform(0.0, epsilon, size=shape))
    
def preprocess_fold_prcp(fold):
    (train_lo, train_hi), (test_lo, test_hi) = fold
    return PrecipitationDataFold((train_lo, train_hi),
                                (test_lo, test_hi))

def eval_metrics(indices, true: tf.Tensor, pred: tf.Tensor, coords):
    # pointwise metrics
    rmse = eval_rmse(true, pred).numpy()
    bias = eval_bias(true, pred).numpy()
    corr = eval_corr(true, pred).numpy()
    # climdex indices
    pred_arr = xr.DataArray(pred.numpy(), coords=coords)
    mrx1 = indices.monthly_rx1day(pred_arr)
    mrx5 = indices.monthly_rx5day(pred_arr)
    ar20 = indices.annual_r20mm(pred_arr)
    atot = indices.prcptot(pred_arr)
    sdii = indices.sdii(pred_arr)
    cdd = indices.cdd(pred_arr)
    cwd = indices.cwd(pred_arr)
    return {'rmse': rmse, 'bias': bias, 'corr': corr,
            'mrx1': mrx1, 'mrx5': mrx5,
            'sdii': sdii, 'cdd': cdd, 'cwd': cwd,
            'ar20': ar20, 'atot': atot}

def plot_indices(metrics):
    indices_avg = {k: v.mean(dim=['lat', 'lon', 'chan']).values for k, v in metrics.items() if k not in ['rmse','bias','corr']}
    fig = plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    sns.boxplot(x=['monthly rx1d', 'monthly rx5d'], y=[indices_avg['mrx1'], indices_avg['mrx5']])
    plt.title('Monthly k-day rainfall')
    plt.subplot(2,2,2)
    sns.boxplot(x=['cdd', 'cwd'], y=[indices_avg['cdd'], indices_avg['cwd']])
    plt.title('Montly consecutive dry/wet days')
    plt.subplot(2,2,3)
    sns.boxplot(x=['sdii'], y=[indices_avg['sdii']])
    plt.title('Montly simple precip intensity index')
    plt.subplot(2,2,4)
    sns.boxplot(x=['atot'], y=[indices_avg['atot']])
    plt.title('Annual total rainfall')
    return fig

def plot_error_maps(metrics, lat, lon):
    fig, axs, plot_fn = image_map_factory(1, 3, figsize=(8,6), cbar_per_subplot=True)
    rmse_max = np.quantile(metrics['rmse'], 0.95).round(2)
    plot_fn(axs[0], metrics['rmse'], lat, lon, title='spatial sparse rmse', cmap='Reds', min_max=(0.0, rmse_max))
    bias_max = np.quantile(metrics['bias'], 0.95).round(2)
    bias_min = np.quantile(metrics['bias'], 0.05).round(2)
    bias_bound = np.maximum(np.abs(bias_max), np.abs(bias_min))
    plot_fn(axs[1], metrics['bias'], lat, lon, title='spatial sparse bias', cmap='bwr', min_max=(-bias_bound, bias_bound))
    qqrsq_min = np.nanmin(metrics['corr'])
    qqrsq_min = 0.0 if not np.isfinite(qqrsq_min) else qqrsq_min
    plot_fn(axs[2], metrics['corr'], lat, lon, title='correlation per dim', cmap='copper', min_max=(qqrsq_min,1.0))
    return fig

import tensorflow as tf
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import utils.metrics as metrics
from utils.plot import image_map_factory
from utils.data import xr_to_tf_dataset
from utils.preprocessing import remove_monthly_means, restore_monthly_means
from experiments.common import prepare_downscaling_data

eval_rmse = metrics.scaled_rmse_metric()
eval_bias = metrics.bias_metric()
eval_qqrsq = metrics.qqrsq_metric()

class TemperatureDataFold:
    def __init__(self, train, test, monthly_means):
        self.train = train
        self.test = test
        self.monthly_means = monthly_means
        
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
    
def preprocess_fold_maxt(fold):
    (train_lo, train_hi), (test_lo, test_hi) = fold
    train_lo, monthly_means_lo = remove_monthly_means(train_lo)
    train_hi, monthly_means_hi = remove_monthly_means(train_hi)
    test_lo,_ = remove_monthly_means(test_lo, monthly_means_lo)
    test_hi,_ = remove_monthly_means(test_hi, monthly_means_hi)
    return TemperatureDataFold((train_lo, train_hi),
                               (test_lo, test_hi),
                               (monthly_means_lo, monthly_means_hi))

def eval_metrics(indices, true: tf.Tensor, pred: tf.Tensor, coords, monthly_means=None):
    # pointwise metrics
    rmse = eval_rmse(true, pred).numpy()
    bias = eval_bias(true, pred).numpy()
    qqrsq = eval_qqrsq(true, pred).numpy()
    # climdex indices
    true_arr = xr.DataArray(true.numpy(), coords=coords)
    pred_arr = xr.DataArray(pred.numpy(), coords=coords)
    if monthly_means is not None:
        true_arr = restore_monthly_means(true_arr, monthly_means)
        pred_arr = restore_monthly_means(pred_arr, monthly_means)
    txx_true = indices.monthly_txx(true_arr)
    txx_pred = indices.monthly_txx(pred_arr)
    txn_true = indices.monthly_txn(true_arr)
    txn_pred = indices.monthly_txn(pred_arr)
    txid_true = indices.annual_icing_days(true_arr)
    txid_pred = indices.annual_icing_days(pred_arr)
    txsd_true = indices.annual_summer_days(true_arr)
    txsd_pred = indices.annual_summer_days(pred_arr)
    txx_bias = txx_pred - txx_true
    txn_bias = txn_pred - txn_true
    txid_bias = txid_pred - txid_true
    txsd_bias = txsd_pred - txsd_true
    return {'rmse': rmse, 'bias': bias, 'qqrsq': qqrsq,
            'txx_bias': txx_bias, 'txn_bias': txn_bias,
            'txid_bias': txid_bias, 'txsd_bias': txsd_bias}

def plot_indices(metrics):
    names = ['txx_bias', 'txn_bias', 'txid_bias', 'txsd_bias']
    indices_avg = {k: v.mean(dim=['lat', 'lon', 'chan']).values for k, v in metrics.items() if k in names}
    fig = plt.figure(figsize=(8,6))
    plt.subplot(1,2,1)
    sns.boxplot(x=['txx bias', 'txn bias'], y=[indices_avg['txx_bias'], indices_avg['txn_bias']])
    plt.title('Monthly max/min temperature, bias')
    plt.subplot(1,2,2)
    sns.boxplot(x=['icing days bias', 'summer days bias'], y=[indices_avg['txid_bias'], indices_avg['txsd_bias']])
    plt.title('Annual number of icing/summer days, bias')
    return fig

def plot_error_maps(metrics, lat, lon):
    fig, axs, plot_fn = image_map_factory(1, 3, figsize=(8,6), cbar_per_subplot=True)
    rmse_max = np.quantile(metrics['rmse'], 0.95).round(2)
    plot_fn(axs[0], metrics['rmse'], lat, lon, title='spatial rmse', cmap='Reds', min_max=(0.0, rmse_max))
    bias_max = np.quantile(metrics['bias'], 0.95).round(2)
    plot_fn(axs[1], metrics['bias'], lat, lon, title='spatial bias', cmap='bwr', min_max=(-bias_max, bias_max))
    qqrsq_min = metrics['qqrsq'].min()
    plot_fn(axs[2], metrics['qqrsq'], lat, lon, title='spatial QQ $R^2$', cmap='copper', min_max=(qqrsq_min,1.0))
    return fig

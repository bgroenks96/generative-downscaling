import click
import mlflow
import logging
import os.path
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import climdex.temperature as tdex
import climdex.precipitation as pdex
import experiments.maxt_experiment_base as maxt
import experiments.prcp_experiment_base as prcp
from utils.plot import image_map_factory
from utils.preprocessing import remove_monthly_means
from utils.data import create_time_series_train_test_generator_v2
from baselines.bcsd import BCSD
from experiments.common import load_data

def fit_bcsd_maxt(fold, i):
    mlflow.log_param('fold', i+1)
    (lr_train, lr_obs_train, hr_train), (lr_test, lr_obs_test, hr_test) = fold
    indices = tdex.indices('Time', convert_units_fn=lambda x: x + 273.15)
    bcsd = BCSD(verbose=True)
    bcsd.fit(lr_train, hr_train, lr_obs_train)
    hr_pred = bcsd.predict(lr_test).compute()
    hr_pred = xr.DataArray(hr_pred, coords=hr_test.coords, dims=hr_test.dims)
    y_true, y_pred = tf.constant(hr_test.values), tf.constant(hr_pred.values)
    metrics = maxt.eval_metrics(indices, y_true, y_pred, hr_test.coords)
    np.savez('/tmp/metrics.npz', **metrics)
    mlflow.log_artifact('/tmp/metrics.npz', 'data')
    avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
    mlflow.log_metrics(avg_metrics)
    # create plots
    fig = maxt.plot_indices(metrics)
    plt.savefig(f'/tmp/indices.png')
    mlflow.log_artifact(f'/tmp/indices.png', 'figures')
    fig = maxt.plot_error_maps(metrics, hr_test.lat, hr_test.lon)
    plt.savefig(f'/tmp/error-maps.png')
    mlflow.log_artifact(f'/tmp/error-maps.png', 'figures')
    
def fit_bcsd_prcp(fold, i):
    mlflow.log_param('fold', i)
    indices = pdex.indices('Time')
    (lr_train, lr_obs_train, hr_train), (lr_test, lr_obs_test, hr_test) = fold
    bcsd = BCSD(verbose=True)
    bcsd.fit(lr_train, hr_train, lr_obs_train)
    hr_pred = bcsd.predict(lr_test).compute()
    assert not np.any(np.isnan(hr_pred.values))
    hr_pred = xr.DataArray(hr_pred, coords=hr_test.coords, dims=hr_test.dims)
    y_true, y_pred = tf.constant(hr_test.values, dtype=tf.float32), tf.constant(hr_pred.values, dtype=tf.float32)
    #y_true, y_pred = tf.math.pow(y_true, 3.0), tf.math.pow(y_pred, 3.0)
    metrics = prcp.eval_metrics(indices, y_true, y_pred, hr_test.coords)
    np.savez('/tmp/metrics.npz', **metrics)
    mlflow.log_artifact('/tmp/metrics.npz', 'data')
    avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
    mlflow.log_metrics(avg_metrics)
    # create plots
    fig = prcp.plot_indices(metrics)
    plt.savefig(f'/tmp/indices.png')
    mlflow.log_artifact(f'/tmp/indices.png', 'figures')
    fig = prcp.plot_error_maps(metrics, hr_test.lat, hr_test.lon)
    plt.savefig(f'/tmp/error-maps.png')
    mlflow.log_artifact(f'/tmp/error-maps.png', 'figures')

@click.command(help="Fits and evaluates BCSD on the ERA-I/Rasmussen dataset")
@click.option("--scale", type=click.INT, required=True, help="Downscaling factor")
@click.option("--region", type=click.STRING, default='southeast_us')
@click.option("--var", type=click.STRING, default='MAXT', help="Dataset var name")
@click.option("--test-size", type=click.INT, default=146, help='size of the test set for each fold')
@click.option("--splits", type=click.INT, default=5, help="Number of CV splits to use")
@click.option("--auth", type=click.STRING, default='gcs.secret.json', help="GCS keyfile")
@click.argument("data_lr", type=click.STRING, default="erai/daily-1deg")
def bcsd(data_lr, scale, region, var, test_size, splits, auth, **kwargs):
    mlflow.log_param('region', region)
    mlflow.log_param('var', var)
    if scale == 2:
        data_hr = 'daily-1-2deg'
    elif scale == 4:
        data_hr = 'daily-1-4deg'
    elif scale == 8:
        data_hr = 'daily-1-8deg'
    else:
        raise NotImplementedError(f'unsupported downscaling factor {scale}')
    logging.info(f'==== Starting run ====')
    data_lo, data_hi = load_data(data_lr, data_hr, region, auth, scale=scale)
    data_obs_lo, _ = load_data('ras/daily-1deg', data_hr, region, auth, scale=scale)
    data_lo = data_lo[[var]].fillna(0.).clip(min=0.0, max=np.inf)
    data_obs_lo = data_obs_lo[[var]].fillna(0.).clip(min=0.0, max=np.inf)
    data_hi = data_hi[[var]].fillna(0.).clip(min=0.0, max=np.inf)
    if var == 'PRCP':
        data_lo = xr.where(data_lo > 1.0, data_lo, 0.0)
        data_obs_lo = xr.where(data_obs_lo > 1.0, data_obs_lo, 0.0)
        data_hi = xr.where(data_hi > 1.0, data_hi, 0.0)
    split_fn = create_time_series_train_test_generator_v2(n_splits=splits, test_size=test_size)
    folds = list(split_fn(data_lo, data_obs_lo, data_hi))
    for i, fold in enumerate(folds):
        logging.info(f'Fold {i+1}/{len(folds)}')
        with mlflow.start_run(nested=True):
            if var == 'MAXT':
                fit_bcsd_maxt(fold, i)
            elif var == 'PRCP':
                fit_bcsd_prcp(fold, i)
            else:
                raise NotImplementError(f'variable {var} not recognized')
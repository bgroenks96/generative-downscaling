import click
import mlflow
import logging
import os
import os.path
import shutil
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
from utils.distributions import normal, bernoulli_gamma
from baselines.dscnn import create_bmd_cnn10
from normalizing_flows.models import VariationalModel
from normalizing_flows.utils import get_metrics
from experiments.common import load_data

def bmd_plot(x, y, mean, samples, latlon_lo, latlon_hi):
    x_max, x_min = np.quantile(x, 0.95), np.quantile(x, 0.05)
    y_max, y_min = np.quantile(y, 0.95), np.quantile(y, 0.05)
    vmin, vmax = np.minimum(x_min, y_min), np.maximum(x_max, y_max)
    fig, axs, plot_fn = image_map_factory(2, 3, figsize=(6,4), cmap='viridis', min_max=(vmin, vmax))
    t = 0 # use first sample; the ordering should be random each epoch
    plot_fn(axs[0,0], x[t].numpy(), latlon_lo[0], latlon_lo[1], title='lo-res true')
    plot_fn(axs[0,1], y[t].numpy(), latlon_hi[0], latlon_hi[1], title='hi-res true')
    plot_fn(axs[0,2], mean[t].numpy(), latlon_hi[0], latlon_hi[1], title='hi-res predicted mean')
    for i, sample in enumerate(samples[t:t+3]):
        plot_fn(axs[1,i], samples[i].numpy(), latlon_hi[0], latlon_hi[1], title=f'sample {i}')
    return fig

def fit_bmd_maxt(fold, epochs, lr, batch_size, buffer_size, validate_freq):
    indices = tdex.indices('Time', convert_units_fn=lambda x: x + 273.15)
    data_fold = maxt.preprocess_fold_maxt(fold)
    train_lo, train_hi = data_fold.train
    test_lo, test_hi = data_fold.test
    N_train, N_test = train_lo.Time.size, test_lo.Time.size
    (ht_lr, wt_lr), (ht_hi, wt_hi) = train_lo.shape[1:3], train_hi.shape[1:3]
    monthly_means_lo, monthly_means_hi = data_fold.monthly_means
    train_ds = data_fold.train_dataset(batch_size=batch_size, buffer_size=buffer_size,
                                       supervised=True)
    test_ds = data_fold.test_dataset(batch_size=batch_size, buffer_size=N_test,
                                     supervised=True)
    scale = wt_hi // wt_lr
    encoder = create_bmd_cnn10(ht_lr, wt_lr, scale=scale, c_out=2)
    model = VariationalModel(encoder, normal(), optimizer=tf.keras.optimizers.Adam(lr=lr), output_shape=(None,ht_hi,wt_hi,1))
    ckpt_dir = f'/tmp/bmd-final'
    os.makedirs(ckpt_dir)
    for j in range(0, epochs, validate_freq):
        hist = model.fit(train_ds, epochs=validate_freq, steps_per_epoch=N_train//batch_size,
                         validation_data=test_ds, validation_steps=N_test//batch_size)
        hist = get_metrics(hist)
        mlflow.log_metrics(hist)
        j += validate_freq
        mlflow.log_metric('epoch', j)
        encoder.save(f'{ckpt_dir}/bmd-epoch{j}.h5')
        mlflow.log_artifact(f'{ckpt_dir}/bmd-epoch{j}.h5', artifact_path=f'model/')
        x_true = []
        y_true = []
        y_mean = []
        y_samples = []
        for x, y in test_ds:
            x_true.append(x)
            y_true.append(y)
            y_mean.append(model.mean(x))
            y_samples.append(model.sample(x))
        x_true = tf.concat(x_true, axis=0)
        y_true = tf.concat(y_true, axis=0)
        y_mean = tf.concat(y_mean, axis=0)
        y_samples = tf.concat(y_samples, axis=0)
        fig = bmd_plot(x_true, y_true, y_mean, y_samples, (test_lo.lat, test_lo.lon), (test_hi.lat, test_hi.lon))
        plt.savefig(f'/tmp/samples-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/samples-epoch{j}.png', 'figures')
        metrics = maxt.eval_metrics(indices, y_true, y_mean, test_hi.coords, monthly_means_hi)
        np.savez(f'/tmp/metrics-epoch{j}.npz', **metrics)
        mlflow.log_artifact(f'/tmp/metrics-epoch{j}.npz', 'data')
        avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
        mlflow.log_metrics(avg_metrics)
        # create plots
        fig = maxt.plot_indices(metrics)
        plt.savefig(f'/tmp/indices-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/indices-epoch{j}.png', 'figures')
        fig = maxt.plot_error_maps(metrics, test_hi.lat, test_hi.lon)
        plt.savefig(f'/tmp/error-maps-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/error-maps-epoch{j}.png', 'figures')
    shutil.rmtree(ckpt_dir)
        
def fit_bmd_prcp(fold, epochs, lr, batch_size, buffer_size, validate_freq):
    indices = pdex.indices('Time')
    data_fold = prcp.preprocess_fold_prcp(fold)
    train_lo, train_hi = data_fold.train
    test_lo, test_hi = data_fold.test
    N_train, N_test = train_lo.Time.size, test_lo.Time.size
    (ht_lr, wt_lr), (ht_hi, wt_hi) = train_lo.shape[1:3], train_hi.shape[1:3]
    train_ds = data_fold.train_dataset(batch_size=batch_size, buffer_size=buffer_size,
                                       supervised=True)
    test_ds = data_fold.test_dataset(batch_size=batch_size, buffer_size=N_test,
                                     supervised=True)
    scale = wt_hi // wt_lr
    encoder = create_bmd_cnn10(ht_lr, wt_lr, scale=scale, c_out=3)
    model = VariationalModel(encoder, bernoulli_gamma(), optimizer=tf.keras.optimizers.Adam(lr=lr), output_shape=(None,ht_hi,wt_hi,1))
    ckpt_dir = f'/tmp/bmd-prcp-final'
    os.makedirs(ckpt_dir)
    for j in range(0, epochs, validate_freq):
        hist = model.fit(train_ds, epochs=validate_freq, steps_per_epoch=N_train//batch_size,
                         validation_data=test_ds, validation_steps=N_test//batch_size)
        hist = get_metrics(hist)
        mlflow.log_metrics(hist)
        j += validate_freq
        mlflow.log_metric('epoch', j)
        encoder.save(f'{ckpt_dir}/bmd-epoch{j}.h5')
        mlflow.log_artifact(f'{ckpt_dir}/bmd-epoch{j}.h5', artifact_path=f'model/')
        x_true = []
        y_true = []
        y_mean = []
        y_samples = []
        for x, y in test_ds:
            x_true.append(x)
            y_true.append(y)
            y_mean.append(model.mean(x))
            y_samples.append(model.sample(x))
        x_true = tf.concat(x_true, axis=0)
        y_true = tf.concat(y_true, axis=0)
        y_mean = tf.concat(y_mean, axis=0)
        y_samples = tf.concat(y_samples, axis=0)
        fig = bmd_plot(x_true, y_true, y_mean, y_samples, (test_lo.lat, test_lo.lon), (test_hi.lat, test_hi.lon))
        plt.savefig(f'/tmp/samples-epoch{j}.png')
        plt.close(fig)
        mlflow.log_artifact(f'/tmp/samples-epoch{j}.png', 'figures')
        y_true = tf.math.pow(y_true, 3.0)
        y_mean = tf.math.pow(y_mean, 3.0)
        metrics = prcp.eval_metrics(indices, y_true, y_mean, test_hi.coords)
        np.savez(f'/tmp/metrics-epoch{j}.npz', **metrics)
        mlflow.log_artifact(f'/tmp/metrics-epoch{j}.npz', 'data')
        avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
        mlflow.log_metrics(avg_metrics)
        # create plots
        fig = prcp.plot_indices(metrics)
        plt.savefig(f'/tmp/indices-epoch{j}.png')
        plt.close(fig)
        mlflow.log_artifact(f'/tmp/indices-epoch{j}.png', 'figures')
        fig = prcp.plot_error_maps(metrics, test_hi.lat, test_hi.lon)
        plt.savefig(f'/tmp/error-maps-epoch{j}.png')
        plt.close(fig)
        mlflow.log_artifact(f'/tmp/error-maps-epoch{j}.png', 'figures')

@click.command(help="Fits and evaluates the Bano-Medina CNN10 on the ERA-I/Rasmussen dataset")
@click.option("--scale", type=click.INT, required=True, help="Downscaling factor")
@click.option("--epochs", type=click.INT, default=50)
@click.option("--learning-rate", type=click.FLOAT, default=1.0E-4)
@click.option("--batch-size", type=click.INT, default=100)
@click.option("--buffer-size", type=click.INT, default=2400)
@click.option("--validate-freq", type=click.INT, default=10)
@click.option("--region", type=click.STRING, default='southeast_us')
@click.option("--var", type=click.STRING, default='MAXT', help="Dataset var name")
@click.option("--auth", type=click.STRING, default='gcs.secret.json', help="GCS keyfile")
@click.argument("data_lr", type=click.STRING, default="ras/daily-1deg")
def bmd(data_lr, scale, epochs, learning_rate, batch_size, buffer_size, validate_freq, region, var, auth, **kwargs):
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
    data_lo = data_lo[[var]].fillna(0.).clip(min=0.0, max=np.inf).to_array(dim='chan').transpose('Time','lat','lon','chan')
    data_hi = data_hi[[var]].fillna(0.).clip(min=0.0, max=np.inf).to_array(dim='chan').transpose('Time','lat','lon','chan')
    if var == 'PRCP':
        data_lo = prcp.preprocess_dataset(data_lo, (data_lo.Time.size, data_lo.lat.size, data_lo.lon.size, 1), epsilon=1.0)
        data_lo = xr.where(data_lo > 1.0, data_lo, 0.0)
        data_hi = prcp.preprocess_dataset(data_hi, (data_hi.Time.size, data_hi.lat.size, data_hi.lon.size, 1), epsilon=1.0)
        data_hi = xr.where(data_hi > 1.0, data_hi, 0.0)
    lr_train = data_lo.isel(Time=slice(0,data_lo.Time.size-2*365))
    lr_test = data_lo.isel(Time=slice(data_lo.Time.size-2*365, data_lo.Time.size+1))
    hr_train = data_hi.isel(Time=slice(0,data_lo.Time.size-2*365))
    hr_test = data_hi.isel(Time=slice(data_lo.Time.size-2*365, data_lo.Time.size+1))
    if var == 'MAXT':
        fit_bmd_maxt(((lr_train, hr_train), (lr_test, hr_test)), epochs, learning_rate,
                     batch_size, buffer_size, validate_freq)
    elif var == 'PRCP':
        fit_bmd_prcp(((lr_train, hr_train), (lr_test, hr_test)), epochs, learning_rate,
                     batch_size, buffer_size, validate_freq)
    else:
        raise NotImplementedError(f'unrecognized variable {var}')
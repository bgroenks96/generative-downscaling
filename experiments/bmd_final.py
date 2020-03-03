import click
import mlflow
import logging
import os.path
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import climdex.temperature as tdex
import experiments.maxt_experiment_base as base
from utils.plot import image_map_factory
from utils.preprocessing import remove_monthly_means
from utils.distributions import normal
from baselines.dscnn import create_bmd_cnn10
from normalizing_flows.models import VariationalModel
from normalizing_flows.utils import get_metrics
from experiments.common import load_data

indices = tdex.indices('Time', convert_units_fn=lambda x: x + 273.15)

def bmd_plot(x, y, mean, median, samples, lat, lon):
    x_max, x_min = np.quantile(x, 0.95), np.quantile(x, 0.05)
    y_max, y_min = np.quantile(y, 0.95), np.quantile(y, 0.05)
    vmin, vmax = np.minimum(x_min, y_min), np.maximum(x_max, y_max)
    fig, axs, plot_fn = image_map_factory(2, 4, figsize=(6,4), cmap='viridis', min_max=(vmin, vmax))
    t = 0 # use first sample; the ordering should be random each epoch
    plot_fn(axs[0,0], x[t].numpy(), lat, lon, title='lo-res true')
    plot_fn(axs[0,1], y[t].numpy(), lat, lon, title='hi-res true')
    plot_fn(axs[0,2], mean[t].numpy(), lat, lon, title='lo-res true')
    plot_fn(axs[0,3], median[t].numpy(), lat, lon, title='lo-res true')
    for i, sample in enumerate(samples[t:t+4]):
        plot_fn(axs[1,i], samples[i].numpy(), lat, lon, title=f'sample {i}')
    return fig

def fit_bmd_cnn(fold, epochs, batch_size, buffer_size, validate_freq):
    data_fold = base.preprocess_fold_maxt(fold)
    train_lo, train_hi = data_fold.train
    test_lo, test_hi = data_fold.test
    N_train, N_test = train_lo.Time.size, test_lo.Time.size
    (ht_lr, wt_lr), (ht_hi, wt_hi) = train_lo.shape[1:3], train_hi.shape[1:3]
    monthly_means_lo, monthly_means_hi = data_fold.monthly_means
    train_ds = data_fold.train_dataset(batch_size=batch_size, buffer_size=buffer_size,
                                       supervised=True)
    test_ds = data_fold.test_dataset(batch_size=10*batch_size, buffer_size=buffer_size,
                                     supervised=True)
    scale = wt_hi // wt
    encoder = create_bmd_cnn10(ht_lr, wt_lr, c_out=2)
    model = VariationalModel(encoder, normal())
    ckpt_dir = f'/tmp/bmd-final'
    ckptm = model.create_checkpoint_manager(ckpt_dir)
    for j in range(0, epochs, validate_freq):
        hist = model.fit(train_ds, epochs=validate_freq, steps_per_epoch=N_train//batch_size,
                         validation_data=test_ds, validation_steps=N_test//batch_size)
        mlflow.log_artifacts(get_metrics(hist))
        ckptm.save()
        j += validate_freq
        mlflow.log_artifacts(os.path.dirname(ckptm.latest_checkpoint), artifact_path=f'model/ckpt-epoch{j}')
        x = []
        y_true = []
        y_mean = []
        y_median = []
        y_samples = []
        for x, y in test_ds:
            x.append(x)
            y_true.append(y)
            y_mean.append(model.mean(x))
            y_median.append(model.quantile(x, 0.5))
            y_samples.append(model.sample(x))
        x = tf.concat(x, axis=0)
        y_true = tf.concat(y_true, axis=0)
        y_mean = tf.concat(y_mean, axis=0)
        y_median = tf.concat(y_median, axis=0)
        y_samples = tf.concat(y_samples, axis=0)
        fig = bmd_plot(x, y_true, y_mean, y_median, y_samples)
        plt.savefig(f'/tmp/samples-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/samples-epoch{j}.png', 'figures')
        metrics = base.eval_metrics(indices, y_true, y_median, hr_test.coords, monthly_means_hi)
        np.savez(f'/tmp/metrics-epoch{j}.npz', **metrics)
        mlflow.log_artifact(f'/tmp/metrics-epoch{j}.npz', 'data')
        avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
        mlflow.log_metrics(avg_metrics)
        # create plots
        fig = base.plot_indices(metrics)
        plt.savefig(f'/tmp/indices-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/indices-epoch{j}.png', 'figures')
        fig = base.plot_error_maps(metrics, test_hi.lat, test_hi.lon)
        plt.savefig(f'/tmp/error-maps-epoch{j}.png')
        mlflow.log_artifact(f'/tmp/error-maps-epoch{j}.png', 'figures')

@click.command(help="Fits and evaluates the Bano-Medina CNN10 on the ERA-I/Rasmussen dataset")
@click.option("--scale", type=click.INT, required=True, help="Downscaling factor")
@click.option("--epochs", type=click.INT, default=20)
@click.option("--batch-size", type=click.INT, default=60)
@click.option("--buffer-size", type=click.INT, default=1200)
@click.option("--validate-freq", type=click.INT, default=5)
@click.option("--region", type=click.STRING, default='southeast_us')
@click.option("--var", type=click.STRING, default='MAXT', help="Dataset var name")
@click.option("--auth", type=click.STRING, default='gcs.secret.json', help="GCS keyfile")
@click.argument("data_lr", type=click.STRING, default="ras/daily-1deg")
def bmd(scale, epochs, batch_size, buffer_size, validate_freq, region, var, auth, **kwargs):
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
    lr_train = data_lo.isel(Time=slice(0,data_lo.Time.size-365))
    lr_test = data_lo.isel(Time=slice(data_lo.Time.size-365, data_lo.Time.size+1))
    hr_train = data_hi.isel(Time=slice(0,data_lo.Time.size-365))
    hr_test = data_hi.isel(Time=slice(data_lo.Time.size-365, data_lo.Time.size+1))
    fit_bmd_cnn(((lr_train, lr_test), (hr_train, hr_test)), epochs, batch_size, buffer_size, validate_freq)
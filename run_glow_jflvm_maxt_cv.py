# Add submodule paths
import sys
sys.path += ['./normalizing_flows', './baselines', './climdex']
# imports
import click
import mlflow
import logging
import os.path
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import utils.data as data_util
import utils.nn_util as nn
import xarray as xr
import gcsfs
import dscnn
import climdex.temperature as tdex
from normalizing_flows.models import JointFlowLVM, adversarial
from normalizing_flows.flows import Transform, Flow, Invert
from normalizing_flows.flows.glow import GlowFlow, coupling_nn_glow
from datasource import EraiRasDataLoader
from utils.pipeline_v2 import Pipeline, fillnan, clip
from utils.preprocessing import remove_monthly_means
from utils.distributions import normal
from utils.plot import image_map_factory
from tensorflow.keras.optimizers import Adamax
from tqdm import tqdm

def southeast_us(dataset, scale_factor=1):
    lats, lons = dataset.lat, dataset.lon
    seus_lat_st = np.abs(lats - 30).argmin().values
    seus_lat_en = seus_lat_st + 8*scale_factor
    seus_lon_st = np.abs(lons - 260).argmin().values
    seus_lon_en = seus_lon_st + 16*scale_factor
    dataset_seus = dataset.isel(lat=slice(seus_lat_st, seus_lat_en),
                                lon=slice(seus_lon_st, seus_lon_en))
    return dataset_seus

def preprocess_vds(data_lo, data_hi, batch_size=100, buffer_size=1000, supervised=True):
    if supervised:
        data = tf.data.Dataset.zip((data_lo, data_hi)).shuffle(buffer_size)
    else:
        data = tf.data.Dataset.zip((data_lo.shuffle(buffer_size), data_hi.shuffle(buffer_size)))
    return data.batch(batch_size)

def load_data(data_lr, data_hr, region_fn, auth_token, scale=4, project='thesis-research-255223'):
    data = EraiRasDataLoader(gcs_bucket='erai-rasmussen', gcs_project=project, auth=auth_token)
    data_lr = xr.open_zarr(data.rasmussen(data_lr), consolidated=True)
    data_hr = xr.open_zarr(data.rasmussen(data_hr), consolidated=True)
    reg_lr = region_fn(data_lr)
    reg_hr = region_fn(data_hr, scale_factor=scale)
    return reg_lr, reg_hr

indices = tdex.indices('Time')
def eval_climdex(true, pred, coords):
    true_arr = xr.DataArray(true, coords=coords)
    pred_arr = xr.DataArray(pred, coords=coords)
    txx_true = indices.monthly_txx(true_arr)
    txx_pred = indices.monthly_txx(pred_arr)
    txn_true = indices.monthly_txn(true_arr)
    txn_pred = indices.monthly_txn(pred_arr)
    txx_bias = txx_pred - txx_true
    txn_bias = txn_pred - txn_true
    return txx_bias, txn_bias

@click.command(help="Fits and validates a Glow FLVM on the ERA-I/Rasmussen dataset")
@click.option("--layers", type=click.INT, default=3, help="Number of Glow layers")
@click.option("--depth", type=click.INT, default=8, help="Number of Glow steps per layer")
@click.option("--min-filters", type=click.INT, default=32, help="Minimum number of filters in Glow affine coupling NNs")
@click.option("--max-filters", type=click.INT, default=256, help="Maximum number of filters in Glow affine coupling NNs")
@click.option("--lam", type=click.FLOAT, default=1.0, help="Weight parameter 'lambda' for MLE loss in joint FLVM")
@click.option("--dnet-layers", type=click.INT, default=3, help="Number of layers in the adversarial critic")
@click.option("--dnet-filters", type=click.INT, default=64, help="Number of filters in the critic")
@click.option("--batch-size", type=click.INT, default=10, help="Sample batch size")
@click.option("--epochs", type=click.INT, default=50, help="Number of epochs to train")
@click.option("--splits", type=click.INT, default=5, help="Number of CV splits to use")
@click.option("--var", type=click.STRING, default='MAXT', help="Dataset var name")
@click.option("--auth", type=click.STRING, default='gcs.secret.json', help="GCS keyfile")
@click.argument("data_lr")
@click.argument("scale", type=click.INT)
def run_cv(data_lr, scale, layers, depth, min_filters, max_filters, lam,
           dnet_layers, dnet_filters, batch_size, epochs, splits, var, auth):
    if scale == 2:
        data_hr = 'daily-1-2deg'
    elif scale == 4:
        data_hr = 'daily-1-4deg'
    elif scale == 8:
        data_hr = 'daily-1-8deg'
    else:
        raise NotImplementedError(f'unsupported downscaling factor {scale}')
    logging.info(f'==== Starting run ====')
    data_lo, data_hi = load_data(data_lr, data_hr, southeast_us, auth, scale=scale)
    data_lo = data_lo[[var]].fillna(0.).clip(min=0.0, max=np.inf).isel(Time=slice(0,data_lo.Time.size-365))
    data_hi = data_hi[[var]].fillna(0.).clip(min=0.0, max=np.inf).isel(Time=slice(0,data_hi.Time.size-365))
    # create train/test splits
    split_fn = data_util.create_time_series_train_test_generator(n_splits=splits)
    folds = list(split_fn(data_lo, data_hi))
    def upsample(new_wt, new_ht, method, scale_factor=1):
        @tf.function
        def _upsample(x):
            return tf.image.resize(x, (new_wt,new_ht), method=method) / scale_factor
        return _upsample
    sample_batch_size = batch_size
    load_batch_size = 400
    n_epochs = epochs
    validate_freq = 5
    warmup_epochs = 1
    models = []
    for i, ((train_lo, train_hi), (test_lo, test_hi)) in enumerate(folds):
        logging.info(f'Fold {i+1}/{len(folds)}')
        with mlflow.start_run(nested=True):
            mlflow.log_param('fold', i+1)
            mlflow.log_param('layers', layers)
            mlflow.log_param('depth', depth)
            mlflow.log_param('lam', lam)
            N_train, N_test = train_lo.Time.size, test_lo.Time.size
            (wt, ht), (wt_hi, ht_hi) = train_lo.shape[1:3], train_hi.shape[1:3]
            scale = wt_hi // wt
            print('{} training samples, {} test samples, {}x{} -> {}x{}'.format(N_train, N_test, wt, ht, wt_hi, ht_hi))
            train_steps = data_util.num_batches(N_train, sample_batch_size)
            test_steps = data_util.num_batches(N_test, sample_batch_size)
            train_lo, monthly_means_train_lo = remove_monthly_means(train_lo)
            train_hi, monthly_means_train_hi = remove_monthly_means(train_hi)
            test_lo, monthly_means_test_lo = remove_monthly_means(test_lo)
            test_hi, monthly_means_test_hi = remove_monthly_means(test_hi)
            train_lo_ds = data_util.xr_to_tf_dataset(train_lo, load_batch_size).map(upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            test_lo_ds = data_util.xr_to_tf_dataset(test_lo, load_batch_size).map(upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            train_hi_ds = data_util.xr_to_tf_dataset(train_hi, load_batch_size)
            test_hi_ds = data_util.xr_to_tf_dataset(test_hi, load_batch_size)
            train_ds = preprocess_vds(train_lo_ds, train_hi_ds, batch_size=sample_batch_size, buffer_size=N_train, supervised=False)
            test_ds = preprocess_vds(test_lo_ds, test_hi_ds, batch_size=sample_batch_size, buffer_size=N_test, supervised=False)
            test_ds_paired = preprocess_vds(test_lo_ds, test_hi_ds, batch_size=1, buffer_size=N_test, supervised=True)
            flow_hr = Invert(GlowFlow(num_layers=layers, depth=depth, coupling_nn_ctor=coupling_nn_glow(min_filters=min_filters, max_filters=max_filters)))
            flow_lr = Invert(GlowFlow(num_layers=layers, depth=depth, coupling_nn_ctor=coupling_nn_glow(min_filters=min_filters, max_filters=max_filters)))
            #learning_rate = LinearWarmupSchedule(1.0E-3, num_warmup_steps=N_train//sample_batch_size*warmup_epochs)
            dx = adversarial.PatchDiscriminator((wt_hi,ht_hi,1), n_layers=dnet_layers, n_filters=dnet_filters)
            dy = adversarial.PatchDiscriminator((wt_hi,ht_hi,1), n_layers=dnet_layers, n_filters=dnet_filters)
            model_joint = JointFlowLVM(flow_lr, flow_hr, dx, dy, input_shape=(None,wt_hi,ht_hi,1))
            ckptm = model_joint.create_checkpoint_manager(f'/tmp/glow-jflvm-fold{i}')
            for j in range(0, n_epochs, validate_freq):
                logging.info(f'Training joint model for {validate_freq} epochs ({j}/{n_epochs} complete)')
                mlflow.log_param('epoch', j+1)
                # train and test model
                model_joint.train(train_ds, steps_per_epoch=N_train//sample_batch_size, num_epochs=validate_freq, lam=lam)
                eval_metrics = model_joint.evaluate(test_ds, N_test//sample_batch_size)
                ckptm.save()
                # extract and log metrics
                mlflow.log_metrics({k: v[0] for k, v in eval_metrics.items()})
                samples_x, samples_y = model_joint.sample(n=4)
                fig, axs, plot_fn = image_map_factory(3, 4, figsize=(6,4), cmap='viridis', min_max=(-6,6))
                for k,x in enumerate(samples_x):
                    plot_fn(axs[0,k], x.numpy(), test_hi.lat, test_hi.lon, title=f'Sample $x_{k} \sim P(X)$')
                for k,y in enumerate(samples_y):
                    plot_fn(axs[1,k], y.numpy(), test_hi.lat, test_hi.lon, title=f'Sample $y_{k} \sim P(Y)$')
                x_t, y_t = next(test_ds_paired.__iter__())
                xp_t = model_joint.predict_x(y_t)
                yp_t = model_joint.predict_y(x_t)
                plot_fn(axs[2,0], x_t[0].numpy(), test_hi.lat, test_hi.lon, title='x true')
                plot_fn(axs[2,1], xp_t[0].numpy(), test_hi.lat, test_hi.lon, title='x predicted')
                plot_fn(axs[2,2], y_t[0].numpy(), test_hi.lat, test_hi.lon, title='y true')
                cs = plot_fn(axs[2,3], yp_t[0].numpy(), test_hi.lat, test_hi.lon, title='y predicted')
                fig.colorbar(cs, ax=axs.ravel().tolist(), orientation='vertical', shrink=0.6, pad=0.01).set_label('daily max tmperature anomalies (K)')
                fig_filename = f'/tmp/samples_fold{i+1}_epoch{j+1}.png'
                plt.savefig(fig_filename)
                mlflow.log_artifact(fig_filename, 'figures')
                logging.info('Evaluating ClimDEX indices on predictions')
                y_true, y_pred = [], []
                for x, y in test_ds:
                    y_true.append(y)
                    y_ = model_joint.predict_y(x)
                    y_pred.append(y_)
                y_true = tf.concat(y_true, axis=0)
                y_pred = tf.concat(y_pred, axis=0)
                txx_bias, txn_bias = eval_climdex(y_true.numpy(), y_pred.numpy(), test_hi.coords)
                txx_bias_mean, txx_bias_std = txx_bias.mean().values, txx_bias.std().values
                txn_bias_mean, txn_bias_std = txn_bias.mean().values, txn_bias.std().values
                mlflow.log_metrics({'txx_bias_avg': float(txx_bias_mean), 'txx_bias_std': float(txx_bias_std)})
                mlflow.log_metrics({'txn_bias_avg': float(txn_bias_mean), 'txn_bias_std': float(txn_bias_std)})
                rmse = tf.math.sqrt(tf.math.reduce_mean((y_true - y_pred)**2))
                mlflow.log_metric('rmse', rmse.numpy())
                fig = plt.figure(figsize=(8,6))
                plt.plot(range(txx_bias.Time.size), txx_bias.mean(dim=['lat','lon']), c='r')
                plt.plot(range(txn_bias.Time.size), txn_bias.mean(dim=['lat','lon']), c='b')
                plt.title('Monthly TX max/min validation bias')
                plt.xlabel('Months from $t_0$')
                plt.ylabel('Bias (K)')
                plt.legend(['txx','txn'])
                fig_filename = f'/tmp/indices_fold{i+1}_epoch{j+1}.png'
                plt.savefig(fig_filename)
                mlflow.log_artifact(fig_filename, 'figures')
                mlflow.log_artifacts(os.path.dirname(ckptm.latest_checkpoint), artifact_path=f'model/ckpt-fold{i+1}-epoch{j+1}')

if __name__=='__main__':
    logging.basicConfig(filename='run.log',level=logging.DEBUG)
    run_cv()
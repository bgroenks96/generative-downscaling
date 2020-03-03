import click
import mlflow
import logging
import os.path
import shutil
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import climdex.temperature as tdex
import experiments.maxt_experiment_base as base
from utils.data import create_time_series_train_test_generator
from utils.plot import image_map_factory
from models.glow import build_jflvm
from experiments.common import load_data, predict_batched, upsample

indices = tdex.indices('Time', convert_units_fn=lambda x: x + 273.15)

def flvm_plot(model, test_data, lat, lon):
    samples_x, samples_y = model.sample(n=4)
    x_max, x_min = np.quantile(samples_x, 0.95), np.quantile(samples_x, 0.05)
    y_max, y_min = np.quantile(samples_y, 0.95), np.quantile(samples_y, 0.05)
    vmin, vmax = np.minimum(x_min, y_min), np.maximum(x_max, y_max)
    fig, axs, plot_fn = image_map_factory(3, 4, figsize=(6,4), cmap='viridis', min_max=(vmin, vmax))
    for k,x in enumerate(samples_x):
        plot_fn(axs[0,k], x.numpy(), lat, lon, title=f'Sample $x_{k} \sim P(X)$')
    for k,y in enumerate(samples_y):
        plot_fn(axs[1,k], y.numpy(), lat, lon, title=f'Sample $y_{k} \sim P(Y)$')
    x_t, y_t = next(test_data.__iter__())
    xp_t = model.predict_x(y_t)
    yp_t = model.predict_y(x_t)
    plot_fn(axs[2,0], x_t[0].numpy(), lat, lon, title='lo-res true')
    plot_fn(axs[2,1], xp_t[0].numpy(), lat, lon, title='lo-res predicted')
    plot_fn(axs[2,2], y_t[0].numpy(), lat, lon, title='hi-res true')
    cs = plot_fn(axs[2,3], yp_t[0].numpy(), lat, lon, title='hi-res predicted')
    fig.colorbar(cs, ax=axs.ravel().tolist(), orientation='vertical', shrink=0.6, pad=0.01).set_label('daily max tmperature anomalies (K)')
    plt.suptitle('Model samples and predictions', y=0.9)
    return fig

def fit_glow_jflvm_maxt(fold, i, layers, depth, min_filters, max_filters, lam, alpha, dnet_layers, dnet_filters,
                        batch_size, buffer_size, epochs, validate_freq, supervised):
        mlflow.log_param('fold', i+1)
        mlflow.log_param('layers', layers)
        mlflow.log_param('depth', depth)
        mlflow.log_param('lam', lam)
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('supervised', supervised)
        data_fold = base.preprocess_fold_maxt(fold)
        train_lo, train_hi = data_fold.train
        test_lo, test_hi = data_fold.test
        N_train, N_test = train_lo.Time.size, test_lo.Time.size
        (wt, ht), (wt_hi, ht_hi) = train_lo.shape[1:3], train_hi.shape[1:3]
        monthly_means_lo, monthly_means_hi = data_fold.monthly_means
        train_ds = data_fold.train_dataset(batch_size=batch_size, buffer_size=buffer_size,
                                           map_fn_lo=upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                           supervised=supervised)
        test_ds = data_fold.test_dataset(batch_size=batch_size, buffer_size=buffer_size,
                                         map_fn_lo=upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                         supervised=supervised)
        test_ds_paired = data_fold.test_dataset(batch_size=10*batch_size, buffer_size=buffer_size,
                                                map_fn_lo=upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                                supervised=True)
        scale = wt_hi // wt
        model_joint = build_jflvm((None,wt_hi,ht_hi,1), scale, layers, depth,
                                    min_filters=min_filters, max_filters=max_filters,
                                    dnet_layers=dnet_layers, dnet_filters=dnet_filters)
        ckpt_dir = f'/tmp/glow-jflvm-fold{i}'
        ckptm = model_joint.create_checkpoint_manager(ckpt_dir)
        for j in range(0, epochs, validate_freq):
            logging.info(f'Training for {validate_freq} epochs, {j}/{epochs} complete')
            mlflow.log_metric('epoch', j)
            # train and test model
            model_joint.train(train_ds, steps_per_epoch=N_train//batch_size, num_epochs=validate_freq, lam=lam, alpha=alpha)
            model_metrics = model_joint.evaluate(test_ds, N_test//batch_size)
            ckptm.save()
            j += validate_freq
            mlflow.log_artifacts(os.path.dirname(ckptm.latest_checkpoint), artifact_path=f'model/ckpt-fold{i}-epoch{j}')
            # extract and log model metrics
            mlflow.log_metrics({k: v[0] for k, v in model_metrics.items()})
            # draw and plot samples
            fig = flvm_plot(model_joint, test_ds_paired, test_hi.lat, test_hi.lon)
            plt.savefig(f'/tmp/samples-fold{i}-epoch{j}.png')
            mlflow.log_artifact(f'/tmp/samples-fold{i}-epoch{j}.png', f'figures')
            # make predictions
            y_true, y_pred = predict_batched(test_ds_paired, lambda x: model_joint.predict_y(x))
            metrics = base.eval_metrics(indices, y_true, y_pred, monthly_means_hi, test_hi.coords)
            avg_metrics = {k: float(np.mean(v)) for k,v in metrics.items()}
            mlflow.log_metrics(avg_metrics)
            # create plots
            fig = base.plot_indices(metrics)
            plt.savefig(f'/tmp/indices-fold{i}-epoch{j}.png')
            mlflow.log_artifact(f'/tmp/indices-fold{i}-epoch{j}.png', 'figures')
            fig = base.plot_error_maps(metrics, test_hi.lat, test_hi.lon)
            plt.savefig(f'/tmp/err-fold{i}-epoch{j}.png')
            mlflow.log_artifact(f'/tmp/err-fold{i}-epoch{j}.png', 'figures')
        # delete checkpoint temp folder after completion
        shutil.rmtree(ckpt_dir)

@click.command(help="Fits and validates a Glow FLVM on the ERA-I/Rasmussen dataset")
@click.option("--scale", type=click.INT, required=True, help="Downscaling factor")
@click.option("--layers", type=click.INT, default=3, help="Number of Glow layers")
@click.option("--depth", type=click.INT, default=8, help="Number of Glow steps per layer")
@click.option("--min-filters", type=click.INT, default=32, help="Minimum number of filters in Glow affine coupling NNs")
@click.option("--max-filters", type=click.INT, default=256, help="Maximum number of filters in Glow affine coupling NNs")
@click.option("--lam", type=click.FLOAT, default=1.0, help="Weight parameter 'lambda' for MLE loss in joint FLVM")
@click.option("--alpha", type=click.FLOAT, default=0.0, help="Weight parameter 'alpha' for auxiliary losses in joint FLVM")
@click.option("--dnet-layers", type=click.INT, default=3, help="Number of layers in the adversarial critic")
@click.option("--dnet-filters", type=click.INT, default=64, help="Number of filters in the critic")
@click.option("--batch-size", type=click.INT, default=10, help="Sample batch size")
@click.option("--buffer-size", type=click.INT, default=1200, help="Load batch size")
@click.option("--validate-freq", type=click.INT, default=5, help="Validate/evaluate frequency (epoch interval)")
@click.option("--epochs", type=click.INT, default=20, help="Number of epochs to train")
@click.option("--splits", type=click.INT, default=3, help="Number of CV splits to use")
@click.option("--region", type=click.STRING, default='southeast_us')
@click.option("--var", type=click.STRING, default='MAXT', help="Dataset var name")
@click.option("--supervised", type=click.BOOL, default=False, help="Whether or not to use supervised training (paired samples)")
@click.option("--auth", type=click.STRING, default='gcs.secret.json', help="GCS keyfile")
@click.argument("data_lr")
def glow_jflvm_cv(data_lr, scale, splits, region, var, auth, **kwargs):
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
    data_lo = data_lo[[var]].fillna(0.).clip(min=0.0, max=np.inf).isel(Time=slice(0,data_lo.Time.size-365))
    data_hi = data_hi[[var]].fillna(0.).clip(min=0.0, max=np.inf).isel(Time=slice(0,data_hi.Time.size-365))
    # create train/test splits
    split_fn = create_time_series_train_test_generator(n_splits=splits)
    folds = list(split_fn(data_lo, data_hi))
    for i, fold in enumerate(folds):
        logging.info(f'Fold {i+1}/{len(folds)}')
        with mlflow.start_run(nested=True):
            if var == 'MAXT':
                fit_glow_jflvm_maxt(fold, i, **kwargs)
            else:
                raise NotImplementedError(f"variable {var} not supported")
import tensorflow as tf
import xarray as xr
from datasource import EraiRasDataLoader
from regions import southeast_us, pacific_nw

def upsample(new_wt, new_ht, method, scale_factor=1):
    @tf.function
    def _upsample(x):
        return tf.image.resize(x, (new_wt,new_ht), method=method) / scale_factor
    return _upsample

def prepare_downscaling_data(data_lo, data_hi, batch_size=100, buffer_size=1000, supervised=True):
    if supervised:
        data = tf.data.Dataset.zip((data_lo, data_hi)).shuffle(buffer_size)
    else:
        data = tf.data.Dataset.zip((data_lo.shuffle(buffer_size), data_hi.shuffle(buffer_size)))
    return data.batch(batch_size)

def load_data(data_lr, data_hr, region, auth_token, scale=4, project='thesis-research-255223'):
    if region == 'southeast_us':
        region_fn = southeast_us
    elif region == 'pacific_nw':
        region_fn = pacific_nw
    data = EraiRasDataLoader(gcs_bucket='erai-rasmussen', gcs_project=project, auth=auth_token)
    if data_lr.startswith('ras/'):
        data_lr = xr.open_zarr(data.rasmussen(data_lr.split('/')[-1]), consolidated=True)
    elif data_lr.startswith('erai/'):
        data_lr = xr.open_zarr(data.erai(data_lr.split('/')[-1]), consolidated=True)
    else:
        raise NotImplementedError(data_lr)
    data_hr = xr.open_zarr(data.rasmussen(data_hr), consolidated=True)
    reg_lr = region_fn(data_lr)
    reg_hr = region_fn(data_hr, scale_factor=scale)
    return reg_lr, reg_hr

def predict_batched(data_xy, predict_fn):
    y_true, y_pred = [], []
    for x, y in data_xy:
        y_true.append(y)
        y_ = predict_fn(x)
        y_pred.append(y_)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    return y_true, y_pred
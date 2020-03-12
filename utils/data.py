import numpy as np
import tensorflow as tf
import xarray as xr
import utils.nn_util as nn
from utils.stats import Stats
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit
from functools import reduce

class DataContainer:
    """
    Container type for train/test partitions.
    """    
    def __init__(self, datasets, train, test, stats):
        self.dataset = datasets if len(datasets) > 1 else datasets[0]
        self.train = train if len(train) > 1 else train[0]
        self.test = test if len(test) > 1 else test[0]
        self.stats = stats if len(stats) > 1 else stats[0]
        
def extract_patches(tf_dataset: tf.data.Dataset, k=1, stride=1, scale=1, batch_size=100, batch_input=False):
    scales = scale if hasattr(scale, '__iter__') else [scale]
    def _extract_patches(*xs):
        assert len(xs) == len(scales), 'inputs must be aligned with scales: got {}, expected {}'.format(len(xs), len(scales))
        if len(xs) > 1:
            return tuple([nn.extract_patches(x, k*scale, stride*scale) for x, scale in zip(xs, scales)])
        else:
            return nn.extract_patches(xs[0], k*scale, stride*scale)
    if batch_input:
        return tf_dataset.map(_extract_patches)
    else:
        return tf_dataset.batch(batch_size).map(_extract_patches).unbatch()

def tensor_data_gen(batch_size, *xs):
    def _gen_tensors(x):
        for i in range(0, x.shape[0], batch_size):
            st, en = i, np.minimum(i+batch_size, x.shape[0])
            yield from (x for x in x[st:en].values)
    if len(xs) > 1:
        yield from zip(*[_gen_tensors(x) for x in xs])
    else:
        yield from _gen_tensors(xs[0])
    
def xr_to_tf_dataset(data, batch_size):
    data = data if isinstance(data, list) else [data]
    data_gen = lambda: tensor_data_gen(batch_size, *data)
    types = tuple([tf.float32 for _ in data]) if len(data) > 1 else tf.float32
    shapes = tuple([x.shape[1:] for x in data]) if len(data) > 1 else data[0].shape[1:]
    return tf.data.Dataset.from_generator(data_gen, types, output_shapes=shapes)
    
def create_time_series_train_test_generator(n_splits=2, time_dim='Time', var_dim='chan'):
    def dataset_to_array(ds):
        arr = ds.to_array(dim=var_dim)
        return arr.transpose(*arr.dims[1:], var_dim)
    def time_series_split(*datasets):
        chunks = datasets[0].chunks[time_dim]
        chunk_size = chunks[0]
        timecv = TimeSeriesSplit(n_splits=n_splits)
        N = datasets[0][time_dim].size
        for train, test in timecv.split(list(range(len(chunks)))):
            train_st, train_en = train[0], train[-1] + 1
            test_st, test_en = test[0], test[-1] + 1
            train_inds = list(range(train_st*chunk_size, np.minimum(train_en*chunk_size, N)))
            test_inds = list(range(test_st*chunk_size, np.minimum(test_en*chunk_size, N)))
            train_data = [dataset_to_array(ds).isel({time_dim: train_inds}) for ds in datasets]
            test_data = [dataset_to_array(ds).isel({time_dim: test_inds}) for ds in datasets]
            yield train_data, test_data
    return time_series_split

def create_time_series_train_test_generator_v2(n_splits=2, test_size=30, time_dim='Time', var_dim='chan'):
    def dataset_to_array(ds):
        arr = ds.to_array(dim=var_dim)
        return arr.transpose(*arr.dims[1:], var_dim)
    def split(n):
        n = int(n)
        inds = list(range(n))
        for i in range(n_splits):
            test_st = n-(n_splits - i)*test_size
            train_inds = inds[i*test_size:test_st]
            test_inds = inds[test_st:test_st+test_size]
            yield train_inds, test_inds
    def time_series_split(*datasets):
        N = datasets[0][time_dim].size
        for train, test in split(N):
            train_data = [dataset_to_array(ds).isel({time_dim: train}) for ds in datasets]
            test_data = [dataset_to_array(ds).isel({time_dim: test}) for ds in datasets]
            yield train_data, test_data
    return time_series_split
        
def generate_seasonal_inds(X, n_folds=1, test_ratio=0.2, rand_seed=None):
    """
    Generates indices for a randomized, uniform train/test split independently for each season.
    If n_folds > 1, test_ratio will be ignored.
    """
    seasons = X.groupby('Time.season')
    for _, season in seasons:
        if n_folds > 1:
            splitter = KFold(n_splits=n_folds, random_state=rand_seed)
        else:
            splitter = ShuffleSplit(n_splits=n_folds, test_size=test_ratio, random_state=rand_seed)
        yield [(np.sort(train), np.sort(test)) for train, test in splitter.split(season)]

def create_seasonal_train_test_generator(seasonal_folds, time_dim='Time'):
    def generate_train_test_splits(*datasets):
        """
        Splits X and y into train/test partitions using the given seasonal folds.
        X and y are expected to be xarray Datasets or DataArrays with a datetime formatted time dimension.
        """
        datasets = list(datasets)
        # broadcast and chunk data before indexing
        for i, ds in enumerate(datasets):
            datasets[i] = xr.broadcast(ds)[0].chunk({k: v[0] for k, v in ds.chunks.items()})
        # group by seasons
        ds_seasons = [list(ds.groupby(f'{time_dim}.season')) for ds in datasets]
        # index and concatenate folds
        for folds in zip(*seasonal_folds):
            def select_seasons(season_data):
                for (_, season), (train_inds, test_inds) in zip(season_data, folds):
                    train, test = season.isel({time_dim: train_inds}), season.isel({time_dim: test_inds})
                    yield train, test
            # select and merge folds for each season
            fold = []
            for ds, seasons in zip(datasets, ds_seasons):
                train_folds, test_folds = zip(*select_seasons(seasons))
                train = reduce(lambda acc, x: xr.concat([acc, x], dim=time_dim), train_folds)
                test = reduce(lambda acc, x: xr.concat([acc, x], dim=time_dim), test_folds)
                fold.append((train, test))
            yield fold
    return generate_train_test_splits

def batch_gen(batch_size, random_seed=None):
    randomizer = np.random.RandomState(seed=random_seed)
    def _gen_batch(x, y=None):
        n = len(x)
        assert y is None or len(y) == n
        inds = np.array(range(n))
        if random_seed is not None:
            randomizer.shuffle(inds)
        for i in range(0, n, batch_size):
            st, en = i, np.minimum(i + batch_size, n)
            if y is not None:
                yield x[inds[st:en]], y[inds[st:en]]
            else:
                yield x[inds[st:en]]
    return _gen_batch

def num_batches(n, batch_size):
    steps = n // batch_size
    return steps if n % batch_size == 0 else steps + 1

def calculate_n_subimages(data, k, stride):
    """
    Calculates the number of subimages generated for a single timestep of 'data'
    given a k x k subimage window and stride.
    """
    return (1 + (data.lat.size - k) // stride)*(1 + (data.lon.size - k) // stride)

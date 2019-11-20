import numpy as np
import xarray as xr

class Pipeline:
    def __init__(self, varnames, *ops):
        self.varnames = varnames if isinstance(varnames, list) else [varnames]
        self.ops = list(ops)
        
    def __call__(self, *inputs):
        return self.transform_all(*inputs)
        
    def transform(self, x):
        chunks = x.chunks
        x = xr.broadcast(x)[0].chunk({k: v[0] for k, v in chunks.items()})
        x = x[self.varnames]
        for op in self.ops:
            x = op(x)
        return x
    
    def transform_all(self, *xs):
        xs_post = []
        for x in xs:
            x_post = self.transform(x)
            xs_post.append(x_post)
        return xs_post if len(xs_post) > 1 else xs_post[0]
        
def fillnan(value=0.0):
    def _fill_op(x):
        return x.fillna(value)
    return _fill_op

def clip(min_val=-np.inf, max_val=np.inf):
    def _clip_op(x):
        return x.clip(min_val, max_val)
    return _clip_op

def remove_monthly_means(time_dim='Time'):
    month_index = f'{time_dim}.month'
    def _remove_means(x):
        monthly_means = x.groupby(month_index).mean(dim='Time')
        for month in np.unique(x[month_index]):
            x = xr.where(x[month_index] == month, x - monthly_means.sel(month=month), x)
        return x
    return _remove_means

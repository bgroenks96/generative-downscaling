import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap

def plot_image(X):
    X = num_to_nan(X, lo_thresh=50.0)
    X_min, X_max = 240, 330
    img = plt.imshow(X.squeeze(), origin='lower', cmap='gist_ncar')
    img.set_clim(vmin=X_min, vmax=X_max)
    plt.colorbar(fraction=0.025, pad=0.05)

def plot_image_map(X, lats, lons, cmap="viridis", figsize=(6,4), title='', min_max=None,  bins=100, gridlines=False):
    X = X.squeeze()
    assert len(X.shape) == 2
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], proj)
    min_val, max_val = (min_max[0], min_max[1]) if min_max is not None else (np.nanmin(X), np.nanmax(X))
    breaks = np.linspace(min_val-1.0E-15, max_val+1.0E-15, bins+1)
    cs = ax.contourf(lons, lats, X, breaks, cmap=cmap, transform=proj, extend='both')
    cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.04)
    ax.coastlines()
    if gridlines:
        ax.gridlines(crs=proj, draw_labels=True)
        ax.set_title(title, y=1.09)
    else:
        ax.set_title(title)
    
def image_map_factory(rows, cols, figsize=(6,4), cmap='viridis', min_max=None, bins=100, gridlines=False,
                      wspace=0.05, hspace=0.05, cbar_per_subplot=False, cbar_orientation='horizontal'):
    proj = ccrs.PlateCarree()
    fig, axarr = plt.subplots(rows, cols, figsize=(cols*figsize[0], rows*figsize[1]), subplot_kw={'projection': proj})
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    def plot_next(ax, X, lats, lons, cmap=cmap, title='', bins=bins, min_max=min_max, gridlines=gridlines):
        X = X.squeeze()
        assert len(X.shape) == 2
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], proj)
        min_val, max_val = (min_max[0]-1.0E-15, min_max[1]+1.0E-15) if min_max is not None else (np.nanmin(X), np.nanmax(X))
        breaks = np.linspace(min_val-1.0E-15, max_val+1.0E-15, bins+1)
        cs = ax.contourf(lons, lats, X, breaks, cmap=cmap, transform=proj, extend='both')
        if cbar_per_subplot:
            plt.colorbar(cs, ax=ax, orientation=cbar_orientation, pad=0.03)
        ax.coastlines()
        if gridlines:
            ax.gridlines(crs=proj, draw_labels=True)
            ax.set_title(title, y=1.09)
        else:
            ax.set_title(title)
        return cs
    return fig, axarr, plot_next

def prcp_cmap():
    colors = [(255.,255.,255.),
              (214.,226.,255.),
              (181.,201.,255.),
              (142.,178.,255.),
              (127.,150.,255.),
              (99.,112.,247.),
              (0.,99.,255.),
              (0.,150.,150.),
              (0.,198.,51.),
              (99.,255.,0.),
              (150.,255.,0.),
              (198.,255.,51.),
              (255.,255.,0.),
              (255.,198.,0.),
              (255.,160.,0.),
              (255.,124.,0.),
              (255.,25.,0.)]
    colors = np.array(colors) / 255.
    return ListedColormap(colors)
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Utilities for common ops
def area_avg(x):
    return x.weighted(np.cos(np.deg2rad(x.lat))).mean(('lat', 'lon'))

def calc_anom(x):
    xclim = x.groupby('time.month').mean()
    return x - xclim.sel(month=x.time.dt.month)

def plot_map(x, title, units, **kwargs):
    ax = plt.axes(projection=ccrs.PlateCarree())
    p = x.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=dict(label=units, orientation='horizontal'), **kwargs)
    ax.set_global()
    ax.coastlines()
    plt.title(title)
    return p


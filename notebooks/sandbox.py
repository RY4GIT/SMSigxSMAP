#!usr/bin/env python
# -*- coding: utf-8 -*-


#%% IMPORTS

import os
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm


#%% DATA IMPORT

dir_name = f"fit_models_py_raraki_2023-11-09"
# dir_name = f"fit_models_py_raraki_2023-10-26_CONUS"
data_dir = "/home/brynmorgan/waves/projects/smap-drydown/data"
datarod_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"

results_file = rf"/home/raraki/waves/projects/smap-drydown/output/{dir_name}/all_results.csv"
_df = pd.read_csv(results_file)
coord_info_file = "/home/raraki/waves/projects/smap-drydown/data/datarods/coord_info.csv"
coord_info = pd.read_csv(coord_info_file)
df = _df.merge(coord_info, on=['EASE_row_index', 'EASE_column_index'], how='left')


# Ancillary data
df_anc = pd.read_csv(
    os.path.join(data_dir, datarod_dir, anc_file)
).drop(["spatial_ref", "latitude", "longitude"], axis=1)
df_anc.loc[df_anc["sand_fraction"] < 0, "sand_fraction"] = np.nan


# Aridity indices
df_ai = pd.read_csv(
    os.path.join(data_dir, datarod_dir, ai_file)
).drop(["latitude", "longitude"], axis=1)
df_ai.loc[df_ai["AI"] < 0, "AI"] = np.nan

# Land cover
IGBPclass = pd.read_csv(
    os.path.join(data_dir, anc_dir, IGBPclass_file)
)

df = df.merge(df_anc, on=['EASE_row_index', 'EASE_column_index'], how='left')
df = df.merge(df_ai, on=['EASE_row_index', 'EASE_column_index'], how='left')
df = pd.merge(df, IGBPclass, left_on='IGBP_landcover', right_on='class', how='left')


df = df.assign(_diff_R2=df["q_r_squared"] - df["exp_r_squared"])

df_filt = df[df['q_r_squared'] >= 0.7].copy()
df_filt2 = df[(df['q_r_squared'] >= 0.7) | (df['exp_r_squared'] >= 0.7)].copy()
df_filt_exp = df[df['exp_r_squared'] >= 0.7].copy()


#%% PLOT PARAMS + COLORS

# Define the three colors in the colormap
colors = ['#d8b365', '#f5f5f5', '#5ab4ac']

# Define the specific order for your categories.
vegetation_orders = ["BAR", "OSH", "CNM", "WSA", "SAV", "GRA", "CRO", "ENF", "CSH", "WET"]
veg_colors = ["#7A422A", "#C99728", "#229954", "#4C6903", "#92BA31", "#13BFB2", "#F7C906", "#022E1F", "#A68F23", "#4D5A6B"]

# Create a color palette dictionary
palette_dict = dict(zip(vegetation_orders, veg_colors))

# Create a custom colormap
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_BrBG', colors, N=256)
cmap = "BrBG"


#%% MAPS

stat_methods = {
    'count': lambda x: x.count(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
    'mean': lambda x: x.mean(),
    'median': lambda x: x.median(),
    'sum': lambda x: x.sum(),
    'std': lambda x: x.std()
}

def aggregate_data(
        df=None, model_type=None, parameter=None, stat='mean'
    ):
    varname = f"{model_type}_{parameter}"
    agg_values = df.groupby(['latitude', 'longitude'])[varname].agg(stat)

    data_array = agg_values.reset_index().pivot(
        index='latitude', columns='longitude', values=varname
    )

    return data_array




def plot_map(ax, data_array=None, parameter=None, norm=None):

    # Create a grid of lat and lon coordinates
    lons, lats = np.meshgrid(data_array.columns, data_array.index)
    # Plot the heatmap using Cartopy
    im = ax.pcolormesh(
        lons, lats, data_array.values, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(),
    ) #, vmin=vmin, vmax=vmax)

    # Add coastlines
    ax.coastlines()
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label=f'Mean {parameter}')
    cbar.ax.set_position([0.92, 0.1, 0.02, 0.8])
    
    # Set plot title and labels
    # ax.set_title(f'Mean {parameter} per pixel')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')

    # plt.show()
    # plt.tight_layout()

    return ax


#%%

da_q = aggregate_data(df=df_filt, model_type="q", parameter="q")


# Create a figure and axes with Cartopy projection
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

norm = TwoSlopeNorm(vmin=1, vcenter=2.5, vmax=4)
ax = plot_map(ax=ax, data_array=da_q, parameter="q", norm=norm)
ax.set_title(r"Mean $q$ per pixel")
plt.show()
plt.tight_layout()

# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=8)
# ax = plot_map(ax=ax, data_array=da_q, parameter="q", norm=norm)

# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# da_q = aggregate_data(df=df_filt, model_type="exp", parameter="tau")
# ax = plot_map(ax=ax, data_array=da_q, parameter="tau")

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

norm = TwoSlopeNorm(vmin=0, vcenter=4, vmax=8)
ax = plot_map(ax=ax, data_array=da_q, parameter="tau", norm=norm)
ax.set_title(r"Mean $\tau$ per pixel")
plt.show()
plt.tight_layout()


df_filt["k_dernormalize"] = df_filt["q_k"] * (df_filt["max_sm"] - df_filt["min_sm"])
df_filt["q_AET/PET"] = df_filt["k_dernormalize"]/df_filt["pet"]

#%% NEW: MAPS OF STD

df_q = aggregate_data(df=df_filt, model_type="q", parameter="q", stat='std')


# Create a figure and axes with Cartopy projection
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

norm = TwoSlopeNorm(vmin=1, vcenter=2.5, vmax=4)
ax = plot_map(ax=ax, data_array=df_q, parameter="q", norm=norm)
ax.set_title(r"$\sigma_{q}$ per pixel")
plt.show()
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

norm = TwoSlopeNorm(vmin=0, vcenter=4, vmax=8)
ax = plot_map(ax=ax, data_array=df_q, parameter="tau", norm=norm)
ax.set_title(r"$\sigma_{\tau}$ per pixel")
plt.show()
plt.tight_layout()






#%% BOX PLOTS

n_bins = 10
sand_bins = [i * 0.1 for i in range(11)]
ai_bins = [i * 0.25 for i in range(9)]
df_filt['sand_bins'] = pd.cut(df_filt['sand_fraction'], bins=sand_bins, include_lowest=True)
df_filt['ai_bins'] = pd.cut(df_filt['AI'], bins=ai_bins, include_lowest=True)


x = 'sand_bins'
x_lab = "Sand fraction"

# x = 'ai_bins'
# x_lab = "Aridity Index"

y = 'q_q'
y_lab = "Nonlinear q parameter"


fig = plt.figure(figsize=(6, 4))
ax = sns.boxplot(x=x, y=y, data=df_filt, boxprops=dict(facecolor='lightgray'))
plt.setp(ax.get_xticklabels(), rotation=45)

ax.set_ylabel(y_lab)
ax.set_xlabel(x_lab)
plt.tight_layout()


x = 'abbreviation'
x_lab = "IGBP Landcover Class"

y = 'q_q'
y_lab = "Nonlinear q parameter"


# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 4))

# Plot the boxplot with specified colors and increased alpha
sns.boxplot(
    x=x,
    y=y,
    data=df_filt,
    # hue = colors, 
    order=vegetation_orders,
    palette=veg_colors,
    ax=ax
)

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

# Optionally, adjust layout
plt.tight_layout()
ax.set_ylabel("Nonlinear q parameter")
ax.set_xlabel("IGBP Landcover Class")
# Show the plot
plt.show()


# %%

pixels = df_filt.groupby(['latitude', 'longitude'])[
    ['abbreviation']
].agg('max').reset_index()

df_agg = df_filt.groupby(['latitude', 'longitude'])['q_q'].agg('median').reset_index()

df_agg = df_agg.merge(pixels, on=['latitude', 'longitude'], how='left')


x = 'abbreviation'
x_lab = "IGBP Landcover Class"

y = 'q_q'
y_lab = "Nonlinear q parameter"

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 4))

# Plot the boxplot with specified colors and increased alpha
sns.boxplot(
    x=x,
    y=y,
    data=df_agg,
    # hue = colors, 
    order=vegetation_orders,
    palette=veg_colors,
    ax=ax
)

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

# Optionally, adjust layout
plt.tight_layout()
ax.set_ylabel("Nonlinear q parameter")
ax.set_xlabel("IGBP Landcover Class")
# Show the plot
plt.show()


# %%

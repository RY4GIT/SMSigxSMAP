# %% Import packages
import os
import getpass

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interpn

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
from matplotlib.colors import Normalize 
import datashader as ds
from datashader.mpl_ext import dsshow

from functions import q_drydown, exponential_drydown, loss_model

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2023-11-25_global_95asmax"
###########################################################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define model acceptabiltiy criteria
success_modelfit_thresh = 0.7
sm_range_thresh = 0.3
z_mm = 50 # Soil thickness

# cmap for sand
sand_bin_list = [i * 0.1 for i in range(11)]
sand_cmap = "Oranges"

# cmap for ai
ai_bin_list = [i * 0.25 for i in range(9)]
ai_cmap = "RdBu"

# Define the specific order for vegetation categories.
vegetation_color_dict = {
    "Barren": "#7A422A", 
    "Open shrublands": "#C99728", 
    "Grasslands": "#13BFB2", 
    "Savannas": "#92BA31",
    "Woody savannas": "#4C6903",
    "Croplands": "#F7C906",
    "Cropland/natural vegetation mosaics": "#229954",
}

var_dict = {
    'theta' : {
        'column_name': 'sm',
        'symbol' : r"$\theta$",
        'label' : r"SMAP soil moisture $\theta$",
        'unit' : r"$[m^3/m^3]$",
        'lim': [0, 4],
    },
    'dtheta' : {
        'column_name': '',
        'symbol' : r"$d\theta/dt$",
        'label' : r"Change in soil moisture $d\theta/dt$",
        'unit' : r"$[m^3/m^3/day]$",
        'lim': [0, 4],
    },
    'q_q' : {
        'column_name': 'q_q',
        'symbol' : r"$q$",
        'label' : r"Nonlinear parameter $q$",
        'unit' : "[-]",
        'lim': [0, 4],
    },
    'ETmax' : {
        'column_name': 'q_ETmax',
        'symbol' : r"$ET_{max}$",
        'label' : r"Estimated $ET_{max}$ by non-linear model",
        'unit' : "[mm/day]",
        'lim': [0, 20],
    },
    's*' : {
        'column_name': 'max_sm',
        'symbol' : r"$s_{*}$",
        'label' : r"Estimated $s_{*}$",
        'unit' : r"$[m^3/m^3]$",
        'lim': [-0.04, 0.04],
    },
    'sand_bins' : {
        'column_name': 'sand_bins',
        'symbol' : r"",
        'label' : r"Sand fraction",
        'unit' : "[-]",
        'lim': [0.0, 1.0],
    },
    'ai_bins' : {
        'column_name': 'ai_bins',
        'symbol' : r"AI",
        'label' : r"Aridity Index",
        'unit' : "[MAP/MAE]",
        'lim': [0.0, 2.0],
    },
    'veg_class' : {
        'column_name': 'name',
        'symbol' : r"",
        'label' : r"IGBP Landcover Class",
        'unit' : "",
        'lim': [0, 1],
    },
}
############################################################################

# %% DATA IMPORT 

# Data dir
user_name = getpass.getuser()
data_dir = rf"/home/{user_name}/waves/projects/smap-drydown/data"
datarod_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"
coord_info_file = "coord_info.csv"

# Read the output
output_dir = rf"/home/{user_name}/waves/projects/smap-drydown/output"
results_file = rf"all_results.csv"
_df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
print('Loaded results file')

# Read coordinate information
coord_info = pd.read_csv(os.path.join(data_dir, datarod_dir, coord_info_file))
df = _df.merge(coord_info, on=['EASE_row_index', 'EASE_column_index'], how='left')
print('Loaded coordinate information')

# Ancillary data
df_anc = pd.read_csv(
    os.path.join(data_dir, datarod_dir, anc_file)
).drop(["spatial_ref", "latitude", "longitude"], axis=1)
df_anc.loc[df_anc["sand_fraction"] < 0, "sand_fraction"] = np.nan
print('Loaded ancillary information (sand fraction and land-cover)')

# Aridity indices
df_ai = pd.read_csv(
    os.path.join(data_dir, datarod_dir, ai_file)
).drop(["latitude", "longitude"], axis=1)
df_ai.loc[df_ai["AI"] < 0, "AI"] = np.nan
print('Loaded ancillary information (aridity index)')

# Land cover
IGBPclass = pd.read_csv(
    os.path.join(data_dir, anc_dir, IGBPclass_file)
)

df = df.merge(df_anc, on=['EASE_row_index', 'EASE_column_index'], how='left')
df = df.merge(df_ai, on=['EASE_row_index', 'EASE_column_index'], how='left')
df = pd.merge(df, IGBPclass, left_on='IGBP_landcover', right_on='class', how='left')
print('Loaded ancillary information (land-cover)')

print(f"Total number of drydown event: {len(df)}")

# %% Get some stats

# Difference between R2 values of two models
df = df.assign(diff_R2=df["q_r_squared"] - df["exp_r_squared"])

# Denormalize k and calculate the estimated ETmax values from k parameter from q model
df["q_ETmax"] = df["q_k"] * (df["max_sm"] - df["min_sm"]) * z_mm

# Get the binned dataset

# sand bins
df['sand_bins'] = pd.cut(df['sand_fraction'], bins=sand_bin_list, include_lowest=True)
first_I = df['sand_bins'].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df['sand_bins'] = df['sand_bins'].cat.rename_categories({first_I: new_I})

# ai_bins
df['ai_bins'] = pd.cut(df['AI'], bins=ai_bin_list, include_lowest=True)
first_I = df['ai_bins'].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df['ai_bins'] = df['ai_bins'].cat.rename_categories({first_I: new_I})

# %%
# Soil mositure range covered by the observation
def calculate_sm_range(row):
    input_string = row.sm

    # Processing the string
    input_string = input_string.replace('\n', ' np.nan')
    input_string = input_string.replace(' nan', ' np.nan')
    input_string = input_string.strip('[]')

    # Converting to numpy array and handling np.nan
    sm = np.array([float(value) if value != 'np.nan' else np.nan for value in input_string.split()])

    # Calculating sm_range
    sm_range = (np.nanmax(sm) - np.nanmin(sm)) / (row.max_sm - row.min_sm) if row.max_sm != row.min_sm else np.nan
    return sm_range

# Applying the function to each row and creating a new column 'sm_range'
df['sm_range'] = df.apply(calculate_sm_range, axis=1)

# %% Exclude model fits failure

# Runs where q model performed good 
df_filt_q = df[df['q_r_squared'] >= success_modelfit_thresh].copy()
df_filt_q_2 = df_filt_q[df_filt_q['sm_range']> sm_range_thresh].copy()
print(f"q model fit was successful: {len(df_filt_q)}")
print(f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_2)}")

# Runs where exponential model performed good
df_filt_exp = df[df['exp_r_squared'] >= success_modelfit_thresh].copy()
df_filt_exp_2 = df_filt_exp[df_filt_exp['sm_range']> sm_range_thresh].copy()
print(f"exp model fit was successful: {len(df_filt_exp)}")
print(f"exp model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_exp_2)}")

# Runs where either of the model performed satisfactory
df_filt_q_or_exp = df[(df['q_r_squared'] >= success_modelfit_thresh) | (df['exp_r_squared'] >= success_modelfit_thresh)].copy()
df_filt_q_or_exp_2 = df_filt_q_or_exp[df_filt_q_or_exp['sm_range']> sm_range_thresh].copy()
print(f"both q and exp model fit was successful: {len(df_filt_q_or_exp)}")
print(f"both q and exp model were successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_or_exp_2)}")

# Runs where both of the model performed satisfactory
df_filt_q_and_exp = df[(df['q_r_squared'] >= success_modelfit_thresh) | (df['exp_r_squared'] >= success_modelfit_thresh)].copy()
df_filt_q_and_exp_2 = df_filt_q_and_exp[df_filt_q_and_exp['sm_range']> sm_range_thresh].copy()
print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")
print(f"both q and exp model were successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_and_exp_2)}")

# %% 
############################################################################
# Model performance comparison
###########################################################################
def using_datashader(ax, x, y, cmap):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        norm="linear",
        aspect="auto",
        vmin=0,
        vmax=30,
        ax=ax,
        cmap=cmap
    )
    plt.colorbar(dsartist, label=f'Gaussian density [-]')


def plot_R2_models(df, R2_threshold, cmap):

    # Read data
    x = df['exp_r_squared'].values
    y = df['q_r_squared'].values

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    # Calculate the point density
    sc = using_datashader(ax, x, y, cmap)

    # plt.title(rf'')
    plt.xlabel(r'$R^2$ of Linear loss model')
    plt.ylabel(r'$R^2$ of Non-linear loss model')

    # Add 1:1 line
    ax.plot([R2_threshold, 1], [R2_threshold, 1], color='k', linestyle='--', label='1:1 line')

    # Add a trendline
    coefficients = np.polyfit(x, y, 1)
    trendline_x = np.array([R2_threshold, 1])
    trendline_y = coefficients[0] * trendline_x + coefficients[1]
    ax.plot(trendline_x, trendline_y, color='k', label='Trendline')

    ax.set_xlim([R2_threshold, 1])
    ax.set_ylim([R2_threshold, 1])
    plt.legend()

# plot_R2_models(df=df, R2_threshold=0.0)

# Plot R2 of q vs exp model, where where both q and exp model performed R2 > 0.7 and covered >30% of the SM range 
plot_R2_models(df=df_filt_q_and_exp_2, R2_threshold=success_modelfit_thresh, cmap="viridis")

# %%
############################################################################
# Map plots 
###########################################################################
def plot_map(df, coord_info, cmap, norm, var_item):

    # Get the mean values of the variable
    stat = df.groupby(['EASE_row_index', 'EASE_column_index'])[var_item["column_name"]].mean()

    # Reindex to the full EASE row/index extent 
    new_index = pd.MultiIndex.from_tuples(
        zip(coord_info['EASE_row_index'], coord_info['EASE_column_index']),
        names=['EASE_row_index', 'EASE_column_index']
    )
    stat_pad = stat.reindex(new_index, fill_value=np.nan)

    # Join latitude and longitude
    merged_data = stat_pad.reset_index().merge(
        coord_info[['EASE_row_index', 'EASE_column_index', 'latitude', 'longitude']],
        on=['EASE_row_index', 'EASE_column_index'],
        how='left'
    ).set_index(['EASE_row_index', 'EASE_column_index'])

    # Create pivot array 
    pivot_array = merged_data.pivot(index='latitude', columns='longitude', values=var_item["column_name"])
    pivot_array[pivot_array.index > -60] # Exclude antarctica in the map (no data)

    # Get lat and lon
    lons = pivot_array.columns.values
    lats = pivot_array.index.values
    
    # Plot in the map
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': ccrs.Robinson()})
    im = ax.pcolormesh(lons, lats, pivot_array, norm=norm, cmap=cmap, transform=ccrs.PlateCarree()) 
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label=f'Median {var_item["label"]}', shrink=0.35, pad=0.02)
    
    # Set plot title and labels
    # ax.set_title(f'Mean {variable_name} per pixel')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


# %% Plot the map of q values, where both q and exp models performed > 0.7 and covered >30% of the SM range 
var_key = "q_q"
norm = Normalize(vmin = var_dict[var_key]['lim'][0], vmax = var_dict[var_key]['lim'][1])
plot_map(df=df_filt_q_2, coord_info=coord_info, cmap='YlGnBu', norm=norm, var_item=var_dict[var_key])

# %% Map of R2 values 
# Plot the map of R2 differences, where both q and exp model performed > 0.7 and covered >30% of the SM range 
var_key = "diff_R2" 
norm = Normalize(vmin = var_dict[var_key]['lim'][0], vmax = var_dict[var_key]['lim'][1])
plot_map(df=df_filt_q_and_exp_2, coord_info=coord_info, cmap='RdBu', norm=norm, var_item=var_dict[var_key])

# %%
############################################################################
# Box plots (might go supplemental)  
###########################################################################

def plot_boxplots(df, x_var, y_var):
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x=x_var["column_name"], y=y_var["column_name"], data=df, boxprops=dict(facecolor='lightgray'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel(f'{x_var["label"]} {x_var["unit"]}')
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1]*5)
    plt.tight_layout()
    
# %% sand
plot_boxplots(df_filt_q_and_exp_2, var_dict["sand_bins"], var_dict["q_q"])

# %% Aridity index 
plot_boxplots(df_filt_q_and_exp_2, var_dict["ai_bins"], var_dict["q_q"])

# %% Vegatation
from textwrap import wrap
def wrap_at_space(text, max_width):
    parts = text.split(" ")
    wrapped_parts = [wrap(part, max_width) for part in parts]
    return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])

def plot_boxplots_categorical(df, x_var, y_var, categories, colors):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the boxplot with specified colors and increased alpha
    sns.boxplot(
        x=x_var['column_name'],
        y=y_var['column_name'],
        data=df,
        # hue=x_var['column_name'],
        legend=False, 
        order=categories,
        palette=colors,
        ax=ax
    )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

    # Optionally, adjust layout
    plt.tight_layout()
    ax.set_xlabel(f'{x_var["label"]}')
    max_label_width = 20
    ax.set_xticklabels([wrap_at_space(label.get_text(), max_label_width) for label in ax.get_xticklabels()])
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    # Show the plot
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1]*2)
    plt.tight_layout()
    plt.show()

# %%
plot_boxplots_categorical(df_filt_q_and_exp_2, var_dict["veg_class"], var_dict["q_q"], categories=vegetation_color_dict.keys(), colors=list(vegetation_color_dict.values()))

# %%

def plot_violin_categorical(df, x_var, y_var, categories, colors):
    fig, ax =  plt.subplots(figsize=(8, 4))
    for i, category in enumerate(categories):
        subset = df[df[x_var["column_name"]] == category]
        sns.violinplot(x=x_var["column_name"], y=y_var["column_name"], data=subset, order=[category], color=colors[i], ax=ax, alpha=0.75, cut=0)

    # ax = sns.violinplot(x='abbreviation', y='q_q', data=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'), 
    ax.set_xlabel(f'{x_var["label"]}')
    max_label_width = 20
    ax.set_xticklabels([wrap_at_space(label.get_text(), max_label_width) for label in ax.get_xticklabels()])
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    # Show the plot
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1]*2)
    plt.tight_layout()
    plt.show()
    

plot_violin_categorical(df_filt_q_and_exp_2, var_dict["veg_class"], var_dict["q_q"], categories=vegetation_color_dict.keys(), colors=list(vegetation_color_dict.values()))

# %%
############################################################################
# Loss function plots  
###########################################################################

# %% Sand 




# %% Vegeation




# %% Aridity index 



# %%
############################################################################
# Scatter plots with error bars
###########################################################################

# %% q vs. k per vegetation 




# %% q vs. s* per vegetation


# %% Import packages
import os
import getpass

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde

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
from textwrap import wrap

from functions import q_drydown, exponential_drydown, loss_model
# %%

!pip install mpl-scatter-density
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2023-11-25_global_95asmax"
###########################################################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define model acceptabiltiy criteria
z_mm = 50  # Soil thickness

# Define the specific order for vegetation categories.
vegetation_color_dict = {
    "Barren": "#808080",  # "#7A422A",
    "Open shrublands": "#C99728",
    "Grasslands": "#13BFB2",
    "Savannas": "#92BA31",
    "Woody savannas": "#4C6903",
    "Croplands": "#F7C906",
    "Cropland/natural vegetation mosaics": "#229954",
}

var_dict = {
    "theta": {
        "column_name": "sm",
        "symbol": r"$\theta$",
        "label": r"SMAP soil moisture",
        "unit": r"$[m^3/m^3]$",
        "lim": [0, 0.50],
    },
    "dtheta": {
        "column_name": "",
        "symbol": r"$-d\theta/dt$",
        "label": r"Change in soil moisture",
        "unit": r"$[m^3/m^3/day]$",
        "lim": [-0.10, 0],
    },
    "q_q": {
        "column_name": "q_q",
        "symbol": r"$q$",
        "label": r"Nonlinear parameter $q$",
        "unit": "[-]",
        "lim": [0.5, 4.0],
    },
    "q_ETmax": {
        "column_name": "q_ETmax",
        "symbol": r"$ET_{max}$",
        "label": r"Estimated $ET_{max}$",
        "unit": "[mm/day]",
        "lim": [0, 17.5],
    },
    "theta_star": {
        "column_name": "max_sm",
        "symbol": r"$\theta*$",
        "label": r"Estimated $\theta*$",
        "unit": r"$[m^3/m^3]$",
        "lim": [0.1, 0.45],
    },
    "sand_bins": {
        "column_name": "sand_bins",
        "symbol": r"",
        "label": r"Sand fraction",
        "unit": "[-]",
        "lim": [0.0, 1.0],
    },
    "ai_bins": {
        "column_name": "ai_bins",
        "symbol": r"AI",
        "label": r"Aridity Index",
        "unit": "[MAP/MAE]",
        "lim": [0.0, 2.0],
    },
    "veg_class": {
        "column_name": "name",
        "symbol": r"",
        "label": r"IGBP Landcover Class",
        "unit": "",
        "lim": [0, 1],
    },
    "ai": {
        "column_name": "AI",
        "symbol": r"AI",
        "label": r"Aridity Index",
        "unit": "[MAP/MAE]",
        "lim": [0.0, 1.1],
    },
    "diff_R2": {
        "column_name": "diff_R2",
        "symbol": r"$R^2$",
        "label": r"$R^2$ (Nonlinear - linear)",
        "unit": "[-]",
        "lim": [-0.02, 0.02],
    },
}


# %% ############################################################################
# DATA IMPORT

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
print("Loaded results file")

# Read coordinate information
coord_info = pd.read_csv(os.path.join(data_dir, datarod_dir, coord_info_file))
df = _df.merge(coord_info, on=["EASE_row_index", "EASE_column_index"], how="left")
print("Loaded coordinate information")

# Ancillary data
df_anc = pd.read_csv(os.path.join(data_dir, datarod_dir, anc_file)).drop(
    ["spatial_ref", "latitude", "longitude"], axis=1
)
df_anc.loc[df_anc["sand_fraction"] < 0, "sand_fraction"] = np.nan
print("Loaded ancillary information (sand fraction and land-cover)")

# Aridity indices
df_ai = pd.read_csv(os.path.join(data_dir, datarod_dir, ai_file)).drop(
    ["latitude", "longitude"], axis=1
)
df_ai.loc[df_ai["AI"] < 0, "AI"] = np.nan
print("Loaded ancillary information (aridity index)")

# Land cover
IGBPclass = pd.read_csv(os.path.join(data_dir, anc_dir, IGBPclass_file))

df = df.merge(df_anc, on=["EASE_row_index", "EASE_column_index"], how="left")
df = df.merge(df_ai, on=["EASE_row_index", "EASE_column_index"], how="left")
df = pd.merge(df, IGBPclass, left_on="IGBP_landcover", right_on="class", how="left")
print("Loaded ancillary information (land-cover)")

print(f"Total number of drydown event: {len(df)}")

# %% Create output directory
fig_dir = os.path.join(output_dir, dir_name, "figs")
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")
# %% Get some stats

# Difference between R2 values of two models
df = df.assign(diff_R2=df["q_r_squared"] - df["exp_r_squared"])

# Denormalize k and calculate the estimated ETmax values from k parameter from q model
df["q_ETmax"] = df["q_k"] * (df["max_sm"] - df["min_sm"]) * z_mm
df["q_k_denormalized"] = df["q_k"] * (df["max_sm"] - df["min_sm"])

# Get the binned dataset

# cmap for sand
sand_bin_list = [i * 0.1 for i in range(11)]
sand_bin_list = sand_bin_list[1:]
sand_cmap = "Oranges"

# cmap for ai
ai_bin_list = [i * 0.25 for i in range(7)]
ai_cmap = "RdBu"

# sand bins
df["sand_bins"] = pd.cut(df["sand_fraction"], bins=sand_bin_list, include_lowest=True)
first_I = df["sand_bins"].cat.categories[0]
new_I = pd.Interval(0.1, first_I.right)
df["sand_bins"] = df["sand_bins"].cat.rename_categories({first_I: new_I})

# ai_bins
df["ai_bins"] = pd.cut(df["AI"], bins=ai_bin_list, include_lowest=True)
first_I = df["ai_bins"].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df["ai_bins"] = df["ai_bins"].cat.rename_categories({first_I: new_I})

# %%
# Soil mositure range covered by the observation
def calculate_sm_range(row):
    input_string = row.sm

    # Processing the string
    input_string = input_string.replace("\n", " np.nan")
    input_string = input_string.replace(" nan", " np.nan")
    input_string = input_string.strip("[]")

    # Converting to numpy array and handling np.nan
    sm = np.array(
        [
            float(value) if value != "np.nan" else np.nan
            for value in input_string.split()
        ]
    )

    # Calculating sm_range
    sm_range = (
        (np.nanmax(sm) - np.nanmin(sm)) / (row.max_sm - row.min_sm)
        if row.max_sm != row.min_sm
        else np.nan
    )
    return sm_range


# Applying the function to each row and creating a new column 'sm_range'
df["sm_range"] = df.apply(calculate_sm_range, axis=1)


# %%
def calculate_n_days(row):
    input_string = row.sm

    # Processing the string
    input_string = input_string.replace("\n", " np.nan")
    input_string = input_string.replace(" nan", " np.nan")
    input_string = input_string.strip("[]")

    # Converting to numpy array and handling np.nan
    sm = np.array(
        [
            float(value) if value != "np.nan" else np.nan
            for value in input_string.split()
        ]
    )

    # Calculating sm_range
    n_days = len(sm)
    return n_days


# Applying the function to each row and creating a new column 'sm_range'
df["n_days"] = df.apply(calculate_n_days, axis=1)
df.columns
# %% Exclude model fits failure
print(f"Total number of events: {len(df)}")

# Defining threshold for q value
q_thresh = 1e-03
success_modelfit_thresh = 0.7
sm_range_thresh = 0.1

# Runs where q model performed reasonablly well
df_filt_q = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["q_q"] > q_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()

print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range, plus extremely small q removed: {len(df_filt_q)}"
)

# Runs where q model performed reasonablly well
df_filt_allq = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()

print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_allq)}"
)

# Runs where exponential model performed good
df_filt_exp = df[
    (df["exp_r_squared"] >= success_modelfit_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()
print(
    f"exp model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_exp)}"
)

# Runs where either of the model performed satisfactory
df_filt_q_or_exp = df[
    (
        (df["q_r_squared"] >= success_modelfit_thresh)
        | (df["exp_r_squared"] >= success_modelfit_thresh)
    )
    & (df["sm_range"] > sm_range_thresh)
].copy()

print(f"either q or exp model fit was successful: {len(df_filt_q_or_exp)}")

# Runs where both of the model performed satisfactory
df_filt_q_and_exp = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["exp_r_squared"] >= success_modelfit_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()

print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")


# %%
###################### Statistics
sample_sand_stat = df_filt_q[["id_x", "sand_bins"]].groupby("sand_bins").count()
print(sample_sand_stat)
sample_sand_stat.to_csv(os.path.join(fig_dir, f"sample_sand_stat.csv"))

sample_ai_stat = df_filt_q[["id_x", "ai_bins"]].groupby("ai_bins").count()
print(sample_ai_stat)
sample_ai_stat.to_csv(os.path.join(fig_dir, f"sample_ai_stat.csv"))

sample_veg_stat = df_filt_q[["id_x", "name"]].groupby("name").count()
print(sample_veg_stat)
sample_veg_stat.to_csv(os.path.join(fig_dir, f"sample_veg_stat.csv"))

# Check no data in sand 
print(sum(pd.isna(df_filt_q["sand_fraction"])==True))

# %%
############################################################################
# PLOTTING FUNCTION STARTS HERE
###########################################################################

############################################################################
# Model performance comparison
###########################################################################

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    
def plot_R2_models_v2(df, R2_threshold):
    plt.rcParams.update({"font.size": 30})
    # Read data
    x = df["exp_r_squared"].values
    y = df["q_r_squared"].values

    # Create a scatter plot
    #$ fig, ax = plt.subplots(figsize=(4.5 * 1.2, 4 * 1.2),)
    fig = plt.figure(figsize=(4.7 * 1.2, 4 * 1.2))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis, vmin=0, vmax=30)
    fig.colorbar(density, label='Number of points per pixel')
    plt.show()

    # plt.title(rf'')
    ax.set_xlabel(r"Linear model")
    ax.set_ylabel(r"Non-linear model")

    # Add 1:1 line
    ax.plot(
        [R2_threshold, 1],
        [R2_threshold, 1],
        color="white",
        linestyle="--",
        label="1:1 line",
        linewidth=3)

    # Add a trendline
    coefficients = np.polyfit(x, y, 1)
    trendline_x = np.array([R2_threshold, 1])
    trendline_y = coefficients[0] * trendline_x + coefficients[1]
    ax.plot(trendline_x, trendline_y, color="white", label="Trendline", linewidth=3)

    ax.set_xlim([R2_threshold, 1])
    ax.set_ylim([R2_threshold, 1])
    ax.set_title(r"$R^2$ comparison")

    fig.savefig(os.path.join(fig_dir, f"R2_scatter.png"), dpi=900, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"R2_scatter.pdf"), dpi=900, bbox_inches="tight")
    return fig, ax

# plot_R2_models(df=df, R2_threshold=0.0)

# Plot R2 of q vs exp model, where where both q and exp model performed R2 > 0.7 and covered >30% of the SM range
plot_R2_models_v2(
    df=df_filt_q_and_exp, R2_threshold=success_modelfit_thresh
)

# %%
############################################################################
# Map plots
###########################################################################
def plot_map(ax, df, coord_info, cmap, norm, var_item, stat_type, title=""):
    plt.rcParams.update({"font.size": 12})

    # Get the mean values of the variable
    if stat_type == "median":
        stat = df.groupby(["EASE_row_index", "EASE_column_index"])[
            var_item["column_name"]
        ].median()
        stat_label = "Median"
    elif stat_type == "mean":
        stat = df.groupby(["EASE_row_index", "EASE_column_index"])[
            var_item["column_name"]
        ].mean()
        stat_label = "Mean"

    # Reindex to the full EASE row/index extent
    new_index = pd.MultiIndex.from_tuples(
        zip(coord_info["EASE_row_index"], coord_info["EASE_column_index"]),
        names=["EASE_row_index", "EASE_column_index"],
    )
    stat_pad = stat.reindex(new_index, fill_value=np.nan)

    # Join latitude and longitude
    merged_data = (
        stat_pad.reset_index()
        .merge(
            coord_info[
                ["EASE_row_index", "EASE_column_index", "latitude", "longitude"]
            ],
            on=["EASE_row_index", "EASE_column_index"],
            how="left",
        )
        .set_index(["EASE_row_index", "EASE_column_index"])
    )

    # Create pivot array
    pivot_array = merged_data.pivot(
        index="latitude", columns="longitude", values=var_item["column_name"]
    )
    pivot_array[pivot_array.index > -60]  # Exclude antarctica in the map (no data)

    # Get lat and lon
    lons = pivot_array.columns.values
    lats = pivot_array.index.values

    # Plot in the map
    im = ax.pcolormesh(
        lons, lats, pivot_array, norm=norm, cmap=cmap, transform=ccrs.PlateCarree()
    )
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        label=f'{stat_label} {var_item["label"]} {var_item["unit"]}',
        shrink=0.35,
        pad=0.02,
    )

    # Set plot title and labels
    # ax.set_title(f'Mean {variable_name} per pixel')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title != "": 
        ax.set_title(title, loc="left")

# %% 
#################################
# Map figures (Main manuscript)
################################

# Plot the map of q values, where both q and exp models performed > 0.7 and covered >30% of the SM range
# Also exclude the extremely small value of q that deviates the analysis
var_key = "q_q"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=var_dict[var_key]["lim"][1])
fig_map_q, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median"
)
fig_map_q.savefig(os.path.join(fig_dir, f"q_map_median.png"), dpi=900, bbox_inches="tight")

print(f"Global median q: {df_filt_q['q_q'].median()}")
print(f"Global mean q: {df_filt_q['q_q'].mean()}")

# %% Map of R2 values
# Plot the map of R2 differences, where both q and exp model performed > 0.7 and covered >30% of the SM range
var_key = "diff_R2"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=var_dict[var_key]["lim"][1])
fig_map_R2, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
plot_map(
    ax=ax, 
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
)
fig_map_R2.savefig(os.path.join(fig_dir, f"R2_map_median.png"), dpi=900, bbox_inches="tight")

# %% 

################################
# Map figures (Supplemental)
################################

var_key = "diff_R2"
norm = Normalize(vmin=var_dict[var_key]["lim"][0]*2, vmax=var_dict[var_key]["lim"][1]*2)
fig_map_R2 = plot_map(
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="mean",
)
fig_map_R2.savefig(os.path.join(fig_dir, f"R2_map_mean.png"), dpi=900, bbox_inches="tight")

print(f"Global median diff R2 (nonlinear - linear): {df_filt_q_and_exp['diff_R2'].median()}")
print(f"Global mean diff R2 (nonlinear - linear): {df_filt_q_and_exp['diff_R2'].mean()}")

# %%
# Map of theta_star
var_key = "theta_star"
norm = Normalize(vmin=0.0, vmax=0.6)
fig_map_theta_star, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="(A)"
)
fig_map_theta_star.savefig(os.path.join(fig_dir, f"sup_map_thetastar.png"), dpi=900, bbox_inches="tight")
# fig_map_q.savefig(os.path.join(fig_dir, f"q_map.pdf"), bbox_inches="tight")

print(f"Global median theta_star: {df_filt_q['max_sm'].median()}")
print(f"Global mean theta_star: {df_filt_q['max_sm'].mean()}")

# %%
# Map of ETmax
var_key = "q_ETmax"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=10)
fig_map_ETmax, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="(B)"
)
fig_map_ETmax.savefig(os.path.join(fig_dir, f"sup_map_ETmax.png"), dpi=900, bbox_inches="tight")
# fig_map_q.savefig(os.path.join(fig_dir, f"q_map.pdf"), bbox_inches="tight")

print(f"Global median ETmax: {df_filt_q['q_ETmax'].median()}")
print(f"Global mean ETmax: {df_filt_q['q_ETmax'].mean()}")


# %%
############################################################################
# Histogram plots
###########################################################################

def plot_hist(df, var_key):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=0.5, color="#2c7fb8", fill=False, linewidth=3, ax=ax
    )

    # Calculate median and mean
    median_value = df[var_key].median()
    mean_value = df[var_key].mean()

    # Add median and mean as vertical lines
    ax.axvline(median_value, color='#2c7fb8', linestyle='-', linewidth=3, label=f'Median')
    ax.axvline(mean_value, color='#2c7fb8', linestyle=':', linewidth=3, label=f'Mean')

    # Setting the x limit
    ax.set_xlim(0, 10)

    # Adding title and labels
    # ax.set_title("Histogram of $q$ values")
    ax.set_xlabel(r"$q$ [-]")
    ax.set_ylabel("Frequency [-]")
    fig.legend(loc='upper right', bbox_to_anchor=(0.93, 0.9))

    return fig, ax


fig_q_hist, _ = plot_hist(df=df_filt_q, var_key="q_q")
fig_q_hist.savefig(os.path.join(fig_dir, f"q_hist.png"), dpi=1200, bbox_inches="tight")
fig_q_hist.savefig(os.path.join(fig_dir, f"q_hist.pdf"), dpi=1200, bbox_inches="tight")

# %%
############################################################################
# Loss function plots
###########################################################################

def plot_loss_func(ax, df, z_var, cmap, title="", plot_legend=False):
    
    # Get unique bins
    bins_in_range = df[z_var["column_name"]].unique()
    bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
    bin_sorted = sorted(bins_list, key=lambda x: x.left)

    # For each row in the subset, calculate the loss for a range of theta values
    for i, category in enumerate(bin_sorted):
        subset = df[df[z_var["column_name"]] == category]

        # Get the median of all the related loss function parameters
        theta_min = subset["min_sm"].median()
        theta_max = subset["max_sm"].median()
        denormalized_k = subset["q_k_denormalized"].median()
        q = subset["q_q"].median()

        # Calculate the loss function
        theta = np.arange(theta_min, theta_max, 0.01)
        dtheta = loss_model(
            theta, q, denormalized_k, theta_wp=theta_min, theta_star=theta_max
        )

        # Plot median line
        ax.plot(
            theta,
            dtheta,
            label=f"{category}",
            color=plt.get_cmap(cmap)(i / len(bins_list)),
            linewidth=3
            )

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(
        f"{var_dict['dtheta']['label']}\n{var_dict['theta']['symbol']} {var_dict['dtheta']['unit']}"
    )
    if title=="":
        title=f'Median loss function by {z_var["label"]} {z_var["unit"]}'
    ax.set_title(title, loc="left")

    # ax.set_xlim(var_dict['theta']['lim'][0],var_dict['theta']['lim'][1])
    # ax.set_ylim(var_dict['dtheta']['lim'][1],var_dict['dtheta']['lim'][0])
    if plot_legend: 
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            title=f'{z_var["label"]}\n{z_var["unit"]}',
        )

# %% Vegeation
def wrap_text(text, width):
    return "\n".join(wrap(text, width))


def plot_loss_func_categorical(ax, df, z_var, categories, colors, title="", plot_legend=True):
    # fig, ax = plt.subplots(figsize=(4.2, 4))

    # For each row in the subset, calculate the loss for a range of theta values
    for i, category in enumerate(categories):
        subset = df[df[z_var["column_name"]] == category]

        # Get the median of all the related loss function parameters
        theta_min = subset["min_sm"].median()
        theta_max = subset["max_sm"].median()
        denormalized_k = subset["q_k_denormalized"].median()
        q = subset["q_q"].median()

        # Calculate the loss function
        theta = np.arange(theta_min, theta_max, 0.01)
        dtheta = loss_model(
            theta, q, denormalized_k, theta_wp=theta_min, theta_star=theta_max
        )

        # Plot median line
        ax.plot(theta, dtheta, label=category, color=colors[i], linewidth=3)

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(
        f"{var_dict['dtheta']['label']}\n{var_dict['dtheta']['symbol']} {var_dict['dtheta']['unit']}"
    )
    if title=="": 
        title = f'Median loss function by {z_var["label"]} {z_var["unit"]}'

    ax.set_title(title, loc="left")

    # Adjust the layout so the subplots fit into the figure area
    # ax.tight_layout()
    # Add a legend
    if plot_legend: 
        legend = ax.legend(bbox_to_anchor=(1, 1))
        for text in legend.get_texts():
            label = text.get_text()
            wrapped_label = wrap_text(label, 16)  # Wrap text after 16 characters
            text.set_text(wrapped_label)

    
fig_lossfnc_veg, ax = plt.subplots(figsize=(4.2, 4))
plot_loss_func_categorical(
    ax,
    df_filt_q,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

# %%
############################################################################
# Scatter plots with error bars
###########################################################################

def plot_scatter_with_errorbar_categorical(
    ax, df, x_var, y_var, z_var, categories, colors, quantile, title="", plot_logscale=False, plot_legend=True
):
    # fig, ax = plt.subplots(figsize=(5, 5))
    stats_dict = {}

    # Calculate median and 90% confidence intervals for each vegetation class
    for i, category in enumerate(categories):
        subset = df[df[z_var["column_name"]] == category]

        # Median calculation
        x_median = subset[x_var["column_name"]].median()
        y_median = subset[y_var["column_name"]].median()

        # 90% CI calculation, using the 5th and 95th percentiles
        x_ci_low, x_ci_high = np.percentile(
            subset[x_var["column_name"]], [quantile, 100 - quantile]
        )
        y_ci_low, y_ci_high = np.percentile(
            subset[y_var["column_name"]], [quantile, 100 - quantile]
        )

        # Store in dict
        stats_dict[category] = {
            "x_median": x_median,
            "y_median": y_median,
            "x_ci": (x_median - x_ci_low, x_ci_high - x_median),
            "y_ci": (y_median - y_ci_low, y_ci_high - y_median),
            "color": colors[i],
        }

    # Now plot medians with CIs
    for category, stats in stats_dict.items():
        ax.errorbar(
            stats["x_median"],
            stats["y_median"],
            xerr=np.array([[stats["x_ci"][0]], [stats["x_ci"][1]]]),
            yerr=np.array([[stats["y_ci"][0]], [stats["y_ci"][1]]]),
            fmt="o",
            label=category,
            capsize=5,
            capthick=2,
            color=stats["color"],
            alpha=0.7,
            markersize=15,
            mec="darkgray",
            mew=1,
            linewidth=3,
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")

    if title=="":
        title = f"Median with {quantile}% confidence interval"
    ax.set_title(title, loc='left')

    # Add a legend
    if plot_legend: 
        ax.legend(bbox_to_anchor=(1, 1))
    if plot_logscale:
        ax.set_yscale("log")
        # ax.set_xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

# %%
def plot_scatter_with_errorbar(ax, df, x_var, y_var, z_var, cmap, quantile, title="", plot_logscale=False, plot_legend=False):
    # fig, ax = plt.subplots(figsize=(5, 5))
    stats_dict = {}

    # Get unique bins
    bins_in_range = df[z_var["column_name"]].unique()
    bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
    bin_sorted = sorted(bins_list, key=lambda x: x.left)
    colors = plt.cm.get_cmap(cmap, len(bin_sorted))
    # Calculate median and 90% confidence intervals for each vegetation class
    for i, category in enumerate(bin_sorted):
        subset = df[df[z_var["column_name"]] == category]

        # Median calculation
        x_median = subset[x_var["column_name"]].median()
        y_median = subset[y_var["column_name"]].median()

        # 90% CI calculation, using the 5th and 95th percentiles
        x_ci_low, x_ci_high = np.percentile(
            subset[x_var["column_name"]], [quantile, 100 - quantile]
        )
        y_ci_low, y_ci_high = np.percentile(
            subset[y_var["column_name"]], [quantile, 100 - quantile]
        )

        color_val = colors(i / (len(bin_sorted) - 1))
        # Store in dict
        stats_dict[category] = {
            "x_median": x_median,
            "y_median": y_median,
            "x_ci": (x_median - x_ci_low, x_ci_high - x_median),
            "y_ci": (y_median - y_ci_low, y_ci_high - y_median),
            "color": color_val,
        }

    # Now plot medians with CIs
    for category, stats in stats_dict.items():
        ax.errorbar(
            stats["x_median"],
            stats["y_median"],
            xerr=np.array([[stats["x_ci"][0]], [stats["x_ci"][1]]]),
            yerr=np.array([[stats["y_ci"][0]], [stats["y_ci"][1]]]),
            fmt="o",
            label=str(category),
            capsize=5,
            capthick=2,
            color=stats["color"],
            alpha=0.7,
            markersize=15,
            mec="darkgray",
            mew=1,
            linewidth=3
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")
    if title=="":
        title=f"Median with {quantile}% confidence interval"

    ax.set_title(title, loc="left")

    # Add a legend
    if plot_legend:
        plt.legend(bbox_to_anchor=(1, 1.5))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])
# %% Loss function plots + parameter scatter plots 

# %%  ##########################
## Loss function plots + parameter scatter plots  (Figure 4)
##########################

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.rcParams.update({"font.size": 18})
# Plotting
plot_loss_func_categorical(
    axs[0],
    df_filt_q,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    title="(A)",
    plot_legend=False
    )

plot_scatter_with_errorbar_categorical(
    ax=axs[1], 
    df=df_filt_q, 
    x_var=var_dict["ai"], 
    y_var=var_dict["q_q"], 
    z_var=var_dict["veg_class"], 
    categories=list(vegetation_color_dict.keys()), 
    colors=list(vegetation_color_dict.values()), 
    quantile=25,
    title="(B)",
    plot_logscale=False,
    plot_legend=False
    )


plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig(os.path.join(fig_dir, "q_veg_ai.png"), dpi=1200, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "q_veg_ai.pdf"), dpi=1200, bbox_inches="tight")

# %%
#####################################
#  Loss function plots + parameter scatter plots  (Supplemental)
#######################################
# Vegetation
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
plt.rcParams.update({"font.size": 18})
plot_loss_func_categorical(
    axs[0, 0],
    df_filt_q,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    title="(A)"
    )

plot_scatter_with_errorbar_categorical(
    axs[0, 1], 
    df_filt_q, 
    var_dict["theta_star"], 
    var_dict["q_q"], 
    var_dict["veg_class"], 
    list(vegetation_color_dict.keys()), 
    list(vegetation_color_dict.values()), 
    25, 
    "(B)",
    False,
    False
    )

plot_scatter_with_errorbar_categorical(
    axs[1, 0], 
    df_filt_q, 
    var_dict["q_ETmax"], 
    var_dict["q_q"], 
    var_dict["veg_class"], 
    list(vegetation_color_dict.keys()), 
    list(vegetation_color_dict.values()), 
    25, 
    "(C)",
    False,
    False
    )
plot_scatter_with_errorbar_categorical(
    axs[1, 1], 
    df_filt_q, 
    var_dict["theta_star"], 
    var_dict["q_ETmax"], 
    var_dict["veg_class"], 
    list(vegetation_color_dict.keys()), 
    list(vegetation_color_dict.values()), 
    25,
    "(D)",
    False,
    False
    )

plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_veg.png"), dpi=1200, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_veg.pdf"), dpi=1200, bbox_inches="tight")

# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_veg_legend.pdf"), dpi=1200, bbox_inches="tight")
# %%
# Aridity Index

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
plt.rcParams.update({"font.size": 18})

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap, 
    plot_legend=False,
    title="(A)"
    )

plot_scatter_with_errorbar(
    ax=axs[0, 1], 
    df=df_filt_q, 
    x_var=var_dict["theta_star"], 
    y_var=var_dict["q_q"], 
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap, 
    quantile=25, 
    title="(B)",
    plot_logscale=False,
    plot_legend=False
    )

plot_scatter_with_errorbar(
    ax=axs[1, 0], 
    df=df_filt_q, 
    x_var=var_dict["q_ETmax"], 
    y_var=var_dict["q_q"], 
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap, 
    quantile=25, 
    title="(C)",
    plot_logscale=False,
    plot_legend=False
    )
plot_scatter_with_errorbar(
    ax=axs[1, 1], 
    df=df_filt_q, 
    x_var=var_dict["q_ETmax"], 
    y_var=var_dict["theta_star"], 
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap, 
    quantile=25, 
    title="(D)",
    plot_logscale=False,
    plot_legend=False
    )

plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai.png"), dpi=1200, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai.pdf"), dpi=1200, bbox_inches="tight")

# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai_legend.png"), dpi=1200, bbox_inches="tight")
# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai_legend.pdf"), dpi=1200, bbox_inches="tight")

# %%
# sand

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
plt.rcParams.update({"font.size": 18})

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap, 
    plot_legend=False,
    title="(A)"
    )

plot_scatter_with_errorbar(
    ax=axs[0, 1], 
    df=df_filt_q, 
    x_var=var_dict["theta_star"], 
    y_var=var_dict["q_q"], 
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap, 
    quantile=25, 
    title="(B)",
    plot_logscale=False,
    plot_legend=False
    )

plot_scatter_with_errorbar(
    ax=axs[1, 0], 
    df=df_filt_q, 
    x_var=var_dict["q_ETmax"], 
    y_var=var_dict["q_q"], 
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap, 
    quantile=25, 
    title="(C)",
    plot_logscale=False,
    plot_legend=False
    )
plot_scatter_with_errorbar(
    ax=axs[1, 1], 
    df=df_filt_q, 
    x_var=var_dict["q_ETmax"], 
    y_var=var_dict["theta_star"], 
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap, 
    quantile=25, 
    title="(D)",
    plot_logscale=False,
    plot_legend=False
    )

plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand.png"), dpi=1200, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand.pdf"), dpi=1200, bbox_inches="tight")

# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.png"), dpi=1200, bbox_inches="tight")
# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.pdf"), dpi=1200, bbox_inches="tight")

# %%
##########################################################################################
# Histogram with mean and median
###########################################################################################

def plot_histograms_with_mean_median_categorical(df, x_var, z_var, categories, colors):
    # Determine the number of rows needed for subplots based on the number of categories
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 3 * n_rows))

    if n_rows == 1:
        axes = [axes]  # Make it iterable even for a single category

    for i, (category, ax) in enumerate(zip(categories, axes)):
        subset = df[df[z_var["column_name"]] == category]

        # Determine bin edges based on bin interval
        bin_interval = 0.1
        min_edge = 0
        max_edge = 10
        bins = np.arange(min_edge, max_edge + bin_interval, bin_interval)

        # Plot histogram
        sns.histplot(
            subset[x_var["column_name"]],
            label="histogram",
            color=colors[i],
            bins=bins,  # You can adjust the number of bins
            kde=False,
            ax=ax,
        )

        # Calculate and plot mean and median lines
        mean_value = subset[x_var["column_name"]].mean()
        median_value = subset[x_var["column_name"]].median()
        ax.axvline(mean_value, color=colors[i], linestyle=":", lw=2, label="mean")
        ax.axvline(median_value, color=colors[i], linestyle="-", lw=2, label="median")

        # Creating a KDE (Kernel Density Estimation) of the data
        kde = gaussian_kde(subset[x_var["column_name"]])

        # Creating a range of values to evaluate the KDE
        kde_values = np.linspace(0, max(subset[x_var["column_name"]]), 1000)

        kde.set_bandwidth(bw_method=kde.factor / 3.0)

        # Evaluating the KDE
        kde_evaluated = kde(kde_values)

        # Finding the peak of the KDE
        peak_kde_value = kde_values[np.argmax(kde_evaluated)]

        # Plotting the KDE
        ax.plot(kde_values, kde_evaluated, color=colors[i])

        # Highlighting the peak of the KDE
        ax.axvline(
            x=peak_kde_value,
            color=colors[i],
            linestyle="--",
            linewidth=2.5,
            label="mode",
        )

        # Set titles and labels for each subplot
        ax.set_title(f"{z_var['label']}: {category}")
        ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
        ax.set_ylabel("Frequency\n[Number of drydown events]")

        ax.set_xlim(0, x_var["lim"][1] * 2)
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    return fig, ax


# %%
fig_hist_q_veg, _ = plot_histograms_with_mean_median_categorical(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

fig_hist_q_veg.savefig(
    os.path.join(fig_dir, f"sup_hist_q_veg.png"), dpi=1200, bbox_inches="tight"
)

# %%

def plot_histograms_with_mean_median(df, x_var, z_var, cmap):
    
    # Get unique bins
    bins_in_range = df[z_var["column_name"]].unique()
    bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
    bin_sorted = sorted(bins_list, key=lambda x: x.left)

    # Determine the number of rows needed for subplots based on the number of categories
    n_rows = len(bin_sorted)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 3 * n_rows))

    if n_rows == 1:
        axes = [axes]  # Make it iterable even for a single category

    # For each row in the subset, calculate the loss for a range of theta values
    for i, (category, ax) in enumerate(zip(bin_sorted, axes)):
        subset = df[df[z_var["column_name"]] == category]
        color = plt.get_cmap(cmap)(i / len(bins_list))

        # Determine bin edges based on bin interval
        bin_interval = 0.1
        min_edge = 0
        max_edge = 10
        bins = np.arange(min_edge, max_edge + bin_interval, bin_interval)
        
        
        # Plot histogram
        sns.histplot(
            subset[x_var["column_name"]],
            label="histogram",
            color=color,
            bins=bins,  # You can adjust the number of bins
            kde=False,
            ax=ax,
        )

        # Calculate and plot mean and median lines
        mean_value = subset[x_var["column_name"]].mean()
        median_value = subset[x_var["column_name"]].median()
        ax.axvline(mean_value, color=color, linestyle=":", lw=2, label="mean")
        ax.axvline(median_value, color=color, linestyle="-", lw=2, label="median")

        # Creating a KDE (Kernel Density Estimation) of the data
        kde = gaussian_kde(subset[x_var["column_name"]])

        # Creating a range of values to evaluate the KDE
        kde_values = np.linspace(0, max(subset[x_var["column_name"]]), 1000)

        kde.set_bandwidth(bw_method=kde.factor / 3.0)

        # Evaluating the KDE
        kde_evaluated = kde(kde_values)

        # Finding the peak of the KDE
        peak_kde_value = kde_values[np.argmax(kde_evaluated)]

        # Plotting the KDE
        ax.plot(kde_values, kde_evaluated, color=color)

        # Highlighting the peak of the KDE
        ax.axvline(
            x=peak_kde_value,
            color=plt.get_cmap(cmap)(i / len(bins_list)),
            linestyle="--",
            linewidth=2.5,
            label="mode",
        )

        # Set titles and labels for each subplot
        ax.set_title(f"{z_var['label']}: {category}")
        ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
        ax.set_ylabel("Frequency\n[Number of drydown events]")

        ax.set_xlim(0, x_var["lim"][1] * 2)
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    return fig, ax

# %%
fig_hist_q_ai, _ = plot_histograms_with_mean_median(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap
)

fig_hist_q_ai.savefig(
    os.path.join(fig_dir, f"sup_hist_q_ai.png"), dpi=1200, bbox_inches="tight"
)

# %%
fig_hist_q_sand, _ = plot_histograms_with_mean_median(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap
)

fig_hist_q_sand.savefig(
    os.path.join(fig_dir, f"sup_hist_q_sand.png"), dpi=1200, bbox_inches="tight"
)

# %% Including extremely small  q values as well 
fig_hist_q_veg2, _ = plot_histograms_with_mean_median_categorical(
    df=df_filt_allq,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

fig_hist_q_veg2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_veg_allq.png"), dpi=1200, bbox_inches="tight"
)

fig_hist_q_ai2, _ = plot_histograms_with_mean_median(
    df=df_filt_allq,
    x_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap
)

fig_hist_q_ai2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_ai_allq.png"), dpi=1200, bbox_inches="tight"
)

fig_hist_q_sand2, _ = plot_histograms_with_mean_median(
    df=df_filt_allq,
    x_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap
)

fig_hist_q_sand2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_sand_allq.png"), dpi=1200, bbox_inches="tight"
)

# %% Loss function parameter by vegetaiton and AI, supplemental (support Figure 4)

def wrap_at_space(text, max_width):
    parts = text.split(" ")
    wrapped_parts = [wrap(part, max_width) for part in parts]
    return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])

def plot_box_ai_veg(df):
    plt.rcParams.update({'font.size': 26})  # Adjust the font size as needed

    fig, ax =  plt.subplots(figsize=(20, 8))
    for i, category in enumerate(vegetation_color_dict.keys()):
        subset = df[df['name'] == category]
        sns.boxplot(x='name', y='AI', data=subset, color=vegetation_color_dict[category], ax=ax, linewidth=2)

    # ax = sns.violinplot(x='abbreviation', y='q_q', data=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'), 
    max_label_width = 20
    ax.set_xticklabels([wrap_at_space(label.get_text(), max_label_width) for label in ax.get_xticklabels()])
    plt.setp(ax.get_xticklabels(), rotation=45)

    # ax.set_xticklabels([textwrap.fill(t.get_text(), 10) for t in ax.get_xticklabels()])
    ax.set_ylabel("Aridity index [MAP/MAE]")
    ax.set_xlabel("IGBP Landcover Class")
    ax.set_ylim(0, 2.0)
    ax.set_title("(A)", loc="left")
    plt.tight_layout()

    return fig, ax

fig_box_ai_veg, _ = plot_box_ai_veg(df_filt_q)
fig_box_ai_veg.savefig(
    os.path.join(fig_dir, f"sup_box_ai_veg.png"), dpi=1200, bbox_inches="tight"
)

# %%
# # %% Vegetation vs AI Boxplot
fig_ai_vs_veg, axs = plt.subplots(1,2, figsize=(12, 6))
plot_scatter_with_errorbar_categorical(
    ax=axs[0],
    df=df_filt_q,
    x_var=var_dict["ai"],
    y_var=var_dict["theta_star"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    title="(B)",
    quantile=25,
    plot_logscale=False,
    plot_legend=False
)

plot_scatter_with_errorbar_categorical(
    ax=axs[1],
    df=df_filt_q,
    x_var=var_dict["ai"],
    y_var=var_dict["q_ETmax"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    title="(C)",
    quantile=25,
    plot_logscale=True,
    plot_legend=False
)

fig_ai_vs_veg.tight_layout()
fig_ai_vs_veg.savefig(
    os.path.join(fig_dir, f"sup_ai_vs_veg.png"), dpi=1200, bbox_inches="tight"
)


# %%
###########################################################################
###########################################################################
############################################################################
# Other plots (sandbox)
###########################################################################
###########################################################################
###########################################################################

# %%
############################################################################
# Box plots (might go supplemental)
###########################################################################


def plot_boxplots(df, x_var, y_var):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        x=x_var["column_name"],
        y=y_var["column_name"],
        data=df,
        boxprops=dict(facecolor="lightgray"),
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel(f'{x_var["label"]} {x_var["unit"]}')
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1] * 5)
    fig.tight_layout()

    return fig, ax


# %% sand
fig_box_sand, _ = plot_boxplots(df_filt_q, var_dict["sand_bins"], var_dict["q_q"])
fig_box_sand.savefig(
    os.path.join(fig_dir, f"box_sand.png"), dpi=600, bbox_inches="tight"
)
# %% Aridity index
fig_box_ai, _ = plot_boxplots(df_filt_q, var_dict["ai_bins"], var_dict["q_q"])
fig_box_ai.savefig(os.path.join(fig_dir, f"box_ai.png"), dpi=600, bbox_inches="tight")


# %% Vegatation
def wrap_at_space(text, max_width):
    parts = text.split(" ")
    wrapped_parts = [wrap(part, max_width) for part in parts]
    return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


def plot_boxplots_categorical(df, x_var, y_var, categories, colors):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the boxplot with specified colors and increased alpha
    sns.boxplot(
        x=x_var["column_name"],
        y=y_var["column_name"],
        data=df,
        # hue=x_var['column_name'],
        legend=False,
        order=categories,
        palette=colors,
        ax=ax,
    )

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

    # Optionally, adjust layout
    plt.tight_layout()
    ax.set_xlabel(f'{x_var["label"]}')
    max_label_width = 20
    ax.set_xticklabels(
        [
            wrap_at_space(label.get_text(), max_label_width)
            for label in ax.get_xticklabels()
        ]
    )
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    # Show the plot
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1] * 3)
    plt.tight_layout()
    plt.show()

    return fig, ax


# %%
fig_box_veg, _ = plot_boxplots_categorical(
    df_filt_q,
    var_dict["veg_class"],
    var_dict["q_q"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)
fig_box_veg.savefig(os.path.join(fig_dir, f"box_veg.png"), dpi=600, bbox_inches="tight")

# %%
def plot_hist_diffR2(df, var_key):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=0.005, color="#2c7fb8", fill=False, linewidth=3, ax=ax
    )

    # Setting the x limit
    ax.set_xlim(-0.1, 0.1)

    # Adding title and labels
    #ax.set_title("Histogram of diffR2 values")
    ax.set_xlabel(r"diffR2")
    ax.set_ylabel("Frequency")

    return fig, ax

plot_hist_diffR2(df=df_filt_q_and_exp, var_key="diff_R2")

fig_thetastar_vs_et_ai.savefig(
    os.path.join(fig_dir, f"thetastar_vs_et_ai.png"), dpi=600, bbox_inches="tight"
)


# %%
pixel_counts = (
    df_filt_q_and_exp.groupby(["EASE_row_index", "EASE_column_index"])
    .size()
    .reset_index(name="count")
)
plt.hist(
    pixel_counts["count"],
    bins=range(min(pixel_counts["count"]), max(pixel_counts["count"]) + 2, 1),
)
pixel_counts["count"].median()

# # %%
# def plot_scatter_per_pixel_categorical(
#     df, x_var, y_var, z_var, categories, colors, plot_logscale
# ):
#     # Get the median values of the variable
#     x_stat = (
#         df.groupby(["EASE_row_index", "EASE_column_index"])[x_var["column_name"]]
#         .median()
#         .reset_index()
#     )

#     y_stat = (
#         df.groupby(["EASE_row_index", "EASE_column_index"])[y_var["column_name"]]
#         .median()
#         .reset_index()
#     )

#     _merged_data = x_stat.merge(
#         df[[z_var["column_name"], "EASE_row_index", "EASE_column_index"]],
#         on=["EASE_row_index", "EASE_column_index"],
#         how="left",
#     )
#     merged_data = y_stat.merge(
#         _merged_data, on=["EASE_row_index", "EASE_column_index"], how="left"
#     )

#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Calculate median and 90% confidence intervals for each vegetation class
#     for i, category in enumerate(categories):
#         # i = 4
#         # category = "Woody savannas"
#         subset = merged_data[merged_data[z_var["column_name"]] == category]

#         plt.scatter(
#             subset[x_var["column_name"]],
#             subset[y_var["column_name"]],
#             color=colors[i],
#             alpha=0.05,
#             s=0.1,
#         )

#     # Add labels and title
#     ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
#     ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")

#     # Add a legend
#     plt.legend(bbox_to_anchor=(1, 1))
#     if plot_logscale:
#         plt.xscale("log")
#     ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

#     # Show the plot
#     return fig, ax


# fig_ai_vs_q, _ = plot_scatter_per_pixel_categorical(
#     df=df_filt_q,
#     x_var=var_dict["ai"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     plot_logscale=False,
# )
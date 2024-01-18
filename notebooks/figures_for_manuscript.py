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

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2023-11-25_global_95asmax"
###########################################################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define model acceptabiltiy criteria
success_modelfit_thresh = 0.7
sm_range_thresh = 0.3
z_mm = 50  # Soil thickness

# cmap for sand
sand_bin_list = [i * 0.1 for i in range(11)]
sand_cmap = "Oranges"

# cmap for ai
ai_bin_list = [i * 0.25 for i in range(8)]
ai_cmap = "RdBu"

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
        "lim": [0.5, 4],
    },
    "q_ETmax": {
        "column_name": "q_ETmax",
        "symbol": r"$ET_{max}$",
        "label": r"Estimated $ET_{max}$ by non-linear model",
        "unit": "[mm/day]",
        "lim": [0, 10],
    },
    "theta_star": {
        "column_name": "max_sm",
        "symbol": r"$\theta*$",
        "label": r"Estimated $\theta*$",
        "unit": r"$[m^3/m^3]$",
        "lim": [0.1, 0.4],
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

# sand bins
df["sand_bins"] = pd.cut(df["sand_fraction"], bins=sand_bin_list, include_lowest=True)
first_I = df["sand_bins"].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
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

# Runs where q model performed reasonablly well
df_filt_q = df[
    (df["q_r_squared"] >= success_modelfit_thresh) & (df["q_q"] > 0.1)
].copy()

# df_filt_q = df[
#     (df["q_r_squared"] >= success_modelfit_thresh)
# ].copy()

# df_filt_q = df[
#     (df["q_r_squared"] >= success_modelfit_thresh) & ((df["q_q"] > 0.1) | ((df["q_q"] < 0.1) & (df["n_days"] > 10)))
# ].copy()


df_filt_q_2 = df_filt_q[(df_filt_q["sm_range"] > sm_range_thresh)].copy()


print(f"q model fit was successful: {len(df_filt_q)}")
print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_2)}"
)
# print(f"q model fit without short drydown:  {len(df_filt_q_3)}")

# Runs where exponential model performed good
df_filt_exp = df[df["exp_r_squared"] >= success_modelfit_thresh].copy()
df_filt_exp_2 = df_filt_exp[df_filt_exp["sm_range"] > sm_range_thresh].copy()
print(f"exp model fit was successful: {len(df_filt_exp)}")
print(
    f"exp model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_exp_2)}"
)

# Runs where either of the model performed satisfactory
df_filt_q_or_exp = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    | (df["exp_r_squared"] >= success_modelfit_thresh)
].copy()
df_filt_q_or_exp_2 = df_filt_q_or_exp[
    df_filt_q_or_exp["sm_range"] > sm_range_thresh
].copy()
print(f"either q or exp model fit was successful: {len(df_filt_q_or_exp)}")
print(
    f"either q or exp model were successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_or_exp_2)}"
)

# Runs where both of the model performed satisfactory
df_filt_q_and_exp = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["exp_r_squared"] >= success_modelfit_thresh)
].copy()
df_filt_q_and_exp_2 = df_filt_q_and_exp[
    df_filt_q_and_exp["sm_range"] > sm_range_thresh
].copy()
print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")
print(
    f"both q and exp model were successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q_and_exp_2)}"
)

# %%
############################################################################
# PLOTTING FUNCTION STARTS HERE
###########################################################################


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
        cmap=cmap,
    )
    plt.colorbar(dsartist, label=f"Point density")


# %%


def plot_R2_models(df, R2_threshold, cmap):
    plt.rcParams.update({"font.size": 30})
    # Read data
    x = df["exp_r_squared"].values
    y = df["q_r_squared"].values

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(4.5 * 1.2, 4 * 1.2))
    # Calculate the point density
    sc = using_datashader(ax, x, y, cmap)

    # plt.title(rf'')
    plt.xlabel(r"Linear model")
    plt.ylabel(r"Non-linear model")

    # Add 1:1 line
    ax.plot(
        [R2_threshold, 1],
        [R2_threshold, 1],
        color="white",
        linestyle="--",
        label="1:1 line",
    )

    # Add a trendline
    coefficients = np.polyfit(x, y, 1)
    trendline_x = np.array([R2_threshold, 1])
    trendline_y = coefficients[0] * trendline_x + coefficients[1]
    ax.plot(trendline_x, trendline_y, color="white", label="Trendline")

    ax.set_xlim([R2_threshold, 1])
    ax.set_ylim([R2_threshold, 1])
    ax.set_title(r"$R^2$ comparison")

    fig.savefig(os.path.join(fig_dir, f"R2_scatter.png"), dpi=600, bbox_inches="tight")
    # return fig, ax


# plot_R2_models(df=df, R2_threshold=0.0)

# Plot R2 of q vs exp model, where where both q and exp model performed R2 > 0.7 and covered >30% of the SM range
plot_R2_models(
    df=df_filt_q_and_exp_2, R2_threshold=success_modelfit_thresh, cmap="viridis"
)
# fig_R2.savefig(os.path.join(fig_dir, f"R2_scatter.pdf"), dpi=600, bbox_inches='tight')


# %%
############################################################################
# Map plots
###########################################################################
def plot_map(df, coord_info, cmap, norm, var_item):
    plt.rcParams.update({"font.size": 12})

    # Get the mean values of the variable
    stat = df.groupby(["EASE_row_index", "EASE_column_index"])[
        var_item["column_name"]
    ].median()

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
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
    im = ax.pcolormesh(
        lons, lats, pivot_array, norm=norm, cmap=cmap, transform=ccrs.PlateCarree()
    )
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()

    # Add colorbar
    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        label=f'Median {var_item["label"]}',
        shrink=0.35,
        pad=0.02,
    )

    # Set plot title and labels
    # ax.set_title(f'Mean {variable_name} per pixel')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig


# %% Plot the map of q values, where both q and exp models performed > 0.7 and covered >30% of the SM range
var_key = "q_q"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=var_dict[var_key]["lim"][1])
fig_map_q = plot_map(
    df=df_filt_q_2,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
)
fig_map_q.savefig(os.path.join(fig_dir, f"q_map.png"), dpi=600, bbox_inches="tight")
# fig_map_q.savefig(os.path.join(fig_dir, f"q_map.pdf"), bbox_inches="tight")

# %% Map of R2 values
# Plot the map of R2 differences, where both q and exp model performed > 0.7 and covered >30% of the SM range
var_key = "diff_R2"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=var_dict[var_key]["lim"][1])
fig_map_R2 = plot_map(
    df=df_filt_q_and_exp_2,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm,
    var_item=var_dict[var_key],
)
fig_map_R2.savefig(os.path.join(fig_dir, f"R2_map.png"), dpi=600, bbox_inches="tight")


# %%
def plot_hist(df, var_key):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=1, color="#2c7fb8", fill=False, linewidth=3, ax=ax
    )

    # Setting the x limit
    ax.set_xlim(0, 10)

    # Adding title and labels
    ax.set_title("Histogram of $q$ values")
    ax.set_xlabel(r"$q$ [-]")
    ax.set_ylabel("Frequency")

    return fig, ax


fig_q_hist, _ = plot_hist(df=df_filt_q_2, var_key="q_q")
fig_q_hist.savefig(os.path.join(fig_dir, f"q_hist.png"), dpi=600, bbox_inches="tight")

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
fig_box_sand, _ = plot_boxplots(df_filt_q_2, var_dict["sand_bins"], var_dict["q_q"])
fig_box_sand.savefig(
    os.path.join(fig_dir, f"box_sand.png"), dpi=600, bbox_inches="tight"
)
# %% Aridity index
fig_box_ai, _ = plot_boxplots(df_filt_q_2, var_dict["ai_bins"], var_dict["q_q"])
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
    df_filt_q_2,
    var_dict["veg_class"],
    var_dict["q_q"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)
fig_box_veg.savefig(os.path.join(fig_dir, f"box_veg.png"), dpi=600, bbox_inches="tight")

# %%
############################################################################
# Loss function plots
###########################################################################


def plot_loss_func(df, z_var, cmap):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(5.8, 4))

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
        )

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(
        f"{var_dict['dtheta']['label']}\n{var_dict['theta']['symbol']} {var_dict['dtheta']['unit']}"
    )
    ax.set_title(f'Median loss function by {z_var["label"]} {z_var["unit"]}')
    # ax.set_xlim(var_dict['theta']['lim'][0],var_dict['theta']['lim'][1])
    # ax.set_ylim(var_dict['dtheta']['lim'][1],var_dict['dtheta']['lim'][0])

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title=f'{z_var["label"]}\n{z_var["unit"]}',
    )

    # Adjust the layout so the subplots fit into the figure area
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust

    # Adjust the layout so the subplots fit into the figure area
    fig.tight_layout()
    # Add a legend
    # fig.legend(bbox_to_anchor=(1, 1))

    return fig, ax


# %% Sand
fig_lossfnc_sand, _ = plot_loss_func(df_filt_q_2, var_dict["sand_bins"], sand_cmap)
fig_lossfnc_sand.savefig(
    os.path.join(fig_dir, f"lossfunc_sand.png"), dpi=600, bbox_inches="tight"
)

df_filt_q_and_exp_2[["id_x", "sand_bins"]].groupby("sand_bins").count()

# %% Aridity index
fig_lossfnc_ai, _ = plot_loss_func(df_filt_q_2, var_dict["ai_bins"], ai_cmap)
fig_lossfnc_ai.savefig(
    os.path.join(fig_dir, f"lossfunc_ai.png"), dpi=600, bbox_inches="tight"
)
df_filt_q_and_exp_2[["id_x", "ai_bins"]].groupby("ai_bins").count()


# %% Vegeation
def wrap_text(text, width):
    return "\n".join(wrap(text, width))


def plot_loss_func_categorical(df, z_var, categories, colors):
    fig, ax = plt.subplots(figsize=(4.2, 4))

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
        ax.plot(theta, dtheta, label=category, color=colors[i])

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(
        f"{var_dict['dtheta']['label']}\n{var_dict['theta']['symbol']} {var_dict['dtheta']['unit']}"
    )
    ax.set_title(f'Median loss function by {z_var["label"]} {z_var["unit"]}')

    # Adjust the layout so the subplots fit into the figure area
    plt.tight_layout()
    # Add a legend
    plt.legend(bbox_to_anchor=(1, 1))
    legend = plt.legend(bbox_to_anchor=(1, 1))
    for text in legend.get_texts():
        label = text.get_text()
        wrapped_label = wrap_text(label, 16)  # Wrap text after 16 characters
        text.set_text(wrapped_label)

    return fig, ax


fig_lossfnc_veg, _ = plot_loss_func_categorical(
    df_filt_q_2,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)
fig_lossfnc_veg.savefig(
    os.path.join(fig_dir, f"lossfunc_veg.png"), dpi=600, bbox_inches="tight"
)
# %%
count_veg_samples = df[df.name.isin(vegetation_color_dict.keys())]
count_veg_samples[["id_x", "name"]].groupby("name").count()


# %%
############################################################################
# Scatter plots with error bars
###########################################################################


def plot_scatter_with_errorbar_categorical(
    df, x_var, y_var, z_var, categories, colors, quantile, plot_logscale
):
    fig, ax = plt.subplots(figsize=(5, 5))
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
        plt.errorbar(
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
            markersize=10,
            mec="darkgray",
            mew=1,
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")
    plt.title(f"Median with {quantile}% confidence interval")

    # Add a legend
    plt.legend(bbox_to_anchor=(1, 1))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

    # Show the plot
    return fig, ax


# %% q vs. k per vegetation
fig_et_vs_q, _ = plot_scatter_with_errorbar_categorical(
    df=df_filt_q_2,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    quantile=33,
    plot_logscale=True,
)
fig_et_vs_q.savefig(
    os.path.join(fig_dir, f"et_vs_q_veg.png"), dpi=600, bbox_inches="tight"
)

# %% q vs. s* per vegetation
fig_thetastar_vs_q, _ = plot_scatter_with_errorbar_categorical(
    df=df_filt_q_2,
    x_var=var_dict["theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    quantile=33,
    plot_logscale=False,
)

fig_thetastar_vs_q.savefig(
    os.path.join(fig_dir, f"thetastar_vs_q_veg.png"), dpi=600, bbox_inches="tight"
)


# %% ETmax vs .s* per vegetation
fig_thetastar_vs_et, _ = plot_scatter_with_errorbar_categorical(
    df=df_filt_q_2,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["theta_star"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    quantile=33,
    plot_logscale=True,
)
fig_thetastar_vs_et.savefig(
    os.path.join(fig_dir, f"thetastar_vs_et_veg.png"), dpi=600, bbox_inches="tight"
)


# %%
def plot_scatter_with_errorbar(df, x_var, y_var, z_var, cmap, quantile, plot_logscale):
    fig, ax = plt.subplots(figsize=(5, 5))
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
        plt.errorbar(
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
            markersize=10,
            mec="darkgray",
            mew=1,
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")
    plt.title(f"Median with {quantile}% confidence interval")

    # Add a legend
    plt.legend(bbox_to_anchor=(1, 1.5))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

    # Show the plot
    plt.show()
    plt.tight_layout()

    return fig, ax


# %%
fig_thetastar_vs_et_ai, _ = plot_scatter_with_errorbar(
    df=df_filt_q_2,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["theta_star"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=33,
    plot_logscale=True,
)

fig_thetastar_vs_et_ai.savefig(
    os.path.join(fig_dir, f"thetastar_vs_et_ai.png"), dpi=600, bbox_inches="tight"
)


# %% Histogram with mean and median


def plot_histograms_with_mean_median(df, x_var, z_var, categories, colors):
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
        ax.set_ylabel("Frequency")

        ax.set_xlim(x_var["lim"][0], x_var["lim"][1] * 2)
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    return fig, ax


# %%

fig_hist_q_veg, _ = plot_histograms_with_mean_median(
    df=df_filt_q_2,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

fig_hist_q_veg.savefig(
    os.path.join(fig_dir, f"hist_q_veg.png"), dpi=600, bbox_inches="tight"
)


# %% Vegetation vs AI
fig_ai_vs_q, _ = plot_scatter_with_errorbar_categorical(
    df=df_filt_q_2,
    x_var=var_dict["ai"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    quantile=25,
    plot_logscale=False,
)

# %%
df_filt_q_2.columns
# %%


def plot_scatter_per_pixel_categorical(
    df, x_var, y_var, z_var, categories, colors, plot_logscale
):
    # Get the median values of the variable
    x_stat = (
        df.groupby(["EASE_row_index", "EASE_column_index"])[x_var["column_name"]]
        .median()
        .reset_index()
    )

    y_stat = (
        df.groupby(["EASE_row_index", "EASE_column_index"])[y_var["column_name"]]
        .median()
        .reset_index()
    )

    _merged_data = x_stat.merge(
        df[[z_var["column_name"], "EASE_row_index", "EASE_column_index"]],
        on=["EASE_row_index", "EASE_column_index"],
        how="left",
    )
    merged_data = y_stat.merge(
        _merged_data, on=["EASE_row_index", "EASE_column_index"], how="left"
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    # Calculate median and 90% confidence intervals for each vegetation class
    for i, category in enumerate(categories):
        subset = merged_data[merged_data[z_var["column_name"]] == category]

        plt.scatter(
            subset[x_var["column_name"]],
            subset[y_var["column_name"]],
            color=colors[i],
            alpha=0.05,
            s=1,
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")

    # Add a legend
    plt.legend(bbox_to_anchor=(1, 1))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

    # Show the plot
    return fig, ax


fig_ai_vs_q, _ = plot_scatter_per_pixel_categorical(
    df=df_filt_q_2,
    x_var=var_dict["ai"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_logscale=False,
)

# %%
# def calc_peak_kde_value(data):
#     kde = gaussian_kde(data)

#     # Creating a range of values to evaluate the KDE
#     kde_values = np.linspace(0, max(data), 1000)

#     kde.set_bandwidth(bw_method=kde.factor / 3.0)

#     # Evaluating the KDE
#     kde_evaluated = kde(kde_values)

#     # Finding the peak of the KDE
#     peak_kde_value = kde_values[np.argmax(kde_evaluated)]

#     return peak_kde_value


# def plot_scatter_with_errorbar_categorical_mode(
#     df, x_var, y_var, z_var, categories, colors, quantile, plot_logscale
# ):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     stats_dict = {}

#     # Calculate median and 90% confidence intervals for each vegetation class
#     for i, category in enumerate(categories):
#         subset = df[df[z_var["column_name"]] == category]

#         # Median calculation
#         # Creating a KDE (Kernel Density Estimation) of the data
#         x_mode = calc_peak_kde_value(subset[x_var["column_name"]])
#         y_mode = calc_peak_kde_value(subset[y_var["column_name"]])

#         # 90% CI calculation, using the 5th and 95th percentiles
#         x_ci_low, x_ci_high = np.percentile(
#             subset[x_var["column_name"]], [quantile, 100 - quantile]
#         )
#         y_ci_low, y_ci_high = np.percentile(
#             subset[y_var["column_name"]], [quantile, 100 - quantile]
#         )

#         x_ci_error = (max(x_mode - x_ci_low, 0), max(x_ci_high - x_mode, 0))
#         y_ci_error = (max(y_mode - y_ci_low, 0), max(y_ci_high - y_mode, 0))

#         # Store in dict
#         stats_dict[category] = {
#             "x_mode": x_mode,
#             "y_mode": y_mode,
#             "x_ci": x_ci_error,
#             "y_ci": y_ci_error,
#             "color": colors[i],
#         }

#     # Now plot medians with CIs
#     for category, stats in stats_dict.items():
#         plt.errorbar(
#             stats["x_mode"],
#             stats["y_mode"],
#             xerr=np.array([[stats["x_ci"][0]], [stats["x_ci"][1]]]),
#             yerr=np.array([[stats["y_ci"][0]], [stats["y_ci"][1]]]),
#             fmt="o",
#             label=category,
#             capsize=5,
#             capthick=2,
#             color=stats["color"],
#             alpha=0.7,
#             markersize=10,
#             mec="darkgray",
#             mew=1,
#         )

#     # Add labels and title
#     ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
#     ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")
#     plt.title(f"Scatter plot of modes with top {100-quantile*2}% confidence interval")

#     # Add a legend
#     plt.legend(bbox_to_anchor=(1, 1))
#     if plot_logscale:
#         plt.xscale("log")
#     ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1])

#     # Show the plot
#     plt.show()


# # %%
# plot_scatter_with_errorbar_categorical_mode(
#     df=df_filt_q_2[df_filt_q_2["q_q"] >= 0.1],
#     x_var=var_dict["q_ETmax"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=40,
#     plot_logscale=True,
# )

# # %% q vs. s* per vegetation
# plot_scatter_with_errorbar_categorical_mode(
#     df=df_filt_q_2[df_filt_q_2["q_q"] >= 0.1],
#     x_var=var_dict["theta_star"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=40,
#     plot_logscale=False,
# )

# # %% ETmax vs .s* per vegetation
# plot_scatter_with_errorbar_categorical_mode(
#     df=df_filt_q_2[df_filt_q_2["q_q"] >= 0.1],
#     x_var=var_dict["q_ETmax"],
#     y_var=var_dict["theta_star"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=40,
#     plot_logscale=True,
# )

# # %%
# small_q = df_filt_q_2[(df_filt_q_2["q_q"] < 0.1)].copy()
# large_q = df_filt_q_2[(df_filt_q_2["q_q"] > 0.1)].copy()
# # %%

# # Plotting both histograms on the same plot without fill
# plt.hist(
#     small_q["n_days"],
#     bins=np.arange(4, 30, 1),
#     alpha=0.7,
#     edgecolor="blue",
#     linewidth=1.5,
#     fill=False,
#     label="Small Q",
# )
# plt.hist(
#     large_q["n_days"],
#     bins=np.arange(4, 30, 1),
#     alpha=0.7,
#     edgecolor="green",
#     linewidth=1.5,
#     fill=False,
#     label="Large Q",
# )

# # Adding labels and legend
# plt.ylabel("Frequency")
# plt.title("Histogram of number of observation days")
# plt.legend()


# %%
# # %% Density plot
# def plot_2d_density(
#     df,
#     x_var,
#     y_var,
#     z_var,
#     categories,
#     colors,
#     quantile,
#     plot_logscale_x=False,
#     plot_logscale_y=False,
# ):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     kde_objects = []

#     # Calculate and plot density for each category
#     for i, category in enumerate(categories):
#         subset = df[df[z_var["column_name"]] == category]
#         x_data = subset[x_var["column_name"]]
#         y_data = subset[y_var["column_name"]]

#         # Calculate KDE
#         kde = gaussian_kde([x_data, y_data])
#         kde.set_bandwidth(bw_method=kde.factor / 5)
#         xmin, xmax = x_var["lim"]
#         ymin, ymax = y_var["lim"]
#         xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#         zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

#         # Find contour levels to fill between for middle 30%
#         levels = np.linspace(zi.min(), zi.max(), 100)
#         middle_levels = np.quantile(levels, [quantile / 100, 1.0])

#         # Plot KDE and fill relevant contours
#         # ax.contour(xi, yi, zi.reshape(xi.shape), levels=10, colors="white")
#         ax.contourf(
#             xi,
#             yi,
#             zi.reshape(xi.shape),
#             levels=middle_levels,
#             colors=colors[i],
#             alpha=0.5,
#             label=category,
#         )

#     # Add labels and title
#     ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
#     ax.set_ylabel(f"{y_var['label']} {y_var['unit']}")
#     plt.title(f"Density > {quantile} percentile")

#     # Set axis scales and limits
#     if plot_logscale_x:
#         ax.set_xscale("log")
#     if plot_logscale_y:
#         ax.set_yscale("log")
#     ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
#     ax.set_ylim(5e-2, 2)  # (y_var["lim"][0], y_var["lim"][1])
#     plt.legend(bbox_to_anchor=(1, 1))
#     legend = plt.legend(bbox_to_anchor=(1, 1))
#     for text in legend.get_texts():
#         label = text.get_text()
#         wrapped_label = wrap_text(label, 16)  # Wrap text after 16 characters
#         text.set_text(wrapped_label)
#     # Show the plot
#     plt.show()


# # %%
# plot_2d_density(
#     df=df_filt_q_2,
#     x_var=var_dict["theta_star"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=70,
#     plot_logscale_y=True,
# )
# # %%
# plot_2d_density(
#     df=df_filt_q_2[df_filt_q_2["q_q"] > 0.1],
#     x_var=var_dict["theta_star"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=90,
#     plot_logscale=False,
# )

# # %%
# plot_2d_density(
#     df=df_filt_q_2,
#     x_var=var_dict["q_ETmax"],
#     y_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=90,
#     plot_logscale=False,
# )

# # %%
# plot_2d_density(
#     df=df_filt_q_2,
#     x_var=var_dict["q_ETmax"],
#     y_var=var_dict["theta_star"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     quantile=90,
#     plot_logscale=False,
# # )

# # %%
# x = df_filt_q_2["n_days"].values
# y = df_filt_q_2["q_q"].values

# fig, ax = plt.subplots()
# img = using_datashader(ax, x, y, "viridis")
# ax.set_xlim(0, 25)
# # Colorbar


# plt.xlabel("Number of drydown days")
# plt.ylabel("q")
# plt.show()
# # ax.set_xlim(0,25)
# # %%

# %%
# def plot_violin_categorical(df, x_var, y_var, categories, colors):
#     fig, ax = plt.subplots(figsize=(8, 4))
#     for i, category in enumerate(categories):
#         subset = df[df[x_var["column_name"]] == category]
#         sns.violinplot(
#             x=x_var["column_name"],
#             y=y_var["column_name"],
#             data=subset,
#             order=[category],
#             color=colors[i],
#             ax=ax,
#             alpha=0.75,
#             cut=0,
#         )

#     # ax = sns.violinplot(x='abbreviation', y='q_q', data=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'),
#     ax.set_xlabel(f'{x_var["label"]}')
#     max_label_width = 20
#     ax.set_xticklabels(
#         [
#             wrap_at_space(label.get_text(), max_label_width)
#             for label in ax.get_xticklabels()
#         ]
#     )
#     plt.setp(ax.get_xticklabels(), rotation=45)
#     ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
#     # Show the plot
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1] * 2)
#     plt.tight_layout()
#     plt.show()


# plot_violin_categorical(
#     df_filt_q_2,
#     var_dict["veg_class"],
#     var_dict["q_q"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
# )

# # %%
# def plot_pdf(df, x_var, z_var, cmap):
#     fig, ax = plt.subplots(figsize=(4, 4))

#     # Get unique bins
#     bins_in_range = df[z_var["column_name"]].unique()
#     bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
#     bin_sorted = sorted(bins_list, key=lambda x: x.left)

#     # For each row in the subset, calculate the loss for a range of theta values
#     for i, category in enumerate(bin_sorted):
#         subset = df[df[z_var["column_name"]] == category]

#         sns.kdeplot(
#             subset[x_var["column_name"]],
#             label=category,
#             bw_adjust=0.5,
#             color=plt.get_cmap(cmap)(i / len(bins_list)),
#             cut=0,
#             ax=ax,
#         )

#     # Set titles and labels
#     plt.title(f"Kernel Density Estimation by {z_var['label']}")
#     ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
#     plt.ylabel("Density [-]")

#     ax.set_xlim(x_var["lim"][0], x_var["lim"][1] * 2)

#     # Show the legend
#     plt.legend()

#     # Show the plot
#     plt.show()


# # %% sand
# plot_pdf(
#     df=df_filt_q_2, x_var=var_dict["q_q"], z_var=var_dict["sand_bins"], cmap=sand_cmap
# )

# # %% aridity index
# plot_pdf(df=df_filt_q_2, x_var=var_dict["q_q"], z_var=var_dict["ai_bins"], cmap=ai_cmap)


# # %%
# def plot_pdf_categorical(df, x_var, z_var, categories, colors):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     for i, category in enumerate(categories):
#         subset = df[df[z_var["column_name"]] == category]
#         sns.kdeplot(
#             subset[x_var["column_name"]],
#             label=category,
#             bw_adjust=0.5,
#             color=colors[i],
#             cut=0,
#             ax=ax,
#         )
#     # Set titles and labels
#     plt.title(f"Kernel Density Estimation by {z_var['label']}")
#     ax.set_xlabel(f"{x_var['label']} {x_var['unit']}")
#     plt.ylabel("Density [-]")

#     ax.set_xlim(x_var["lim"][0], x_var["lim"][1] * 2)

#     # Show the legend
#     plt.legend()

#     # Show the plot
#     plt.show()


# # %%
# plot_pdf_categorical(
#     df=df_filt_q_2,
#     x_var=var_dict["q_q"],
#     z_var=var_dict["veg_class"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
# )

# %%

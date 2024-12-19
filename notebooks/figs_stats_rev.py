# %% Import packages
import os
import sys
import getpass
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu, ks_2samp, median_test
from scipy.interpolate import griddata
import statsmodels.api as statsm
from functions import q_model, loss_model

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import (
    Normalize,
    LinearSegmentedColormap,
    ListedColormap,
    BoundaryNorm,
)
import cartopy.crs as ccrs
from textwrap import wrap

# !pip install mpl-scatter-density
import mpl_scatter_density

# Math font
import matplotlib as mpl

plt.rcParams["font.family"] = "DejaVu Sans"  # Or any other available font
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Ensure the font is set correctly
# mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams["font.sans-serif"] = "Myriad Pro"
mpl.rcParams["font.size"] = 12.0
mpl.rcParams["axes.titlesize"] = 12.0
plt.rcParams["mathtext.fontset"] = (
    "stixsans"  #'stix'  # Or 'cm' (Computer Modern), 'stixsans', etc.
)

# Ryoko do not have this font on my system
# mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# font_files = mpl.font_manager.findSystemFonts(fontpaths=['/home/brynmorgan/Fonts/'])
# for font_file in font_files:
#     mpl.font_manager.fontManager.addfont(font_file)

# %% Plot config

################ CHANGE HERE FOR PATH CONFIG ##################################
############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-12-03_revision"  # f"raraki_2024-12-03_revision"  # "raraki_2024-02-02"  # f"raraki_2023-11-25_global_95asmax"
###########################################################################
# f"raraki_2024-05-13_global_piecewise" was used for the 1st version of the manuscript
# f"raraki_2024-12-03_revision" was used for the revised version of the manuscript

# Ryoko Araki, Bryn Morgan, Hilary K McMillan, Kelly K Caylor.
# Nonlinear Soil Moisture Loss Function Reveals Vegetation Responses to Water Availability. ESS Open Archive . August 01, 2024.
# DOI: 10.22541/essoar.172251989.99347091/v1

# Data dir
user_name = getpass.getuser()
data_dir = rf"/home/{user_name}/waves/projects/smap-drydown/data"

# Read the model output (results)
output_dir = rf"/home/{user_name}/waves/projects/smap-drydown/output"
results_file = rf"all_results_processed.csv"

datarods_dir = "datarods"
anc_rangeland_processed_file = "anc_info_rangeland_processed.csv"
coord_info_file = "coord_info.csv"

################ CHANGE HERE FOR PLOT VISUAL CONFIG & SETTING VARIABLES #########################

## Define parameters
save = True

note_dir = rf"/home/{user_name}/smap-drydown/notebooks"
with open(os.path.join(note_dir, "fig_veg_colors_lim.json"), "r") as file:
    vegetation_color_dict = json.load(file)

# Load variable settings
with open(os.path.join(note_dir, "fig_variable_labels.json"), "r") as file:
    var_dict = json.load(file)

# %% ############################################################################
# Data import & preps

# ############################################################################
# Create figure output directory in the model output directory
fig_dir = os.path.join(output_dir, dir_name, "figs")
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")

# Open the output file
f = open(os.path.join(fig_dir, "log.txt"), "w")
original_stdout = sys.stdout  # Save the original stdout
sys.stdout = f  # Change the stdout to the file handle

# ############################################################################
# DATA IMPORT

df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
print("Loaded results file\n")

coord_info = pd.read_csv(os.path.join(data_dir, datarods_dir, coord_info_file))

# %%
# Get bins for ancillary data
sand_bin_list = [i * 0.1 for i in range(11)]
sand_bin_list = sand_bin_list[1:]
sand_cmap = "Oranges_r"

df["sand_bins"] = pd.cut(df["sand_fraction"], bins=sand_bin_list, include_lowest=True)
first_I = df["sand_bins"].cat.categories[0]
new_I = pd.Interval(0.1, first_I.right)
df["sand_bins"] = df["sand_bins"].cat.rename_categories({first_I: new_I})

# ai
ai_bin_list = [i * 0.25 for i in range(7)]
ai_cmap = "RdBu"

df["ai_bins"] = pd.cut(df["AI"], bins=ai_bin_list, include_lowest=True)
first_I = df["ai_bins"].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df["ai_bins"] = df["ai_bins"].cat.rename_categories({first_I: new_I})


# %% ###################################################
# Exclude model fits failure
def count_median_number_of_events_perGrid(df):
    grouped = df.groupby(["EASE_row_index", "EASE_column_index"]).agg(
        median_diff_R2_q_tauexp=("diff_R2_q_tauexp", "median"),
        count=("diff_R2_q_tauexp", "count"),
    )
    print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}\n")


print(f"Total number of events: {len(df)}")
count_median_number_of_events_perGrid(df)

###################################################
# Defining model acceptabiltiy criteria
R2_thresh = 0.8
sm_range_thresh = 0.20
small_q_thresh = 1.0e-04
large_q_thresh = 0.8
###################################################


###################################################
# Model criteria used in the manuscript
# R2_thresh = 0.8
# sm_range_thresh = 0.20
# small_q_thresh = 1.0e-04
# large_q_thresh = 0.8
###################################################


def filter_df(df, criteria):
    return df[criteria].copy()


def print_model_success(message, df):
    print(f"{message} {len(df)}")
    count_median_number_of_events_perGrid(df)


# Filtering dfframes based on various model criteria
criteria_q = (df["q_r_squared"] > R2_thresh) & (df["sm_range"] > sm_range_thresh)
criteria_specific_q = (
    criteria_q
    & (df["q_q"] > small_q_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
    & (df["first3_avail2"])
)
criteria_tauexp = (df["tauexp_r_squared"] > R2_thresh) & (
    df["sm_range"] > sm_range_thresh
)
criteria_exp = (df["exp_r_squared"] > R2_thresh) & (df["sm_range"] > sm_range_thresh)

df_filt_q = filter_df(df, criteria_specific_q)
df_filt_allq = filter_df(df, criteria_q)
df_filt_tauexp = filter_df(df, criteria_tauexp)
df_filt_exp = filter_df(df, criteria_exp)
df_filt_q_or_tauexp = filter_df(df, criteria_q | criteria_tauexp)
df_filt_q_and_tauexp = filter_df(df, criteria_q & criteria_tauexp)
df_filt_q_or_exp = filter_df(df, criteria_q | criteria_exp)
df_filt_q_and_exp = filter_df(df, criteria_q & criteria_exp)

# Printing success messages and calculating events
print_model_success("q model fit successful:", df_filt_q)
print_model_success("exp model fit successful:", df_filt_exp)
print_model_success("tau-exp model fit successful:", df_filt_tauexp)
print_model_success(
    "both q and exp model fit successful (used for comparison):", df_filt_q_and_exp
)
print_model_success(
    "both q and tau-exp model fit successful (used for comparison):",
    df_filt_q_and_tauexp,
)
# print_model_success("either q or tau-exp:", df_filt_q_or_tauexp)
# print_model_success("either q or exp model fit successful:", df_filt_q_or_exp)
print_model_success(
    "q model fit successful (not filtering q parameter values):", df_filt_allq
)


def print_performance_comparison(df, model1, model2):
    n_better = sum(df[model1] > df[model2])
    percentage_better = n_better / len(df) * 100
    print(
        f"Of successful fits, {model1} performed better in {percentage_better:.1f} percent of events: {n_better}"
    )


# %% In terms of AIC, BIC, and p-values
print("============ EVENT-WISE COMPARISON ==============")
print(r"In terms of $R^2$")
print("tau-linear vs nonlinear")
print_performance_comparison(df_filt_q_and_tauexp, "q_r_squared", "tauexp_r_squared")
print("linear vs nonlinear")
print_performance_comparison(df_filt_q_and_exp, "q_r_squared", "exp_r_squared")


print("\n")
print(r"In terms of $AIC$")
print("tau-linear vs nonlinear")
print_performance_comparison(df_filt_q_and_tauexp, "q_aic", "tauexp_aic")
print("linear vs nonlinear")
print_performance_comparison(df_filt_q_and_exp, "q_aic", "exp_aic")

print("\n")
print(r"In terms of $AIC$")
print("tau-linear vs nonlinear")
print_performance_comparison(df_filt_q_and_tauexp, "q_bic", "tauexp_bic")
print("linear vs nonlinear")
print_performance_comparison(df_filt_q_and_exp, "q_bic", "exp_bic")


# %%
def print_p_value(df, varname, thresh):
    n_better = sum(df[varname] < thresh)
    percentage_better = n_better / len(df) * 100
    print(
        f"Of successful fits, {varname} < {thresh} in {percentage_better:.1f} percent of events: {n_better}"
    )


print("\n")
print(r"In terms of $p$-value")
print("Pre-filtered q values")
print_p_value(df_filt_q, "q_eq_1_p", 0.05)
print("All q values")
print_p_value(df_filt_allq, "q_eq_1_p", 0.05)


# %%
def save_figure(fig, fig_dir, filename, save_format, dpi):
    path = os.path.join(fig_dir, f"{filename}.{save_format}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close()


# %%
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#
# Plots & Stats
#
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

###################################################################
# Number of samples
################################################################0
# How much percent area (based on SMAP pixels) had better R2


def count_model_performance(df, varname):
    grouped = df.groupby(["EASE_row_index", "EASE_column_index"]).agg(
        median_diff=(varname, "median"), count=(varname, "count")
    )
    print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}")
    print(f"Number of SMAP grids with df: {len(grouped)}")
    pos_median_diff = (grouped["median_diff"] > 0).sum()
    print(
        f"Number of SMAP grids with bettter nonlinear model fits: {pos_median_diff} ({(pos_median_diff/len(grouped))*100:.1f} percent)"
    )
    # sns.histplot(grouped["count"], binwidth=0.5, color="#2c7fb8", fill=False, linewidth=3)


print("============ GRID-WISE COMPARISON ==============")
print(r"In terms of $R^2$")
print("tau-linear vs nonlinear")
count_model_performance(df_filt_q_and_tauexp, "diff_R2_q_tauexp")
print("linear vs nonlinear")
count_model_performance(df_filt_q_and_exp, "diff_R2_q_exp")

print("\n")
print(r"In terms of $AIC$")
print("tau-linear vs nonlinear")
count_model_performance(df_filt_q_and_tauexp, "diff_aic_q_tauexp")
print("linear vs nonlinear")
count_model_performance(df_filt_q_and_exp, "diff_aic_q_exp")

print("\n")
print(r"In terms of $BIC$")
print("tau-linear vs nonlinear")
count_model_performance(df_filt_q_and_tauexp, "diff_bic_q_tauexp")
print("linear vs nonlinear")
count_model_performance(df_filt_q_and_exp, "diff_bic_q_exp")

print("\n")
print(r"In terms of $AICc$")
print("tau-linear vs nonlinear")
count_model_performance(df_filt_q_and_tauexp, "diff_aicc_q_tauexp")
print("linear vs nonlinear")
count_model_performance(df_filt_q_and_exp, "diff_aicc_q_exp")

# %%
############################################################################
# PLOTTING FUNCTION STARTS HERE
###########################################################################

############################################################################
# Model performance comparison
###########################################################################

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label="Number of points per pixel")


def plot_model_metrics(df, linearmodel, metric_name, threshold=None, save=False):
    plt.rcParams.update({"font.size": 30})

    # Read df
    x_varname = f"{linearmodel}_{metric_name}"
    y_varname = f"q_{metric_name}"

    df = df[df[x_varname].notna() & np.isfinite(df[x_varname])]
    df = df[df[y_varname].notna() & np.isfinite(df[y_varname])]

    x = df[x_varname].values
    y = df[y_varname].values

    # Create a scatter plot
    # $ fig, ax = plt.subplots(figsize=(4.5 * 1.2, 4 * 1.2),)
    fig = plt.figure(figsize=(4.7 * 1.2, 4 * 1.2))
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(x, y, cmap=white_viridis, vmin=0, vmax=30)
    fig.colorbar(density, label="Frequency")
    plt.show()

    # plt.title(rf'')
    if metric_name == "r_squared":
        if linearmodel == "tauexp":
            ax.set_xlabel(r"$R^2$ ($\tau$-based linear)")
        else:
            ax.set_xlabel(r"$R^2$ (linear)")
        ax.set_ylabel(r"$R^2$ (Nonlinear)")
    elif metric_name in ["aic", "bic"]:
        ax.set_xlabel(f"{metric_name.upper()} (linear)")
        ax.set_ylabel(f"{metric_name.upper()} (Nonlinear)")

    if threshold is not None and not np.isnan(threshold):
        max_val = 1
        min_val = threshold
    else:
        max_val = max(x.max(), y.max())
        min_val = min(x.min(), y.min())

    # Add 1:1 line
    linecolor = "white"

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=linecolor,
        linestyle="--",
        label="1:1 line",
        linewidth=3,
    )

    # Add a trendline
    coefficients = np.polyfit(x, y, 1)
    trendline_x = np.array([min_val, max_val])
    trendline_y = coefficients[0] * trendline_x + coefficients[1]

    # Display the R2 values where nonlinear model got better
    x_intersect = coefficients[1] / (1 - coefficients[0])
    print(f"The trendline intersects with 1:1 line at {x_intersect:.2f}")
    ax.plot(trendline_x, trendline_y, color=linecolor, label="Trendline", linewidth=3)

    # Adjust axis limits
    # ax.set_xlim([min_val,max_val])
    # ax.set_ylim([min_val,max_val])
    # ax.set_title(r"$R^2$ comparison")

    if save:
        fig.savefig(
            os.path.join(fig_dir, f"{metric_name}_scatter_{linearmodel}.png"),
            dpi=900,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(fig_dir, f"{metric_name}_scatter_{linearmodel}.pdf"),
            dpi=1200,
            bbox_inches="tight",
        )
    return fig, ax


# %%

# Plot R2 of q vs exp model, where where both q and exp model performed R2 > 0.7 and covered >30% of the SM range
plot_model_metrics(
    df=df_filt_q_and_exp,
    linearmodel="exp",
    metric_name="r_squared",
    threshold=R2_thresh,
    save=save,
)
plot_model_metrics(
    df=df_filt_q_and_tauexp,
    linearmodel="tauexp",
    metric_name="r_squared",
    threshold=R2_thresh,
    save=save,
)

# %%
plot_model_metrics(
    df=df_filt_q_and_exp, linearmodel="exp", metric_name="aic", save=save
)
plot_model_metrics(
    df=df_filt_q_and_tauexp, linearmodel="tauexp", metric_name="aic", save=save
)

plot_model_metrics(
    df=df_filt_q_and_exp, linearmodel="exp", metric_name="bic", save=save
)
plot_model_metrics(
    df=df_filt_q_and_tauexp, linearmodel="tauexp", metric_name="bic", save=save
)

plot_model_metrics(
    df=df_filt_q_and_exp, linearmodel="exp", metric_name="aicc", save=save
)
plot_model_metrics(
    df=df_filt_q_and_tauexp, linearmodel="tauexp", metric_name="aicc", save=save
)


# %% Define plot_map
############################################################################
# Map plots
###########################################################################
def plot_map(
    ax, df, coord_info, cmap, norm, var_item, stat_type, title="", bar_label=None
):
    plt.setp(ax.spines.values(), linewidth=0.5)

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
    merged_df = (
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
    pivot_array = merged_df.pivot(
        index="latitude", columns="longitude", values=var_item["column_name"]
    )
    pivot_array[pivot_array.index > -60]  # Exclude antarctica in the map (no df)

    # Get lat and lon
    lons = pivot_array.columns.values
    lats = pivot_array.index.values

    # Plot in the map
    im = ax.pcolormesh(
        lons, lats, pivot_array, norm=norm, cmap=cmap, transform=ccrs.PlateCarree()
    )
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)

    if not bar_label:
        bar_label = f'{var_item["symbol"]} {var_item["unit"]}'

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        # label=f'{stat_label} {var_item["label"]} {var_item["unit"]}',
        label=bar_label,
        shrink=0.25,
        # width=0.1,
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


# %% Map of differences in R2 values
stat_type = "median"


def print_global_stats(df, diff_var, model_desc):
    print(f"Global median {diff_var} ({model_desc}): {df[diff_var].median()}")
    print(f"Global mean {diff_var} ({model_desc}): {df[diff_var].mean()}")


# %%
# Setup common variables
var_key_exp = "diff_aic_q_exp"
var_key_tauexp = "diff_aic_q_tauexp"
norm_exp = Normalize(
    vmin=var_dict[var_key_exp]["lim"][0], vmax=var_dict[var_key_exp]["lim"][1]
)
norm_tauexp = Normalize(
    vmin=var_dict[var_key_tauexp]["lim"][0], vmax=var_dict[var_key_tauexp]["lim"][1]
)

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_aic, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_exp,
    var_item=var_dict[var_key_exp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_exp]["label"],
)
if save:
    save_figure(fig_map_aic, fig_dir, f"aic_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_aic_q_exp", "nonlinear - linear")

# Plot and save maps for tauexp model
fig_map_aic, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_tauexp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_tauexp,
    var_item=var_dict[var_key_tauexp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_tauexp]["label"],
)
if save:
    save_figure(fig_map_aic, fig_dir, f"aic_map_{stat_type}_and_tauexp", "png", 900)

# Print statistical summaries for tauexp model
print_global_stats(
    df_filt_q_and_tauexp, "diff_aic_q_exp", "nonlinear - tau-based linear"
)

# %%
var_key_exp = "diff_aicc_q_exp"
var_key_tauexp = "diff_aicc_q_tauexp"

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_aicc, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_exp,
    var_item=var_dict[var_key_exp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_exp]["label"],
)
if save:
    save_figure(fig_map_aicc, fig_dir, f"aicc_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_aicc_q_exp", "nonlinear - linear")

# Plot and save maps for tauexp model
fig_map_aicc, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_tauexp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_tauexp,
    var_item=var_dict[var_key_tauexp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_tauexp]["label"],
)
if save:
    save_figure(fig_map_aicc, fig_dir, f"aicc_map_{stat_type}_and_tauexp", "png", 900)

# Print statistical summaries for tauexp model
print_global_stats(
    df_filt_q_and_tauexp, "diff_aicc_q_exp", "nonlinear - tau-based linear"
)
# %%
# Setup common variables
var_key_exp = "diff_bic_q_exp"
var_key_tauexp = "diff_bic_q_tauexp"
norm_exp = Normalize(
    vmin=var_dict[var_key_exp]["lim"][0], vmax=var_dict[var_key_exp]["lim"][1]
)
norm_tauexp = Normalize(
    vmin=var_dict[var_key_tauexp]["lim"][0], vmax=var_dict[var_key_tauexp]["lim"][1]
)

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_bic, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_exp,
    var_item=var_dict[var_key_exp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_exp]["label"],
)
if save:
    save_figure(fig_map_bic, fig_dir, f"bic_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_bic_q_exp", "nonlinear - linear")

# Plot and save maps for tauexp model
fig_map_bic, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_tauexp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_tauexp,
    var_item=var_dict[var_key_tauexp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_tauexp]["label"],
)
if save:
    save_figure(fig_map_bic, fig_dir, f"bic_map_{stat_type}_and_tauexp", "png", 900)

# Print statistical summaries for tauexp model
print_global_stats(
    df_filt_q_and_tauexp, "diff_bic_q_exp", "nonlinear - tau-based linear"
)


# %% Map of q
plt.rcParams.update({"font.size": 12})
var_key = "tauexp_tau"
norm = Normalize(vmin=2.0, vmax=6.0)
fig_map_tau, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
stat_type = "median"
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type=stat_type,
)

save_figure(fig_map_tau, fig_dir, f"tau_map_{stat_type}", "png", 900)
# save_figure(fig_map_q, fig_dir, f"q_map_{stat_type}", "pdf", 1200)


# %%
def plot_hist(df, var_key):
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=0.2, color="tab:blue", fill=False, linewidth=3, ax=ax
    )

    # Calculate median and mean
    median_value = df[var_key].median()
    mean_value = df[var_key].mean()

    # Add median and mean as vertical lines
    ax.axvline(
        median_value, color="tab:grey", linestyle="--", linewidth=3, label=f"Median"
    )
    ax.axvline(mean_value, color="tab:grey", linestyle=":", linewidth=3, label=f"Mean")

    # Setting the x limit
    ax.set_xlim(0, 10)

    # Adding title and labels
    ax.set_xlabel(var_dict[var_key]["symbol"])
    ax.set_ylabel("Frequency")
    fig.legend(loc="upper right", bbox_to_anchor=(0.93, 0.9), fontsize="small")

    return fig, ax


plt.rcParams.update({"font.size": 30})
fig_q_hist, _ = plot_hist(df=df_filt_q, var_key="tauexp_tau")
if save:
    save_figure(fig_q_hist, fig_dir, f"tau_hist", "png", 900)
    save_figure(fig_q_hist, fig_dir, f"tau_hist", "pdf", 1200)
# %%
print(f"Global median q: {df_filt_q['tauexp_tau'].median()}")
print(f"Global mean q: {df_filt_q['tauexp_tau'].mean()}")

# %%
df.columns
# %%


def plot_map_counts(ax, df, coord_info, cmap, norm, title=""):
    plt.setp(ax.spines.values(), linewidth=0.5)

    # Get the statistic of the variable
    stat = df.groupby(["EASE_row_index", "EASE_column_index"])["q_q"].count()
    stat_label = "Count"

    # Reindex to the full EASE row/index extent
    new_index = pd.MultiIndex.from_tuples(
        zip(coord_info["EASE_row_index"], coord_info["EASE_column_index"]),
        names=["EASE_row_index", "EASE_column_index"],
    )
    stat_pad = stat.reindex(new_index, fill_value=np.nan)

    # Join latitude and longitude
    merged_df = (
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
    pivot_array = merged_df.pivot(index="latitude", columns="longitude", values="q_q")
    pivot_array[pivot_array.index > -60]  # Exclude Antarctica in the map

    # Get lat and lon
    lons = pivot_array.columns.values
    lats = pivot_array.index.values

    # Plot in the map
    im = ax.pcolormesh(
        lons, lats, pivot_array, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
    )
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        label="Number of events",
        shrink=0.25,
        pad=0.02,
    )

    # Set plot title and labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title != "":
        ax.set_title(title, loc="left")


# Map of theta_star
norm = Normalize(vmin=0, vmax=40)
fig_map_events, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map_counts(
    ax=ax, df=df_filt_q, coord_info=coord_info, norm=norm, cmap="GnBu", title=""
)
plt.show()
if save:
    save_figure(fig_map_events, fig_dir, f"map_eventcounts", "png", 900)


# %%
import matplotlib.cm as cm


def plot_boxplots(df, x_var, y_var, cmap_name):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extract unique categories for the x variable
    categories = df[x_var["column_name"]].dropna().unique()
    categories = sorted(
        categories, key=lambda x: x.left
    )  # Sort intervals by their left edge

    # Generate a colormap for the categories
    cmap = cm.get_cmap(cmap_name, len(categories))
    colors = [cmap(i) for i in range(len(categories))]
    palette = dict(zip(categories, colors))  # Map categories to colors

    sns.boxplot(
        x=x_var["column_name"],
        y=y_var["column_name"],
        data=df,
        palette=cmap_name,
        # boxprops=dict(facecolor="lightgray"),
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel(f'{x_var["label"]} {x_var["unit"]}')
    ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
    ax.set_ylim(y_var["lim"][0] * 5, y_var["lim"][1] * 5)
    fig.tight_layout()

    return fig, ax


# %% sand
fig_box_sand, _ = plot_boxplots(
    df_filt_q_and_tauexp,
    var_dict["sand_bins"],
    var_dict["diff_R2_tauexp"],
    cmap_name=sand_cmap,
)
fig_box_sand.savefig(
    os.path.join(fig_dir, f"box_diff_R2_tauexp_sand.png"),
    dpi=600,
    bbox_inches="tight",
)

# %% Aridity index
# %% sand
fig_box_ai, _ = plot_boxplots(
    df_filt_q_and_tauexp,
    var_dict["ai_bins"],
    var_dict["diff_R2_tauexp"],
    cmap_name=ai_cmap,
)
fig_box_ai.savefig(
    os.path.join(fig_dir, f"box_diff_R2_tauexp_ai.png"),
    dpi=600,
    bbox_inches="tight",
)
# %% sand
fig_box_sand, _ = plot_boxplots(
    df_filt_q_and_exp,
    var_dict["sand_bins"],
    var_dict["diff_aic_q_exp"],
    cmap_name=sand_cmap,
)
fig_box_sand.savefig(
    os.path.join(fig_dir, f"box_diff_aic_exp_sand.png"),
    dpi=600,
    bbox_inches="tight",
)

# %% Aridity index
fig_box_ai, _ = plot_boxplots(
    df_filt_q_and_exp,
    var_dict["ai_bins"],
    var_dict["diff_aic_q_exp"],
    cmap_name=ai_cmap,
)
fig_box_ai.savefig(
    os.path.join(fig_dir, f"box_diff_aic_exp_ai.png"),
    dpi=600,
    bbox_inches="tight",
)


# %% Vegatation
def wrap_at_space(text, max_width):
    parts = text.split(" ")
    wrapped_parts = [wrap(part, max_width) for part in parts]
    return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


def plot_boxplots_categorical(df, x_var, y_var, categories, colors):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

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
    ax.set_ylim(y_var["lim"][0] * 5, y_var["lim"][1] * 5)
    plt.tight_layout()
    plt.show()

    return fig, ax


# %%
fig_box_veg, _ = plot_boxplots_categorical(
    df_filt_q_and_tauexp,
    var_dict["veg_class"],
    var_dict["diff_R2_tauexp"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)
fig_box_veg.savefig(
    os.path.join(fig_dir, f"box_diff_R2_tauexp_veg.png"), dpi=600, bbox_inches="tight"
)

# %%
fig_box_veg, _ = plot_boxplots_categorical(
    df_filt_q_and_exp,
    var_dict["veg_class"],
    var_dict["diff_aic_q_exp"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)
fig_box_veg.savefig(
    os.path.join(fig_dir, f"box_diff_aic_exp_veg.png"), dpi=600, bbox_inches="tight"
)


# def plot_box_ai_veg(df):
#     plt.rcParams.update({"font.size": 26})  # Adjust the font size as needed

#     fig, ax = plt.subplots(figsize=(20, 8))
#     for i, category in enumerate(vegetation_color_dict.keys()):
#         subset = df[df["name"] == category]
#         sns.boxplot(
#             x="name",
#             y="AI",
#             df=subset,
#             color=vegetation_color_dict[category],
#             ax=ax,
#             linewidth=2,
#         )

#     # ax = sns.violinplot(x='abbreviation', y='q_q', df=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'),
#     max_label_width = 20
#     ax.set_xticklabels(
#         [
#             wrap_at_space(label.get_text(), max_label_width)
#             for label in ax.get_xticklabels()
#         ]
#     )
#     plt.setp(ax.get_xticklabels(), rotation=45)

#     # ax.set_xticklabels([textwrap.fill(t.get_text(), 10) for t in ax.get_xticklabels()])
#     ax.set_ylabel("Aridity index [MAP/MAE]")
#     ax.set_xlabel("IGBP Landcover Class")
#     ax.set_ylim(0, 2.0)
#     ax.set_title("(a)", loc="left")
#     plt.tight_layout()

#     return fig, ax


# fig_box_ai_veg, _ = plot_box_ai_veg(df_filt_q)
# fig_box_ai_veg.savefig(
#     os.path.join(fig_dir, f"sup_box_ai_veg.png"), dpi=1200, bbox_inches="tight"
# )

# %%

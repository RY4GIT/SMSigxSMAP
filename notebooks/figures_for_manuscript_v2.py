# %% Import packages
import os
import getpass

import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import cartopy.crs as ccrs

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colors import Normalize
from textwrap import wrap

from functions_v2 import q_model, loss_model
from matplotlib.colors import LinearSegmentedColormap
!pip install mpl-scatter-density
import mpl_scatter_density
import matplotlib.colors as mcolors
import json
# Math font
plt.rcParams["mathtext.fontset"] = (
    "stixsans"  #'stix'  # Or 'cm' (Computer Modern), 'stixsans', etc.
)
# Ryoko do not have this font on my system
# import matplotlib as mpl

# mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# font_files = mpl.font_manager.findSystemFonts(fontpaths=['/home/brynmorgan/Fonts/'])

# for font_file in font_files:
#     mpl.font_manager.fontManager.addfont(font_file)

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Myriad Pro'

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-05-12_global_piecewise" #"raraki_2024-02-02"  # f"raraki_2023-11-25_global_95asmax"
############################|###############################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define parameters
z_mm = 50  # Soil thickness

with open('fig_veg_colors.json', 'r') as file:
    vegetation_color_dict = json.load(file)

# Load variable settings
with open('fig_variable_labels.json', 'r') as file:
    var_dict = json.load(file)

# %% ############################################################################
# DATA IMPORT & PATH CONFIGS

# Data dir
user_name = getpass.getuser()
data_dir = rf"/home/{user_name}/waves/projects/smap-drydown/data"
datarod_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
anc_rangeland_file = "anc_info_rangeland.csv"
anc_rangeland_processed_file = "anc_info_rangeland_processed.csv"
anc_Bassiouni_params_file = "anc_info_Bassiouni.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"
coord_info_file = "coord_info.csv"

# Read the model output (results)
output_dir = rf"/home/{user_name}/waves/projects/smap-drydown/output"
results_file = rf"all_results.csv"
_df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
_df["year"] = pd.to_datetime(_df["event_start"]).dt.year
print("Loaded results file")

# Create figure output directory in the model output directory
fig_dir = os.path.join(output_dir, dir_name, "figs")
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")

# ANCILLARY DATA IMPORT
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

# Calculate some metrics for evaluation

# Difference between R2 values of two models
df = df.assign(diff_R2=df["q_r_squared"] - df["exp_r_squared"])


# Get the binned dataset

# cmap for sand
sand_bin_list = [i * 0.1 for i in range(11)]
sand_bin_list = sand_bin_list[1:]
sand_cmap = "Oranges_r"

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

# # Ancillary data
# df_anc_Bassiouni = pd.read_csv(os.path.join(data_dir, datarod_dir, anc_Bassiouni_params_file)).drop(
#     ["Unnamed: 0", "latitude", "longitude"], axis=1
# )
# print("Loaded ancillary information (parameters from Bassiouni)")
# df_anc_Bassiouni.head()
# df = df.merge(df_anc_Bassiouni, on=["EASE_row_index", "EASE_column_index"], how="left")
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

def calculate_sm_range_abs(row):
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
    sm_range_abs = (
        (np.nanmax(sm) - np.nanmin(sm))
        if row.max_sm != row.min_sm
        else np.nan
    )
    return sm_range_abs

def check_1ts_range(row, verbose=False):
    common_params = {
        "q": row.q_q,
        "ETmax": row.q_ETmax,
        "theta_star": row.q_theta_star,
        "theta_w": row.q_theta_w,
    }

    s_t_0 = q_model(t=0, theta_0=row.q_theta_star, **common_params)
    s_t_1 = q_model(t=1, theta_0=row.q_theta_star, **common_params)

    dsdt_0 = loss_model(theta=s_t_0, **common_params)
    dsdt_1 = loss_model(theta=s_t_1, **common_params)

    if verbose:
        print(f"{(dsdt_0 - dsdt_1) / (row.q_ETmax / z_mm)*100:.1f} percent")
    return (dsdt_0 - dsdt_1) / (row["q_ETmax"] / z_mm) * (-1)

# Applying the function to each row and creating a new column 'sm_range'
df["sm_range"] = df.apply(calculate_sm_range, axis=1)
df["sm_range_abs"] = df.apply(calculate_sm_range_abs, axis=1)
df["event_length"] = (
    pd.to_datetime(df["event_end"]) - pd.to_datetime(df["event_start"])
).dt.days
df["large_q_criteria"] = df.apply(check_1ts_range, axis=1)

# # %%
# def calculate_n_days(row):
#     input_string = row.sm

#     # Processing the string
#     input_string = input_string.replace("\n", " np.nan")
#     input_string = input_string.replace(" nan", " np.nan")
#     input_string = input_string.strip("[]")

#     # Converting to numpy array and handling np.nan
#     sm = np.array(
#         [
#             float(value) if value != "np.nan" else np.nan
#             for value in input_string.split()
#         ]
#     )

#     # Calculating sm_range
#     n_days = len(sm)
#     return n_days


# # Applying the function to each row and creating a new column 'sm_range'
# df["n_days"] = df.apply(calculate_n_days, axis=1)


#%%
def filter_by_data_availability(df):
    # Define a helper function to convert string to list
    def str_to_list(s):
        return list(map(int, s.strip('[]').split()))

    # Convert the 'time' column from string of lists to actual lists
    df['time_list'] = df['time'].apply(str_to_list)

    # Filter condition 1: Check if first three items are [0, 1, 2]
    # condition = df['time_list'].apply(lambda x: x[:3] == [0, 1, 2])

    condition = df['time_list'].apply(
        lambda x: len(set(x[:4]).intersection({0, 1, 2})) >= 2
    )

    # Apply the first filter
    filtered_df = df[condition]

    return filtered_df

print(len(df))
df = filter_by_data_availability(df)
print(len(df))

# %% Exclude model fits failure

def count_median_number_of_events_perGrid(df):
    grouped = df.groupby(["EASE_row_index", "EASE_column_index"]).agg(
        median_diff_R2=("diff_R2", "median"), count=("diff_R2", "count")
    )
    print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}")


print(f"Total number of events: {len(df)}")
count_median_number_of_events_perGrid(df)

###################################################
# Defining model acceptabiltiy criteria
q_thresh = 1.0e-03
R2_thresh = 0.8
sm_range_thresh = 0.1 #0.1
# event_length_thresh = 3
large_q_thresh = 0.7
###################################################

# Runs where q model performed reasonablly well
df_filt_q = df[
    (df["q_r_squared"] >= R2_thresh)
    & (df["q_q"] > q_thresh)
    & (df["sm_range"] > sm_range_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
].copy()

print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range, plus extremely small q removed: {len(df_filt_q)}"
)
count_median_number_of_events_perGrid(df_filt_q)

# Runs where q model performed reasonablly well
df_filt_allq = df[
    (df["q_r_squared"] >= R2_thresh)
    & (df["sm_range"] > sm_range_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
].copy()

print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_allq)}"
)
count_median_number_of_events_perGrid(df_filt_allq)

# Runs where exponential model performed good
df_filt_exp = df[
    (df["exp_r_squared"] >= R2_thresh)
    & (df["sm_range"] > sm_range_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
].copy()
print(
    f"exp model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_exp)}"
)
count_median_number_of_events_perGrid(df_filt_exp)

# Runs where either of the model performed satisfactory
df_filt_q_or_exp = df[
    (
        (df["q_r_squared"] >= R2_thresh)
        | (df["exp_r_squared"] >= R2_thresh)
    )
    & (df["sm_range"] > sm_range_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
].copy()

print(f"either q or exp model fit was successful: {len(df_filt_q_or_exp)}")
count_median_number_of_events_perGrid(df_filt_q_or_exp)

# Runs where both of the model performed satisfactory
df_filt_q_and_exp = df[
    (df["q_r_squared"] >= R2_thresh)
    & (df["exp_r_squared"] >= R2_thresh)
    & (df["sm_range"] > sm_range_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
].copy()

print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")
count_median_number_of_events_perGrid(df_filt_q_and_exp)

# How many events showed better R2?
n_nonlinear_better_events = sum(
    df_filt_q_and_exp["q_r_squared"] > df_filt_q_and_exp["exp_r_squared"]
)
print(
    f"Of successful fits, nonlinear model performed better in {n_nonlinear_better_events/len(df_filt_q_and_exp)*100:.0f} percent of events: {n_nonlinear_better_events}"
)


# %%
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#
# Rangeland analysis
#
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

# Read data 
_rangeland_info = pd.read_csv(
    os.path.join(data_dir, datarod_dir, anc_rangeland_processed_file)
).drop(["Unnamed: 0"], axis=1)

rangeland_info = _rangeland_info.merge(coord_info, on=["EASE_row_index", "EASE_column_index"])

# merge with results dataframe
df_filt_q_conus = df_filt_q.merge(
    rangeland_info, on=["EASE_row_index", "EASE_column_index", "year"], how="left"
)

# Bin AI values
df_filt_q_conus["AI_binned2"] = pd.cut(
    df_filt_q_conus["AI"],
    # bins=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, np.inf],
    # labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0","1.0-1.25", "1.25-1.5", "1.5-"],
    bins=[0, 0.5, 1.0, 1.5, np.inf],
    labels=["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-"],
)

# Change to percentage (TODO: fix this in the data management)
df_filt_q_conus["fractional_wood"] = df_filt_q_conus["fractional_wood"] * 100
df_filt_q_conus["fractional_herb"] = df_filt_q_conus["fractional_herb"] * 100

# Print some statistics
print(f"Total number of drydown event with successful q fits: {len(df_filt_q)}")
print(
    f"Total number of drydown event with successful q fits & within CONUS: {sum(~pd.isna(df_filt_q_conus['fractional_wood']))}"
)
print(f"{sum(~pd.isna(df_filt_q_conus['fractional_wood']))/len(df_filt_q)*100:.2f}%")


# %%
# Get the statistics on the proportion of q>1 and q<1 events 
def get_df_percentage_q(
    df,
    var_name,
    weight_by, 
    bins=[0, 20, 40, 60, 80, 100],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
):

    new_varname = var_name + "_pct"
    df[new_varname] = pd.cut(df[var_name], bins=bins, labels=labels)

    # Calculating percentage of q>1 events for each AI bin and fractional_wood_pct
    q_greater_1 = (
        df[df["q_q"] > 1]
        .groupby(["AI_binned2", new_varname])
        .agg(
            count_greater_1=('q_q', 'size'),  # Count the occurrences
            sum_weightfact_q_gt_1=(weight_by, 'sum')  # Sum the event_length
        )
        .reset_index()
    )

    total_counts = (
        df.groupby(["AI_binned2", new_varname]).agg(
            total_count=('q_q', 'size'),  # Count the occurrences
            sum_weightfact_total=(weight_by, 'sum')  # Sum the event_length
        )
        .reset_index()
    )
    percentage_df = pd.merge(q_greater_1, total_counts, on=["AI_binned2", new_varname])

    percentage_df["percentage_q_gt_1"] = (
        percentage_df["count_greater_1"] / percentage_df["total_count"]
    ) * 100
    percentage_df["percentage_q_le_1"] = 100 - percentage_df["percentage_q_gt_1"]
    percentage_df["percentage_q_le_1"] = percentage_df["percentage_q_le_1"].fillna(0)
    percentage_df["percentage_q_gt_1"] = percentage_df["percentage_q_gt_1"].fillna(0)

    # count percentage * days percentage
    # percentage_df["weighted_percentage_q_gt_1"] = percentage_df["percentage_q_gt_1"] * percentage_df["sum_event_length_q_gt_1"]/ percentage_df["sum_event_length_total"]
    # percentage_df["weighted_percentage_q_le_1"] = percentage_df["percentage_q_le_1"] * (percentage_df["sum_event_length_total"]-percentage_df["sum_event_length_q_gt_1"])/ percentage_df["sum_event_length_total"]

    # Percentage of count * days
    percentage_df["count_x_weight_q_gt_1"] = percentage_df["count_greater_1"] * percentage_df["sum_weightfact_q_gt_1"]
    percentage_df["count_x_weight_q_le_1"] = (percentage_df["total_count"] - percentage_df["count_greater_1"]) * (percentage_df["sum_weightfact_total"]-percentage_df["sum_weightfact_q_gt_1"])
    percentage_df["weighted_percentage_q_gt_1"] = percentage_df["count_x_weight_q_gt_1"] / (percentage_df["count_x_weight_q_gt_1"]  + percentage_df["count_x_weight_q_le_1"]) * 100
    percentage_df["weighted_percentage_q_le_1"] = percentage_df["count_x_weight_q_le_1"] / (percentage_df["count_x_weight_q_gt_1"]  + percentage_df["count_x_weight_q_le_1"]) * 100
    return percentage_df

percentage_df = get_df_percentage_q(df_filt_q_conus, "fractional_wood", "sm_range_abs")

print(percentage_df)

#%%
# Plot the q<1 and q>1 proportion with aridity and fractional woody vegetation cover 

def darken_hex_color(hex_color, darken_factor=0.7):
    # Convert hex to RGB
    rgb_color = mcolors.hex2color(hex_color)
    # Convert RGB to HSV
    hsv_color = mcolors.rgb_to_hsv(rgb_color)
    # Darken the color by reducing its V (Value) component
    hsv_color[2] *= darken_factor
    # Convert back to RGB, then to hex
    darkened_rgb_color = mcolors.hsv_to_rgb(hsv_color)
    return mcolors.to_hex(darkened_rgb_color)
    
def plot_grouped_stacked_bar(ax, df, x_column_to_plot, z_var, var_name, title_name, weighted=False):

    # Determine unique groups and categories
    # Define the width of the bars and the space between groups
    bar_width = 0.2
    space_between_bars  = 0.025
    space_between_groups = 0.2
    
    # Determine unique values for grouping
    # Aridity bins
    x_unique = df[x_column_to_plot].unique()[:-1]
    # Vegetation bins
    z_unique = df[z_var].unique()
    n_groups = len(z_unique)
    
    # Define original colors
    base_colors  = ['#FFE268', '#22BBA9'] # (q<1, q>1)
    min_darken_factor = 0.85

    # Setup for weighted or unweighted percentages
    if weighted:
        y_vars = ['weighted_percentage_q_le_1', 'weighted_percentage_q_gt_1']
    else:
        y_vars = ['percentage_q_le_1', 'percentage_q_gt_1']
    
    # Create the grouped and stacked bars
    for z_i, z in enumerate(z_unique):
        for x_i, x in enumerate(x_unique):

            # Darken colors for this group
            darken_factor = max(np.sqrt(np.sqrt(np.sqrt(1 - (x_i / len(x_unique))))), min_darken_factor)
            colors = [darken_hex_color(color, darken_factor) for color in base_colors]
    
            # Calculate the x position for each group
            group_offset = (bar_width + space_between_bars) * n_groups
            x_pos = x_i * (group_offset + space_between_groups) + (bar_width + space_between_bars) * z_i
            
            # Get the subset of data for this group
            subset = df[(df[x_column_to_plot] == x) & (df[z_var] == z)]
            
            # Get bottom values for stacked bars
            bottom_value = 0

            for i, (y_var, color) in enumerate(zip(y_vars, colors)):
                ax.bar(
                    x_pos,
                    subset[y_var].values[0],
                    bar_width,
                    bottom=bottom_value,
                    color=color,
                    edgecolor='white',
                    label=f'{z} - {y_var.split("_")[-1]}' if x_i == 0 and i == 0 else ""
                )

                bottom_value += subset[y_var].values[0]
    
    # Set the x-ticks to the middle of the groups
    ax.set_xticks([i * (group_offset + space_between_groups) + group_offset / 2 for i in range(len(x_unique))])
    ax.set_xticklabels(x_unique, rotation=45)
    ax.set_xlabel(f"{var_dict["ai_bins"]['label']} {var_dict["ai_bins"]['unit']}")

    # Set the y-axis
    if weighted:
        ax.set_ylabel("Weighted proportion of\ndrydown events\nby event length (%)")
        ax.set_ylim([0, 40])
    else:
        ax.set_ylabel("Proportion of\ndrydown events (%)")
        ax.set_ylim([0, 40])

    # Set the second x-ticks
    # Replicate the z_var labels for the number of x_column_to_plot labels
    z_labels = np.tile(z_unique, len(x_unique))

    # Adjust the tick positions for the replicated z_var labels
    new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

    # Hide the original x-axis ticks and labels
    ax.tick_params(axis='x', which='both', length=0)

    # Create a secondary x-axis for the new labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_positions)
    ax2.set_xlabel(f"{var_dict[var_name]['label']} {var_dict[var_name]['unit']}")

    # Adjust the secondary x-axis to appear below the primary x-axis
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.set_xticklabels(z_labels, rotation=45)

    # Set plot title and legend
    ax.set_title(title_name)

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(4, 4))
plot_grouped_stacked_bar(
    ax=ax,
    df=percentage_df,
    x_column_to_plot="AI_binned2",
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=False
)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(4, 4))
plot_grouped_stacked_bar(
    ax=ax,
    df=percentage_df,
    x_column_to_plot="AI_binned2",
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=True
)
plt.tight_layout()

# %%
def plot_grouped_stacked_bar_uni(ax, df, z_var, var_name, title_name, weighted=False):

    # Determine unique groups and categories
    # Define the width of the bars and the space between groups
    bar_width = 0.2
    space_between_bars  = 0.025
    space_between_groups = 0.2
    
    # Determine unique values for grouping

    # Vegetation bins
    z_unique = df[z_var].unique()
    n_groups = len(z_unique)

    # exclude = df["AI_binned2"].unique()[-1]
    
    # Define original colors
    base_colors  = ['#FFE268', '#22BBA9'] # (q<1, q>1)
    min_darken_factor = 0.85

    # Setup for weighted or unweighted percentages
    if weighted:
        y_vars = ['weighted_percentage_q_le_1', 'weighted_percentage_q_gt_1']
    else:
        y_vars = ['percentage_q_le_1', 'percentage_q_gt_1']
    
    # Create the grouped and stacked bars
    for z_i, z in enumerate(z_unique):

        # Get the subset of data for this group
        subset = df[(df[z_var] == z)] #&(df["AI_binned2"] != exclude)]
        
        # Get bottom values for stacked bars
        bottom_value = 0
        # Calculate the x position for each group
        group_offset = (bar_width + space_between_bars) * n_groups
        x_pos = (group_offset + space_between_groups) + (bar_width + space_between_bars) * z_i
            

        for i, (y_var, color) in enumerate(zip(y_vars, base_colors)):
            ax.bar(
                x_pos,
                subset[y_var].values[0],
                bar_width,
                bottom=bottom_value,
                color=color,
                edgecolor='white',
            )

            bottom_value += subset[y_var].values[0]

    # Set the y-axis
    if weighted:
        ax.set_ylabel("Weighted proportion of\ndrydown events\nby event length (%)")
        ax.set_ylim([0, 20])
    else:
        ax.set_ylabel("Proportion of\ndrydown events (%)")
        ax.set_ylim([0, 40])

    # Set the second x-ticks
    # Replicate the z_var labels for the number of x_column_to_plot labels
    z_labels = z_unique

    # Adjust the tick positions for the replicated z_var labels
    new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

    # Hide the original x-axis ticks and labels
    ax.tick_params(axis='x', which='both', length=0)

    # Create a secondary x-axis for the new labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_positions)
    ax2.set_xlabel(f"{var_dict[var_name]['label']} {var_dict[var_name]['unit']}")

    # Adjust the secondary x-axis to appear below the primary x-axis
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.set_xticklabels(z_labels, rotation=45)

    # Set plot title and legend
    ax.set_title(title_name)

percentage_df_10 = get_df_percentage_q(df_filt_q_conus, "fractional_wood", "sm_range_abs", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ["0-10%","10-20%", "20-30%", "30-40%", "40-50%","50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])

fig, ax = plt.subplots(figsize=(4, 4))
plot_grouped_stacked_bar_uni(
    ax=ax,
    df=percentage_df_10,
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=True
)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(4, 4))
plot_grouped_stacked_bar_uni(
    ax=ax,
    df=percentage_df_10,
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=False
)
plt.tight_layout()

# %% 
# #################################################################
# Relationship between q and the length of the drydown events
#################################################################
# Calculate the number of bins

def plot_eventlength_hist(df):
    min_value = df['event_length'].min()
    max_value = df['event_length'].max()
    bin_width = 1
    n_bins = int((max_value - min_value) / bin_width) + 1  # Adding 1 to include the max value

    hist = df['event_length'].hist(bins=n_bins)
    hist.set_xlabel('Length of the event [days]')
    hist.set_ylabel('Frequency')
    # hist.set_xlim([0, 20])

# plot_eventlength_hist(df_filt_q)
plot_eventlength_hist(df_filt_q_conus[~pd.isna(df_filt_q_conus["barren_percent"])])

# %%

def plot_eventlength_vs_q(df):
    plt.scatter(df['event_length'], df['q_q'], marker='.', alpha=0.3)
    plt.ylabel(r'$q$')
    plt.xlabel('Length of the event [days]')
    # plt.xlim([0, 20])

# plot_eventlength_vs_q(df_filt_q)
plot_eventlength_vs_q(df_filt_q_conus[~pd.isna(df_filt_q_conus["barren_percent"])])


# %%
##################################################################
##### Statistics
#################################################################

# %%
###################################################################
# Number of samples
################################################################


# How much percent area (based on SMAP pixels) had better R2
grouped = df_filt_q_and_exp.groupby(["EASE_row_index", "EASE_column_index"]).agg(
    median_diff_R2=("diff_R2", "median"), count=("diff_R2", "count")
)
print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}")
print(f"Number of SMAP grids with data: {len(grouped)}")
num_positive_median_diff_R2 = (grouped["median_diff_R2"] > 0).sum()
print(
    f"Number of SMAP grids with bettter nonlinear model fits: {num_positive_median_diff_R2} ({(num_positive_median_diff_R2/len(grouped))*100:.1f} percent)"
)

sns.histplot(grouped["count"], binwidth=0.5, color="#2c7fb8", fill=False, linewidth=3)


###################################################################
# Number of samples
###################################################################
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
print(sum(pd.isna(df_filt_q["sand_fraction"]) == True))

###################################################################
# Stats on q 
###################################################################

print(f"Global q<1 median: {df_filt_q[df_filt_q["q_q"] < 1]["q_q"].median():.2f}")
print(f"Global q<1 mean: {df_filt_q[df_filt_q["q_q"] < 1]["q_q"].mean():.2f}")
print(f"Global q>1 median: {df_filt_q[df_filt_q["q_q"] > 1]["q_q"].median():.2f}")
print(f"Global q<1 mean: {df_filt_q[df_filt_q["q_q"] > 1]["q_q"].mean():.2f}")

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


def plot_R2_models_v2(df, R2_threshold, save=False):
    plt.rcParams.update({"font.size": 30})
    # Read data
    x = df["exp_r_squared"].values
    y = df["q_r_squared"].values

    # Create a scatter plot
    # $ fig, ax = plt.subplots(figsize=(4.5 * 1.2, 4 * 1.2),)
    fig = plt.figure(figsize=(4.7 * 1.2, 4 * 1.2))
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(x, y, cmap=white_viridis, vmin=0, vmax=30)
    fig.colorbar(density, label="Number of points per pixel")
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
        linewidth=3,
    )

    # Add a trendline
    coefficients = np.polyfit(x, y, 1)
    trendline_x = np.array([R2_threshold, 1])
    trendline_y = coefficients[0] * trendline_x + coefficients[1]

    # Display the R2 values where nonlinear model got better
    x_intersect = coefficients[1] / (1 - coefficients[0])
    print(f"The trendline intersects with 1:1 line at {x_intersect:.2f}")
    ax.plot(trendline_x, trendline_y, color="white", label="Trendline", linewidth=3)

    ax.set_xlim([R2_threshold, 1])
    ax.set_ylim([R2_threshold, 1])
    ax.set_title(r"$R^2$ comparison")

    if save:
        fig.savefig(
            os.path.join(fig_dir, f"R2_scatter.png"), dpi=900, bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(fig_dir, f"R2_scatter.pdf"), dpi=1200, bbox_inches="tight"
        )
    return fig, ax


# Plot R2 of q vs exp model, where where both q and exp model performed R2 > 0.7 and covered >30% of the SM range
plot_R2_models_v2(df=df_filt_q_and_exp, R2_threshold=R2_thresh, save=True)


# %%
############################################################################
# Map plots
###########################################################################
def plot_map(
    ax, df, coord_info, cmap, norm, var_item, stat_type, title="", bar_label=None
):
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

    if not bar_label:
        bar_label = f'{stat_label} {var_item["label"]}'

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        # label=f'{stat_label} {var_item["label"]} {var_item["unit"]}',
        label=bar_label,
        shrink=0.35,
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
save = False
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
    stat_type="median",
)
if save:
    fig_map_q.savefig(
        os.path.join(fig_dir, f"q_map_median.png"),
        dpi=900,
        bbox_inches="tight",
        transparent=True,
    )
    fig_map_q.savefig(
        os.path.join(fig_dir, f"q_map_median.pdf"),
        dpi=1200,
        bbox_inches="tight",
        transparent=True,
    )

print(f"Global median q: {df_filt_q['q_q'].median()}")
print(f"Global mean q: {df_filt_q['q_q'].mean()}")

# %% Map of differences in R2 values

save = True
stat_type = "median"
# Plot the map of R2 differences, where both q and exp model performed > 0.7 and covered >30% of the SM range
var_key = "diff_R2"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=var_dict[var_key]["lim"][1])
fig_map_R2, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type=stat_type,
    bar_label= stat_type.capitalize() + " differences\nin " + var_dict[var_key]["label"],
)
if save:
    fig_map_R2.savefig(
        os.path.join(fig_dir, f"R2_map_{stat_type}.png"),
        dpi=900,
        bbox_inches="tight",
        transparent=True,
    )

print(
    f"Global median diff R2 (nonlinear - linear): {df_filt_q_and_exp['diff_R2'].median()}"
)
print(
    f"Global mean diff R2 (nonlinear - linear): {df_filt_q_and_exp['diff_R2'].mean()}"
)

# %%
save = save
# Map of theta_star
var_key = "theta_star"
norm = Normalize(vmin=0.0, vmax=0.6)
fig_map_theta_star, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="A",
)
if save:
    fig_map_theta_star.savefig(
        os.path.join(fig_dir, f"sup_map_thetastar.png"), dpi=900, bbox_inches="tight"
    )

print(f"Global median theta_star: {df_filt_q['max_sm'].median()}")
print(f"Global mean theta_star: {df_filt_q['max_sm'].mean()}")

# %%
# Map of ETmax
var_key = "q_ETmax"
norm = Normalize(vmin=var_dict[var_key]["lim"][0], vmax=10)
fig_map_ETmax, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="B",
)
if save:
    fig_map_ETmax.savefig(
        os.path.join(fig_dir, f"sup_map_ETmax.png"), dpi=900, bbox_inches="tight"
    )

print(f"Global median ETmax: {df_filt_q['q_ETmax'].median()}")
print(f"Global mean ETmax: {df_filt_q['q_ETmax'].mean()}")


# %%
############################################################################
# Histogram of q values (global)
###########################################################################

save=True
def plot_hist(df, var_key):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=0.2, color="#2c7fb8", fill=False, linewidth=3, ax=ax
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
    # ax.set_title("Histogram of $q$ values")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel("Frequency")
    fig.legend(loc="upper right", bbox_to_anchor=(0.93, 0.9))

    return fig, ax


fig_q_hist, _ = plot_hist(df=df_filt_q, var_key="q_q")
if save:
    fig_q_hist.savefig(
        os.path.join(fig_dir, f"q_hist.png"),
        dpi=1200,
        bbox_inches="tight",
        transparent=True,
    )
    fig_q_hist.savefig(
        os.path.join(fig_dir, f"q_hist.pdf"),
        dpi=1200,
        bbox_inches="tight",
        transparent=True,
    )

# %%
############################################################################
# Loss function plot
###########################################################################

def wrap_text(text, width):
    return "\n".join(wrap(text, width))

def plot_loss_func(ax, df, z_var, categories=None, colors=None, cmap=None, title="", plot_legend=False):
    
    # Get category/histogram bins to enumerate
    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)

        # Get colors according to the number of bins
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted = categories

    # For each row in the subset, calculate the loss for a range of theta values
    for i, category in enumerate(bins_sorted):
        # Get subset 
        subset = df[df[z_var["column_name"]] == category]

        # Get the median of all the related loss function parameters
        theta_w = subset["q_theta_w"].median()
        theta_star = subset["q_theta_star"].median()
        ETmax = subset["q_ETmax"].median()
        q = subset["q_q"].median()

        # Calculate the loss function
        theta = np.arange(theta_w, theta_star, 0.01)
        dtheta = loss_model(
            theta=theta, q=q, ETmax=ETmax, theta_w=theta_w, theta_star=theta_star
        )

        # Plot loss function from median parameters 
        ax.plot(
            theta,
            dtheta,
            label=f"{category}",
            color=colors[i],
            linewidth=3,
        )

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(
        f"{var_dict['dtheta']['label']}\n{var_dict['theta']['symbol']} {var_dict['dtheta']['unit']}"
    )
    if title == "":
        title = f'Median loss function by {z_var["label"]} {z_var["unit"]}'
    ax.set_title(title, loc="left")

    if plot_legend:
        if categories is None:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                title=f'{z_var["label"]}\n{z_var["unit"]}',
            )
        else:
            legend = ax.legend(bbox_to_anchor=(1, 1))
            for text in legend.get_texts():
                label = text.get_text()
                wrapped_label = wrap_text(label, 16)  # Wrap text after 16 characters
                text.set_text(wrapped_label)

############################################################################
# Scatter plots with error bars
###########################################################################

def plot_scatter_with_errorbar(
    ax,
    df,
    x_var,
    y_var,
    z_var,
    quantile,
    cmap=None,
    categories=None,
    colors=None,
    title="",
    plot_logscale=False,
    plot_legend=False,
):

    stats_dict = {}

    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted=categories

    # Calculate median and 90% confidence intervals for each vegetation class
    for i, category in enumerate(bins_sorted):
        subset = df[df[z_var["column_name"]] == category]

        # Median calculation
        x_median = subset[x_var["column_name"]].median()
        y_median = subset[y_var["column_name"]].median()

        # Confidence interval calculation
        x_ci_low, x_ci_high = np.nanpercentile(
            subset[x_var["column_name"]], [quantile, 100 - quantile]
        )
        y_ci_low, y_ci_high = np.nanpercentile(
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
            label=str(category),
            capsize=5,
            capthick=2,
            color=stats["color"],
            alpha=0.7,
            markersize=10,
            mec=stats["color"],
            mew=1,
            linewidth=3,
        )

    # Add labels and title
    ax.set_xlabel(f"Estimated {x_var['symbol']} {x_var['unit']}")
    ax.set_ylabel(f"Estimated {y_var['symbol']} {y_var['unit']}")
    if title == "":
        title = f"Median with {quantile}% confidence interval"

    ax.set_title(title, loc="left")

    # Add a legend
    if plot_legend:
        plt.legend(bbox_to_anchor=(1, 1.5))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])


# %%
#####################################
#  4-grid Loss function plots + parameter scatter plots
#######################################
# Vegetation
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot_loss_func(
    axs[0, 0],
    df_filt_q,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    title="A",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="B",
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="C",
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["theta_star"],
    y_var=var_dict["q_ETmax"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="D",
)

plt.tight_layout()
plt.show()

if save:
    # Save the combined figure
    fig.savefig(
        os.path.join(fig_dir, "sup_lossfnc_veg.png"), dpi=1200, bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(fig_dir, "sup_lossfnc_veg.pdf"), dpi=1200, bbox_inches="tight"
    )

# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_veg_legend.pdf"), dpi=1200, bbox_inches="tight")
# %%
# Aridity Index

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.rcParams.update({"font.size": 18})

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    title="A",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="B",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="C",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["theta_star"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="D",
    plot_logscale=False,
    plot_legend=False,
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

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    title="A",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="B",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="C",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["theta_star"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="D",
    plot_logscale=False,
    plot_legend=False,
)

plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig(
    os.path.join(fig_dir, "sup_lossfnc_sand.png"), dpi=1200, bbox_inches="tight"
)
fig.savefig(
    os.path.join(fig_dir, "sup_lossfnc_sand.pdf"), dpi=1200, bbox_inches="tight"
)

# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.png"), dpi=1200, bbox_inches="tight")
# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.pdf"), dpi=1200, bbox_inches="tight")

# %%
##########################################################################################
# Histogram with mean and median
###########################################################################################

def plot_histograms_with_mean_median(df, x_var, z_var, cmap=None, categories=None, colors=None):
    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted=categories

    # Determine the number of rows needed for subplots based on the number of categories
    n_rows = len(bins_sorted)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 3 * n_rows))

    if n_rows == 1:
        axes = [axes]  # Make it iterable even for a single category

    # For each row in the subset, calculate the loss for a range of theta values
    for i, (category, ax) in enumerate(zip(bins_sorted, axes)):
        subset = df[df[z_var["column_name"]] == category]

        # Determine bin edges based on bin interval
        bin_interval = 0.05
        min_edge = 0
        max_edge = 10
        bins = np.arange(min_edge, max_edge + bin_interval, bin_interval)

        # Plot histogram
        sns.histplot(
            subset[x_var["column_name"]],
            label="histogram",
            color=colors[i],
            bins=bins,
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


# %% df_filt_allq: Including extremely small  q values as well
plt.rcParams.update({"font.size": 8})
fig_hist_q_veg, _ = plot_histograms_with_mean_median(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

fig_hist_q_veg.savefig(
    os.path.join(fig_dir, f"sup_hist_q_veg_allq.png"), dpi=1200, bbox_inches="tight"
)

#%%
fig_hist_q_ai2, _ = plot_histograms_with_mean_median(
    df=df_filt_q, x_var=var_dict["q_q"], z_var=var_dict["ai_bins"], cmap=ai_cmap
)

fig_hist_q_ai2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_ai_allq.png"), dpi=1200, bbox_inches="tight"
)

#%%
fig_hist_q_sand2, _ = plot_histograms_with_mean_median(
    df=df_filt_q, x_var=var_dict["q_q"], z_var=var_dict["sand_bins"], cmap=sand_cmap
)

fig_hist_q_sand2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_sand_allq.png"), dpi=1200, bbox_inches="tight"
)

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

# # %% Loss function parameter by vegetaiton and AI, supplemental (support Figure 4)
# def wrap_at_space(text, max_width):
#     parts = text.split(" ")
#     wrapped_parts = [wrap(part, max_width) for part in parts]
#     return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


# def plot_box_ai_veg(df):
#     plt.rcParams.update({"font.size": 26})  # Adjust the font size as needed

#     fig, ax = plt.subplots(figsize=(20, 8))
#     for i, category in enumerate(vegetation_color_dict.keys()):
#         subset = df[df["name"] == category]
#         sns.boxplot(
#             x="name",
#             y="AI",
#             data=subset,
#             color=vegetation_color_dict[category],
#             ax=ax,
#             linewidth=2,
#         )

#     # ax = sns.violinplot(x='abbreviation', y='q_q', data=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'),
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
#     ax.set_title("A", loc="left")
#     plt.tight_layout()

#     return fig, ax


# fig_box_ai_veg, _ = plot_box_ai_veg(df_filt_q)
# fig_box_ai_veg.savefig(
#     os.path.join(fig_dir, f"sup_box_ai_veg.png"), dpi=1200, bbox_inches="tight"
# )

# # # %%
# # # # %% Vegetation vs AI Boxplot
# # fig_ai_vs_veg, axs = plt.subplots(1, 2, figsize=(12, 6))
# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[0],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["theta_star"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title="B",
# #     quantile=25,
# #     plot_logscale=False,
# #     plot_legend=False,
# # )

# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[1],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["q_ETmax"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title="C",
# #     quantile=25,
# #     plot_logscale=True,
# #     plot_legend=False,
# # )

# # fig_ai_vs_veg.tight_layout()
# # fig_ai_vs_veg.savefig(
# #     os.path.join(fig_dir, f"sup_ai_vs_veg.png"), dpi=1200, bbox_inches="tight"
# # )


# # %%
# ###########################################################################
# ###########################################################################
# ############################################################################
# # Other plots (sandbox)
# ###########################################################################
# ###########################################################################
# ###########################################################################


# # %%
# ############################################################################
# # Box plots (might go supplemental)
# ###########################################################################
# def plot_boxplots(df, x_var, y_var):
#     plt.rcParams.update({"font.size": 12})
#     fig, ax = plt.subplots(figsize=(6, 4))

#     sns.boxplot(
#         x=x_var["column_name"],
#         y=y_var["column_name"],
#         data=df,
#         boxprops=dict(facecolor="lightgray"),
#         ax=ax,
#     )

#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     ax.set_xlabel(f'{x_var["label"]} {x_var["unit"]}')
#     ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1])
#     fig.tight_layout()

#     return fig, ax


# # %% sand
# fig_box_sand, _ = plot_boxplots(df_filt_q, var_dict["sand_bins"], var_dict["q_q"])
# fig_box_sand.savefig(
#     os.path.join(fig_dir, f"box_sand.png"), dpi=600, bbox_inches="tight"
# )
# # %% Aridity index
# fig_box_ai, _ = plot_boxplots(df_filt_q, var_dict["ai_bins"], var_dict["q_q"])
# fig_box_ai.savefig(os.path.join(fig_dir, f"box_ai.png"), dpi=600, bbox_inches="tight")


# # %% Vegatation
# def wrap_at_space(text, max_width):
#     parts = text.split(" ")
#     wrapped_parts = [wrap(part, max_width) for part in parts]
#     return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


# def plot_boxplots_categorical(df, x_var, y_var, categories, colors):
#     # Create the figure and axes
#     fig, ax = plt.subplots(figsize=(8, 4))

#     # Plot the boxplot with specified colors and increased alpha
#     sns.boxplot(
#         x=x_var["column_name"],
#         y=y_var["column_name"],
#         data=df,
#         # hue=x_var['column_name'],
#         legend=False,
#         order=categories,
#         palette=colors,
#         ax=ax,
#     )

#     for patch in ax.artists:
#         r, g, b, a = patch.get_facecolor()
#         patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

#     # Optionally, adjust layout
#     plt.tight_layout()
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
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1] * 3)
#     plt.tight_layout()
#     plt.show()

#     return fig, ax


# # %%
# fig_box_veg, _ = plot_boxplots_categorical(
#     df_filt_q,
#     var_dict["veg_class"],
#     var_dict["q_q"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
# )
# fig_box_veg.savefig(os.path.join(fig_dir, f"box_veg.png"), dpi=600, bbox_inches="tight")


# # %%
# def plot_hist_diffR2(df, var_key):
#     plt.rcParams.update({"font.size": 30})
#     fig, ax = plt.subplots(figsize=(5.5, 5))

#     # Create the histogram with a bin width of 1
#     sns.histplot(
#         df[var_key], binwidth=0.005, color="#2c7fb8", fill=False, linewidth=3, ax=ax
#     )

#     # Setting the x limit
#     ax.set_xlim(-0.1, 0.1)

#     # Adding title and labels
#     # ax.set_title("Histogram of diffR2 values")
#     ax.set_xlabel(r"diffR2")
#     ax.set_ylabel("Frequency")

#     return fig, ax


# plot_hist_diffR2(df=df_filt_q_and_exp, var_key="diff_R2")

# # fig_thetastar_vs_et_ai.savefig(
# #     os.path.join(fig_dir, f"thetastar_vs_et_ai.png"), dpi=600, bbox_inches="tight"
# # )


# # %%
# pixel_counts = (
#     df_filt_q_and_exp.groupby(["EASE_row_index", "EASE_column_index"])
#     .size()
#     .reset_index(name="count")
# )
# plt.hist(
#     pixel_counts["count"],
#     bins=range(min(pixel_counts["count"]), max(pixel_counts["count"]) + 2, 1),
# )
# pixel_counts["count"].median()


# # # %% Ridgeplot for poster
# # def plot_ridgeplot(df, x_var, z_var, categories, colors):
# #     # # Create a figure
# #     # fig, ax = plt.subplots(figsize=(10, 10))

# #     # Create a FacetGrid varying by the categorical variable, using the order and palette defined
# #     g = sns.FacetGrid(
# #         df,
# #         row=z_var["column_name"],
# #         hue=z_var["column_name"],
# #         aspect=2.5,
# #         height=1.5,
# #         palette=colors,
# #         row_order=categories,
# #     )
# #     # https://stackoverflow.com/questions/45911709/limit-the-range-of-x-in-seaborn-distplot-kde-estimation

# #     # Map the kdeplot for the variable of interest across the FacetGrid
# #     def plot_kde_and_lines(x, color, label):
# #         ax = plt.gca()  # Get current axis
# #         sns.kdeplot(
# #             x,
# #             bw_adjust=0.1,
# #             clip_on=False,
# #             fill=True,
# #             alpha=0.5,
# #             clip=[0, 5],
# #             linewidth=0,
# #             color=color,
# #             ax=ax,
# #         )
# #         sns.kdeplot(
# #             x,
# #             bw_adjust=0.1,
# #             clip_on=False,
# #             clip=[0, 5],
# #             linewidth=2.5,
# #             color="w",
# #             ax=ax,
# #         )
# #         # Median
# #         median_value = x.median()
# #         ax.axvline(median_value, color=color, linestyle=":", lw=2, label="Median")
# #         # Mode (using KDE peak as a proxy)
# #         kde = gaussian_kde(x, bw_method=0.1)
# #         kde_values = np.linspace(x.min(), x.max(), 1000)
# #         mode_value = kde_values[np.argmax(kde(kde_values))]
# #         ax.axvline(mode_value, color=color, linestyle="--", lw=2, label="Mode")

# #     #
# #     g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# #     # Map the custom plotting function
# #     g.map(plot_kde_and_lines, x_var["column_name"])

# #     # Set the subplots to overlap
# #     g.fig.subplots_adjust(hspace=-2)

# #     # Add a horizontal line for each plot
# #     g.map(plt.axhline, y=0, lw=2, clip_on=False)

# #     # Define and use a simple function to label the plot in axes coordinates
# #     def label(x, color, label):
# #         ax = plt.gca()
# #         ax.text(
# #             0,
# #             0.2,
# #             label,
# #             fontweight="bold",
# #             color=color,
# #             ha="left",
# #             va="center",
# #             transform=ax.transAxes,
# #             size=16,
# #         )

# #     g.map(label, x_var["column_name"])

# #     # Remove axes details that don't play well with overlap
# #     g.set_titles("")
# #     g.set(yticks=[], ylabel="", xlabel=r"$q$ [-]")
# #     g.despine(bottom=True, left=True)

# #     # Adjust the layout
# #     plt.tight_layout()
# #     plt.show()

# #     return g


# # vegetation_color_dict_limit = {
# #     "Open shrublands": "#C99728",
# #     "Grasslands": "#13BFB2",
# #     "Savannas": "#92BA31",
# #     "Woody savannas": "#4C6903",
# #     "Croplands": "#F7C906",
# # }

# # fig_ridge_veg = plot_ridgeplot(
# #     df=df_filt_q,
# #     x_var=var_dict["q_q"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict_limit.keys(),
# #     colors=list(vegetation_color_dict_limit.values()),
# # )

# # fig_ridge_veg.savefig(
# #     os.path.join(fig_dir, f"sup_hist_ridge_veg.pdf"), dpi=1200, bbox_inches="tight"
# # )

# # fig_ridge_veg.savefig(
# #     os.path.join(fig_dir, f"sup_hist_ridge_veg.png"), dpi=1200, bbox_inches="tight"
# # )



# # %%


# # %%
# def plot_q_ai_wood_scatter(df):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sc=ax.scatter(df["fractional_wood"], df["q_q"], c=df["AI"], cmap="RdBu", alpha=0.3)
#     ax.set_ylim(0,1.5)
#     ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
#     ax.set_ylabel(var_dict["q_q"]["label"]+" "+var_dict["q_q"]["symbol"])
#     # Create a colorbar with the scatter plot's color mapping
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label('Aridity Index (MAP/MAE')

# plot_q_ai_wood_scatter(df_filt_q_conus)

#     # %%
# # ##########################################################
# # # Similar plot but in scatter plot format
# # ##########################################################

# # Get unique AI_binned2 values and assign colors
# cmap = plt.get_cmap('RdBu')
# ai_bins_list = percentage_df['AI_binned2'].unique()
# colors = cmap(np.linspace(0, 1, len(ai_bins_list)))
# color_map = dict(zip(ai_bins_list, colors))

# plt.rcParams.update({"font.size": 16})
# fig, ax = plt.subplots(figsize=(8, 5))
# # for (ai_bin, group) in percentage_df.groupby('AI_binned2'):
# #     ax.plot(group['fractional_wood_pct'], group['percentage_q_gt_1'], label=ai_bin, alpha=0.7, marker='o', color=color_map[ai_bin])
# for (ai_bin, group) in percentage_df.groupby('AI_binned2'):
#     ax.plot(group['fractional_wood_pct'], group['percentage_q_gt_1'], label=ai_bin, alpha=0.7, marker='o', color=color_map[ai_bin])

# ax.set_xlabel('Fractional Wood Coverage (%)')
# ax.set_ylabel('Proportion of events with\n'+r'$q>1$'+'(convex non-linearity) (%)')
# ax.legend(title='Aridity Index\n[MAP/MAE]', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
# ax.set_ylim([60, 100])  # Adjusting y-axis limits to 0-100% for percentage
# plt.xticks(rotation=45)
# fig.tight_layout()


# # %%
# #############################################################
# # Get the Barren + Litter + Other percentage
# #############################################################
# # plt_idx= ~pd.isna(df_filt_q_conus["barren_percent"])
# # Get Barren percent
# df_filt_q_conus["nonveg_percent"] = df_filt_q_conus["barren_percent"] + (
#     100 - df_filt_q_conus["totalrangeland_percent"]
# )
# percentage_df2 = get_df_percentage_q(df_filt_q_conus, "barren_percent")

# # # Plotting the first set of bars (percentage_q_gt_1)
# def plot_fracq_by_pct(ax, df, x_column_to_plot, var_name, title_name, weighted=False):

#     if weighted:
#         y_var_q_le_1 = "weighted_percentage_q_le_1"
#         y_var_q_gt_1 = "weighted_percentage_q_gt_1"
#     else:
#         y_var_q_le_1 = "percentage_q_le_1"
#         y_var_q_gt_1 = "percentage_q_gt_1"

#     sns.barplot(
#         x=x_column_to_plot,
#         y=y_var_q_le_1,
#         data=df,
#         color="#FFE268",
#         label=y_var_q_le_1,
#         ax=ax,
#         width=0.98,
#         edgecolor="white",
#         linewidth=3,
#     )

#     sns.barplot(
#         x=x_column_to_plot,
#         y=y_var_q_gt_1,
#         data=df,
#         color="#22BBA9",
#         label=y_var_q_gt_1,
#         ax=ax,
#         width=0.98,
#         edgecolor="white",
#         linewidth=3,
#         bottom=df[y_var_q_le_1],
#     )

#     ax.set_xlabel(f"{var_dict[var_name]['label']} {var_dict[var_name]['unit']}")
#     ax.set_ylabel("Proportion of drydown events (%)")
#     # plt.legend(title='Aridity Index [MAP/MAE]', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
#     if weighted:
#         ymax=20
#     else:
#         ymax=50
#     ax.set_ylim([0, ymax])
#     plt.xticks(rotation=45)
#     ax.set_title(title_name, loc="left")

#     ax.legend_ = None
# # %% Barren plot of q>1 vs q<1 by vegetation and aridity, for "other" land-uses
# fig = plt.figure(figsize=(8, 4))

# plt.rcParams.update({"font.size": 12})
# ax1 = plt.subplot(121)
# subset_df = percentage_df2[percentage_df2["AI_binned2"] == "0-0.5"]
# plot_fracq_by_pct(
#     ax1,
#     subset_df,
#     "barren_percent_pct",
#     "rangeland_barren",
#     "A.              P/PET < 0.5",
# )

# ax1 = plt.subplot(122)
# subset_df2 = percentage_df2[percentage_df2["AI_binned2"] == "1.5-"]
# plot_fracq_by_pct(
#     ax1,
#     subset_df2,
#     "barren_percent_pct",
#     "rangeland_barren",
#     "B.             P/PET > 1.5",
# )
# plt.tight_layout()

# plt.savefig(
#     os.path.join(fig_dir, f"fracq_fracbarren_ai.pdf"), dpi=1200, bbox_inches="tight"
# )

# # %%
# # Get "other" land-use percent in vegetated area
# percentage_df3 = get_df_percentage_q(
#     df_filt_q_conus[df_filt_q_conus["nonveg_percent"] < 20],
#     "nonveg_percent",
#     bins=[0, 5, 10, 15, 20],
#     labels=["0-5%", "5-10%", "10-15%", "15-20%"],
# )
# percentage_df3
# # %%
# # Statistics of q>1 vs q<1 by vegetation and aridity, for "other" land-uses
# fig = plt.figure(figsize=(8, 4))

# plt.rcParams.update({"font.size": 12})
# ax1 = plt.subplot(121)
# subset_df = percentage_df3[percentage_df3["AI_binned2"] == "0-0.5"]
# plot_fracq_by_pct(
#     ax1,
#     subset_df,
#     "nonveg_percent_pct",
#     "rangeland_20%",
#     "A.              P/PET < 0.5",
# )

# ax1 = plt.subplot(122)
# subset_df2 = percentage_df3[percentage_df3["AI_binned2"] == "1.5-"]
# plot_fracq_by_pct(
#     ax1,
#     subset_df2,
#     "nonveg_percent_pct",
#     "rangeland_20%",
#     "B.             P/PET > 1.5",
# )
# plt.tight_layout()

# plt.savefig(
#     os.path.join(fig_dir, f"fracq_frac20pctNonveg_ai.pdf"),
#     dpi=1200,
#     bbox_inches="tight",
# )


# #

# #%%
# from matplotlib.cm import get_cmap

# def sort_percentages(labels):
#     return sorted(labels, key=lambda x: int(x.split('-')[0]))


# def plot_bars(ax, df, x_var, y_var, z_var, title):

#     # Determine unique groups and categories
#     # Define the width of the bars and the space between groups
#     bar_width = 0.2
#     space_between_bars  = 0.025
#     space_between_groups = 0.2
    
#     # Determine unique values for grouping
#     x_unique = df[x_var].unique()
#     z_unique = sort_percentages(df[z_var].unique())
#     n_groups = len(z_unique)

#     # Generate a green colormap
#     colormap = get_cmap('Greens')
#     color_list = [colormap(i / n_groups) for i in range(n_groups)]
    
#     # Create the grouped and stacked bars
#     for z_i, z in enumerate(z_unique):
#         for x_i, x in enumerate(x_unique):
    
#             # Calculate the x position for each group
#             group_offset = (bar_width + space_between_bars) * n_groups
#             x_pos = x_i * (group_offset + space_between_groups) + (bar_width + space_between_bars) * z_i
            
#             # Get the subset of data for this group
#             subset = df[(df[x_var] == x) & (df[z_var] == z)]
            
#             # Get bottom values for stacked bars
#             ax.bar(
#                 x_pos,
#                 subset[y_var["column_name"]].median(),
#                 bar_width,
#                 edgecolor='white',
#                 color=color_list[z_i],
#             )
    
#     # Set the x-ticks to the middle of the groups
#     ax.set_xticks([i * (group_offset + space_between_groups) + group_offset / 2 for i in range(len(x_unique))])
#     ax.set_xticklabels(x_unique, rotation=45)
#     ax.set_xlabel(f"{var_dict["ai_bins"]['label']} {var_dict["ai_bins"]['unit']}")
#     ax.set_ylabel(f"{y_var['label']} {y_var['unit']}, Median")
#     ax.set_title(title)

#     # Set the second x-ticks
#     # Replicate the z_var labels for the number of x_column_to_plot labels
#     z_labels = np.tile(z_unique, len(x_unique))

#     # Adjust the tick positions for the replicated z_var labels
#     new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

#     # Hide the original x-axis ticks and labels
#     ax.tick_params(axis='x', which='both', length=0)

#     # Create a secondary x-axis for the new labels
#     ax2 = ax.twiny()
#     ax2.set_xlim(ax.get_xlim())
#     ax2.set_xticks(new_tick_positions)

#     # Adjust the secondary x-axis to appear below the primary x-axis
#     ax2.spines['top'].set_visible(False)
#     ax2.xaxis.set_ticks_position('bottom')
#     ax2.xaxis.set_label_position('bottom')
#     ax2.spines['bottom'].set_position(('outward', 60))
#     ax2.tick_params(axis='x', which='both', length=0)
#     ax2.set_xticklabels(z_labels, rotation=45)

# plt.rcParams.update({"font.size": 11})

# var_to_test = ["event_length", "q_ETmax", "theta_star", "theta_w", "sm_range", "q_q", "q_delta_theta", "sm_range_abs"]

# for var_name in var_to_test:
#     for i, condition in enumerate(["q>1", "q<1"]):
#         fig, ax = plt.subplots(figsize=(6, 3))
#         if condition == "q<1":
#             subset = df_filt_q_conus[(df_filt_q_conus["q_q"]<1)&~pd.isna(df_filt_q_conus["fractional_wood"])]
#             title = r"$q < 1$"
#         elif condition == "q>1":
#             subset = df_filt_q_conus[(df_filt_q_conus["q_q"]>1)&~pd.isna(df_filt_q_conus["fractional_wood"])]
#             title = r"$q > 1$"
#         bins=[0, 20, 40, 60, 80, 100]
#         labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
#         subset["fractional_wood_pct"] = pd.cut(subset["fractional_wood"], bins=bins, labels=labels)
#         plot_bars(
#             ax=ax,
#             df=subset,
#             y_var=var_dict[var_name],
#             x_var="AI_binned2",
#             z_var="fractional_wood_pct",
#             title=title
#         )

# #%%
# subset.head()

# # %%
# df_filt_q_conus.columns

# # %%
# # # %%
# # longest_events = df_filt_q_conus.sort_values(by='event_length', ascending=False).groupby(['EASE_row_index', 'EASE_column_index']).first().reset_index()
# # group_sizes = df_filt_q_conus.groupby(['EASE_row_index', 'EASE_column_index']).size().reset_index(name='size')
# # filtered_groups = group_sizes[group_sizes['size'] >= 5]

# # # Now, only keep rows from longest_events that have groups with size >= 5
# # longest_events_filtered = longest_events[longest_events.set_index(['EASE_row_index', 'EASE_column_index']).index.isin(filtered_groups.set_index(['EASE_row_index', 'EASE_column_index']).index)]

# # # %%
# # percentage_df_longest_event = get_df_percentage_q(longest_events, "fractional_wood")
# # fig, ax = plt.subplots(figsize=(6, 4))

# # plt.rcParams.update({"font.size": 11})
# # plot_grouped_stacked_bar(
# #     ax=ax,
# #     df=percentage_df_longest_event,
# #     x_column_to_plot="AI_binned2",
# #     z_var="fractional_wood_pct",
# #     var_name="rangeland_wood",
# #     title_name="",
# #     weighted=True
# # )
# # plt.tight_layout()

# # # %%
# # percentage_df_longest_event = get_df_percentage_q(longest_events_filtered, "fractional_wood")
# # fig, ax = plt.subplots(figsize=(6, 4))
# # plot_grouped_stacked_bar(
# #     ax=ax,
# #     df=percentage_df_longest_event,
# #     x_column_to_plot="AI_binned2",
# #     z_var="fractional_wood_pct",
# #     var_name="rangeland_wood",
# #     title_name="",
# #     weighted=False
# # )
# # plt.tight_layout()


# # # %%

# # def plot_q_ai_wood_scatter(df):
# #     fig, ax = plt.subplots(figsize=(6, 4))
# #     sc=ax.scatter(df["fractional_wood"], df["q_q"], c=df["AI"], cmap="RdBu", alpha=0.3)
# #     ax.set_ylim(0,10)
# #     ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
# #     ax.set_ylabel(var_dict["q_q"]["label"]+" "+var_dict["q_q"]["symbol"])
# #     # Create a colorbar with the scatter plot's color mapping
# #     cbar = plt.colorbar(sc, ax=ax)
# #     cbar.set_label('Aridity Index (MAP/MAE')

# # plot_q_ai_wood_scatter(longest_events_filtered)
# # # %%
# # # Assuming df is the DataFrame with the relevant data and it contains a column named 'data'
# # # for which we want to calculate the coefficient of variation.
# # # Group by 'EASE_row_index' and 'EASE_column_index' and filter groups with count 16

# # # First, group by the indices and filter out the groups with exactly 16 observations
# # long_events = df_filt_q_conus.groupby(['EASE_row_index', 'EASE_column_index']).filter(lambda x: len(x) >=16)

# # # Now, calculate the coefficient of variation for each group
# # def coefficient_of_variation(x):
# #     return x.std() / x.mean()

# # stat = long_events.groupby(['EASE_row_index', 'EASE_column_index']).agg(fractional_wood=("fractional_wood", "median"), AI=("AI", "median"))
# # stat["q_cv"] = long_events.groupby(['EASE_row_index', 'EASE_column_index'])["q_q"].agg(coefficient_of_variation)

# # # %%
# # fig, ax = plt.subplots(figsize=(6, 4))
# # sc=ax.scatter(stat["fractional_wood"], stat["q_cv"], c=stat["AI"], cmap="RdBu", alpha=0.4)
# # ax.set_ylim(0,4)
# # ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
# # ax.set_ylabel("Coefficient of variation of" +var_dict["q_q"]["symbol"])
# # # Create a colorbar with the scatter plot's color mapping
# # cbar = plt.colorbar(sc, ax=ax)
# # cbar.set_label('Aridity Index (MAP/MAE')
# # # %%
# # from pylab import *
# # def plot_rangeland_map(ax, df, var_item, cmap):
# #     plt.rcParams.update({"font.size": 12})

# #     df = df.drop_duplicates(subset=['latitude', 'longitude'])
# #     pivot_array = df.pivot(
# #         index="latitude", columns="longitude", values=var_item
# #     ) 
# #     pivot_array[pivot_array.index > -60]

# #     # Get lat and lon
# #     lons = pivot_array.columns.values
# #     lats = pivot_array.index.values

# #     # Plot in the map
# #     custom_cmap = cm.get_cmap(cmap, 5)

# #     im = ax.pcolormesh(
# #         lons, lats, pivot_array, cmap=custom_cmap, transform=ccrs.PlateCarree()
# #     )
    
# #     ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
# #     ax.coastlines()
# #     ax.set_extent([-125, -66.93457, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# #     # Add colorbar
# #     plt.colorbar(
# #         im,
# #         ax=ax,
# #         orientation="vertical",
# #         shrink=0.35,
# #         pad=0.02,
# #     )

# #     ax.set_xlabel("Longitude")
# #     ax.set_ylabel("Latitude")
# #     ax.set_title(var_item)

# # fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# # plot_rangeland_map(ax, rangeland_info, "fractional_wood", cmap="BuGn")

# # fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# # plot_rangeland_map(ax, rangeland_info, "fractional_herb", cmap="BuGn")

# # #%%
# # # Assuming rangeland_info is your DataFrame
# # # This will mark all rows that are duplicates as True, keeping the first occurrence as False (not a duplicate by default)
# # duplicates = rangeland_info.duplicated(keep=False)

# # # To show the duplicate rows
# # duplicate_rows = rangeland_info[duplicates]

# # duplicate_rows
# # # %%




# # fig = plt.figure(figsize=(8, 4))

# # plt.rcParams.update({"font.size": 12})
# # ax1 = plt.subplot(121)
# # subset_df = percentage_df[percentage_df["AI_binned2"] == "0-0.5"]
# # plot_fracq_by_pct(
# #     ax1,
# #     subset_df,
# #     "fractional_wood_pct",
# #     "rangeland_wood",
# #     "A.              P/PET < 0.5",
# # )

# # ax1 = plt.subplot(122)
# # subset_df2 = percentage_df[percentage_df["AI_binned2"] == "1.5-"]
# # plot_fracq_by_pct(
# #     ax1,
# #     subset_df2,
# #     "fractional_wood_pct",
# #     "rangeland_wood",
# #     "B.             P/PET > 1.5",
# # )
# # plt.tight_layout()

# # plt.savefig(
# #     os.path.join(fig_dir, f"fracq_fracwood_ai.pdf"), dpi=1200, bbox_inches="tight"
# # )
# # %%

# # %% Loss function plots + parameter scatter plots

# # # %%  ##########################
# # ## Loss function plots + parameter scatter plots  (Figure 4)
# # ##########################
# # save = True
# # # Create a figure with subplots
# # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# # # mpl.rcParams['font.size'] = 18
# # # # plt.rcParams.update({'axes.labelsize' : 14})
# # # plt.rcParams.update({"font.size": 18})
# # # Plotting
# # plot_loss_func_categorical(
# #     axs[0],
# #     df_filt_q,
# #     var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title=None,
# #     plot_legend=False,
# # )

# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[1],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["q_q"],
# #     z_var=var_dict["veg_class"],
# #     categories=list(vegetation_color_dict.keys()),
# #     colors=list(vegetation_color_dict.values()),
# #     quantile=25,
# #     title=None,
# #     plot_logscale=False,
# #     plot_legend=False,
# # )

# # axs[1].set_ylim([0, 0.75])
# # axs[1].set_xlim([0, 1.0])

# # # axs[0].set_yticks(
# #     [-0.1, -0.08, -0.06, -0.04, -0.02, 0.0], ["0.10", 0.08, 0.06, 0.04, 0.02, "0.00"]
# # )
# # axs[1].set_yticks(
# #     [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], ["", 1.0, "", 2.0, "", 3.0, "", 4.0]
# # )


# plt.tight_layout()
# plt.show()

# if save:
#     # Save the combined figure
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.png"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.pdf"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.svg"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
# %% Run this befroe running fig_stats.py
# %% Import packages
import os
import getpass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import cartopy.crs as ccrs

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from textwrap import wrap

from functions import q_model, loss_model
from matplotlib.colors import LinearSegmentedColormap

# !pip install mpl-scatter-density
import mpl_scatter_density
from scipy.stats import spearmanr
import statsmodels.api as statsm
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import json
from scipy.stats import mannwhitneyu, ks_2samp, median_test
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-05-13_global_piecewise"  # "raraki_2024-02-02"  # f"raraki_2023-11-25_global_95asmax"
############################|###############################################
# %%
# Define some variables
z_mm = 50  # Soil thickness

# %% ############################################################################
# DATA IMPORT & PATH CONFIGS

# Data dir
user_name = getpass.getuser()
data_dir = rf"/home/{user_name}/waves/projects/smap-drydown/data"
datarods_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
anc_rangeland_file = "anc_info_rangeland.csv"
anc_rangeland_processed_file = "anc_info_rangeland_processed.csv"
anc_Bassiouni_params_file = "anc_info_Bassiouni.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"
coord_info_file = "coord_info.csv"

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################
note_dir = f"/home/{user_name}/smap-drydown/notebooks"
## Define parameters
with open(os.path.join(note_dir, "fig_veg_colors_lim.json"), "r") as file:
    vegetation_color_dict = json.load(file)

# Load variable settings
with open(os.path.join(note_dir, "fig_variable_labels.json"), "r") as file:
    var_dict = json.load(file)


################ Read the model output (results) ################
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

# Open the output file
f = open(os.path.join(fig_dir, "log.txt"), "w")
original_stdout = sys.stdout  # Save the original stdout
sys.stdout = f  # Change the stdout to the file handle

# ANCILLARY DATA IMPORT
# Read coordinate information
coord_info = pd.read_csv(os.path.join(data_dir, datarods_dir, coord_info_file))
df = _df.merge(coord_info, on=["EASE_row_index", "EASE_column_index"], how="left")
print("Loaded coordinate information")

# Ancillary df
df_anc = pd.read_csv(os.path.join(data_dir, datarods_dir, anc_file)).drop(
    ["spatial_ref", "latitude", "longitude"], axis=1
)
df_anc.loc[df_anc["sand_fraction"] < 0, "sand_fraction"] = np.nan
print("Loaded ancillary information (sand fraction and land-cover)")

# Aridity indices
df_ai = pd.read_csv(os.path.join(data_dir, datarods_dir, ai_file)).drop(
    ["latitude", "longitude"], axis=1
)
df_ai.loc[df_ai["AI"] < 0, "AI"] = np.nan
print("Loaded ancillary information (aridity index)")

# Land cover
IGBPclass = pd.read_csv(os.path.join(data_dir, anc_dir, IGBPclass_file))

df = df.merge(df_anc, on=["EASE_row_index", "EASE_column_index"], how="left")
df = df.merge(df_ai, on=["EASE_row_index", "EASE_column_index"], how="left")
df = pd.merge(df, IGBPclass, left_on="IGBP_landcover", right_on="class", how="left")
# df["name"][df["name"]=="Croplands"] = "Cropland/natural vegetation"
# df["name"][df["name"]=="Cropland/natural vegetation mosaics"] = "Cropland/natural vegetation"
print("Loaded ancillary information (land-cover)")
print(df["name"].unique())
# Get the binned ancillary information


# %% ############################################################################
# Calculate some stats for evaluation

# Difference between R2 values of two models
df = df.assign(diff_R2_q_tauexp=df["q_r_squared"] - df["tauexp_r_squared"])
df = df.assign(diff_R2_q_exp=df["q_r_squared"] - df["exp_r_squared"])


def df_availability(row):
    # Check df point availability in the first 3 time steps of observation
    # Define a helper function to convert string to list
    def str_to_list(s):
        return list(map(int, s.strip("[]").split()))

    # Convert the 'time' column from string of lists to actual lists
    time_list = str_to_list(row["time"])

    # Check if first three items are [0, 1, 2]
    # condition = time_list[:3] == [0, 1, 2]

    # Check if at least 2 elements exist in any combination (0 and 1, 0 and 2, or 1 and 2) in the first 3 elements
    first_3_elements = set(time_list[:3])
    required_elements = [{0, 1}, {0, 2}, {1, 2}]

    condition = any(
        required.issubset(first_3_elements) for required in required_elements
    )

    return condition


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
    return np.abs(sm_range)


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


# Create new columns
df["first3_avail2"] = df.apply(df_availability, axis=1)
df["sm_range"] = df.apply(calculate_sm_range, axis=1)
df["event_length"] = (
    pd.to_datetime(df["event_end"]) - pd.to_datetime(df["event_start"])
).dt.days + 1
df["large_q_criteria"] = df.apply(check_1ts_range, axis=1)


# %%
output_path = os.path.join(output_dir, dir_name, "all_results_processed.csv")
df.to_csv(output_path)
print(output_path)
# %%

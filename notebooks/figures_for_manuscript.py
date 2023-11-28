# %% Import packages
import os
import getpass

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
from functions import q_drydown, exponential_drydown, loss_model

import utils_figs as utils

# %% DATA IMPORT 

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
output_dir_name = f"raraki_2023-11-25_global_95asmax"
###########################################################################

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

# Read coordinate information
coord_info = pd.read_csv(os.path.join(data_dir, datarod_dir, coord_info_file))
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

print(f"Total number of drydown event: {len(df)}")

# %% Get some stats
df = df.assign(_diff_R2=df["q_r_squared"] - df["exp_r_squared"])

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

df_filt_q = df[df['q_r_squared'] >= utils.success_modelfit_thresh].copy()
df_filt_q2 = df_filt_q[df_filt_q['sm_range']> utils.sm_range_thresh].copy()
df_filt_q_and_exp = df[(df['q_r_squared'] >= utils.success_modelfit_thresh) | (df['exp_r_squared'] >= success_modelfit_thresh)].copy()
df_filt_exp = df[df['exp_r_squared'] >= utils.success_modelfit_thresh].copy()

print(f"q model fit was successful: {len(df_filt_q)}")
print(f"q model fit was successful & fit over {utils.sm_range_thresh*100} percent of the soil mositure range: {len(df_filt_q2)}")
print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")
print(f"exp model fit was successful: {len(df_filt_exp)}")

# %%
############################################################################
# Map plots 
###########################################################################

# %% Map of parameters 


# %% Map of R2 values 


# %% 
############################################################################
# Scatter plots
###########################################################################

# %% Scatter plots of R2 values 


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


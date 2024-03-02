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
from datetime import datetime
from functions import q_drydown, exponential_drydown, loss_model

!pip install mpl-scatter-density
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import ast

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-02-02"
###########################################################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define model acceptabiltiy criteria
z_mm = 50  # Soil thickness

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
    }
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

# %%
# Land cover
IGBPclass = pd.read_csv(os.path.join(data_dir, anc_dir, IGBPclass_file))
IGBPclass.rename({"name": "landcover_name"}, inplace=True)

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

# %%
df["event_length"] = (pd.to_datetime(df['event_end']) - pd.to_datetime(df['event_start'])).dt.days
df["event_length"]
# %%
###################################################
# Defining thresholds
q_thresh = 1e-03
success_modelfit_thresh = 0.7
sm_range_thresh = 0.1
###################################################

# Runs where q model performed reasonablly well
df_filt_q = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["q_q"] > q_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()
print(
    f"q model fit was successful & fit over {sm_range_thresh*100} percent of the soil mositure range, plus extremely small q removed: {len(df_filt_q)}"
)

# Runs where both of the model performed satisfactory
df_filt_q_and_exp = df[
    (df["q_r_squared"] >= success_modelfit_thresh)
    & (df["exp_r_squared"] >= success_modelfit_thresh)
    & (df["sm_range"] > sm_range_thresh)
].copy()
print(f"both q and exp model fit was successful: {len(df_filt_q_and_exp)}")

# %%
# Filter out drydown events and identify the best one to present

# Get general
# print(df[(df["q_q"] > 1)& (df["sm_range"]>0.1) & ((df["q_r_squared"] - df["exp_r_squared"])>0.1) & (df["q_r_squared"]>0.8)].index)

# %

def plot_drydown(event_id, ax=None, plot_precip=False, save=False):

    # Assuming 'df' is your DataFrame and 'event_id' is defined
    event = df.loc[event_id]

    # Convert the modified string to a NumPy array
    # Replace '\n' with ' ' (space) to ensure all numbers are separated by spaces
    input_string = event.sm.replace('\n', ' np.nan').replace(' nan', ' np.nan').strip('[]')
    sm = np.array([float(value) if value != 'np.nan' else np.nan for value in input_string.split()])
    values = event.time.strip('[]').split()
    t_d = np.array([int(value) for value in values])

    # Calculating n_days
    n_days = (pd.to_datetime(event.event_end) - pd.to_datetime(event.event_start)).days

    # Define variables and parameters
    theta = np.arange(0, 1, 1/24)
    t = np.arange(0, n_days, 1/24)
    k = event.q_k
    q0 = 1
    q = event.q_q
    delta_theta = event.q_delta_theta
    min_sm = event.min_sm
    max_sm = event.max_sm
    exp_delta_theta = event.exp_delta_theta
    theta_w = event.exp_theta_w
    tau = event.exp_tau
    norm_sm = (sm - min_sm) / (max_sm - min_sm)
    y_obs = norm_sm[~np.isnan(norm_sm)] * (max_sm - min_sm) + min_sm 
    y_nonlinear = q_drydown(t=t, k=k, q=q, delta_theta=delta_theta) * (max_sm - min_sm) + min_sm
    y_exp = exponential_drydown(t, exp_delta_theta, theta_w, tau)


    ############################################################
    # Get timeseries of SM data
    ############################################################

    def get_filename(varname, EASE_row_index, EASE_column_index):
        """Get the filename of the datarod"""
        filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
        return filename

    def set_time_index(df, index_name="time"):
        """Set the datetime index to the pandas dataframe"""
        df[index_name] = pd.to_datetime(df[index_name])
        return df.set_index("time")

    data_dir = r"/home/waves/projects/smap-drydown/data"
    datarods_dir = "datarods"

    def get_dataframe(varname, event):
        """Get the pandas dataframe for a datarod of interest

        Args:
            varname (string): name of the variable: "SPL3SMP", "PET", "SPL4SMGP"

        Returns:
            dataframe: Return dataframe with datetime index, cropped for the timeperiod for a variable
        """

        fn = get_filename(
            varname,
            EASE_row_index=event.EASE_row_index,
            EASE_column_index=event.EASE_column_index,
        )
        _df = pd.read_csv(os.path.join(data_dir, datarods_dir, varname, fn))

        # Set time index and crop
        df = set_time_index(_df, index_name="time")
        return df

    def get_soil_moisture(varname="SPL3SMP", event=None):
        """Get a datarod of soil moisture data for a pixel"""

        # Get variable dataframe
        _df = get_dataframe(varname=varname, event=event)

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (
            _df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 8.0)
        condition_bad_data_pm = (
            _df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 8.0)
        _df.loc[
            condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
        ] = np.nan
        _df.loc[
            condition_bad_data_pm, "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"
        ] = np.nan

        # If there is two different versions of 2015-03-31 data --- remove this
        df = _df.loc[~_df.index.duplicated(keep="first")]

        # Resample to regular time interval
        df = df.resample("D").asfreq()

        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df["soil_moisture_daily"] = df[
            [
                "Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
            ]
        ].mean(axis=1, skipna=True)

        return df["soil_moisture_daily"]

    def get_precipitation(varname="SPL4SMGP", event=None):
        """Get a datarod of precipitation data for a pixel"""

        # Get variable dataframe
        _df = get_dataframe(varname=varname, event=event)

        # Drop unnccesary dimension and change variable name
        _df = _df.drop(columns=["x", "y"]).rename(
            {"precipitation_total_surface_flux": "precip"}, axis="columns"
        )

        # Convert precipitation from kg/m2/s to mm/day -> 1 kg/m2/s = 86400 mm/day
        _df.precip = _df.precip * 86400

        # Resample to regular time interval
        return _df.resample("D").asfreq()

    df_ts = get_soil_moisture(event=event)
    df_p = get_precipitation(event=event)

    # Plotting settings 
    nonlinear_label = rf'Nonlinear model ($R^2$={event.q_r_squared:.2f}, $q$={q:.1f})'
    #; $\theta^*$={max_sm:.2f}; $\Delta \theta$={event.q_delta_theta:.2f}; $ET_max$={event.q_ETmax:.2f}'
    linear_label = rf'Linear model ($R^2$={event.exp_r_squared:.2f}, $\tau$={tau:.2f})'

    # 
    start_date = pd.to_datetime(event.event_start) - pd.Timedelta(7, 'D')
    end_date = pd.to_datetime(event.event_end) + pd.Timedelta(7, 'D')
    date_range = pd.date_range(start=pd.to_datetime(event.event_start), end=pd.to_datetime(event.event_end), freq="H")

    # 
    # Plotting
    if plot_precip:
        # Create a figure
        fig = plt.figure(figsize=(7, 3.5))

        import matplotlib.gridspec as gridspec
        # Set up a GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # Create the subplots
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.scatter(df_ts[start_date:end_date].index, df_ts[start_date:end_date].values, color='grey', label='SMAP observation')
        ax1.plot(date_range[:-1], y_nonlinear, label=nonlinear_label, color='darkorange')
        ax1.plot(date_range[:-1], y_exp, label=linear_label, color='darkblue', alpha=0.5)
        ax2.bar(df_p[start_date:end_date].index, df_p[start_date:end_date].values.flatten(), color='grey')
        ax2.set_ylabel("Precipitation \n[mm/d]")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Soil moisture content' + '\n' + rf'$\theta$ $[m3/m3]$')
        ax1.legend(loc='upper right')
        ax1.set_title(f"Latitude: {event.latitude:.1f}; Longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand)")

        # Optional: Hide x-ticks for ax1 if they're redundant
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Adjust the subplots to prevent overlap
        plt.subplots_adjust(hspace=0.1)  # Adjust the space between plots if necessary

        fig.tight_layout()
        fig.autofmt_xdate()
    
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.2, 4))
        
        ax.scatter(df_ts[start_date:end_date].index, df_ts[start_date:end_date].values, color='k', label='SMAP observation')
        ax.plot(date_range[:-1], y_nonlinear, label=nonlinear_label, color='darkorange')
        ax.plot(date_range[:-1], y_exp, label=linear_label, color='darkblue', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Soil moisture content' + '\n' + rf'$\theta$ $[m3/m3]$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))
        ax.set_title(f"Latitude: {event.latitude:.1f}; Longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand)")

        plt.tight_layout()
        # ax.autofmt_xdate()
        # Optional: Hide x-ticks for ax1 if they're redundant
        # plt.setp(ax1.get_xticklabels(), visible=False)

        # Adjust the subplots to prevent overlap
        # plt.subplots_adjust(hspace=0.1)  # Adjust the space between plots if necessary

        # fig.tight_layout()
        # fig.autofmt_xdate()


    if save:
        fig.savefig(
            os.path.join(fig_dir, f"event_{event_id}.png"), dpi=1200, bbox_inches="tight"
        )


#%%
        
def plot_drydown(event_id, ax=None, legend=True, save=False):

    # Assuming 'df' is your DataFrame and 'event_id' is defined
    event = df.loc[event_id]

    # Convert the modified string to a NumPy array
    # Replace '\n' with ' ' (space) to ensure all numbers are separated by spaces
    input_string = event.sm.replace('\n', ' np.nan').replace(' nan', ' np.nan').strip('[]')
    sm = np.array([float(value) if value != 'np.nan' else np.nan for value in input_string.split()])
    values = event.time.strip('[]').split()
    t_d = np.array([int(value) for value in values])

    # Calculating n_days
    n_days = (pd.to_datetime(event.event_end) - pd.to_datetime(event.event_start)).days

    # Define variables and parameters
    theta = np.arange(0, 1, 1/24)
    t = np.arange(0, n_days, 1/24)
    k = event.q_k
    q0 = 1
    q = event.q_q
    delta_theta = event.q_delta_theta
    min_sm = event.min_sm
    max_sm = event.max_sm
    exp_delta_theta = event.exp_delta_theta
    theta_w = event.exp_theta_w
    tau = event.exp_tau
    norm_sm = (sm - min_sm) / (max_sm - min_sm)
    y_obs = norm_sm[~np.isnan(norm_sm)] * (max_sm - min_sm) + min_sm 
    y_nonlinear = q_drydown(t=t, k=k, q=q, delta_theta=delta_theta) * (max_sm - min_sm) + min_sm
    y_exp = exponential_drydown(t, exp_delta_theta, theta_w, tau)


    ############################################################
    # Get timeseries of SM data
    ############################################################

    def get_filename(varname, EASE_row_index, EASE_column_index):
        """Get the filename of the datarod"""
        filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
        return filename

    def set_time_index(df, index_name="time"):
        """Set the datetime index to the pandas dataframe"""
        df[index_name] = pd.to_datetime(df[index_name])
        return df.set_index("time")

    data_dir = r"/home/waves/projects/smap-drydown/data"
    datarods_dir = "datarods"

    def get_dataframe(varname, event):
        """Get the pandas dataframe for a datarod of interest

        Args:
            varname (string): name of the variable: "SPL3SMP", "PET", "SPL4SMGP"

        Returns:
            dataframe: Return dataframe with datetime index, cropped for the timeperiod for a variable
        """

        fn = get_filename(
            varname,
            EASE_row_index=event.EASE_row_index,
            EASE_column_index=event.EASE_column_index,
        )
        _df = pd.read_csv(os.path.join(data_dir, datarods_dir, varname, fn))

        # Set time index and crop
        df = set_time_index(_df, index_name="time")
        return df

    def get_soil_moisture(varname="SPL3SMP", event=None):
        """Get a datarod of soil moisture data for a pixel"""

        # Get variable dataframe
        _df = get_dataframe(varname=varname, event=event)

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (
            _df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 8.0)
        condition_bad_data_pm = (
            _df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 8.0)
        _df.loc[
            condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
        ] = np.nan
        _df.loc[
            condition_bad_data_pm, "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"
        ] = np.nan

        # If there is two different versions of 2015-03-31 data --- remove this
        df = _df.loc[~_df.index.duplicated(keep="first")]

        # Resample to regular time interval
        df = df.resample("D").asfreq()

        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df["soil_moisture_daily"] = df[
            [
                "Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
            ]
        ].mean(axis=1, skipna=True)

        return df["soil_moisture_daily"]


    df_ts = get_soil_moisture(event=event)


    # Plotting settings 
    nonlinear_label = rf'Nonlinear model ($R^2$={event.q_r_squared:.2f}, $q$={q:.1f})'
    #; $\theta^*$={max_sm:.2f}; $\Delta \theta$={event.q_delta_theta:.2f}; $ET_max$={event.q_ETmax:.2f}'
    linear_label = rf'Linear model ($R^2$={event.exp_r_squared:.2f}, $\tau$={tau:.2f})'

    # 
    start_date = pd.to_datetime(event.event_start) #- pd.Timedelta(7, 'D')
    end_date = pd.to_datetime(event.event_end) #+ pd.Timedelta(7, 'D')
    date_range = pd.date_range(start=pd.to_datetime(event.event_start), end=pd.to_datetime(event.event_end), freq="H")


    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4))
    
    ax.scatter(df_ts[start_date:end_date].index, df_ts[start_date:end_date].values, color='k', label='SMAP observation')
    ax.plot(date_range[:-1], y_nonlinear, label=nonlinear_label, color='darkorange')
    ax.plot(date_range[:-1], y_exp, label=linear_label, color='darkblue', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Soil moisture content' + '\n' + rf'$\theta$ $[m3/m3]$')
    # ax.set_title(f"Latitude: {event.latitude:.1f}; Longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand)")

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))


    # plt.tight_layout()
    # ax.autofmt_xdate()
    # Optional: Hide x-ticks for ax1 if they're redundant
    # plt.setp(ax1.get_xticklabels(), visible=False)

    # Adjust the subplots to prevent overlap
    # plt.subplots_adjust(hspace=0.1)  # Adjust the space between plots if necessary

    # fig.tight_layout()
    # fig.autofmt_xdate()


    if save:
        fig.savefig(
            os.path.join(fig_dir, f"event_{event_id}.png"), dpi=1200, bbox_inches="tight"
        )



# %%
# Get grasslands
def print_goodones(df):
    print(df[(df["name"]=="Woody savannas") & (df["q_q"]>1) & (df["latitude"]<0) & (df["q_q"]<4) & (df["event_length"] <15) & (df["sm_range"]>0.1) & ((df["q_r_squared"] - df["exp_r_squared"])>0.05) & (df["q_r_squared"]>0.8)].index[100:120])

print_goodones(df_filt_q_and_exp)


# %%
################################################

event_id = 2170
################################################
plot_drydown(event_id=event_id)

# %%
print(df[(df["event_length"] > 50)].index)
# print(df[(df["q_q"] > 100)& (df["sm_range"]>0.1)& (df["q_r_squared"]>0.7)].index)

# %%
# The number of days with soil moisture record, divided by the length of the event
(df["n_days"]/df["event_length"]).hist()

# %%
df[((df["n_days"]/df["event_length"])>0.33)&((df["n_days"]/df["event_length"])<0.5)&(df['q_r_squared']>0.7)].index
# %%


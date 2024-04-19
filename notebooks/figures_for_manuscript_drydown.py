# %% Import packages
import os
import getpass
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import q_drydown, exponential_drydown
import matplotlib.gridspec as gridspec

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-04-18_conus_fc_as_cutoff"
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

IGBPclass = pd.read_csv(os.path.join(data_dir, anc_dir, IGBPclass_file))
IGBPclass.rename({"name": "landcover_name"}, inplace=True)

df = df.merge(df_anc, on=["EASE_row_index", "EASE_column_index"], how="left")
df = df.merge(df_ai, on=["EASE_row_index", "EASE_column_index"], how="left")
df = pd.merge(df, IGBPclass, left_on="IGBP_landcover", right_on="class", how="left")
print("Loaded ancillary information (land-cover)")

print(f"Total number of drydown event: {len(df)}")

# %% Create output directory
fig_dir = os.path.join(output_dir, dir_name, "figs", "events")
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")
# %% Get some stats
###################################
# Get some stats

# Difference between R2 values of two models
df = df.assign(diff_R2=df["q_r_squared"] - df["exp_r_squared"])

# Denormalize k and calculate the estimated ETmax values from k parameter from q model
df["q_ETmax"] = df["q_k"] * (df["max_sm"] - df["min_sm"]) * z_mm
df["q_k_denormalized"] = df["q_k"] * (df["max_sm"] - df["min_sm"])


def filter_by_data_availability(df):
    # Define a helper function to convert string to list
    def str_to_list(s):
        return list(map(int, s.strip("[]").split()))

    # Convert the 'time' column from string of lists to actual lists
    df["time_list"] = df["time"].apply(str_to_list)

    # Filter condition 1: Check if first three items are [0, 1, 2]
    condition = df["time_list"].apply(lambda x: x[:3] == [0, 1, 2])

    # condition = df['time_list'].apply(
    #     lambda x: len(set(x[:4]).intersection({0, 1, 2, 3})) >= 3
    # )

    # Apply the first filter
    filtered_df = df[condition]

    return filtered_df


print(len(df))
df = filter_by_data_availability(df)
print(len(df))


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

df["event_length"] = (
    pd.to_datetime(df["event_end"]) - pd.to_datetime(df["event_start"])
).dt.days


# %%
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
    _df = pd.read_csv(os.path.join(data_dir, datarod_dir, varname, fn))

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
    _df.loc[condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"] = (
        np.nan
    )
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


# %%
def plot_drydown(df, event_id, ax=None, save=False):

    # Assuming 'df' is your DataFrame and 'event_id' is defined
    event = df.loc[event_id]

    ####################################################
    # Get the event data
    ####################################################
    # Convert the modified string to a NumPy array
    # Replace '\n' with ' ' (space) to ensure all numbers are separated by spaces
    input_string = (
        event.sm.replace("\n", " np.nan").replace(" nan", " np.nan").strip("[]")
    )
    sm = np.array(
        [
            float(value) if value != "np.nan" else np.nan
            for value in input_string.split()
        ]
    )
    values = event.time.strip("[]").split()
    # Calculating n_days
    n_days = (pd.to_datetime(event.event_end) - pd.to_datetime(event.event_start)).days

    # Define variables and parameters
    t = np.arange(0, n_days, 1 / 24)
    k = event.q_k
    q = event.q_q

    min_sm = event.min_sm
    max_sm = event.max_sm
    exp_delta_theta = event.exp_delta_theta
    theta_w = event.exp_theta_w
    tau = event.exp_tau
    z = 50
    delta_theta = event.q_theta_0
    delta_theta_denorm = (event.q_delta_theta ) * (max_sm - min_sm)+ min_sm
    ETmax = k * (max_sm - min_sm) * z
    y_nonlinear = (
        q_drydown(t=t, k=k, q=q, delta_theta=delta_theta) * (max_sm - min_sm) + min_sm
    )
    y_exp = exponential_drydown(t, exp_delta_theta, theta_w, tau)

    # Get soil moisture and precipitation timeseries
    df_ts = get_soil_moisture(event=event)
    df_p = get_precipitation(event=event)

    # Plotting settings
    nonlinear_label = rf"Nonlinear model ($R^2$={event.q_r_squared:.2f}, $q$={q:.1f}, $ETmax$={ETmax:.1f}, $\Delta \theta$={delta_theta_denorm:.2f})"
    linear_label = rf"Linear model ($R^2$={event.exp_r_squared:.2f}, $\tau$={tau:.2f}, $\Delta \theta$={exp_delta_theta:.2f})"

    start_date = pd.to_datetime(event.event_start) - pd.Timedelta(7, "D")
    end_date = pd.to_datetime(event.event_end) + pd.Timedelta(7, "D")
    date_range = pd.date_range(
        start=pd.to_datetime(event.event_start),
        end=pd.to_datetime(event.event_end),
        freq="H",
    )

    ####################################################
    # Create a figure
    ####################################################

    fig = plt.figure(figsize=(10, 3.5))

    # Set up a GridSpec layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot observed & fitted soil moisture
    ax1.scatter(
        df_ts[start_date:end_date].index,
        df_ts[start_date:end_date].values,
        color="grey",
        label="SMAP observation",
    )
    ax1.plot(date_range[:-1], y_nonlinear, label=nonlinear_label, color="darkorange")
    ax1.plot(date_range[:-1], y_exp, label=linear_label, color="darkblue", alpha=0.5)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Soil moisture content" + "\n" + rf"$\theta$ $[m3/m3]$")
    ax1.legend(loc="upper right")
    ax1.set_title(
        f"Latitude: {event.latitude:.1f}; Longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand)"
    )

    ax1.axhline(y=max_sm, color="tab:grey", linestyle="--", alpha=0.5)
    # Plot preciptation
    ax2.bar(
        df_p[start_date:end_date].index,
        df_p[start_date:end_date].values.flatten(),
        color="grey",
    )
    ax2.set_ylabel("Precipitation \n[mm/d]")

    # Formatting
    # Optional: Hide x-ticks for ax1 if they're redundant
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Adjust the subplots to prevent overlap
    plt.subplots_adjust(hspace=0.1)  # Adjust the space between plots if necessary

    fig.tight_layout()
    fig.autofmt_xdate()

    if save:
        fig.savefig(
            os.path.join(fig_dir, f"event_{event_id}.png"),
            dpi=1200,
            bbox_inches="tight",
        )


# %%
# Select the events to plot here
###################################################
# Defining thresholds
q_thresh = 0
success_modelfit_thresh = 0.7
sm_range_thresh = 0.3
###################################################

# CONUS
lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.000000, -66.934570

df_filt = df[
    (df["q_r_squared"] > success_modelfit_thresh)
    & (df["event_length"] > 7)
    & (df["q_q"] > 10)
    & (df["sm_range"] > sm_range_thresh)
    & (df["AI"] > 1)
    # & (df["longitude"] >= lon_min)
    # & (df["longitude"] <= lon_max)
]

print(df_filt.index)
print(f"Try: {df_filt.sample(n=5).index}")

# %%
################################################
event_id = 392295
################################################
plot_drydown(df=df_filt, event_id=event_id)
print(df_filt.loc[event_id])
print(f"Next to try: {df_filt.sample(n=1).index}")
# %%
df.columns

# %%
df["AI"]
# %%
df.loc[event_id]
# %%

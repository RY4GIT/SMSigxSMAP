# %% Import packages
import os
import getpass
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import (
    drydown_piecewise,
    q_model,
    exp_model,
    tau_exp_model,
    loss_model,
    q_model_piecewise,
    exp_model_piecewise,
    tau_exp_dash,
)
import matplotlib.gridspec as gridspec
import json

# %% Plot config

############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-05-13_global_piecewise"  # f"raraki_2024-04-26"
###########################################################################

################ CHANGE HERE FOR PLOT VISUAL CONFIG #########################

## Define model acceptabiltiy criteria
z_mm = 50  # Soil thickness

with open("fig_variable_labels.json", "r") as file:
    var_dict = json.load(file)


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


def data_availability(row):
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


# Assuming df is your DataFrame
print("Checking first 3 days data availability")
# df["first3_avail2"] = df.apply(data_availability, axis=1)
df["first3_avail2"] = df.apply(data_availability, axis=1)
print("done")


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


print("Checking the soil moisture range covered by each event")
# Applying the function to each row and creating a new column 'sm_range'
df["sm_range"] = df.apply(calculate_sm_range, axis=1)
print("done")


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


print("Checking the number data point available in each event")
# Applying the function to each row and creating a new column 'sm_range'
df["n_days"] = df.apply(calculate_n_days, axis=1)
print("done")

print("Checking the length of the drydown event")
df["event_length"] = (
    pd.to_datetime(df["event_end"]) - pd.to_datetime(df["event_start"])
).dt.days
print("done")


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
    # sm = np.array(
    #     [
    #         float(value) if value != "np.nan" else np.nan
    #         for value in input_string.split()
    #     ]
    # )
    # values = event.time.strip("[]").split()
    # Calculating n_days
    n_days = (pd.to_datetime(event.event_end) - pd.to_datetime(event.event_start)).days

    # Define variables and parameters
    t = np.arange(0, n_days, 1 / 24)

    start_date = pd.to_datetime(event.event_start) - pd.Timedelta(3, "D")
    end_date = pd.to_datetime(event.event_end) + pd.Timedelta(7, "D")
    date_range = pd.date_range(
        start=pd.to_datetime(event.event_start),
        end=pd.to_datetime(event.event_end),
        freq="H",
    )

    ####################################################
    # Drydown plot
    ####################################################

    fig = plt.figure(figsize=(15, 3.5))

    # Set up a GridSpec layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ___________________________________________________
    # SOIL MOISTURE
    df_ts = get_soil_moisture(event=event)
    ax1.scatter(
        df_ts[start_date:end_date].index,
        df_ts[start_date:end_date].values,
        color="grey",
        label="SMAP observation",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Soil moisture content" + "\n" + rf"$\theta$ $[m3/m3]$")
    ax1.set_title(
        f"Latitude: {event.latitude:.1f}; Longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand; PET= {event.pet:.1f} mm)"
    )

    # ___________________________________________________
    # PRECIPITATION
    df_p = get_precipitation(event=event)
    ax2.bar(
        df_p[start_date:end_date].index,
        df_p[start_date:end_date].values.flatten(),
        color="grey",
    )
    ax2.set_ylabel("Precipitation \n[mm/d]")

    # ___________________________________________________
    # TAU-EXPONENTIAL
    y_tauexp = tau_exp_model(
        t, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    tauexp_label = rf"$\tau$-based Linear model ($R^2$={event.tauexp_r_squared:.2f}, $\tau$={event.tauexp_tau:.2f}, $\Delta \theta$={event.tauexp_delta_theta:.2f}), $\theta_w$={event.tauexp_theta_w:.2f})"
    ax1.plot(
        date_range[:-1],
        y_tauexp,
        label=tauexp_label,
        color="darkblue",
        alpha=0.5,
        linestyle="--",
    )

    # ___________________________________________________
    # EXPONENTIAL
    y_exp = exp_model_piecewise(
        t=t,
        ETmax=event.exp_ETmax,
        theta_0=event.exp_theta_0,
        theta_star=event.exp_theta_star,
        theta_w=event.exp_theta_w,
    )
    exp_label = rf"Linear model ($R^2$={event.exp_r_squared:.2f}, $ETmax$={event.exp_ETmax:.1f}, $\theta^*$={event.exp_theta_star:.2f}, $\theta_w$={event.exp_theta_w:.2f}, $\theta_0$={event.exp_theta_0:.2f})"
    ax1.plot(date_range[:-1], y_exp, label=exp_label, color="darkblue", alpha=0.5)

    # ___________________________________________________
    # Q MODEL
    y_q = q_model_piecewise(
        t=t,
        q=event.q_q,
        ETmax=event.q_ETmax,
        theta_0=event.q_theta_0,
        theta_star=event.q_theta_star,
        theta_w=event.q_theta_w,
    )
    q_label = rf"Nonlinear model ($R^2$={event.q_r_squared:.2f}, $q$={event.q_q:.1f}, $ETmax$={event.q_ETmax:.1f}, $\theta^*$={event.q_theta_star:.2f}, $\theta_w$={event.q_theta_w:.2f}, $\theta_0$={event.q_theta_0:.2f})"
    ax1.plot(date_range[:-1], y_q, label=q_label, color="darkorange")

    # ___________________________________________________
    # Estimated theta_fc
    ax1.axhline(
        y=event.est_theta_fc,
        color="tab:grey",
        linestyle="--",
        alpha=0.5,
        label=r"Estimated $\theta_{fc}$",
    )

    # ___________________________________________________
    # Formatting
    # Optional: Hide x-ticks for ax1 if they're redundant
    ax1.legend(loc="upper right")
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

    # ####################################################
    # # Loss function
    # ####################################################

    fig2, ax3 = plt.subplots(figsize=(3.5, 3.5))

    if np.isnan(event.est_theta_fc):
        est_fc = event.max_sm * 0.95
    else:
        est_fc = event.est_theta_fc
    nonlinear_theta_plot = np.arange(event.q_theta_w, est_fc, 0.001)
    linear_theta_plot = np.arange(event.exp_theta_w, est_fc, 0.001)
    theta_obs = df_ts[
        pd.to_datetime(event.event_start) : pd.to_datetime(event.event_end)
    ].values
    t_obs = np.where(~np.isnan(theta_obs))[0]

    # ___________________________________________________
    # LINEAR (EXPONENTIAL) MODEL
    x_tau_in_L = tau_exp_model(
        t_obs, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    y_tau_in_L = tau_exp_dash(
        t_obs, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    ax3.scatter(
        x_tau_in_L,
        y_tau_in_L,
        color="blue",
        facecolors="none",
        alpha=0.5,
    )

    # Fit the linear regression model
    slope, intercept = np.polyfit(x_tau_in_L, y_tau_in_L, 1)
    tauexp_theta_plot = np.arange(event.tauexp_theta_w, est_fc, 0.001)
    y_vals = slope * tauexp_theta_plot + intercept
    ax3.plot(
        tauexp_theta_plot,
        y_vals,
        color="darkblue",
        linestyle="--",
        alpha=0.9,
        label=r"$\tau$-based Linear model",
    )

    # ___________________________________________________
    # LINEAR MODEL
    # Plot observed & fitted soil moisture
    ax3.plot(
        linear_theta_plot,
        loss_model(
            linear_theta_plot,
            1,
            event.exp_ETmax,
            theta_w=event.exp_theta_w,
            theta_star=event.exp_theta_star,
        ),
        color="darkblue",
        alpha=0.9,
        label="Linear model",
    )

    # Plot observed & fitted soil moisture
    linear_est_theta_obs = exp_model_piecewise(
        t=t_obs,
        ETmax=event.exp_ETmax,
        theta_0=event.exp_theta_0,
        theta_star=event.exp_theta_star,
        theta_w=event.exp_theta_w,
    )
    ax3.scatter(
        linear_est_theta_obs,
        loss_model(
            linear_est_theta_obs,
            1,
            event.exp_ETmax,
            theta_w=event.exp_theta_w,
            theta_star=event.exp_theta_star,
        ),
        color="darkblue",
        alpha=0.5,
        # label=r"Observed $\theta$" + "\n" + r"($d\theta/dt$ is estimated)",
    )

    # ___________________________________________________
    # NONLINEAR (Q) MODEL
    # Plot observed & fitted soil moisture
    ax3.plot(
        nonlinear_theta_plot,
        loss_model(
            nonlinear_theta_plot,
            event.q_q,
            event.q_ETmax,
            theta_w=event.q_theta_w,
            theta_star=event.q_theta_star,
        ),
        color="darkorange",
        alpha=0.9,
        label="Nonlinear model",
    )

    nonlinear_est_theta_obs = q_model_piecewise(
        t=t_obs,
        q=event.q_q,
        ETmax=event.q_ETmax,
        theta_0=event.q_theta_0,
        theta_star=event.q_theta_star,
        theta_w=event.q_theta_w,
    )
    ax3.scatter(
        nonlinear_est_theta_obs,
        loss_model(
            nonlinear_est_theta_obs,
            event.q_q,
            event.q_ETmax,
            theta_w=event.q_theta_w,
            theta_star=event.q_theta_star,
        ),
        color="darkorange",
        alpha=0.5,
    )

    # ___________________________________________________
    # FORMATTING
    ax3.set_xlabel(r"$\theta$ [$m^3$ $m^{-3}$]")
    ax3.set_ylabel(r"$d\theta/dt$ [$m^3$ $m^{-3}$ $day^{-1}$]")
    ax3.legend(loc="upper left")
    title_value = check_1ts_range(df.loc[event_id], verbose=True)
    ax3.set_title(
        f"1st timestep drydown covers {title_value*100:.0f}% of the range"
        + "\n"
        + f"sm range covers {event.sm_range*100:.0f}% of the historical"
    )
    ax3.invert_yaxis()


# %%
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
        print(f"{(dsdt_0 - dsdt_1) / (row.q_ETmax / 50)*100:.1f} percent")
    return (dsdt_0 - dsdt_1) / (row["q_ETmax"] / 50) * (-1)


print("Checking the potential first step drydown")
df["large_q_criteria"] = df.apply(check_1ts_range, axis=1)
print("done")
# %%
# Select the events to plot here

# # CONUS
# lat_min, lat_max = 24.396308, 49.384358
# lon_min, lon_max = -125.000000, -66.934570

df_filt = df[
    (df["q_r_squared"] > 0.8)
    & (df["diff_R2"] > 0)
    & (df["sm_range"] > 0.15)
    & (df["large_q_criteria"] < 0.6)
    & (df["first3_avail2"])
    & (df["q_q"] > 1.0e-04)
    & (df["q_q"] > 1.4)
]
# df_filt = df[(df["q_r_squared"] < 0.8) & (df["q_r_squared"] > 0.7)]
print(df_filt.index)
print(f"Try: {df_filt.sample(n=5).index}")

# Get the indices that are NOT in df_filt
not_in_filt_indices = df[~df.index.isin(df_filt.index)].index

# Display the indices
print(not_in_filt_indices)

# Ensure there are enough indices to sample from
if len(not_in_filt_indices) >= 5:
    print(f"Try: {not_in_filt_indices.to_series().sample(n=5).index}")
else:
    print("Not enough indices to sample 5.")

# %%
################################################
event_id = 297033
################################################
plot_drydown(df=df, event_id=event_id)
# print(df.loc[event_id])
print(f"Next to try (in df): {df_filt.sample(n=1).index}")
print(f"Next to try (not in df): {not_in_filt_indices.to_series().sample(n=1).index}")
# check_1ts_range(df.loc[event_id], verbose=True)
# %%
plt.scatter(df["event_length"], df["q_q"])
plt.scatter(df_filt["event_length"], df_filt["q_q"])

# %%
plt.scatter(df["large_q_criteria"], df["q_q"])
plt.scatter(df_filt["large_q_criteria"], df_filt["q_q"])
# %%
# %%
df_filt["q_r_squared"].hist()
plt.xlim([0, 1])
# %%
plt.scatter(df_filt["q_q"], df_filt["q_ETmax"])
# %%

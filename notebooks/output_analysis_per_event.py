# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ast

from functions import q_drydown, exponential_drydown, loss_model

# %%
################################# CHANGE HERE ######################################
dir_name = f"raraki_2023-11-25_global_95asmax"
####################################################################################
# %%  Read data
input_file = (
    rf"/home/raraki/waves/projects/smap-drydown/output/{dir_name}/all_results.csv"
)
_df = pd.read_csv(input_file)
coord_info_file = (
    "/home/raraki/waves/projects/smap-drydown/data/datarods/coord_info.csv"
)
coord_info = pd.read_csv(coord_info_file)
df = _df.merge(coord_info, on=["EASE_row_index", "EASE_column_index"], how="left")
print(f"Number of events: {len(df)}")
df.head()


# %%
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
################## FILTER TIMESERIES HERE TO CHECK WHICH EVENT YOU WANT TO PLOT ##################
# print(df[df["q_r_squared"]>df["exp_r_squared"]].index)
# print(df[(df["q_r_squared"]>0.7) & (df["q_k"]>10) & (df["sm_range"]>0.5)].index)
print(df[(df["q_q"] < 0.1) & (df["sm_range"] > 0.3) & (df["q_r_squared"] > 0.7)].index)
filtered_indices = df[
    (df["q_q"] < 0.1) & (df["sm_range"] > 0.3) & (df["q_r_squared"] > 0.7)
]
file_path = "filtered_indices.txt"
with open(file_path, "w") as file:
    for index in filtered_indices:
        file.write(str(index) + "\n")
##################################################################################################
# %%
################## SPECIFY THE EVENT ID OF INTEREST HERE ##################
event_id = 82
###########################################################################

# %% Get the event data

event = df.loc[event_id]


def preprocess_string(input_string):
    # Replace '\n' and 'nan' with ' np.nan' and remove square brackets
    return input_string.replace("\n", " np.nan").replace(" nan", " np.nan").strip("[]")


def string_to_float_array(input_string):
    # Convert the preprocessed string to a NumPy array of floats
    return np.array(
        [
            float(value) if value != "np.nan" else np.nan
            for value in input_string.split()
        ]
    )


def string_to_int_array(input_string):
    # Convert the preprocessed string to a NumPy array of integers
    return np.array([int(value) for value in input_string.split()])


def string_to_numpy_array(string):
    try:
        # Safely evaluate the string as a list and convert to a NumPy array
        return np.array(ast.literal_eval(string))
    except (SyntaxError, ValueError):
        return np.nan  # Return NaN if the string cannot be converted


# Get the event data
event = df.loc[event_id]


# %%

var_dict = {
    "theta": {
        "column_name": "",
        "symbol": r"$\theta$",
        "label": r"Soil moisture $\theta$",
        "unit": r"$[m^3/m^3]$",
        "lim": [0, 0.50],
    },
    "dtheta": {
        "column_name": "",
        "symbol": r"$-d\theta/dt$",
        "label": r"$-d\theta/dt$",
        "unit": r"$[m^3/m^3/day]$",
        "lim": [-0.10, 0],
    },
    "theta_norm": {
        "column_name": "",
        "symbol": r"$\theta_{norm}$",
        "label": r"Normalized soil moisture $\theta_{norm}$",
        "unit": r"$[-]$",
        "lim": [0, 1.0],
    },
    "dtheta_norm": {
        "column_name": "",
        "symbol": r"$-d\theta/dt$",
        "label": r"$-d\theta_{norm}/dt$",
        "unit": r"$[-/day]$",
        "lim": [-0.15, 0],
    },
    "t": {
        "column_name": "",
        "symbol": r"$t$",
        "label": r"Timestep",
        "unit": r"$[day]$",
    },
}


def plot_event(event, normalize=False):
    # _________________________________________________________________________
    # Convert strings to NumPy arrays
    sm = string_to_float_array(preprocess_string(event.sm))
    t_d = string_to_int_array(preprocess_string(event.time))
    n_days = t_d[-1] - t_d[0] + 1

    # Prepare variables and parameters
    t = np.arange(0, n_days + 1, 0.1)
    min_sm, max_sm = event.min_sm, event.max_sm
    if normalize:
        theta = np.arange(0, 1, 0.01)
    else:
        theta = np.arange(min_sm, max_sm, 0.01)
    plot_sm = (sm - min_sm) / (max_sm - min_sm) if normalize else sm
    _exp_y_opt = np.array(ast.literal_eval(event.exp_y_opt))
    _q_y_opt = np.array(ast.literal_eval(event.q_y_opt))
    exp_y_opt = (_exp_y_opt - min_sm) / (max_sm - min_sm) if normalize else _exp_y_opt
    q_y_opt = (_q_y_opt - min_sm) / (max_sm - min_sm) if normalize else _q_y_opt

    k = event.q_k if normalize else event.q_k * (max_sm - min_sm)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))

    if normalize:
        theta_vardict = var_dict["theta_norm"]
        dtheta_vardict = var_dict["dtheta_norm"]
    else:
        theta_vardict = var_dict["theta"]
        dtheta_vardict = var_dict["dtheta"]
    # _________________________________________________________________________
    # Drawdown plot
    q_delta_theta = (
        event.q_delta_theta
        if normalize
        else event.q_delta_theta * (max_sm - min_sm) + min_sm
    )
    ax1.plot(t_d, plot_sm[~np.isnan(plot_sm)], ".", color="gray", label="Observed")
    ax1.plot(
        t,
        q_drydown(t, k, event.q_q, q_delta_theta),
        label=f"q model R2={event.q_r_squared:.2f}",
        color="darkorange",
    )
    _exp_y_opt2 = exponential_drydown(
        t, event.exp_delta_theta, event.exp_theta_w, event.exp_tau
    )
    exp_y_opt2 = (
        (_exp_y_opt2 - min_sm) / (max_sm - min_sm) if normalize else _exp_y_opt2
    )
    ax1.plot(
        t,
        exp_y_opt2,
        label=f"exp model R2={event.exp_r_squared:.2f}",
        color="darkblue",
        alpha=0.5,
    )
    ax1.set(
        xlabel=f'{var_dict["t"]["label"]} {var_dict["t"]["unit"]}',
        ylabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
        xlim=(t_d[0], t_d[-1] + 1),
        ylim=(0, 1),
    )
    ax1.set_title("Drydown event")
    ax1.set_ylim(theta[0], theta[-1])
    ax1.legend()

    # _________________________________________________________________________
    # Loss function plot
    if normalize:
        theta_star = 1
        theta_wp = 0
    else:
        theta_star = max_sm
        theta_wp = min_sm
    ax2.plot(
        theta,
        loss_model(theta, event.q_q, k, theta_wp, theta_star),
        label=f"q model (q={event.q_q:.4f})",
        color="pink",
    )
    ax2.plot(
        q_y_opt,
        loss_model(q_y_opt, event.q_q, k, theta_wp, theta_star),
        ".",
        label="Points fitted",
        color="darkorange",
    )
    ax2.plot(
        theta,
        loss_model(theta, 1, k, theta_wp, theta_star),
        label=f"exp model (tau={event.exp_tau:.2f})",
        color="skyblue",
    )
    ax2.plot(
        exp_y_opt,
        loss_model(exp_y_opt, 1, k, theta_wp, theta_star),
        ".",
        label="Points fitted",
        color="darkblue",
    )
    ax2.set(
        xlabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
        ylabel=f'{dtheta_vardict["label"]} {dtheta_vardict["unit"]}',
    )
    ax2.invert_yaxis()
    ax2.legend()
    ax2.set_title("Loss function")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'lat={event["latitude"]:.1f}, lon={event["longitude"]:.1f}')
    plt.show()


# %%
plot_event(event, normalize=True)
plot_event(event, normalize=False)


# %% compare sm range b/w q < 0.1 and not
small_q = df[
    (df["q_q"] < 0.1) & (df["sm_range"] > 0.3) & (df["q_r_squared"] > 0.7)
].copy()

large_q = df[
    (df["q_q"] > 0.1) & (df["sm_range"] > 0.3) & (df["q_r_squared"] > 0.7)
].copy()
# %%
small_q_diff = small_q["max_sm"] - small_q["min_sm"]
large_q_diff = large_q["max_sm"] - large_q["min_sm"]

# Plotting both histograms on the same plot without fill
plt.hist(
    small_q_diff,
    bins=np.arange(0, 0.5, 0.01),
    alpha=0.7,
    edgecolor="blue",
    linewidth=1.5,
    fill=False,
    label="Small Q",
)
plt.hist(
    large_q_diff,
    bins=np.arange(0, 0.5, 0.01),
    alpha=0.7,
    edgecolor="green",
    linewidth=1.5,
    fill=False,
    label="Large Q",
)

# Adding labels and legend
plt.xlabel(f"Soil moisture range (max - min values) $[m3/m3]$")
plt.ylabel("Frequency")
plt.title("Histogram of Differences for Small q and Large q")
plt.legend()
# %%
# %%
# _________________________________________________________________________
# Get the Timeseries of soil moisture data

data_dir = r"/home/waves/projects/smap-drydown/data/datarods"
datarods_dir r"datarods"


def get_sm(EASE_row_index, EASE_column_index, start_date, end_date):
    
    varname = r"SPL3SMP"
    filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
    _df = pd.read_csv(os.path.join(data_dir, datarods_dir, varname, filename))

    # Set time index and crop
    _df = set_time_index(_df, index_name="time")
    _df = _df[start_date : end_date].copy()

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

# _________________________________________________________________________
# Plot within timeseries
sm_ts = get_sm(event.EASE_row_index, event.EASE_column_index, sevent.start_date, event.end_date)
sm_ts.plot()
# %%

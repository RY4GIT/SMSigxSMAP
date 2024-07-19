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
import matplotlib as mpl

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
output_dir = rf"/home/{user_name}/waves/projects/smap-drydown/output"
datarods_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"
coord_info_file = "coord_info.csv"
results_file = rf"all_results_processed.csv"

# %%
df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
print("Loaded results file")
coord_info = pd.read_csv(os.path.join(data_dir, datarods_dir, coord_info_file))

# %% Create output directory
fig_dir = os.path.join(output_dir, dir_name, "figs", "events")
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")

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

base_fontsize = 15
plt.rcParams["font.family"] = "DejaVu Sans"  # Or any other available font
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Ensure the font is set correctly
# mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams["font.sans-serif"] = "Myriad Pro"
mpl.rcParams["font.size"] = base_fontsize  # 12.0
mpl.rcParams["axes.titlesize"] = 12.0
plt.rcParams["mathtext.fontset"] = (
    "stixsans"  #'stix'  # Or 'cm' (Computer Modern), 'stixsans', etc.
)

base_linewidth = 2
markersize = 30


def plot_drydown(
    df, event_id, days_after_to_plot=5, ax=None, plot_precip=True, save=False
):

    linear_color = "tab:grey"  # "#377eb8"

    # Assuming 'df' is your DataFrame and 'event_id' is defined
    event = df.loc[event_id]
    if event.q_q < 1:
        nonlinear_color = "#F7CA0D"
    else:
        nonlinear_color = "#62AD5F"

    ####################################################
    # Get the event data
    ####################################################
    # Convert the modified string to a NumPy array
    # Replace '\n' with ' ' (space) to ensure all numbers are separated by spaces
    input_string = (
        event.sm.replace("\n", " np.nan").replace(" nan", " np.nan").strip("[]")
    )

    n_days = (pd.to_datetime(event.event_end) - pd.to_datetime(event.event_start)).days

    # Define variables and parameters
    t = np.arange(0, n_days, 1 / 24)

    start_date = pd.to_datetime(event.event_start) - pd.Timedelta(1, "D")
    end_date = pd.to_datetime(event.event_end) + pd.Timedelta(days_after_to_plot, "D")
    date_range = pd.date_range(
        start=pd.to_datetime(event.event_start),
        end=pd.to_datetime(event.event_end),
        freq="H",
    )

    ####################################################
    # Drydown plot
    ####################################################

    # Create the subplots
    if plot_precip:
        fig = plt.figure(figsize=(15, 4))

        # Set up a GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        date_factor = (end_date - start_date).days
        plot_height = 3.75
        first_plot_width = 10  # date_factor * 0.4
        second_plot_width = 3.5
        fig = plt.figure(figsize=(first_plot_width + second_plot_width, plot_height))

        gs = gridspec.GridSpec(1, 2, width_ratios=[first_plot_width, second_plot_width])

        ax1 = fig.add_subplot(gs[0, 0])  # First subplot
        ax3 = fig.add_subplot(gs[0, 1])  # Second subplot

    # ___________________________________________________
    # PRECIPITATION
    if plot_precip:
        df_p = get_precipitation(event=event)
        ax2.bar(
            df_p[start_date:end_date].index,
            df_p[start_date:end_date].values.flatten(),
            color="grey",
        )
        ax2.set_ylabel("Precipitation \n[mm/d]")

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
    q_label = (
        rf"Nonlinear model, $R^2$={event.q_r_squared:.3f} ("
        + r"$\hat{q}$"
        + f"={event.q_q:.1f}, "
        + "\n"
        + r"($\hat{\mathrm{ET}_{\mathrm{max}}}$"
        + f"={event.q_ETmax:.1f}, "
        + r"$\hat{\theta_*}$"
        + f"={event.q_theta_star:.2f}, "
        + r"$\hat{\theta_{\mathrm{wp}}}$"
        + f"={event.q_theta_w:.2f}, "
        + r"$\hat{\theta_{\mathrm{0}}}$"
        + f"={event.q_theta_0:.2f})"
    )
    ax1.plot(
        date_range[:-1],
        y_q,
        label=q_label,
        color=nonlinear_color,
        linewidth=base_linewidth * 2,
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
    exp_label = (
        rf"Linear model, $R^2$={event.exp_r_squared:.3f}"
        + "\n"
        + r"($\hat{\mathrm{ET}_{\mathrm{max}}}$"
        + f"={event.exp_ETmax:.1f}, "
        + r"$\hat{\theta_*}$"
        + f"={event.exp_theta_star:.2f}, "
        + r"$\hat{\theta_{\mathrm{wp}}}$"
        + f"={event.exp_theta_w:.2f}, "
        + r"$\hat{\theta_{\mathrm{0}}}$"
        + f"={event.exp_theta_0:.2f})"
    )
    ax1.plot(
        date_range[:-1],
        y_exp,
        label=exp_label,
        color=linear_color,
        linewidth=base_linewidth,
        alpha=0.8,
    )

    # ___________________________________________________
    # TAU-EXPONENTIAL
    y_tauexp = tau_exp_model(
        t, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    tauexp_label = (
        rf"$\tau$-based Linear model, $R^2$={event.tauexp_r_squared:.3f}"
        + "\n("
        + r"$\hat{\tau}$"
        + f"={event.tauexp_tau:.2f}, "
        + r"$\hat{\Delta \theta}$"
        + f"={event.tauexp_delta_theta:.2f}, "
        + r"$\hat{\theta_{\mathrm{wp}}}$"
        + f"={event.tauexp_theta_w:.2f})"
    )
    ax1.plot(
        date_range[:-1],
        y_tauexp,
        label=tauexp_label,
        color=linear_color,
        linewidth=base_linewidth,
        alpha=0.7,
        linestyle="--",
    )

    # ___________________________________________________
    # Estimated theta_fc
    ax1.axhline(
        y=event.est_theta_fc,
        color="tab:grey",
        linestyle=":",
        linewidth=base_linewidth,
        label=r"Estimated $\hat{\theta_{\mathrm{fc}}}$",
    )

    # ___________________________________________________
    # SOIL MOISTURE
    df_ts = get_soil_moisture(event=event)
    ax1.scatter(
        df_ts[start_date:end_date].index,
        df_ts[start_date:end_date].values,
        color="grey",
        label="SMAP observation",
        s=markersize,
    )
    ax1.set_xlabel(f"Date in {start_date.year}")
    ax1.set_ylabel(
        "Soil moisture content" + "\n" + r"$\theta$ ($\mathrm{m}^3$ $\mathrm{m}^{-3}$)"
    )
    ax1.set_title(
        f"latitude: {event.latitude:.1f}, longitude: {event.longitude:.1f} ({event['name']}; aridity index {event.AI:.1f}; {event.sand_fraction*100:.0f}% sand)",  # PET= {event.pet:.1f} mm)
        fontsize=base_fontsize,
    )

    # ___________________________________________________
    # Formatting
    # Optional: Hide x-ticks for ax1 if they're redundant
    legend_fontsize = 10.5
    ax1.legend(loc="upper right", fontsize=legend_fontsize)
    # Optional: Hide x-ticks for ax1 if they're redundant
    if plot_precip:
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        plt.setp(ax1.get_xticklabels(), visible=True)

    fig.tight_layout()
    import matplotlib.dates as mdates

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    # fig.autofmt_xdate()

    # ####################################################
    # # Loss function
    # ####################################################

    if plot_precip:
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
        color=nonlinear_color,
        linewidth=base_linewidth * 2,
        label="Loss model:\nNonlinear",
    )

    # nonlinear_est_theta_obs = q_model_piecewise(
    #     t=t_obs,
    #     q=event.q_q,
    #     ETmax=event.q_ETmax,
    #     theta_0=event.q_theta_0,
    #     theta_star=event.q_theta_star,
    #     theta_w=event.q_theta_w,
    # )
    # ax3.scatter(
    #     nonlinear_est_theta_obs,
    #     loss_model(
    #         nonlinear_est_theta_obs,
    #         event.q_q,
    #         event.q_ETmax,
    #         theta_w=event.q_theta_w,
    #         theta_star=event.q_theta_star,
    #     ),
    #     color=nonlinear_color,
    #     alpha=0.5,
    # )

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
        color=linear_color,
        linewidth=base_linewidth,
        label="Linear",
        alpha=0.8,
    )

    # # Plot observed & fitted soil moisture
    # linear_est_theta_obs = exp_model_piecewise(
    #     t=t_obs,
    #     ETmax=event.exp_ETmax,
    #     theta_0=event.exp_theta_0,
    #     theta_star=event.exp_theta_star,
    #     theta_w=event.exp_theta_w,
    # )
    # ax3.scatter(
    #     linear_est_theta_obs,
    #     loss_model(
    #         linear_est_theta_obs,
    #         1,
    #         event.exp_ETmax,
    #         theta_w=event.exp_theta_w,
    #         theta_star=event.exp_theta_star,
    #     ),
    #     color=linear_color,
    #     alpha=0.5,
    #     # label=r"Observed $\theta$" + "\n" + r"($d\theta/dt$ is estimated)",
    # )

    # ___________________________________________________
    # LINEAR (EXPONENTIAL) MODEL
    x_tau_in_L = tau_exp_model(
        t_obs, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    y_tau_in_L = tau_exp_dash(
        t_obs, event.tauexp_delta_theta, event.tauexp_theta_w, event.tauexp_tau
    )
    # ax3.scatter(
    #     x_tau_in_L,
    #     y_tau_in_L,
    #     color=linear_color,
    #     facecolors="none",
    #     alpha=0.5,
    # )

    # Fit the linear regression model
    slope, intercept = np.polyfit(x_tau_in_L, y_tau_in_L, 1)
    tauexp_theta_plot = np.arange(event.tauexp_theta_w, est_fc, 0.001)
    y_vals = slope * tauexp_theta_plot + intercept
    ax3.plot(
        tauexp_theta_plot,
        y_vals,
        color=linear_color,
        linestyle="--",
        linewidth=base_linewidth,
        alpha=0.7,
        label=r"$\tau$-based" + "\nLinear",
    )

    # ___________________________________________________
    # FORMATTING
    ax3.set_xlabel(r"$\theta$ ($\mathrm{m}^3$ $\mathrm{m}^{-3}$)")
    ax3.set_ylabel(
        r"$d\theta/dt$ ($\mathrm{m}^3$ $\mathrm{m}^{-3}$ $\mathrm{day}^{-1}$)"
    )
    ax3.legend(loc="upper left", fontsize=legend_fontsize)
    # title_value = check_1ts_range(df.loc[event_id], verbose=True)
    # ax3.set_title("(b)", loc="left")
    # print(
    #     f"1st timestep drydown covers {title_value*100:.0f}% of the range"
    #     + "\n"
    #     + f"sm range covers {event.sm_range*100:.0f}% of the historical"
    # )
    print(start_date)
    ax3.invert_yaxis()

    fig.tight_layout()
    if save:
        fig.savefig(
            os.path.join(fig_dir, f"event_{event_id}.png"),
            dpi=1200,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(fig_dir, f"event_{event_id}.pdf"),
            dpi=1200,
            bbox_inches="tight",
        )


save = True
plot_drydown(
    df=df, event_id=155510, plot_precip=False, save=save, days_after_to_plot=14
)
plot_drydown(df=df, event_id=548528, plot_precip=False, save=save, days_after_to_plot=9)
plot_drydown(df=df, event_id=665086, plot_precip=False, save=save, days_after_to_plot=7)
plot_drydown(df=df, event_id=135492, plot_precip=False, save=save, days_after_to_plot=6)
# %%
plot_drydown(df=df, event_id=683982, plot_precip=False, save=save, days_after_to_plot=6)

# %%
##########################################################
# Select the events to plot here

# # CONUS
# lat_min, lat_max = 24.396308, 49.384358
# lon_min, lon_max = -125.000000, -66.934570

df_filt = df[
    (df["q_r_squared"] > 0.80)
    & (df["sm_range"] > 0.20)
    & (df["large_q_criteria"] < 0.8)
    # & (df["first3_avail2"])
    & (df["q_q"] > 1.0e-04)
    & (df["q_q"] <= 0.8)
    & (df["q_r_squared"] > df["tauexp_r_squared"])
    & (df["event_length"] >= 7)
    & (df["name"] == "Grasslands")
    # & (df["q_q"] > 1.7)
    # & (df["q_q"] < 2.0)
]

print(f"Try (in df): {df_filt.sample(n=5).index}")

# %%
# Get the indices that are NOT in df_filt

# Display the indices
not_in_filt_indices = df[~df.index.isin(df_filt.index)].index
print(not_in_filt_indices)

# Ensure there are enough indices to sample from
if len(not_in_filt_indices) >= 5:
    print(f"Try (not in df): {not_in_filt_indices.to_series().sample(n=5).index}")
else:
    print("Not enough indices to sample 5.")

# %%|
################################################
event_id = 190323
################################################

# q < 1: 743974, 155510, 278218
# q > 1: 799057, 683982, 648455, 547425, 271697
save = True
plot_drydown(
    df=df, event_id=event_id, plot_precip=False, save=save, days_after_to_plot=13
)

# print(df.loc[event_id])
print(f"Next to try (in df): {df_filt.sample(n=1).index}")
print(f"Next to try (not in df): {not_in_filt_indices.to_series().sample(n=1).index}")
# %%
# # check_1ts_range(df.loc[event_id], verbose=True)
# # %%
# plt.scatter(df["event_length"], df["q_q"])
# plt.scatter(df_filt["event_length"], df_filt["q_q"])

# # %%
# plt.scatter(df["large_q_criteria"], df["q_q"])
# plt.scatter(df_filt["large_q_criteria"], df_filt["q_q"])
# # %%
# # %%
# df_filt["q_r_squared"].hist()
# plt.xlim([0, 1])
# # %%
# plt.scatter(df_filt["q_q"], df_filt["q_ETmax"])
# # %%
# ax = df["q_q"].hist(vmin=0, vmax=4)
# ax.set_xlim([0, 3])
# %%

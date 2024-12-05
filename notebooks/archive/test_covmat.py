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
datarod_dir = "datarods"
anc_dir = "SMAP_L1_L3_ANC_STATIC"
anc_file = "anc_info.csv"
IGBPclass_file = "IGBP_class.csv"
ai_file = "AridityIndex_from_datarods.csv"
coord_info_file = "coord_info.csv"
results_file = rf"all_results_processed.csv"

# %%
df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
print("Loaded results file")
coord_info = pd.read_csv(os.path.join(data_dir, datarod_dir, coord_info_file))

# # %% Create output directory
# fig_dir = os.path.join(output_dir, dir_name, "figs", "events_temp")
# if not os.path.exists(fig_dir):
#     os.makedirs(fig_dir)
#     print(f"Created dir: {fig_dir}")
# else:
#     print(f"Already exists: {fig_dir}")

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
event_id=548528
event = df.loc[event_id]
df_ts = get_soil_moisture(event=event)
start_date = pd.to_datetime(event.event_start)# - pd.Timedelta(1, "D")
end_date = pd.to_datetime(event.event_end)# + pd.Timedelta(days_after_to_plot, "D")

# %%
event
# %%
_x_obs = event.time #df_ts[start_date:end_date].index
x_obs = np.array([int(num) for num in _x_obs.strip('[]').split()])
_y_obs = event.sm # df_ts[start_date:end_date].values
y_obs = np.array([float(num) for num in _y_obs.strip('[]').split()])
y_obs


# %%
from scipy.optimize import curve_fit

### q ###
min_q = 0  # -np.inf
max_q = np.inf
ini_q = 1.0 + 1.0e-03

### ETmax ###
max_ETmax = 3
min_ETmax = 0
ini_ETmax = max_ETmax * 0.5

### theta_0 ###
first_non_nan = y_obs[0]
min_theta_0 = y_obs[0] - 0.04
max_theta_0 = y_obs[0] + 0.04
ini_theta_0 = y_obs[0] - 0.04

### theta_star ###
max_theta_star = 0.6
min_theta_star = y_obs[~np.isnan(y_obs)][1]
ini_theta_star = (max_theta_star + min_theta_star) / 2

bounds = [
    (min_q, min_ETmax, min_theta_0, min_theta_star),
    (max_q, max_ETmax, max_theta_0, max_theta_star),
]
p0 = [ini_q, ini_ETmax, ini_theta_0, ini_theta_star]
z = 50
theta_w = 0.02
model=lambda t, q, ETmax, theta_0, theta_star: q_model_piecewise(
                    t=t,
                    q=q,
                    ETmax=ETmax,
                    theta_0=theta_0,
                    theta_star=theta_star,
                    theta_w=theta_w,
                    z=z,
                )

popt, pcov = curve_fit(
                f=model, xdata=x_obs, ydata=y_obs, p0=p0, bounds=bounds
            )

# %%
def print_popt(popt, param_names):
    """
    Prints and plots each parameter in the popt array with labels.
    
    Parameters:
        popt (array-like): Optimized parameters returned from curve_fit.
        param_names (list): List of parameter names corresponding to popt.
    """
    if len(popt) != len(param_names):
        raise ValueError("Length of `popt` and `param_names` must match.")
    
    # Print each parameter
    print("Optimized Parameters:")
    for name, value in zip(param_names, popt):
        print(f"{name}: {value:.4f}")
param_names = ["q", "ETmax", "theta_0", "theta_star"]
print(popt, param_names)


def plot_pcov(pcov, param_names):
    """
    Prints and visualizes the covariance matrix returned from curve_fit.
    
    Parameters:
        pcov (array-like): Covariance matrix returned from curve_fit.
        param_names (list): List of parameter names corresponding to the rows/columns of pcov.
    """
    if pcov.shape[0] != pcov.shape[1] or pcov.shape[0] != len(param_names):
        raise ValueError("Covariance matrix dimensions and parameter names must match.")
    
    # Print the covariance matrix
    print("Covariance Matrix:")
    for i, row in enumerate(pcov):
        row_str = "  ".join(f"{val:.4e}" for val in row)
        print(f"{param_names[i]:<10}: {row_str}")

    print("Covariance Matrix Elements:")
    for i in range(len(param_names)):
        for j in range(i, len(param_names)):  # Only iterate over upper triangular (i <= j)
            print(f"Cov({param_names[i]}, {param_names[j]}): {pcov[i, j]:.4e}")
    
plot_pcov(pcov, param_names)
# %%8733 2.06836362 0.30474529 0.34462084] ['q', 'ETmax', 'theta_0', 'theta_star'
y_pred = model(x_obs, *popt)

# Residual sum of squares
ss_res = np.sum((y_obs - y_pred) ** 2)

# Total sum of squares
ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)

# R^2 calculation
r_squared = 1 - (ss_res / ss_tot)

# Number of observations and parameters
n = len(y_obs)
k = len(popt)

# AIC and BIC
aic = n * np.log(ss_res / n) + 2 * k
bic = n * np.log(ss_res / n) + k * np.log(n)

# t-scores
# Find the index of parameter "q"
q_index = param_names.index("q")

# Extract estimate and standard error for "q"
q_estimate = popt[q_index]
q_variance = pcov[q_index, q_index]
q_se = np.sqrt(q_variance)

# Compute the t-statistic: t_stat = param - 1 / param_SE
t_stat = (q_estimate - 1) / q_se

# Degrees of freedom
n = len(y_obs)
k = len(popt)
dof = n - k

# Compute the p-value
from scipy.stats import t
p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
p_value_v2 = 2 * t.sf(np.abs(t_stat), dof)

# parameter_errors = np.sqrt(np.diag(pcov))
# t_scores = popt / parameter_errors

# Print results
print(f"R^2: {r_squared:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
print(f"Test for H0: q = 1")
print(f"q estimate: {q_estimate:.4f}")
print(f"Standard error: {q_se:.4f}")
# print(f"t-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4e}")
print(f"p-value v2: {p_value_v2:.4e}")
# %%
plt.plot(x_obs, y_obs)
plt.plot(x_obs, y_pred)
# %%
print(y_pred)
# %%

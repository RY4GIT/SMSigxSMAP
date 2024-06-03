# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ast

from functions import (
    q_model,
    q_model_piecewise,
    exp_model,
    exp_model_piecewise,
    tau_exp_model,
    loss_model,
)


# %%

# Define parameters
tau1 = 6
tau2 = 8
tau3 = 10

theta_w1 = 0.0
theta_w2 = 0.1
theta_w3 = 0.2

theta_star = 0.6
fc_minus_star = 0.05
theta_fc = theta_star + fc_minus_star
theta_0 = theta_fc


c3 = "#108883"  # "#108883"  # f"#2c7fb8"
c2 = "#2EBB9D"  # "#2EBB9D"  # f"#41b6c4"
c1 = "#269A81"  # "#F7CA0D"  # f"#a1dab4"

d3 = "#2c7fb8"
d2 = "#41b6c4"
d1 = "#a1dab4"
linewidth = 3


# Define plot format
var_dict = {
    "theta": {
        "column_name": "sm",
        "symbol": r"$\theta$",
        # "label": r"Soil moisture",
        "label": r"Soil moisture, $\theta$",
        "unit": r"($m^3$ $m^{-3}$)",
        "lim": [0, theta_fc],
    },
    "dtheta": {
        "column_name": "",
        "symbol": r"$-\frac{d\theta}{dt}$",
        "label": r"$\minus\frac{d\theta}{dt}$",
        "unit": r"($m^3$ $m^{-3}$ $day^{-1}$)",
        "lim": [-0.08, 0],
    },
    "t": {
        "column_name": "",
        "symbol": r"$t$",
        "label": r"$t$",
        "unit": r"($day$)",
    },
}
theta_vardict = var_dict["theta"]
dtheta_vardict = var_dict["dtheta"]

plt.rcParams["mathtext.fontset"] = (
    "stixsans"  #'stix'  # Or 'cm' (Computer Modern), 'stixsans', etc.
)
# %%
# _________________________________________________________________________
# Loss function plot


fig = plt.figure(figsize=(8, 4))
plt.rcParams.update({"font.size": 14})
ax1 = fig.add_subplot(1, 2, 1)

# List of (q, color) pairs
tau_colors = [(tau1, c1), (tau2, c2)]
theta_w_colors = [(theta_w1, d1), (theta_w2, d2), (theta_w3, d3)]


def tau_loss_func(theta, tau, theta_w):
    return -1 / tau * (theta - theta_w)


# Loop through each pair and plot
for tau, color in tau_colors:
    theta = np.arange(theta_w1, theta_fc, 1e-03)
    ax1.plot(
        theta,
        tau_loss_func(theta=theta, tau=tau, theta_w=theta_w1),
        label=rf"$\tau$={tau}",
        linewidth=linewidth,
        color=color,
    )

for theta_w, color in theta_w_colors:
    theta = np.arange(theta_w, theta_fc, 1e-03)
    ax1.plot(
        theta,
        tau_loss_func(theta=theta, tau=tau3, theta_w=theta_w),
        label=rf"$\tau$={tau}",
        linewidth=linewidth,
        color=color,
    )

ax1.set_ylim([0.0, -(theta_fc - theta_w1) / tau1])
ax1.set_xlim(var_dict["theta"]["lim"])
ax1.set(
    xlabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
    ylabel=f'{dtheta_vardict["label"]} {dtheta_vardict["unit"]}',
)
ax1.set_title(label="C", loc="left")
ax1.set_xticks(
    [theta_w1, theta_star, theta_star + fc_minus_star],
    [r"$\theta_{wp}$", r"$\theta^{*}$", r"$\theta_{fc}$"],
)
ax1.set_yticks(
    [0, -(theta_fc - theta_w1) / tau1], [0, r"$\frac{(\theta_{fc}-\theta_{wp})}{\tau}$"]
)

# _________________________________________________________________________
# Drydown plot

ax2 = fig.add_subplot(1, 2, 2)
tmax = 15
t = np.arange(0, tmax, 1e-03)

for tau, color in tau_colors:
    delta_theta = theta_0 - theta_w1
    ax2.plot(
        t,
        tau_exp_model(t=t, delta_theta=delta_theta, theta_w=theta_w1, tau=tau),
        linewidth=linewidth,
        color=color,
    )

for theta_w, color in theta_w_colors:
    delta_theta = theta_0 - theta_w
    ax2.plot(
        t,
        tau_exp_model(t=t, delta_theta=delta_theta, theta_w=theta_w, tau=tau3),
        linewidth=linewidth,
        color=color,
    )

ax2.set_ylim([0.0, theta_fc])
ax2.set_xlim([0, tmax])
ax2.set(
    xlabel=f'{var_dict["t"]["label"]} {var_dict["t"]["unit"]}',
    ylabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
)
ax2.set_title(label="D", loc="left")  # "Soil moisture drydown",
# ax2.set_xticks([5, 15], [" ", " "])
ax2.set_yticks(
    [theta_w, theta_star, theta_fc], [r"$\theta_{wp}$", r"$\theta^*$", r"$\theta_{fc}$"]
)

fig.tight_layout()


# %% Output

out_path = r"/home/raraki/waves/projects/smap-drydown/output/"
out_dir = os.path.join(out_path, "figs_method")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f"Created dir: {out_dir}")


# %%
fig.savefig(os.path.join(out_dir, f"theory_tau.pdf"), dpi=600, bbox_inches="tight")
# %%

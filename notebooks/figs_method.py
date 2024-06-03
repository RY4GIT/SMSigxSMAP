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

# Non-linearity parameters
q0 = 1
q1 = 1.5
q2 = 0.70
# q1 = 2.47
# q2 = 0.59

# Common parameters
ETmax = 6
z = 50
k = ETmax / z

# k = 0.1
theta_w = 0.02
theta_star = 0.6
fc_minus_star = 0.05
theta_fc = theta_star + fc_minus_star
theta_0 = theta_fc
theta_0_ii = theta_star

# Define variables
theta = np.arange(theta_w, theta_star, 1e-03)
theta_under_wp = np.arange(0, theta_w, 1e-03)
theta_above_star = np.arange(theta_star, theta_star + fc_minus_star, 1e-03)

tmax = 15
t_star = (theta_fc - theta_star) / k
t_before_star = np.arange(0, t_star, 1e-03)
t = np.arange(0, tmax, 1e-03)

# %%
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


# %% Plot
fig = plt.figure(figsize=(8, 4))
plt.rcParams.update({"font.size": 14})
c1 = "#108883"  # "#108883"  # f"#2c7fb8"
c2 = "#2EBB9D"  # "#2EBB9D"  # f"#41b6c4"
c3 = "#F7CA0D"  # "#F7CA0D"  # f"#a1dab4"
linewidth = 3

ax1 = fig.add_subplot(1, 2, 1)

# List of (q, color) pairs
q_colors = [(q1, c1), (q2, c3), (q0, c2)]

# _________________________________________________________________________
# Loss function plot

# Loop through each pair and plot
for q, color in q_colors:
    ax1.plot(
        theta,
        -1
        * loss_model(
            theta=theta, q=q, ETmax=ETmax, theta_w=theta_w, theta_star=theta_star
        ),
        label=f"q={q}",
        linewidth=linewidth,
        color=color,
    )


for q, color in q_colors:
    ax1.plot(
        theta_under_wp,
        np.zeros_like(theta_under_wp),
        linewidth=linewidth,
        color=color,
    )
    ax1.plot(
        theta_above_star,
        np.ones_like(theta_above_star) * k,
        linewidth=linewidth,
        color=color,
    )

ax1.set_ylim([0.0, k + 0.005])
ax1.set_xlim(var_dict["theta"]["lim"])
ax1.set(
    xlabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
    ylabel=f'{dtheta_vardict["label"]} {dtheta_vardict["unit"]}',
)
ax1.set_title(
    label="A", loc="left"
)  # rf"Normalized loss function $L(\theta)/\Delta z$",)
ax1.set_xticks(
    [theta_w, theta_star, theta_star + fc_minus_star],
    [r"$\theta_{wp}$", r"$\theta^{*}$", r"$\theta_{fc}$"],
)
ax1.set_yticks([0, k], [0, r"$\frac{ET_{max}}{\Delta z}$"])

# _________________________________________________________________________
# Drydown plot

ax2 = fig.add_subplot(1, 2, 2)

# Calculate & plot d_theta
# q > 1
ax2.plot(
    t,
    q_model_piecewise(
        t=t,
        q=q1,
        ETmax=ETmax,
        theta_0=theta_fc,
        theta_star=theta_star,
        theta_w=theta_w,
    ),
    label=f"q={q1}",
    linewidth=linewidth,
    color=c1,
)
# q < 1
ax2.plot(
    t,
    q_model_piecewise(
        t=t,
        q=q2,
        ETmax=ETmax,
        theta_0=theta_fc,
        theta_star=theta_star,
        theta_w=theta_w,
    ),
    label=f"q={q2}",
    linewidth=linewidth,
    color=c3,
)
# q = 1
ax2.plot(
    t,
    exp_model_piecewise(
        t=t,
        theta_0=theta_fc,
        theta_w=theta_w,
        theta_star=theta_star,
        ETmax=ETmax,
    ),
    label=f"q={q0}",
    linewidth=linewidth,
    color=c2,
)

# for q, color in q_colors:
#     ax2.plot(
#         t_before_star,
#         theta_star - ETmax / z * t_before_star,
#         label=f"q={q0}",
#         linewidth=linewidth,
#         color=color,
# )

# # theta reaches to zero earlier for q > 1
# # Hard to get the value analytically ..
# t_after_star = np.arange(6.05, tmax, 1e-03)
# ax2.plot(
#     t_after_star,
#     np.ones_like(t_after_star) * theta_w,
#     label=f"q={q0}",
#     linewidth=linewidth,
#     color=c3,
# )

ax2.set_ylim([0.0, theta_fc])
ax2.set_xlim([0, tmax])
ax2.set(
    xlabel=f'{var_dict["t"]["label"]} {var_dict["t"]["unit"]}',
    ylabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
)
ax2.set_title(label="B", loc="left")  # "Soil moisture drydown",
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
fig.savefig(os.path.join(out_dir, f"theory_q.pdf"), dpi=600, bbox_inches="tight")
# %%

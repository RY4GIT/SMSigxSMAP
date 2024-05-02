import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger

# Create a logger
log = getLogger(__name__)

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""


class Event:
    def __init__(self, index, event_dict):

        # Read the data
        self.index = index
        self.start_date = event_dict["event_start"]
        self.end_date = event_dict["event_end"]
        sm_subset = np.asarray(event_dict["sm_masked"])
        self.pet = np.nanmax(event_dict["PET"])
        self.min_sm = event_dict["min_sm"]
        self.max_sm = event_dict["max_sm"]
        self.est_theta_fc = event_dict["est_theta_fc"]
        self.est_theta_star = event_dict["est_theta_star"]

        # Prepare the attributes
        self.subset_sm_range = np.nanmax(sm_subset) - np.nanmin(sm_subset)
        self.subset_min_sm = np.nanmin(sm_subset)

        # Prepare the inputs
        t = np.arange(0, len(sm_subset), 1)
        self.x = t[~np.isnan(sm_subset)]
        self.y = sm_subset[~np.isnan(sm_subset)]

    def add_attributes(
        self,
        model_type="",
        popt=[],
        r_squared=np.nan,
        y_opt=[],
        est_theta_star=np.nan,
        est_theta_w=np.nan,
    ):
        if model_type == "tau_exp":
            self.tau_exp = {
                "delta_theta": popt[0],
                "theta_w": popt[1],
                "tau": popt[2],
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }

        if model_type == "exp":
            self.exp = {
                "ETmax": popt[0],
                "theta_0": popt[1],
                "theta_star": est_theta_star,
                "theta_w": est_theta_w,
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }

        if model_type == "q":
            self.q = {
                "q": popt[0],
                "ETmax": popt[1],
                "theta_0": popt[2],
                "theta_star": est_theta_star,
                "theta_w": est_theta_w,
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }

        if model_type == "sgm":
            self.sgm = {
                "theta50": popt[0],
                "k": popt[1],
                "a": popt[2],
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }

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



# Define the three colors in the colormap
colors = ['#d8b365', '#f5f5f5', '#5ab4ac']
# Create a custom colormap
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_BrBG', colors, N=256)
cmap = "BrBG"

# Define the specific order for your categories.
vegetation_orders = [
    "BAR", "OSH", "CNM", "WSA", "SAV", 
    "GRA", "CRO", "ENF", "CSH", "WET"
]
veg_colors = [
    "#7A422A", "#C99728", "#229954", "#4C6903", "#92BA31", 
    "#13BFB2", "#F7C906", "#022E1F", "#A68F23", "#4D5A6B"
]

# Create a color palette dictionary
palette_dict = dict(zip(vegetation_orders, veg_colors))


var_dict = {
    'q_q' : {
        'symbol' : r"$q$",
        'label' : r"$q$",
        # 'unit' : 
    },
}
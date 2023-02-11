# %%
import os
import numpy as np
import pandas as pd
import datetime
import rasterio as rio
import xarray as xr
import rioxarray
import json
import requests
from pyproj import CRS
import pyproj
from osgeo import gdal
from math import ceil, floor
import matplotlib.pyplot as plt
from odc.stac import stac_load
import planetary_computer
import pystac_client
import rich.table
import warnings
warnings.filterwarnings("ignore")

# %%
###### CHANGE HERE ###########
network_name = "Oznet_Alabama_area"
minx = 147.291085
miny = -35.903334
maxx = 148.172738
maxy = -35.190465

###### CHANGE HERE ###########
input_path = r"..\1_data"
output_path = r"..\3_data_out"
SMAPL3_path = "SPL3SMP_E"
SMAPL4_path = "SPL4SMGP"
SMAPL4_grid_path = "SMAPL4SMGP_EASEreference"
PET_path = "PET"

# %%
# 1. Load EASE grid
print('Load EASE grid')
fn = "SMAP_L4_SM_lmc_00000000T000000_Vv7032_001.h5"
file_path = os.path.join(input_path, SMAPL4_grid_path, fn)
if os.path.exists(file_path):
    print('The file exists, loaded EASE grid')
else:
    print('The file does NOT exist')
    print(file_path)
g = gdal.Open(file_path)
subdatasets = g.GetSubDatasets()

varname_lat = "cell_lat"
full_varname_lat = f'HDF5:"{file_path}"://{varname_lat}'

varname_lon = "cell_lon"
full_varname_lon = f'HDF5:"{file_path}"://{varname_lon}'

varname_ease_column = "cell_column"
full_varname_ease_column = f'HDF5:"{file_path}"://{varname_ease_column}'

varname_ease_row = "cell_row"
full_varname_ease_row = f'HDF5:"{file_path}"://{varname_ease_row}'

ease_lat = rioxarray.open_rasterio(full_varname_lat)
ease_lon = rioxarray.open_rasterio(full_varname_lon)
ease_column = rioxarray.open_rasterio(full_varname_ease_column)
ease_row = rioxarray.open_rasterio(full_varname_ease_row)

# %%
target_area_mask = (ease_lat[0].values >= miny) & (ease_lat[0].values <= maxy) & (ease_lon[0].values >= minx) & (ease_lon[0].values <= maxx)
target_coordinate = pd.DataFrame()
target_coordinate['latitude'] = ease_lat[0].values[target_area_mask].flatten()
target_coordinate['longitude'] = ease_lon[0].values[target_area_mask].flatten()
target_coordinate['ease_column'] = ease_column[0].values[target_area_mask].flatten()
target_coordinate['ease_row'] = ease_row[0].values[target_area_mask].flatten()

print(target_coordinate.head())

print(len(target_coordinate))
# %%
# 2. SMAP L4
# 2.1. Get a snapshot of SMAP L4 data for the area of interest
# 
# 2.2. Get SMAPL4 data according to the EASE GRID point 


# %%
# 3. SMAP L3
# 3.1. Get a snapshot of SMAP L3 data for the area of interest
# 
# 
# 
# 3.2. Get SMAPL3 data according to the EASE GRID point

# %% 
# 4. MODIS LAI
# 4.1. Get a snapshot of MODIS LAI data for the area of interest
# 
# 
# 
# 4.2. Get MODIS LAI data according to the EASE GRID point

# %% 
# 4. Singer PET 
# 4.1. Get a snapshot ofSinger PET data for the area of interest
# 
# 
# 
# 4.2. Get Singer PET data according to the EASE GRID point



# %% 5. Sync all the data and save 




# %% 6. Analyze 





# %% 7. Plot and save the snapshot of PET vs. Ltheta

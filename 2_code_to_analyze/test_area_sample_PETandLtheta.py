# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
bbox = {'minx':minx, 'maxx':maxx, 'miny':miny, 'maxy':maxy}

startDate = datetime(2016, 1, 1)
endDate = datetime(2016, 2, 1)
currentDate = startDate

###### CHANGE HERE ###########
input_path = r"..\1_data"
output_path = r"..\3_data_out"
SMAPL3_path = "SPL3SMP_E"
SMAPL4_path = "SPL4SMGP"
SMAPL4_grid_path = "SMAPL4SMGP_EASEreference"
PET_path = "PET"

# %%
# 2. SMAP L4
# 2.1. Get a snapshot of SMAP L4 data and stack of the data along time axis the area of interest

# %%
def load_SMAPL4_precip(bbox=None, currentDate=None):

    SMAPL4_times = ['0130', '0430', '0730', '1030', '1330', '1630', '1930', '2230'] # 3-hourly data
    minx = bbox['minx']
    maxx = bbox['maxx']
    miny = bbox['miny']
    maxy = bbox['maxy']

    # Loop for 3 hourly data
    for SMAPL4_time in SMAPL4_times: 

        ### Some notes ###
        # open_rasterio is fast, as it is lazy load
        # https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.open_rasterio

        # Y axis accidentally flipped when loading data. The following solution suggested in stackoverflow did not work :(
        # ds_SMAPL4 = ds_SMAPL4.reindex(y=ds_SMAPL4.y*(-1))
        # ds_SMAPL4 = ds_SMAPL4.isel(y=slice(None, None, -1)).copy()
        #################

        # Check files
        fn = os.path.join(input_path, SMAPL4_path, f'SMAP_L4_SM_gph_{currentDate.strftime("%Y%m%d")}T{SMAPL4_time}00_Vv7032_001_HEGOUT.nc')
        if not os.path.exists(fn):
            print(f'File does not exist SMAP_L4_SM_gph_{currentDate.strftime("%Y%m%d")}T{SMAPL4_time}00_Vv7032_001_HEGOUT.nc')
            continue

        # Open dataset and clip to the area of interest
        ds_SMAPL4 = rioxarray.open_rasterio(fn)
        ds_SMAPL4_clipped = ds_SMAPL4.rio.clip_box(minx=minx, miny=maxy*(-1), maxx=maxx, maxy=miny*(-1)).copy()
        ds_SMAPL4_clipped_array = ds_SMAPL4_clipped['precipitation_total_surface_flux'][0].values
        ds_SMAPL4_clipped_array[ds_SMAPL4_clipped_array==-9999] = np.nan

        # Stack up to get daily values 
        if not 'ds_SMAPL4_stack' in locals():
            ds_SMAPL4_stack = ds_SMAPL4_clipped_array
        else: 
            ds_SMAPL4_stack = np.dstack((ds_SMAPL4_stack, ds_SMAPL4_clipped_array))

        y_coord = ds_SMAPL4_clipped.y.values * (-1)
        x_coord = ds_SMAPL4_clipped.x.values

        del ds_SMAPL4_clipped_array, ds_SMAPL4, ds_SMAPL4_clipped

    # Get daily average precipitation field
    ds_SMAPL4_avg = np.nanmean(ds_SMAPL4_stack, axis=2)

    # Create new dataarray with data corrected for the flipped y axis
    ds_SMAPL4_P = xr.DataArray(
                data = ds_SMAPL4_avg,
                dims=['y', 'x'],
                coords=dict(
                    y = y_coord,
                    x = x_coord,
                    time = currentDate
                    )
            )
    ds_SMAPL4_P.rio.write_crs('epsg:4326', inplace=True) 
    del ds_SMAPL4_stack, ds_SMAPL4_avg, x_coord, y_coord

    return ds_SMAPL4_P

# %%

# Loop for the timeperiod
delta = timedelta(days=1)
data_list = []
while currentDate <= endDate:
    print(currentDate)
    data_list.append(load_SMAPL4_precip(bbox=bbox, currentDate=currentDate))
    currentDate += delta

stacked_ds_SMAPL4_P = xr.concat(data_list, dim='time')

#%%
# https://docs.xarray.dev/en/stable/user-guide/plotting.html
# https://docs.xarray.dev/en/stable/user-guide/time-series.html
# stacked_ds_SMAPL4_P.plot()
# Plot just in case
import cartopy.crs as ccrs

p = stacked_ds_SMAPL4_P.sel(time=currentDate-delta).plot(
    transform=ccrs.PlateCarree(),
    subplot_kws=dict(projection=ccrs.Orthographic(minx, miny), facecolor="gray")
)

p.axes.set_global()
p.axes.coastlines()

plt.draw()

# %%
# 3. SMAP L3
# 3.1. Get a snapshot of SMAP L4 data and stack of the data along time axis the area of interest
def load_SMAPL3_SM(bbox=None, currentDate=None):
    
    minx = bbox['minx']
    maxx = bbox['maxx']
    miny = bbox['miny']
    maxy = bbox['maxy']

    # Load file 
    fn = os.path.join(input_path, SMAPL3_path, f'SMAP_L3_SM_P_E_{currentDate.strftime("%Y%m%d")}_R18290_001_HEGOUT.nc')
    if not os.path.exists(fn):
        fn = os.path.join(input_path, SMAPL3_path, f'SMAP_L3_SM_P_E_{currentDate.strftime("%Y%m%d")}_R18290_002_HEGOUT.nc')
        if not os.path.exists(fn):
            print(f'The file does not exist: {fn}')
    ds_SMAPL3_0 = rioxarray.open_rasterio(fn)

    # Clip and mask data using quality flag
    # Thankfully the y axis is not flipped for SMAPL3 dataset
    ds_SMAPL3_clipped = ds_SMAPL3_0.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy).copy()
    mask_am = (ds_SMAPL3_clipped.retrieval_qual_flag == 0) | (ds_SMAPL3_clipped.retrieval_qual_flag == 8)
    ds_SMAPL3_am = ds_SMAPL3_clipped['soil_moisture'].where(mask_am).copy()
    mask_pm = (ds_SMAPL3_clipped.retrieval_qual_flag_pm == 0) | (ds_SMAPL3_clipped.retrieval_qual_flag_pm == 8)
    ds_SMAPL3_pm = ds_SMAPL3_clipped['soil_moisture_pm'].where(mask_pm).copy()
    del mask_am, mask_pm

    # Concatenate AM and PM data, take average between AM and PM data, and assign a time coordinate
    ds_SMAPL3 = xr.concat([ds_SMAPL3_am, ds_SMAPL3_pm], dim='band').mean(dim='band', skipna=True).assign_coords({"time": currentDate})
    del ds_SMAPL3_am, ds_SMAPL3_pm

    return ds_SMAPL3
# %%
# Loop for the timeperiod
delta = timedelta(days=1)
data_list = []
while currentDate <= endDate:
    print(currentDate)
    data_list.append(load_SMAPL3_SM(bbox=bbox, currentDate=currentDate))
    currentDate += delta
stacked_ds_SMAPL3_SM = xr.concat(data_list, dim='time')

#%%
import cartopy.crs as ccrs

p = stacked_ds_SMAPL3_SM.sel(time=currentDate-delta).plot(
    transform=ccrs.PlateCarree(),
    subplot_kws=dict(projection=ccrs.Orthographic(minx, miny), facecolor="gray")
)

p.axes.set_global()
p.axes.coastlines()

plt.draw()

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



# %% Make sure xarrays plot on the same location on earth
# https://xarray.pydata.org/en/v0.7.2/plotting.html



# %% 7. Plot and save the snapshot of PET vs. Ltheta
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
target_coordinates = pd.DataFrame()
target_coordinates['latitude'] = ease_lat[0].values[target_area_mask].flatten()
target_coordinates['longitude'] = ease_lon[0].values[target_area_mask].flatten()
target_coordinates['ease_column'] = ease_column[0].values[target_area_mask].flatten()
target_coordinates['ease_row'] = ease_row[0].values[target_area_mask].flatten()

del ease_lat, ease_lon, ease_column, ease_row

# %%
print(target_coordinates.head())
print(target_coordinates.tail())
print(len(target_coordinates))
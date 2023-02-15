# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime
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

startDate = datetime(2016, 1, 1)
endDate = datetime(2016, 7, 1)
currentDate = startDate

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
# %%
# 2. SMAP L4
# Stack of the data for the mini raster



# Read data
def load_SMAPL4(target_coordinate, currentDate, print_results=False):
    SMAPL4_times = ['0130', '0430', '0730', '1030', '1330', '1630', '1930', '2230'] # 3-hourly data
    precip_value = list()
    for SMAPL4_time in SMAPL4_times: 
        fn = os.path.join(input_path, SMAPL4_path, f'SMAP_L4_SM_gph_{currentDate.strftime("%Y%m%d")}T{SMAPL4_time}00_Vv7032_001_HEGOUT.nc')

        # if ~os.path.exists(fn):
        #     print('File doesnt exist')
        #     continue

        full_varname_precip = f'HDF5:"{fn}"://Geophysical_Data/precipitation_total_surface_flux'

        # TODO: Slice the data rather than reading the entire dataset for faster processing 
        ds_SMAPL4_precip = rioxarray.open_rasterio(full_varname_precip)
        ds_SMAPL4_coord = xr.open_dataset(fn)

        # #%%
        # Selecting the ease row did not work ... #1280 did not exist. 
        # Allow for SMAP L4, do not allow for SMAP L3 
        ds = xr.DataArray(
            data = ds_SMAPL4_precip[0].values,
            dims=['y', 'x'],
            coords={
                'y': ds_SMAPL4_coord.y.values,
                'x': ds_SMAPL4_coord.x.values
                },
            attrs={'_FillValue': -9999.0}
        )

        try:
            pixel_precip = ds.sel(x=target_coordinate.longitude, y=target_coordinate.latitude, method='nearest', tolerance=0.25)
            precip_value.append(pixel_precip.values)
        except:
            warnings('No data found near the cell')
            precip_value.append(np.nan)

        if print_results: 
            pixel_precip_coord = ds_SMAPL4_coord.sel(x=target_coordinate.longitude, y=target_coordinate.latitude, method='nearest', tolerance=0.25)
            print(f'Target coord: Lon {target_coordinate.longitude}, Lat {target_coordinate.latitude}')
            print(f'Result coord: Lon {pixel_precip.x.values}, Lat {pixel_precip.y.values}')
            print(f'Result coord2: Lon {pixel_precip_coord.x.values}, Lat {pixel_precip_coord.y.values}')
            print(f'Target EASE: Col {target_coordinate.ease_column}, Row {target_coordinate.ease_row}')
            print(f'Result EASE: Col {pixel_precip_coord.cell_column.values}, Row {pixel_precip_coord.cell_row.values}')

        del ds_SMAPL4_precip, ds_SMAPL4_coord, ds

    return np.nanmean(precip_value)

# %%
# 2.1. Get a snapshot of SMAP L4 data for the area of interest
daily_precips = np.empty([1])
for i in range(len(target_coordinates)):
    print(f'Currently processing {i+1}/{len(target_coordinates)}')
    target_coordinate = target_coordinates.iloc[i]
    precip = load_SMAPL4(target_coordinate=target_coordinate, currentDate=datetime(2016, 1, 1))
    daily_precips = np.append(daily_precips, precip)
daily_precips = daily_precips[1:]
plt.scatter(target_coordinates.longitude.values, target_coordinates.latitude.values, c=daily_precips)

# %% 
# 2.1. Get a timeseries of SMAP L4 data for the area of interest
daily_precip_ts = np.empty([1])

# %%
for i in range(len(target_coordinates)):
    print(f'Currently processing {i+1}/{len(target_coordinates)}')
    target_coordinate = target_coordinates.iloc[i]
    precip = load_SMAPL4(target_coordinate=target_coordinate, currentDate=datetime(2016, 1, 1))
    daily_precip_ts = np.append(daily_precips, precip)
daily_precip_ts = daily_precips[1:]
plt.scatter(daily_precip_ts)

# %%
# 3. SMAP L3
# Stack of the data for the mini raster
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

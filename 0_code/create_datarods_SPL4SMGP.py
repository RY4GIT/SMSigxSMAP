# %%
# https://nsidc.org/data/smap_l1_l3_anc_static/versions/1
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
import os
from datetime import datetime
import glob
from tqdm import tqdm

# %% [markdown]
# # Configuration

# %%
data_dir = r"G:\Araki\SMSigxSMAP\1_data"
SMAPL3_dir = "SPL3SMP"
datarods_dir = "datarods"
SMAPL4_dir = "SPL4SMGP"
SMAPL4_grid_dir = "SMAPL4SMGP_EASEreference"
PET_dir = "PET"
SMAPL3_grid_sample = os.path.join(data_dir, r"SPL3SMP\SMAP_L3_SM_P_20150331_R18290_001.h5")

# %% [markdown]
# # Investigate netCDF4 file

# %%
ncf = netCDF4.Dataset(SMAPL3_grid_sample, diskless=True, persist=False)
nch_am = ncf.groups.get('Soil_Moisture_Retrieval_Data_AM')
nch_pm = ncf.groups.get('Soil_Moisture_Retrieval_Data_PM')

# %%
nch_am.variables

# %%
nch_pm.variables

# %% [markdown]
# # Prepare geocoordinate matrix 

# %%
# Return as regular numpy array rather than masked array
_latitude = ma.getdata(nch_am.variables['latitude'][:].filled(fill_value=np.nan), subok=True)
_longitude = ma.getdata(nch_am.variables['longitude'][:].filled(fill_value=np.nan), subok=True)
_EASE_column_index = ma.getdata(nch_am.variables['EASE_column_index'][:].astype(int).filled(fill_value=-1), subok=True)
_EASE_row_index = ma.getdata(nch_am.variables['EASE_row_index'][:].astype(int).filled(fill_value=-1), subok=True)

# %%
# Coordinates with no data are skipped --- fill them
latitude = np.nanmax(_latitude, axis=1)
EASE_row_index = np.nanmax(_EASE_row_index, axis=1)
longitude = np.nanmax(_longitude, axis=0)
EASE_column_index = np.nanmax(_EASE_column_index, axis=0)

# %%
coord_info_column = pd.DataFrame({"latitude":latitude, "EASE_column_index":EASE_row_index})
coord_info_row = pd.DataFrame({"longitude":longitude, "EASE_row_index":EASE_column_index})
coord_info_column.to_csv(os.path.join(data_dir, 'coord_info_unique_column.csv'), index=False)
coord_info_row.to_csv(os.path.join(data_dir, 'coord_info_unique_row.csv'),  index=False)
coord_info = coord_info_row.assign(key=1).merge(coord_info_column.assign(key=1), on='key').drop('key', axis=1)
coord_info.index.name = 'id'
coord_info.to_csv(os.path.join(data_dir, 'coord_info.csv'))

# %%
# Get a list of files 
SMAPL3_fn_pattern = f'SMAP_L3_SM_P_*.h5'
SMAPL3_file_paths = glob.glob(rf'{data_dir}/{SMAPL3_dir}/{SMAPL3_fn_pattern}')
print(f"{SMAPL3_fn_pattern}: {len(SMAPL3_file_paths)} ... {len(SMAPL3_file_paths)/365:.1f} yrs of data available")

# %%
_ds_SMAPL3 = xr.open_dataset(SMAPL3_file_paths[0], engine='rasterio', group='Soil_Moisture_Retrieval_Data_AM', variable=['soil_moisture'])
ds_SMAPL3_coord_template = _ds_SMAPL3.assign_coords({'x':longitude, 'y':latitude}).rio.write_crs("epsg:4326")
ds_SMAPL3_coord_template

# %% [markdown]
# # Process SMAP L4 precip data

# %% [markdown]
# ## Read SMAP L4 data
from rasterio.enums import Resampling
SMAPL4_template_fn = r"G:\Araki\SMSigxSMAP\1_data\SPL4SMGP\SMAP_L4_SM_gph_20180911T103000_Vv7032_001_HEGOUT.nc"
SMAPL4_template = xr.open_dataset(SMAPL4_template_fn)
# %%
def preprocess_SMAPL4(ds):
    # Assign missing time dimension
    startTime = datetime.strptime(ds.rangeBeginningDateTime.split(".")[0], '%Y-%m-%dT%H:%M:%S')
    endTime = datetime.strptime(ds.rangeEndingDateTime.split(".")[0], '%Y-%m-%dT%H:%M:%S')
    midTime = startTime + (startTime - endTime)/2
    ds = ds.assign_coords(time=midTime)

    # Reassign coordinates 
    ds = ds.assign_coords(x=SMAPL4_template['x'][:], y=SMAPL4_template['y'][:]*(-1))

    # Resample according to SMAPL3 grid
    ds.rio.write_crs('epsg:4326', inplace=True)
    ds_SAMPL4_resampled = ds.sel(band=1).rio.reproject_match(ds_SMAPL3_coord_template.sel(band=1), resampling=Resampling.average)

    # Fillnan 
    _FillValue = _FillValue = 3.4028235e+38
    ds_SMAPL4_3hrly = ds_SAMPL4_resampled.where(ds_SAMPL4_resampled.precipitation_total_surface_flux != _FillValue, np.nan)
    
    return ds_SMAPL4_3hrly

# %%
chunks = {'x': 1200, 'y': 1200, 'time':1, 'band':1}
SMAPL4_fn_pattern = f'SMAP_L4_SM_gph_*.nc'
SMAPL4_file_paths = glob.glob(rf'{data_dir}/{SMAPL4_dir}/{SMAPL4_fn_pattern}')
print(f"{SMAPL4_fn_pattern}: {len(SMAPL4_file_paths)} ... {len(SMAPL4_file_paths)/6/365:.1f} yrs of data available")
# file_path = r"G:\Araki\SMSigxSMAP\1_data\SPL4SMGP\SMAP_L4_SM_gph_20150331T013000_Vv7032_001_HEGOUT.nc"
# ds_SMAPL4_3hrly = xr.open_mfdataset(file_path, group='Geophysical_Data', engine="rasterio", preprocess=preprocess_SMAPL4, chunks=chunks)
# ds_SMAPL4_3hrly
# ds_SMAPL4_3hrly = xr.open_mfdataset(SMAPL4_file_paths, group='Geophysical_Data', engine="rasterio", preprocess=preprocess_SMAPL4, chunks=chunks)

# %%
# Get a list of files 
# Load dataset
import warnings
warnings.filterwarnings("ignore")
for filename in tqdm(SMAPL4_file_paths):
    try:
        _ds_SMAPL4 = xr.open_dataset(filename, group='Geophysical_Data', engine="rasterio")
        _ds_SMAPL4 = preprocess_SMAPL4(_ds_SMAPL4)
        if 'ds_SMAPL4' in locals():
            ds_SMAPL4 = xr.concat([ds_SMAPL4, _ds_SMAPL4], dim="time")
        else:
            ds_SMAPL4 = _ds_SMAPL4
    except Exception as e:
        print(f"An error occurred: {e}")
        continue
# Cannot use open_mfdataset --- can't skip if there is error also reproject_match have too much issues 

# %%
# Re-assign x and y coordinates

# Fill values 
# Skip this because this is potentially too much computation
# _FillValue = 3.4028235e+38 # ds_SMAPL4_3hrly.precipitation_total_surface_flux._FillValue
# ds_SMAPL4_3hrly = ds_SMAPL4_3hrly.where(ds_SMAPL4_3hrly.precipitation_total_surface_flux != _FillValue, np.nan)

# Resample to daily
print("Start resampling")
ds_SMAPL4_daily = ds_SMAPL4.resample(time='D', skipna=True, keep_attrs=True).mean()
print("Done resampling")
# %% [markdown]
# ## Resample to SMAPL3 grid



# %%
# ds_SAMPL4_resampled.precipitation_total_surface_flux.sel(time="2016-10-31").plot()

# %%
# Can combine all data together --- but memory is not enough
# combined_ds = _ds_SMAPL3_list_stacked.sel(band=1)
# combined_ds["precipitation_total_surface_flux"] = ds_SAMPL4_resampled.precipitation_total_surface_flux

# %% [markdown]
# ## Create datarods

# %%
# Create and save the datarods  
out_dir = os.path.join(data_dir, datarods_dir, SMAPL4_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

from itertools import product
print("CREATING SMAPL4 DATARODS")
for y_i, x_j in tqdm(product(EASE_row_index, EASE_column_index)):
    try:
        df_SMAPL3 = ds_SMAPL4_daily.isel(x=x_j, y=y_i,band=0).to_dataframe().drop(['band', 'spatial_ref', 'projection_information'], axis=1)
        filename = f'{SMAPL4_dir}_{y_i:03}_{x_j:03}.csv'
        df_SMAPL3.to_csv(os.path.join(out_dir, filename))
    except Exception as e:
        print(f"An error occurred: {e}")
        continue



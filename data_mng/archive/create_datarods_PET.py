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
data_dir = r"/home/waves/projects/smap-drydown/data"
SMAPL3_dir = "SPL3SMP"
datarods_dir = "datarods"
SMAPL4_dir = "SPL4SMGP"
SMAPL4_grid_dir = "SMAPL4SMGP_EASEreference"
PET_dir = "PET"
SMAPL3_grid_sample = os.path.join(data_dir, r"SPL3SMP/SMAP_L3_SM_P_20150331_R18290_001.h5")

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
# # Process PET data

# %% [markdown]
# ## Read SMAP L4 data
from rasterio.enums import Resampling


# %%
pet_chunks = {'x': 1200, 'y': 1200, 'time':1, 'band':1}
PET_fn_pattern = f'*_daily_pet.nc'
PET_file_paths = glob.glob(rf'{data_dir}/{PET_dir}/{PET_fn_pattern}')
print(f"{PET_fn_pattern}: {len(PET_file_paths)} ... {len(PET_file_paths):.1f} yrs of data available")
print(f"Start reading dataset")
ds_PET = xr.open_mfdataset(PET_file_paths, combine="nested", chunks=pet_chunks, concat_dim="time", parallel=True)
print(f"End reading dataset")

# Interpolate to SMAP grid
ds_PET = ds_PET.rename({'longitude':'x', 'latitude':'y'})
ds_PET.rio.write_crs('epsg:4326', inplace=True)

# print(f"Start filling NaN values")
# _FillValue = ds_PET._FillValue
# ds_PET["pet"] = ds_PET.pet.where(ds_PET.pet != _FillValue, np.nan)
# print(f"End filling NaN values")

print(f"Start interpolating to SMAP grid")
PET_resampled = ds_PET['pet'].interp(coords={'x': ds_SMAPL3_coord_template['x'], 'y': ds_SMAPL3_coord_template['y']}, method='linear', kwargs={'fill_value': np.nan})
print(f"End interpolating to SMAP grid")


# Create and save the datarods  
out_dir = os.path.join(data_dir, datarods_dir, PET_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

from itertools import product
print("CREATING PET DATARODS")
for y_i, x_j in tqdm(product(EASE_row_index, EASE_column_index)):
    try:
        df_PET = PET_resampled.isel(x=x_j, y=y_i).to_dataframe().drop(['spatial_ref'], axis=1)
        filename = f'{PET_dir}_{y_i:03}_{x_j:03}.csv'
        df_PET.to_csv(os.path.join(out_dir, filename))
    except Exception as e:
        print(f"An error occurred: {e}")
        continue

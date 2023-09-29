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
# # Process PET data

# %% [markdown]
# ## Read SMAP L4 data
from rasterio.enums import Resampling
from multiprocessing import Pool
# %% [markdown]
# # Configuration
from itertools import product

# %%
data_dir = r"/home/waves/projects/smap-drydown/data"
SMAPL3_dir = "SPL3SMP"
datarods_dir = "datarods"
SMAPL4_dir = "SPL4SMGP"
SMAPL4_grid_dir = "SMAPL4SMGP_EASEreference"
# PET_dir = "PET"
# out_dir = os.path.join(data_dir, datarods_dir, PET_dir)

# %% [markdown]
# # Investigate netCDF4 file

def create_output_dir(out_dir):
    # Create and save the datarods  
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"created {out_dir}")
    return out_dir

def create_datarods(y_i, x_j, ds, out_dir, data_variable, variables_to_drop=['spatial_ref']):
    try:
        df = ds.isel(x=x_j, y=y_i).to_dataframe().drop(variables_to_drop, axis=1)
        filename = f'{data_variable}_{y_i:03}_{x_j:03}.csv'
        df.to_csv(os.path.join(out_dir, filename))
    except Exception as e:
        print(f"An error occurred: {e}")
        
def get_filepath(filename_pattern, directory):
    file_paths = glob.glob(os.path.join(directory, filename_pattern))
    print(f"{filename_pattern}: {len(file_paths)} ... {len(file_paths):.1f} yrs of data available")
    return file_paths

class EASEgrid_template():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.row_index, self.column_index, self.longitude, self.latitude = self.get_grid_coordinate()
        self.varname = "SPL3SMP"
        self.timestep = 365
        self.filenames = self.get_filepath()
        self.data = self.get_template_dataset()
        
    def get_grid_coordinate(self):
        SMAPL3_grid_sample = os.path.join(self.data_dir, r"SPL3SMP/SMAP_L3_SM_P_20150331_R18290_001.h5")

        # %%
        ncf = netCDF4.Dataset(SMAPL3_grid_sample, diskless=True, persist=False)
        nch_am = ncf.groups.get('Soil_Moisture_Retrieval_Data_AM')
        nch_pm = ncf.groups.get('Soil_Moisture_Retrieval_Data_PM')

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
        
        return EASE_row_index, EASE_column_index, longitude, latitude
    
    def get_filepath(self):
        filepaths = get_filepath(filename_pattern = f'SMAP_L3_SM_P_*.h5', directory=f"{data_dir}/{self.varname}")
        return filepaths
    
    def get_template_dataset(self):
        _ds_SMAPL3 = xr.open_dataset(self.filenames[0], engine='rasterio', group='Soil_Moisture_Retrieval_Data_AM', variable=['soil_moisture'])
        ds_SMAPL3_coord_template = _ds_SMAPL3.assign_coords({'x':self.longitude, 'y':self.latitude}).rio.write_crs("epsg:4326")
        return ds_SMAPL3_coord_template

class PET():
    def __init__(self, data_dir):
        self.varname = "PET"
        self.filenames = self.get_filepath(data_dir)
        self.out_dir = create_output_dir(os.path.join(data_dir, datarods_dir, self.varname))
        
    def get_filepath(self, data_dir):
        file_paths = get_filepath(filename_pattern=f'*_daily_pet.nc', directory=os.path.join(data_dir, self.varname))
        return file_paths

    def read_data(self, resample_target=None):
        print(f"Start reading dataset")

        for filename in tqdm(self.filenames):
            try:
                _ds = xr.open_dataset(filename)

                # The dataset is huge, resample before stacking
                _ds = _ds.rename({'longitude':'x', 'latitude':'y'})
                _ds.rio.write_crs('epsg:4326', inplace=True)
                _ds_resampled = _ds.pet.interp_like(resample_target, method='linear', kwargs={'fill_value': np.nan})
                
                # Stacking data
                if 'ds_PET' in locals():
                    ds = xr.concat([ds, _ds_resampled], dim="time")
                else:
                    ds = _ds_resampled
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
                
        print(f"End reading dataset")
        
        ds = ds.sortby('time')
        self.data = ds
        return ds
    
    def create_datarods(self, y_i, x_j):
        print(f"Processing: {y_i}, {x_j}")
        create_datarods(y_i=y_i, x_j=x_j, ds=self.data, out_dir=self.out_dir, data_variable=self.varname, variables_to_drop=['spatial_ref'])


def main():

    easegrid_template = EASEgrid_template(data_dir=data_dir)
    pet = PET(data_dir=data_dir)
    pet.read_data(resample_target=easegrid_template.data)
    
    # Create a multiprocessing Pool
    num_processes = 4
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.starmap(pet.create_datarods, product(easegrid_template.row_index, easegrid_template.column_index))):
            pass

if __name__ == "__main__":
    main()

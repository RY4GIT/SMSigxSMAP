# %%
# https://nsidc.org/data/smap_l1_l3_anc_static/versions/1
import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

# Define configurations
start_year = '2016'
end_year = '2021'
data_dir = r'/home/waves/projects/smap-drydown/data'
datarods_dir = 'datarods'
SMAPL4_varname = 'SPL4SMGP'
PET_varname = 'PET'
# 
# %%
# Loop through all the rows
def process_row(row_index, row_data):
    EASE_row_index = int(row_data['EASE_row_index'])
    EASE_column_index = int(row_data['EASE_column_index'])

    # %%
    # Read Precipitation data
    filename = os.path.join(data_dir, datarods_dir, SMAPL4_varname, f"{SMAPL4_varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv")
    df_p = pd.read_csv(filename).drop(columns=['x','y']).rename({'precipitation_total_surface_flux': 'precip'}, axis='columns')
    df_p['time'] = pd.to_datetime(df_p['time'])
    df_p.set_index('time', inplace=True)
    df_p.precip = df_p.precip * 86400
    df_p.head()

    # %%
    # Read PET data
    filename = os.path.join(data_dir, datarods_dir, PET_varname, f"{PET_varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv")
    df_pet = pd.read_csv(filename).drop(columns=['x','y']) #.rename({'precipitation_total_surface_flux': 'precip'}, axis='columns')
    df_pet['time'] = pd.to_datetime(df_pet['time'])
    df_pet.set_index('time', inplace=True)
    df_pet.head()

    # %%
    # Join both timeseries
    _df = pd.merge(df_p, df_pet, how='inner', left_index=True, right_index=True)
    df = _df[start_year:end_year].copy()
    # df
    # %%
    # Get the MAP and MAE (mean annuap preciptiation and evapotranspiration)
    df_annual = df.resample('A').sum()
    # df_annual
    # %%
    # Calculate aridity index each year 
    df_annual["AI"] = df_annual.precip / df_annual.pet
    # df_annual
    # %%
    # Get the average values across the years 
    avg_AI = df_annual.AI.mean()
    return row_index, EASE_row_index, EASE_column_index, avg_AI

# The main function to setup and run the multiprocessing pool
def main():
        
    # %%
    # Read coordinate information
    file_path = os.path.join(data_dir, datarods_dir, "coord_info.csv")
    coord_info = pd.read_csv(file_path)
    coord_info.head()
    print('read coordinate info')

    # %%
    # Initialize AI column
    coord_info["AI"] = np.nan

    # Convert the DataFrame to a list of tuples (each tuple is a row)
    rows = [(index, row) for index, row in coord_info.iterrows()]
    
    # Setup the multiprocessing pool. You can specify the number of processes or
    # leave it empty to use os.cpu_count()
    pool = multiprocessing.Pool(processes=20)
    print('Start multiprocessing')
    # Use starmap to pass multiple arguments to the process_row function
    results = list(tqdm(pool.starmap(process_row, rows), total=len(rows)))
    print('End multiprocessing')

    # Close the pool to free up resources
    pool.close()
    pool.join()
    
    # Update the DataFrame with the results from the multiprocessing
    print('Start saving results')
    for row_index, EASE_row_index, EASE_column_index, avg_AI in results:
        coord_info.at[row_index, "AI"] = avg_AI
    print('End saving results')

    # Save the updated DataFrame
    filename = os.path.join(data_dir, datarods_dir, "AridityIndex_from_datarods.csv")
    coord_info.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
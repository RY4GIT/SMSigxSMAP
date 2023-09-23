# Script from https://github.com/maoyab/data_management_for_global_PIEP/blob/master/combine_L3_soil_moisture_data.py

import numpy as np
from pandas import date_range
import h5py
import os
import xarray as xr
from datetime import datetime
import calendar

drive = r'G:/Araki/SMSigxSMAP/1_data'


def get_filename_from_date_L3(date):
    hdf_folder = os.path.join(drive, 'SPL3SMP')
    d = os.path.join(hdf_folder, '%d.%02d.%02d' % (date.year, date.month, date.day))
    if os.path.exists(d):
        f = [fi for fi in os.listdir(d) if fi.endswith('.h5')][0]
        f = os.path.join(d, f)
        return f
    else:
        return None


def sm_day_ds(day, x_s_flags_i=[], x_qr_flags_i=[]):
    v_am = ['/Soil_Moisture_Retrieval_Data_AM/soil_moisture',
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
            '/Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag']
    v_pm = ['/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
            '/Soil_Moisture_Retrieval_Data_PM/retrieval_qual_flag_pm']
    f_i = get_filename_from_date_L3(day)
    data = []
    if f_i is not None:
        with h5py.File(f_i, mode='r') as f:
            for [v_name, s_flag_name, qr_flag_name] in [v_am, v_pm]:
                data_i = f[v_name][:]
                s_flag = f[s_flag_name][:]
                qr_flag = f[qr_flag_name][:]
                _FillValue = f[v_name].attrs['_FillValue']
                valid_max = f[v_name].attrs['valid_max']
                valid_min = f[v_name].attrs['valid_min']

                x_s_i_list = []
                for vv in np.unique(s_flag):
                    for fi in x_s_flags_i:
                        if (str('{0:016b}'.format(vv))[-fi- 1] == '1'):
                            x_s_i_list.append(vv)
                x_s_i_list = np.unique(x_s_i_list)

                x_qr_i_list = []
                for vv in np.unique(qr_flag):
                    for fi in x_qr_flags_i:
                        if (str('{0:016b}'.format(vv))[-fi - 1] == '1'):
                            x_qr_i_list.append(vv)
                x_qr_i_list = np.unique(x_qr_i_list)

                invalid = np.logical_or(data_i > valid_max,
                                        data_i < valid_min)
                invalid = np.logical_or(invalid, data_i == _FillValue)

                for ix in x_s_i_list:
                    invalid = np.logical_or(invalid, s_flag == ix)
                for ix in x_qr_i_list:
                    invalid = np.logical_or(invalid, qr_flag == ix)

                data_i[invalid] = np.nan
                data_i = np.ma.masked_where(np.isnan(data_i), data_i)

                data.append(data_i)
        time = [day, day]
        rows_rs = range(np.shape(data_i)[0])
        cols_rs = range(np.shape(data_i)[1])
        ds = xr.Dataset({'soil_moisture': (['time', 'y', 'x'], data)},
                          coords={'row': rows_rs, 'col': cols_rs, 'time': time})
        ds = ds.resample(time='D').mean()
    else:
        print ('x', day)
        data = np.zeros((1, 406, 964))
        data[data==0] = np.nan
        ds = xr.Dataset({'soil_moisture': (['time', 'y', 'x'], data)},
                  coords={'row': range(406), 'col': range(964), 'time': [day, ]})
    return ds


if __name__ == "__main__":

    output_dir = r'SPL3SMP_processed'
    if ~os.path.exists(output_dir):
        os.makedirs(output_dir)
        
# makes sm daily nc and filter by quality / surface flags combine am + pm retrievals ...............................

    # days = date_range('2015-01-01', '2019-12-31', freq='1D')
    # for day in days:
    #     ds = sm_day_ds(day, x_s_flags_i=[], x_qr_flags_i=[0, ])
    #     nc_filename_out = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d%02d%02d.nc' % (day.year, day.month, day.day))
    #     ds.to_netcdf(nc_filename_out)


# combine sm daily ts by month nc files ............................................................

    for year in [2015, 2016, 2017, 2018, 2019]:
        for month in range(1, 13):
            print(year, month)
            n_days = calendar.monthrange(year, month)[1]
            days = date_range('%d-%02d-01' % (year, month), '%d-%02d-%02d' % (year, month, n_days), freq='1D')
            ds = []
            for day in days:
                f = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d%02d%02d.nc' \
                    % (day.year, day.month, day.day))
                dsi = xr.open_dataset(f)
                ds.append(dsi)
                dsi.close()


            ds = xr.merge(ds)
            f = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d%02d_m.nc' % (year, month))
            ds.to_netcdf(f)


# combine month nc files to year nc files............................................................

    for year in [2015, 2016, 2017, 2018, 2019]:
        ds = []
        for month in range(1, 13):
            print(year, month)
            f = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d%02d_m.nc' % (year, month))
            dsi = xr.open_dataset(f, engine='rasterio')
            ds.append(dsi)
            dsi.close()
        ds = xr.merge(ds)
        f = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d_y.nc' % (year))
        ds.to_netcdf(f)


# combine yearly nc files to all nc file ............................................................

    ds = []
    for year in [2015, 2016, 2017, 2018, 2019]:
        f = os.path.join(drive, output_dir, 'smapL3_daily_sm_%d_y.nc' % (year))
        dsi = xr.open_dataset(f)
        ds.append(dsi)
        dsi.close()

    ds = xr.merge(ds)
    f = os.path.join(drive, output_dir, 'smapL3_daily_sm_all.nc')
    ds.to_netcdf(f)


from netCDF4 import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt


def main(lat_var,lon_var):
	nc=Dataset('2020_daily_pet.nc')
	lats=nc.variables['latitude'][:]
	lons=nc.variables['longitude'][:]
	lati,loni=nearest_point(lat_var, lon_var, lats, lons)
	print(lati)
	print(loni)
	pet=nc.variables['pet'][:3,:,:]
	#maskval=pet.mask
	#maskval=np.concatenate((maskval[:1,:,1800:],maskval[:1,:,:1800]),axis=2)
	#lons=np.arange(0,360,0.1)
	#nc_write(maskval,lats,lons,'mask','days since 1981-01-01','mask.nc')
	mval1=pet[0,0,0]
	mval2=pet[0,5,0]
	print(mval1)
	print(mval2)
	#pet = np.ma.masked_where(pet == mval1, pet)
	#pet = np.ma.masked_where(pet == mval2, pet)
	plt.imshow(pet[1,:,:])
	plt.colorbar()
	plt.show()

	#print(pet)

def nc_write(data, lat, lon, varname, tunits, filename):
    """
    this function write the PET on a netCDF file.

    :param: data: data to be written (time,lat,lon)
    :param: lat: latitude
    :param: lon: longitude
    :param: varname: name of the variable to be written (e.g. 'pet')
    :param: tunits: time units for the data (e.g. 'days since 1981-01-01')
    :param:filename: the file name to write the values with .nc extension

    :return:  produce a netCDF file in the same directory.
    """
    
    ds = Dataset(filename, mode='w', format='NETCDF4_CLASSIC')

    time = ds.createDimension('time', None)
    latitude = ds.createDimension('latitude', len(lat))
    longitude = ds.createDimension('longitude', len(lon))
   
    time = ds.createVariable('time', np.float32, ('time',))
    latitude = ds.createVariable('latitude', np.float32, ('latitude',))
    longitude = ds.createVariable('longitude', np.float32, ('longitude',))

    # check if the data is 2d or 3d
    if len(data.shape) == 3: # 3D array
        pet_val = ds.createVariable(varname, 'f4', ('time','latitude','longitude'))
        time.units = tunits  
        time.calendar = 'proleptic_gregorian'
        time[:] = np.arange(data.shape[0])
        latitude[:] = lat
        longitude [:] = lon
        pet_val[:,:,:] = data
    
    elif len(data.shape) == 2: # 2D array
        pet_val = ds.createVariable(varname, 'f4', ('latitude','longitude'))
        latitude[:] = lat
        longitude [:] = lon
        pet_val[:,:] = data
    else:
        raise ValueError('the function can only write a 2D or 3D array data!')

    ds.close()
    
    return None    
# data=h5py.File('1982_erapet.hdf5','r')
# y=data['pev'][:,805,3592]

# plt.plot(y[:72])
# plt.show()

def nearest_point(lat_var, lon_var, lats, lons):
    """
    This function identify the nearest grid location index for a specific lat-lon
    point.
    :param lat_var: the latitude
    :param lon_var: the longitude
    :param lats: all available latitude locations in the data
    :param lons: all available longitude locations in the data
    :return: the lat_index and lon_index
    """
    # this part is to handle if lons are givn 0-360 or -180-180
    if any(lons > 180.0) and (lon_var < 0.0):
        lon_var = lon_var + 360.0
    else:
        lon_var = lon_var
        
    lat = lats
    lon = lons

    if lat.ndim == 2:
        lat = lat[:, 0]
    else:
        pass
    if lon.ndim == 2:
        lon = lon[0, :]
    else:
        pass

    index_a = np.where(lat >= lat_var)[0][-1]
    index_b = np.where(lat <= lat_var)[0][-1]

    if abs(lat[index_a] - lat_var) >= abs(lat[index_b] - lat_var):
        index_lat = index_b
    else:
        index_lat = index_a

    index_a = np.where(lon >= lon_var)[0][0]
    index_b = np.where(lon <= lon_var)[0][0]
    if abs(lon[index_a] - lon_var) >= abs(lon[index_b] - lon_var):
        index_lon = index_b
    else:
        index_lon = index_a

    return index_lat, index_lon

def check_pet():
	years=np.arange(1981,2021)
	for year in years:
		command="ncdump -h %s"%(str(year)+'_hourly_pet.nc')
		os.system(command)	

if __name__ == '__main__':
	check_pet()	
#main(1.0,35.0)

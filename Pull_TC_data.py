from __future__ import division  # makes division not round with integers 
import os
import pygrib
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
from datetime import datetime, timedelta
from skimage.feature import peak_local_max
#matplotlib.use('pdf')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib.patches as mpatches
#import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
#import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import gaussian_filter
import ctypes
import numpy.ctypeslib as ctl

# Pull_data.py contains functions that are used by AEW_Tracks.py. All of the functions in Pull_Data.py aid in the pulling of variables and formatting
# them to be in a common format so AEW_Tracks.py does not need to know what model the data came from.

# This function takes a common_object and assigns the lat, lon, lat/lon indices over Africa/the Atlantic, and dt value based on the data source
# which is the model attribute of the common_object. 
def get_common_track_data(common_object):
	# box of interest over the Atlantic/Africa (values from Chris Patricola)
	north_lat = 50.5
	south_lat = 3.5
	west_lon = -124.5
	east_lon = 45.5 

	if common_object.model == 'WRF':
		dt = 6 # time between history output files (e.g. is the data 6 hourly, 3 hourly, etc.)
		# get the latitude and longitude and the north, south, east, and west indices of a rectangle over Africa and the Atlantic 
		file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/Historical/wrfout_d01_2008-07-01_00_00_00'
		data = Dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat = wrf.getvar(data, "lat", meta=False) # ordered lat, lon
		lon = wrf.getvar(data, "lon", meta=False) # ordered lat, lon
		#print(lat)
		#print(lon)
		#print(lat.shape)
		#print(lon.shape)
		# get north, south, east, west indices
		lon_index_west, lat_index_south = wrf.ll_to_xy(data,south_lat,west_lon, meta=False) 
		lon_index_east, lat_index_north = wrf.ll_to_xy(data,north_lat,east_lon, meta=False) 
#		print(lon_index_west)
#		print(lon_index_east)
		#lat_crop = lat.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		#lon_crop = lon.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		# the following two lines are to correct for the weird negative indexing that comes back from the wrf.ll_to_xy function
#		lon_index_west = lon.shape[1] + lon_index_west # this stopped being necessary 
		lon_index_east = lon.shape[1] + lon_index_east

		# the total number of degrees in the longitude dimension; this needs to be changed if not using the WRF TCM
		lon_degrees = 360.
		
	elif common_object.model == 'ERA5':
		# dt for ERA5 is 1 hour (data is hourly), but set dt to whatever the dt is for the dataset to be compared with
		# Eg dt=3 to compare with CAM5 or dt=6 to compare with WRF
		dt = 6 # time between files
		file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/200509/e5.oper.an.pl.128_131_u.ll025uv.2005090100_2005090123.nc'
		data = xr.open_dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat_1d_n_s = data.latitude.values # ordered lat, and going from north to south (so 90, 89, 88, .....-88, -89, -90)
		lon_1d_360 = data.longitude.values # ordered lon and goes from 0-360 degrees
		# make the lat array go from south to north 
		lat_1d = np.flip(lat_1d_n_s)
		# make the longitude go from -180 to 180 degrees
		lon_1d = np.array([x - 180.0 for x in lon_1d_360])

		# get north, south, east, west indices for cropping 
		lat_index_north_crop = (np.abs(lat_1d - north_lat_crop)).argmin()
		lat_index_south_crop = (np.abs(lat_1d - south_lat_crop)).argmin()
		lon_index_west_crop = (np.abs(lon_1d - west_lon_crop)).argmin()
		lon_index_east_crop = (np.abs(lon_1d - east_lon_crop)).argmin()

		# set the lat and lon cropping indices in common_object
		common_object.lat_index_north_crop = lat_index_north_crop
		common_object.lat_index_south_crop = lat_index_south_crop
		common_object.lon_index_east_crop = lon_index_east_crop
		common_object.lon_index_west_crop = lon_index_west_crop

		# crop the lat and lon arrays. We don't need the entire global dataset
		lat_1d_crop = lat_1d[lat_index_south_crop:lat_index_north_crop+1]
		lon_1d_crop = lon_1d[lon_index_west_crop:lon_index_east_crop+1]

		# make the lat and lon arrays from the GCM 2D (ordered lat, lon)
		lon = np.tile(lon_1d_crop, (lat_1d_crop.shape[0],1))
		lat_2d = np.tile(lat_1d_crop, (len(lon_1d_crop),1))
		lat = np.rot90(lat_2d,3)
		# switch lat and lon arrays to float32 instead of float64
		lat = np.float32(lat)
		lon = np.float32(lon)
		# make lat and lon arrays C continguous 
		lat = np.asarray(lat, order='C')
		lon = np.asarray(lon, order='C')

		# get north, south, east, west indices for tracking
		lat_index_north = (np.abs(lat_1d_crop - north_lat)).argmin()
		lat_index_south = (np.abs(lat_1d_crop - south_lat)).argmin()
		lon_index_west = (np.abs(lon_1d_crop - west_lon)).argmin()
		lon_index_east = (np.abs(lon_1d_crop - east_lon)).argmin()

		# the total number of degrees in the longitude dimension
		lon_degrees = np.abs(lon[0,0] - lon[0,-1])

	# set dt in the common_object
	common_object.dt = dt
	# set lat and lon in common_object
	common_object.lat = lat # switch from float64 to float32
	common_object.lon = lon # switch from float64 to float32
	# set the lat and lon indices in common_object
	common_object.lat_index_north = lat_index_north
	common_object.lat_index_south = lat_index_south
	common_object.lon_index_east = lon_index_east
	common_object.lon_index_west = lon_index_west
	# set the total number of degrees longitude in common_object
	common_object.total_lon_degrees = lon_degrees
	print(common_object.total_lon_degrees)
	return

# This is a function to get the WRF variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_WRF_variables(common_object, scenario_type, date_time): #, lon_index_west, lat_index_south, lon_index_east, lat_index_north):
	# location of WRF file
	file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/' + scenario_type + '/' + date_time.strftime('%Y') + '/wrfout_d01_'
	# open file
	data = Dataset(file_location + date_time.strftime("%Y-%m-%d_%H_%M_%S"))
	# get u, v, and p
	print("Pulling variables...")
	p_3d = wrf.getvar(data, 'pressure') # pressure in hPa
	u_3d = wrf.getvar(data, 'ua') # zonal wind in m/s
	v_3d = wrf.getvar(data, 'va') # meridional wind in m/s
	t_3d = wrf.getvar(data, 'tk') # temperature in K
	wspd_10m = wrf.getvar(data, 'wspd_wdir10')[0] # 10 m windspeed in m/s
	slp = wrf.getvar(data, 'slp') # sea level pressure in hPa

	# get u and v at the pressure levels 850, and 300 hPa
	u_levels = calc_var_pres_levels(p_3d, u_3d, [850., 600., 300.])
	v_levels = calc_var_pres_levels(p_3d, v_3d, [850., 600., 300.])
	# get t at the pressure levels 850, 700, 500, and 300 hPa
	t_levels = calc_var_pres_levels(p_3d, t_3d, [850., 700., 500., 300.])

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels.values,v_levels.values,common_object.lat,common_object.lon)
	

	return u_levels.values, v_levels.values, t_levels.values, rel_vort_levels, wspd_10m.values, slp.values

# This function interpolates WRF variables to specific pressure levels
# This function takes the pressure and the variable to be interpolated and the 
# pressure levels to be interpolated to.
# The fucntion returns a three dimensional array ordered lev (pressure), lat, lon
def calc_var_pres_levels(p, var, pressure_levels):
	# interpolate the variable to the above pressure levels
	# returns an array with the lev dim the length of pressure_levels
	var_levels = wrf.interplevel(var, p, pressure_levels)
	# get rid of any nans
	# linearly interpolate the missing values
#	print("Any nans?")
#	print(np.isnan(var_levels).any())
	mask = np.isnan(var_levels.values)
	var_levels.values[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), var_levels.values[~mask]) 
#	print("Any nans now?")
#	print(np.isnan(var_levels).any())

	return var_levels

# This is a function to get the ERA5 variables required for tracking
# The function takes the common_object that holds lat/lon information, the scenario type, and the date and time for the desired file
# The function returns u, v, relative vorticity, and curvature vorticity on specific pressure levels
def get_ERA5_variables(common_object,date_time):
	# location of ERA5 files
	u_file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/' + date_time.strftime("%Y%m") + '/e5.oper.an.pl.128_131_u.ll025uv.'
	v_file_location = '/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.an.pl/' + date_time.strftime("%Y%m") + '/e5.oper.an.pl.128_132_v.ll025uv.'
	# open files 
	u_data = xr.open_dataset(u_file_location + date_time.strftime("%Y%m%d") + '00_' + date_time.strftime("%Y%m%d") + '23.nc')
	v_data = xr.open_dataset(v_file_location + date_time.strftime("%Y%m%d") + '00_' + date_time.strftime("%Y%m%d") + '23.nc')
	# get u and v
	print("Pulling variables...")
	# ERA5 is hourly. To compare with CAM5, which is 3 hourly, average every three hours together (eg 00, 01, and 02 are averaged)
	# to make the comparison easier. To compare with WRF, which is 6 hourly, average every six hours. The averaging is controlled by
	# what is set at dt in the common_object (eg dt=3 for CAM5 or dt=6 for WRF)
	u_3d = u_data.U[int(date_time.strftime("%H")),:,:,:]
	v_3d = v_data.V[int(date_time.strftime("%H")),:,:,:]
	# get u and v only on the levels 850, 700, and 600 hPa
	lev_list = [850, 700, 600]
	u_levels_360 = np.zeros([3,u_3d.shape[1],u_3d.shape[2]])
	v_levels_360 = np.zeros([3,v_3d.shape[1],v_3d.shape[2]])
	for level_index in range(0,3):
		# the ERA5 data goes from north to south. Use flip to flip it 180 degrees in the latitude dimension so that
		# the array now goes from south to north like the other datasets.
		u_levels_360[level_index,:,:] = np.flip(u_3d.sel(level=lev_list[level_index]),axis=0)
		v_levels_360[level_index,:,:] = np.flip(v_3d.sel(level=lev_list[level_index]),axis=0)
	# need to roll the u and v variables on the longitude axis because the longitudes were changed from 
	# 0-360 to -180 to 180
	u_levels_full = np.roll(u_levels_360, int(u_levels_360.shape[2]/2), axis=2)
	v_levels_full = np.roll(v_levels_360, int(v_levels_360.shape[2]/2), axis=2)

	# Crop the data. This is a global dataset and we don't need to calculate vorticity values over the entire globe, only over the region of interest. 
	# The tracking algorithm only looks over Africa/the Atlantic, so it's unnecessary to have a global dataset.
	u_levels = u_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	v_levels = v_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]

	# get rid of any NANs
	if np.isnan(u_levels).any():
		mask_u = np.isnan(u_levels)
		u_levels[mask_u] = np.interp(np.flatnonzero(mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u]) 
	if np.isnan(v_levels).any():
		mask_v = np.isnan(v_levels)
		v_levels[mask_v] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v]) 

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)
	# calculate the curvature voriticty
	curve_vort_levels = calc_curve_vort(common_object,u_levels,v_levels,rel_vort_levels)

	# switch the arrays to be float32 instead of float64
	u_levels = np.float32(u_levels)
	v_levels = np.float32(v_levels)
	rel_vort_levels = np.float32(rel_vort_levels)
	curve_vort_levels = np.float32(curve_vort_levels)

	# make the arrays C contiguous (will need this later for the wrapped C smoothing function)
	u_levels = np.asarray(u_levels, order='C')
	v_levels = np.asarray(v_levels, order='C')
	rel_vort_levels = np.asarray(rel_vort_levels, order='C')
	curve_vort_levels = np.asarray(curve_vort_levels, order='C')

	return u_levels, v_levels, rel_vort_levels, curve_vort_levels


# Calculate and return the relative vorticity. Relative vorticity is defined as
# rel vort = dv/dx - du/dy. This function takes the 3-dimensional variables u and v,
# ordered (lev, lat, lon), and the 2-dimensional variables latitude and longitude, ordered
# (lat, lon), as parameters and returns the relative voriticty.
def calc_rel_vort(u,v,lat,lon):
	# take derivatives of u and v
	dv_dx = x_derivative(v, lat, lon)
	du_dy = y_derivative(u, lat)
	# subtract derivatives to calculate relative vorticity
	rel_vort = dv_dx - du_dy
#	print("rel_vort shape =", rel_vort.shape)
	return rel_vort
	
# This function takes the derivative with respect to x (longitude).
# The function takes a three-dimensional variable ordered lev, lat, lon
# and returns d/dx of the variable.
def x_derivative(variable, lat, lon):
	# subtract neighboring longitude points to get delta lon
	# then switch to radians
	dlon = np.radians(lon[0,2]-lon[0,1])
#	print("dlon =", dlon)
	# allocate space for d/dx array
	d_dx = np.zeros_like(variable)
#	print(d_dx.shape)
	# loop through latitudes
	for nlat in range(0,len(lat[:,0])):
			# calculate dx by multiplying dlon by the radius of the Earth, 6367500 m, and the cos of the lat
			dx = 6367500.0*np.cos(np.radians(lat[nlat,0]))*dlon  # constant at this latitude
			#print dx
			# the variable will have dimensions lev, lat, lon
			grad = np.gradient(variable[:,nlat,:], dx)
			d_dx[:,nlat,:] = grad[1]
#	print(d_dx.shape)
	return d_dx

# This function takes the derivative with respect to y (latitude).
# The function takes a 3 dimensional variable (lev, lat, lon) and
# returns d/dy of the variable.
def y_derivative(variable,lat):
	# subtract neighboring latitude points to get delta lat
	# then switch to radians
	dlat = np.radians(lat[2,0]-lat[1,0])
#	print("dlat =", dlat)
	# calculate dy by multiplying dlat by the radius of the Earth, 6367500 m
	dy = 6367500.0*dlat
#	print("dy =", dy)
	# calculate the d/dy derivative using the gradient function
	# the gradient function will return a list of arrays of the same dimensions as the
	# WRF variable, where each array is a derivative with respect to one of the dimensions
	d_dy = np.gradient(variable, dy)
	#print d_dy.shape
	#print d_dy[1].shape
	# return the second item in the list, which is the d/dy array
	return d_dy[1]


# This function is called from AEW_Tracks.py and is what begins the process of acquiring u, v, relative vorticity, and curvature vorticity
# at various pressure levels. The function takes the common_object, the scenario type and date and time and returns the previoiusly mentioned variables.
def get_variables(common_object, scenario_type, date_time):
	if common_object.model == 'WRF':
		u_levels, v_levels, t_levels, rel_vort_levels, wspd_10m, slp = get_WRF_variables(common_object,scenario_type,date_time)
	elif common_object.model == 'ERA5':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_ERA5_variables(common_object,date_time)
	

	return u_levels, v_levels, t_levels, rel_vort_levels, wspd_10m, slp


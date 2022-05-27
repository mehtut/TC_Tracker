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
def get_common_track_data(common_object, highresmip_models):
	# box of interest over the Atlantic/Africa (values from Chris Patricola)
	north_lat = 50.5
	south_lat = 3.5
	west_lon = -124.5
	east_lon = 45.5 

	# lat/lon values to crop data to speed up vorticity calculations
	north_lat_crop = 60. 
	south_lat_crop = -10.
	west_lon_crop = -140.
	east_lon_crop = 50.

	if common_object.model == 'WRF':
		dt = 3 # time between history outputs (e.g. is the data 6 hourly, 3 hourly, etc.)
		# get the latitude and longitude and the north, south, east, and west indices of a rectangle over Africa and the Atlantic 
		file_location = '/global/cscratch1/sd/ebercosh/AEW_Suppression/2003/WRF/control/E0_0515/wrf/wrfout_d01_2003-11-30_00_00_00'
#		file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/Historical/wrfout_d01_2008-07-01_00_00_00'
		data = Dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat = wrf.getvar(data, "lat", meta=False) # ordered lat, lon
		lon = wrf.getvar(data, "lon", meta=False) # ordered lat, lon
		# check to see if the box values (north_lat, etc.) exceed the domain of the WRF data
		if np.amax(lat) < north_lat:
			north_lat = np.amax(lat)
		if np.amin(lat) > south_lat:
			south_lat = np.amin(lat)
		if np.amax(lon) < east_lon:
			east_lon = np.amax(lon)
		if np.amin(lon) > west_lon:
			west_lon = np.amin(lon)
		print(north_lat, south_lat, east_lon, west_lon)
		#print(lat)
		#print(lon)
		# print(lat.shape)
		# print(lon.shape)
		# get north, south, east, west indices
		lon_index_west, lat_index_south = wrf.ll_to_xy(data,south_lat,west_lon, meta=False) 
		lon_index_east, lat_index_north = wrf.ll_to_xy(data,north_lat,east_lon, meta=False) 
#		print(lon_index_west)
#		print(lon_index_east)
		#lat_crop = lat.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		#lon_crop = lon.values[lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]
		# the following two lines are to correct for the weird negative indexing that comes back from the wrf.ll_to_xy function
#		lon_index_west = lon.shape[1] + lon_index_west # this stopped being necessary 
#		lon_index_east = lon.shape[1] + lon_index_east # only for TCM runs

		# the total number of degrees in the longitude dimension; this needs to be changed if not using the WRF TCM
#		lon_degrees = 360. # for TCM
		lon_degrees = np.abs(np.amin(lon) - np.amax(lon))
		
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

	elif common_object.model in highresmip_models:
		dt = 6 # time between files
		u_dir_location = create_full_highresmip_path(common_object,'ua') # get the directory for the current HighResMIP model and scenario
		file_location = os.path.join(u_dir_location, os.listdir(u_dir_location)[0]) # take the first file (just need one file to get the lat/lon information)
#		model_dir_loc_dic = {'CNRM-CM6-1-HR':'CNRM-CERFACS/CNRM-CM6-1-HR/highresSST-present/r1i1p1f2/6hrPlevPt/ua/gr/v20190311/ua_6hrPlevPt_CNRM-CM6-1-HR_highresSST-present_r1i1p1f2_gr_200607010600-200701010000.nc'}
#		file_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/' + model_dir_loc_dic[common_object.model]
#		file_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/ECMWF/ECMWF-IFS-HR/hist-1950/r5i1p1f1/6hrPlevPt/ua/gr/v20190417/ua_6hrPlevPt_ECMWF-IFS-HR_hist-1950_r5i1p1f1_gr_199002010000-199002281800.nc'
		data = xr.open_dataset(file_location)
		# get lat and lon values
		# get the latitude and longitude at a single time (since they don't change with time)
		lat_1d = data.lat.values # ordered lat, and going from south to north (so -90, -89, -88, .....88, 89, 90)
		lon_1d_360 = data.lon.values # ordered lon and goes from 0-360 degrees
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
def get_WRF_variables(common_object, scenario_type, ensemble_number, date_time): #, lon_index_west, lat_index_south, lon_index_east, lat_index_north):
	# location of WRF file
	file_location = '/global/cscratch1/sd/ebercosh/AEW_Suppression/' + date_time.strftime('%Y') + '/WRF/' + scenario_type + '/' + ensemble_number + '/wrf/wrfout_d01_'
#	file_location = '/global/cscratch1/sd/ebercosh/WRF_TCM/' + scenario_type + '/' + date_time.strftime('%Y') + '/wrfout_d01_'
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

def get_highresmip_variables(common_object,date_time):
	print("Pulling variables...")
	# The HighResMIP data is not written out in six (or three) hourly files, but rather in months of data. The u and v for the entire year have already been
	# pulled and stored in the common object. Use the current time to get the correct value from the u and v arrays.
	if common_object.model == 'CNRM-CM6-1-HR' or common_object.model == 'EC-Earth3P-HR':
		u_3d = common_object.highres_u.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
		v_3d = common_object.highres_v.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
		t_3d = common_object.highres_t.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
		uas_2d = common_object.highres_uas.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
		vas_2d = common_object.highres_vas.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
		psl_2d = common_object.highres_psl.sel(time=date_time.strftime("%Y-%m-%d")+'T'+date_time.strftime("%H"))
	elif common_object.model == 'CMCC-CM2-VHR4':
		u_3d = common_object.highres_u.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
		v_3d = common_object.highres_v.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
		t_3d = common_object.highres_t.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
		uas_2d = common_object.highres_uas.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
		vas_2d = common_object.highres_vas.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
		psl_2d = common_object.highres_psl.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/6)]
	elif common_object.model == 'HadGEM3-GC31-HM':
		u_3d = common_object.highres_u.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours
		v_3d = common_object.highres_v.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours
		t_3d = common_object.highres_t.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours
		uas_2d = common_object.highres_uas.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours
		vas_2d = common_object.highres_vas.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours
		psl_2d = common_object.highres_psl.sel(time=date_time.strftime("%Y-%m-%d"))[int(int(date_time.strftime("%H"))/3)] # data is every three hours

	# get u and v only on the levels 850., 600., 300. hPa
	lev_list = [85000, 60000, 30000] # in Pa 
	t_lev_list = [850., 700., 500., 300.] # in Pa, for temperature
	u_levels_360 = np.zeros([3,u_3d.shape[1],u_3d.shape[2]])
	v_levels_360 = np.zeros([3,v_3d.shape[1],v_3d.shape[2]])
	t_levels_360 = np.zeros([4,t_3d.shape[1],t_3d.shape[2]])
	for level_index in range(0,3):
		u_levels_360[level_index,:,:] = u_3d.sel(plev=lev_list[level_index])
		v_levels_360[level_index,:,:] = v_3d.sel(plev=lev_list[level_index])
	for level_index in range(0,4):
		t_levels_360[level_index,:,:] = t_3d.sel(plev=t_lev_list[level_index])
	# need to roll the u and v variables on the longitude axis because the longitudes were changed from 
	# 0-360 to -180 to 180
	u_levels_full = np.roll(u_levels_360, int(u_levels_360.shape[2]/2), axis=2)
	v_levels_full = np.roll(v_levels_360, int(v_levels_360.shape[2]/2), axis=2)
	t_levels_full = np.roll(t_levels_360, int(t_levels_360.shape[2]/2), axis=2)
	uas_full = np.roll(uas_2d, int(uas_2d.shape[1]/2), axis=1)
	vas_full = np.roll(vas_2d, int(vas_2d.shape[1]/2), axis=1)
	psl_full = np.roll(psl_2d, int(psl_2d.shape[1]/2), axis=1)

	# Crop the data. This is a global dataset and we don't need to calculate vorticity values over the entire globe, only over the region of interest. 
	# The tracking algorithm only looks over Africa/the Atlantic, so it's unnecessary to have a global dataset.
	u_levels = u_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	v_levels = v_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	t_levels = t_levels_full[:,common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	uas = uas_full[common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	vas = vas_full[common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]
	psl = psl_full[common_object.lat_index_south_crop:common_object.lat_index_north_crop+1,common_object.lon_index_west_crop:common_object.lon_index_east_crop+1]

	# get rid of any NANs
	if np.isnan(u_levels).any():
		mask_u = np.isnan(u_levels)
		u_levels[mask_u] = np.interp(np.flatnonzero(mask_u), np.flatnonzero(~mask_u), u_levels[~mask_u]) 
	if np.isnan(v_levels).any():
		mask_v = np.isnan(v_levels)
		v_levels[mask_v] = np.interp(np.flatnonzero(mask_v), np.flatnonzero(~mask_v), v_levels[~mask_v]) 
	if np.isnan(t_levels).any():
		mask_t = np.isnan(t_levels)
		t_levels[mask_t] = np.interp(np.flatnonzero(mask_t), np.flatnonzero(~mask_t), t_levels[~mask_t]) 
	if np.isnan(uas).any():
		mask_uas = np.isnan(uas)
		uas[mask_uas] = np.interp(np.flatnonzero(mask_uas), np.flatnonzero(~mask_uas), uas[~mask_uas]) 
	if np.isnan(vas).any():
		mask_vas = np.isnan(vas)
		vas[mask_vas] = np.interp(np.flatnonzero(mask_vas), np.flatnonzero(~mask_vas), vas[~mask_vas]) 
	if np.isnan(psl).any():
		mask_psl = np.isnan(psl)
		psl[mask_psl] = np.interp(np.flatnonzero(mask_psl), np.flatnonzero(~mask_psl), psl[~mask_psl]) 

	# calculate the relative vorticity
	rel_vort_levels = calc_rel_vort(u_levels,v_levels,common_object.lat,common_object.lon)

	# calculate the surface windspeed from uas and vas
	windspeed = np.sqrt(np.square(uas) + np.square(vas))


	return u_levels, v_levels, t_levels, rel_vort_levels, windspeed, psl

def get_highresmip_data(year,common_object,times):
	# model_dir_info_dic = {\
	# 	'CNRM-CM6-1-HR':\
	# 	['CNRM-CERFACS/','/r1i1p1f2/6hrPlevPt/','/gr/v20190221/'],\
	# 	'ECMWF-IFS-HR':\
	# 	['ECMWF/']}

#	model_dir_info_dic = {'CNRM-CM6-1-HR':['CNRM-CERFACS/','/r1i1p1f2/6hrPlevPt/','/gr/' + + '/'],'ECMWF-IFS-HR':'ECMWF/'}
	u_dir_location = create_full_highresmip_path(common_object,'ua')
	v_dir_location = create_full_highresmip_path(common_object,'va')
	t_dir_location = create_full_highresmip_path(common_object,'ta') # units in K
	uas_dir_location = create_full_highresmip_path(common_object,'uas')
	vas_dir_location = create_full_highresmip_path(common_object,'vas')
	psl_dir_location = create_full_highresmip_path(common_object,'psl') # sea level pressure in Pa

	
#	u_dir_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/' + model_dir_info_dic[common_object.model][0] + common_object.model + '/' + common_object.scenario + model_dir_info_dic[common_object.model][1] + 'ua' + model_dir_info_dic[common_object.model][2]
#	v_dir_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/' + model_dir_info_dic[common_object.model][0] + common_object.model + '/' + common_object.scenario + model_dir_info_dic[common_object.model][1] + 'va' + model_dir_info_dic[common_object.model][2]
#	u_dir_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/' + 'ECMWF/ECMWF-IFS-HR/hist-1950/r5i1p1f1/6hrPlevPt/ua/gr/v20190417/'
#	v_dir_location = '/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/' + 'ECMWF/ECMWF-IFS-HR/hist-1950/r5i1p1f1/6hrPlevPt/va/gr/v20190417/'
	u_file_list = []
	v_file_list = []
	t_file_list = []
	uas_file_list = []
	vas_file_list = []
	psl_file_list = []
	# combine the year and the month (e.g. 201005)
	# check the first time in times and see what month it is to determine which months to combine with the year
	# if times[0].strftime('%m') == '05': # May-October 
	# 	year_month_list = [year+'05',year+'06',year+'07',year+'08',year+'09',year+'10']
	# elif times[0].strftime('%m') == '06': # June-November
	# 	year_month_list = [year+'06',year+'07',year+'08',year+'09',year+'10',year+'11']
	year_month_list = [year+'05',year+'06',year+'07',year+'08',year+'09',year+'10',year+'11'] # May-November
	# get all the u files for the given year
	for root, dirs, files in os.walk(u_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				u_file_list.append(file)
	# get all the v files for the given year
	for root, dirs, files in os.walk(v_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				v_file_list.append(file)
	# get all the t files for the given year
	for root, dirs, files in os.walk(t_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				t_file_list.append(file)
	# get all the uas files for the given year
	for root, dirs, files in os.walk(uas_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				uas_file_list.append(file)
	# get all the vas files for the given year
	for root, dirs, files in os.walk(vas_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				vas_file_list.append(file)
	# get all the psl files for the given year
	for root, dirs, files in os.walk(psl_dir_location):
		for file in files:
			if any(substring in file for substring in year_month_list) or (year+'0101' in file and year+'1231' in file):
				#print(file)
				psl_file_list.append(file)
	# The list of files in u/v_file_list will likely not be sorted chronologically, so sort them by date and time
	# The split function takes just part of the file name string (e.g. '2512') and
	# the sorted function sorted based on numbers (so 1, 3, 2 becomes 1, 2, 3).
	u_sorted_list = sorted(list(set(u_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del u_file_list
	v_sorted_list = sorted(list(set(v_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del v_file_list
	t_sorted_list = sorted(list(set(t_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del t_file_list
	uas_sorted_list = sorted(list(set(uas_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del uas_file_list
	vas_sorted_list = sorted(list(set(vas_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del vas_file_list
	psl_sorted_list = sorted(list(set(psl_file_list)), key=lambda x: x.split('00-')[-1].split('.nc')[0])
	del psl_file_list
	# Open the files as xarrays and then concatenate them all together. The sorted files need to be attached to their locations.
	u_data_list = []
	for sorted_file in u_sorted_list:
		u_data_list.append(xr.open_dataset(os.path.join(u_dir_location, sorted_file))) # need to include the directory location for the files
	del u_sorted_list
	v_data_list = []
	for sorted_file in v_sorted_list:
		v_data_list.append(xr.open_dataset(os.path.join(v_dir_location, sorted_file))) # need to include the directory location for the files
	del v_sorted_list
	t_data_list = []
	for sorted_file in t_sorted_list:
		t_data_list.append(xr.open_dataset(os.path.join(t_dir_location, sorted_file))) # need to include the directory location for the files
	del t_sorted_list
	uas_data_list = []
	for sorted_file in uas_sorted_list:
		uas_data_list.append(xr.open_dataset(os.path.join(uas_dir_location, sorted_file))) # need to include the directory location for the files
	del uas_sorted_list
	vas_data_list = []
	for sorted_file in vas_sorted_list:
		vas_data_list.append(xr.open_dataset(os.path.join(vas_dir_location, sorted_file))) # need to include the directory location for the files
	del vas_sorted_list
	psl_data_list = []
	for sorted_file in psl_sorted_list:
		psl_data_list.append(xr.open_dataset(os.path.join(psl_dir_location, sorted_file))) # need to include the directory location for the files
	del psl_sorted_list
	# concatenate files
	u_dataset = xr.concat(u_data_list, dim='time')
	v_dataset = xr.concat(v_data_list, dim='time')
	t_dataset = xr.concat(t_data_list, dim='time')
	uas_dataset = xr.concat(uas_data_list, dim='time')
	vas_dataset = xr.concat(vas_data_list, dim='time')
	psl_dataset = xr.concat(psl_data_list, dim='time')
	# get u and v
	u_4d = u_dataset.ua
	v_4d = v_dataset.va
	t_4d = t_dataset.ta
	uas_3d = uas_dataset.uas
	vas_3d = vas_dataset.vas
	psl_3d = psl_dataset.psl
	return u_4d, v_4d, t_4d, uas_3d, vas_3d, psl_3d


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


# get HighResMIP directory paths
def create_full_highresmip_path(common_object,variable_type):
	# stem directory of all the HighResMIP data
	dir_stem = "/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/"
	# dictionary to build the directory paths
	model_dict = {
	"CNRM-CM6-1-HR": {
		"base_dir": "CNRM-CERFACS",
		"output_interval": "6hrPlevPt",
		"grid_type": "gr",
		"scenario": {
			"hist-1950": {
				"ensemble": "r1i1p1f2",
				"version": "v20190221"
			},
			"highresSST-present": {
				"ensemble": "r1i1p1f2",
				"version": "v20190311"
			},
			"highres-future": {
				"ensemble": "r1i1p1f2",
				"version": "v20190920"
			},
			"highresSST-future": {
				"ensemble": "r1i1p1f2",
				"version": "v20190903"
			}
		}
	},
	"EC-Earth3P-HR": {
		"base_dir": "EC-Earth-Consortium",
		"output_interval": "6hrPlevPt",
		"grid_type": "gr",
		"scenario": {
			"hist-1950": {
				"ensemble": "r1i1p2f1",
				"version": "v20181212"
			},
			"highresSST-present": {
				"ensemble": "r2i1p1f1",
				"version": "v20190514"
			},
			"highres-future": {
				"ensemble": "r1i1p2f1",
				"version": "v20190802"
			},
			"highresSST-future": {
				"ensemble": "r3i1p1f1",
				"version": "v20190713"
			}
		}
	},
	"HadGEM3-GC31-HM": {
		"base_dir": "MOHC",
		"output_interval": "E3hrPt",
		"grid_type": "gn",
		"scenario": {
			"hist-1950": {
				"ensemble": "r1i3p1f1",
				"version": "v20190710"
			},
			"highresSST-present": {
				"ensemble": "r1i3p1f1",
				"version": "v20180605"
			},
			"highres-future": {
				"ensemble": "r1i3p1f1",
				"version": "v20190902"
			},
			"highresSST-future": {
				"ensemble": "r1i3p1f1",
				"version": "v20190710"
			}
		}
	},
	"CMCC-CM2-VHR4": {
		"base_dir": "CMCC",
		"output_interval": "6hrPlevPt",
		"grid_type": "gn",
		"scenario": {
			"hist-1950": {
				"ensemble": "r1i1p1f1",
				"version": "v20180705"
			},
			"highresSST-present": {
				"ensemble": "r1i1p1f1",
				"version": "v20170927"
			},
			"highres-future": {
				"ensemble": "r1i1p1f1",
				"version": "v20190509"
			},
			"highresSST-future": {
				"ensemble": "r1i1p1f1",
				"version": "v20190725"
			}
		}
	},
	"CMCC-CM2-HR4": {
		"base_dir": "CMCC",
		"output_interval": "6hrPlevPt",
		"grid_type": "gn",
		"scenario": {
			"hist-1950": {
				"ensemble": "r1i1p1f1",
				"version": "v20190105"
			},
			"highresSST-present": {
				"ensemble": "r1i1p1f1",
				"version": "v20170927"
			},
			"highres-future": {
				"ensemble": "r1i1p1f1",
				"version": "v20190509"
			},
			"highresSST-future": {
				"ensemble": "r1i1p1f1",
				"version": "v20190725"
			}
		}
	}
	}

	# use os.path.join to create directory location
	full_path = os.path.join(dir_stem,
							model_dict[common_object.model]["base_dir"],
							common_object.model,
							common_object.scenario,
							model_dict[common_object.model]["scenario"][common_object.scenario]["ensemble"],
							model_dict[common_object.model]["output_interval"],
							variable_type,
							model_dict[common_object.model]["grid_type"],
							model_dict[common_object.model]["scenario"][common_object.scenario]["version"])
#	print(full_path)
	return full_path


# This function is called from AEW_Tracks.py and is what begins the process of acquiring u, v, relative vorticity, and curvature vorticity
# at various pressure levels. The function takes the common_object, the scenario type and date and time and returns the previoiusly mentioned variables.
#def get_variables(common_object, scenario_type, ensemble_number, date_time, highresmip_models):
def get_variables(common_object, scenario_type, date_time, highresmip_models):
	if common_object.model == 'WRF':
		u_levels, v_levels, t_levels, rel_vort_levels, wspd_10m, slp = get_WRF_variables(common_object,scenario_type,ensemble_number,date_time)
	elif common_object.model == 'ERA5':
		u_levels, v_levels, rel_vort_levels, curve_vort_levels = get_ERA5_variables(common_object,date_time)
	elif common_object.model in highresmip_models:
		u_levels, v_levels, t_levels, rel_vort_levels, windspeed, psl = get_highresmip_variables(common_object,date_time)
	

	return u_levels, v_levels, t_levels, rel_vort_levels, wspd_10m, slp


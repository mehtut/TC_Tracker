from __future__ import division  # makes division not round with integers 
import os
import pickle
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
import argparse
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
#from Circle_functions import *
from TC_Tracking_functions import *
from Pull_TC_data import *
from scipy.ndimage import gaussian_filter
import ctypes
import numpy.ctypeslib as ctl

# TC_Tracks.py is the main program for the TC tracking algorithm. The model type, scenario type, and year are parsed from the command line.
# The timeframe over which TC tracks are found can be modified in main and is typically set to run from June-November. The output of this 
# program is a TC track object and a figure of all tracks found over the given time period. To run the program, for example, type:
# python TC_Tracks.py --model 'WRF' --scenario 'Historical' --year '2001'


# Common info class. Each instance/object of Common_track_data is an object that holds the latitude and longitude arrays,
# the lat and lon indices for the boundaries over Africa/the Atlantic, the time step (dt) for the model data, the min_threshold
# value, the radius, and the model type
class Common_track_data:
	def __init__(self):
		self.model = None
		self.lat = None
		self.lon = None
		self.lat_index_north = None
		self.lat_index_south = None
		self.lon_index_east = None
		self.lon_index_west = None
		self.lat_index_north_crop = None
		self.lat_index_south_crop = None
		self.lon_index_east_crop = None
		self.lon_index_west_crop = None
		self.total_lon_degrees = None
		self.dt = None
		self.wspd_10m_threshold = None
		self.rel_vort_threshold = None
		self.slp_threshold = None
		self.warm_core_threshold = None
		self.radius = None
	# function to add the model type to self.model
	def add_model(self, model_type):
		self.model = model_type

# TC track class. Each instance/object of TC_track is a a TC track and has corresponding lat/lon points
# and magnitudes of the vorticity at those points. The lat/lon and magnitudes are stored in lists that are 
# associated with each object. The lat/lon and magnitude lists show the progression of the TC in time, so the 
# first lat/lon point in the list is where the TC starts, and then the following lat/lon points are where it 
# travels to over time.
class TC_track:
	def __init__(self):
		self.latlon_list = [] # creates a new empty list for lat/lon tuples
		self.magnitude_list = [] # creates a new empty list for the magnitude of the vorticity at each lat/lon point
		self.time_list = [] # creates a new empty list for the times for each new location in the track
		self.counter = 0 # creates a counter that starts set at zero. This counter will be used to see how many times a track object finds no matching points in unique_track_locations.

	def add_latlon(self, latlon):
		self.latlon_list.append(latlon)

	def remove_latlon(self, latlon):
		self.latlon_list.remove(latlon)

	def add_magnitude(self, magnitude):
		self.magnitude_list.append(magnitude)

	def add_time(self, time):
		self.time_list.append(time)

	def remove_time(self, time):
		self.time_list.remove(time)


def main():

	# set up argparse, which lets command line entries define variable_type and scenario_type
	parser = argparse.ArgumentParser(description='TC Tracker.')
	parser.add_argument('--model', dest='model_type', help='Get the model type for the data (WRF, CAM5, ERA5)', required=True)
	parser.add_argument('--scenario', dest='scenario_type', help='Get the scenario type to be used (Historical, late_century, Plus30)', required=True)
	parser.add_argument('--year', dest='year', help='Get the year of interest', required=True)
	args = parser.parse_args()

	# set the model type that is parsed from the command line
	model_type = args.model_type 
	# set the scenario type that is parsed from the command line
	scenario_type = args.scenario_type 
	# set the year that is parsed from the command line
	year = args.year

	# Set threshold values. These are based on Chris Patricola's tracker that was used on 27 km WRF data.
	# minimum 10m windspeed
#	wspd_10m_threshold = 18.5 # m/s
	wspd_10m_threshold = 17.5 # m/s
#	wspd_10m_threshold = 16.0 # m/s
	# 850mb vorticity threshold 
	rel_vort_threshold = 0.0002 # originally it was at 0.0002
	# sea level pressure threshold
	slp_threshold = 1010 # hPa
	# TTOT ( TTOT = midtropospheric temperature anomaly) threshold (K).  This is 2.0K in Walsh, 1997. 
	warm_core_threshold = 2.0 # K, originally was set at 2K

	# set radius (this is my new addition to the code methodology)
	radius_km = 100 # km

	# create an object from Common_track_data that will hold the model_type, lat, lon and dt information
	common_object = Common_track_data()
	# assign the model type to the common_object
	common_object.add_model(model_type)
	# assign the minimum 10m windspeed threshold to the common_object
	common_object.wspd_10m_threshold = wspd_10m_threshold
	# assign the relative vorticity threshold to the common_object
	common_object.rel_vort_threshold = rel_vort_threshold
	# assign the slp thrshold to the common_object
	common_object.slp_threshold = slp_threshold
	# assign the warm core threshold to the common_object
	common_object.warm_core_threshold = warm_core_threshold
	# assign the radius in km to the common object 
	common_object.radius = radius_km
	# get the common track data (like lat and lon) and assign it to the appropriate attributes in common_object
	get_common_track_data(common_object)

	# set time information
#	times = np.arange(datetime(int(year),5,1,0), datetime(int(year),11,1,0), timedelta(hours=common_object.dt)).astype(datetime) # May - October (AEW seasn)
	times = np.arange(datetime(int(year),6,1,0), datetime(int(year),12,1,0), timedelta(hours=common_object.dt)).astype(datetime) # June - November (tropical cyclone season)
#	times = np.arange(datetime(int(year),8,1,0), datetime(int(year),9,1,0), timedelta(hours=common_object.dt)).astype(datetime) # month of August
#	times = np.arange(datetime(int(year),9,1,0), datetime(int(year),10,1,0), timedelta(hours=common_object.dt)).astype(datetime) # September

	# create a working list for TC tracks
	TC_tracks_list = []
	finished_TC_tracks_list = []

	# loop through all times and find TC tracks
	for time_index in range(0,times.shape[0]):
		print(times[time_index].strftime('%Y-%m-%d_%H'))

		# get variables for each time step
		u_3d, v_3d, t_3d, rel_vort_3d, wspd_10m_2d, slp_2d = get_variables(common_object, scenario_type, times[time_index])

		# create some empty lists that may or may not be filled in later depending on if any TC points are found
		rel_vort_locs = []
		slp_locs = []
		unique_adjusted_tc_locs = []
		adjust_tc_locs_wspd = []
		adjust_tc_locs_warm_core = []
		adjust_tc_locs_shear = []

		# Walsh, 1997 condition 1 
		# TC candidates must exceed a minimum relative vorticity of rel_vort_threshold (rel_vort_threshold may depend on the model resolution)
		# Find new starting points 
#		if (rel_vort_3d[0,common_object.lat_index_south:common_object.lat_index_north+1,common_object.lon_index_west:common_object.lon_index_east+1]>common_object.rel_vort_threshold).any():
#			print("found a possible center!!!!!!!!!!!")
		print("Get starting targets...")
		print("Condition 1")
		rel_vort_locs = get_rel_vort_targets(common_object,rel_vort_3d) 
		slp_locs = get_slp_targets(common_object,slp_2d)
#		print(unique_max_locs)
#		print(len(unique_max_locs))

		# Walsh, 1997 condition 2 
		# There must be a closed pressure minimum within a user-speciﬁed distance of a point satisfying condition 1
		# This minimum pressure is taken as the center of the storm.
		# need to make sure these lists from condition 1 aren't empty to move on to condition 2
		if rel_vort_locs and slp_locs: 
			print("Condition 2")
			unique_adjusted_tc_locs = adjust_tc_locs(common_object, rel_vort_locs, slp_locs)
#			print(unique_adjusted_tc_locs)
#			print(len(unique_adjusted_tc_locs))

		# Walsh, 1997 condition 3 
		# A minimum 10m wind speed of wspd_10m_threshold (Walsh uses 14m/s at 1.125 degree resolution.)
		# Check for minimum 10m wind speed within 2 degrees in each direction of TC center.
		# need to make sure the list from condition 2 isn't empty to move on to condition 3
		if unique_adjusted_tc_locs:
			print("Condition 3")
			adjust_tc_locs_wspd = windspeed_check(common_object, unique_adjusted_tc_locs, wspd_10m_2d)
#			print(adjust_tc_locs_wspd)
#			print(len(adjust_tc_locs_wspd))

		# Walsh, 1997 condition 4 
		# A minimum total tropospheric temperature anomaly of TTOT.
		# Walsh, 1997 condition 6 
		# The temperature anomaly at 300 hPa must be greater than at 850 hPa at the center of the storm.
		# TTOT was determined by ﬁrst calculating a mean reference temperature at a level over a band 2 grid points north and south 
		# of the pressure minimum and 13 grid points east and west. 
		# TTOT was taken to be the sum of the temperature anomalies relative to the reference temperature at each levels, 
		# when calculated at the center of the storm.  
		# Note that Walsh used ECMWF at 1.125 degree resolution. (~14.5 degrees in e/w and 2.25 degrees in n/s on each side of center)
		# need to make sure the list from condition 3 isn't empty to move on to conditions 4 and 6
		if adjust_tc_locs_wspd:
			print("Conditions 4 and 6")
			adjust_tc_locs_warm_core = warm_core_check(common_object, adjust_tc_locs_wspd, t_3d)
#			print(adjust_tc_locs_warm_core)
#			print(len(adjust_tc_locs_warm_core))

		# Walsh, 1997 condition 5 
		# Mean wind speed around the center of the storm at 850 hPa must be greater than at 300 hPa
		# Calculate average wind speeds within 2.5 degrees on each side of TC center 
		# need to make sure the list from conditions 4 and 6 isn't empty to move on to condition 5
#		if adjust_tc_locs_wspd:
		if adjust_tc_locs_warm_core:
			print("Condition 5")
			adjust_tc_locs_shear = shear_check(common_object, adjust_tc_locs_warm_core, u_3d, v_3d)
			print(adjust_tc_locs_shear)
#			print(len(adjust_tc_locs_shear))

		# Compare unique_adjusted_tc_locs and adjust_tc_locs_shear at time t with the track object locations at time t.
		# If there are locations that are too close together (so duplicates between the track object and the new 
		# track locations that were generated for this time step), use the the new lat/lon pair to
		# replace the old lat/lon value in the existing track object and remove the duplicate lat/lon location from either 
		# unique_adjusted_tc_locs or adjust_tc_locs_shear.
		if TC_tracks_list and unique_adjusted_tc_locs: # only enter if the lists aren't empty
			for track_object in TC_tracks_list:
				# get the index of current time from the time list associated with the track object
				current_time_index = track_object.time_list.index(times[time_index])
				# use the time index to get the corresponding track object lat/lon location and then check to make sure that
				# the new track locations aren't duplicates of existing track objects
				# first check and see if any of the points in unique_adjusted_tc_locs are close to a current track object's lat/lon
				# at the current time. This will catch cases where the unique_adjusted_tc_locs lat/lon don't quite meet conditions
				# 3-6, but it's still clearly part of the track. Then check to see if any points in adjust_tc_locs_shear are close to
				# a current track object's lat/lon at hte current time. This will replace the track object's lat/lon pair at the 
				# current time with the lat/lon from adjust_tc_locs_shear, which has satisfied all 6 conditions for TCs.
				unique_track_locations(track_object,unique_adjusted_tc_locs,current_time_index,200.)
				if adjust_tc_locs_shear: # need to make sure this list isn't empty
					unique_track_locations(track_object,adjust_tc_locs_shear,current_time_index,200.)

		# for the locations in adjust_tc_locs_shear (assuming it isn't empty), create new
		# TC track objects. For reach object, add the lat/lon pair from adjust_tc_locs_shear and 
		# also add the time. Then append the new track object to TC_tracks_list
		if adjust_tc_locs_shear: # only enter if the list isn't empty
			for lat_lon_pair in adjust_tc_locs_shear:
				tc_track = TC_track()
				tc_track.add_latlon(lat_lon_pair)
				tc_track.add_time(times[time_index])
#				print(tc_track.latlon_list)
				TC_tracks_list.append(tc_track)
				del tc_track

		# filter out tracks that are outside of the Atlantic Basin area
		# advect tracks
		# add any new track locations in the Atlantic to existing TC tracks in final_TC_tracks_list or if they are not
		# part of an existing track, create a new TC track in final_TC_tracks_list
		if TC_tracks_list: # only enter if the list isn't empty
			for tc_track in list(TC_tracks_list):
				print("filtering...")
				# if farthest east longitude value is greater than 10 (10E), which means east of 10E, then remove the track 
				# because it doesn't doesn't start far enough west
				if tc_track.latlon_list[0][1] > 10:
					print("not far enough west")
		#			print(aew_track.latlon_list)
		#			print(aew_track.time_list)
					TC_tracks_list.remove(tc_track)
					continue

				# if the farthest east longitude value is less than -100 (100W), which means west of 100W, then remove the track
				# because it started too far west
				if tc_track.latlon_list[0][1] < -100:
					print("doesn't start far enough east")
		#			print(aew_track.latlon_list)
		#			print(aew_track.time_list)
					TC_tracks_list.remove(tc_track)
					continue

				# if the farthest south latitude value is greater than 40 (40N), which means north of 40N, then remove the track
				# because it started too far north
				if tc_track.latlon_list[0][0] > 40:
					print("not far enough south")
					TC_tracks_list.remove(tc_track)
					continue

				# remove any tracks that form west of south and central America
				# first remove any tracks that formed south of 10N and west of 80W
				# second remove any tracks that formed south of 15N and west of 90W
				if tc_track.latlon_list[0][0] < 10 and tc_track.latlon_list[0][1] < -80: # south of 10N and west of 80W
					print("west of South America")
					TC_tracks_list.remove(tc_track)
					continue
				if tc_track.latlon_list[0][0] < 15 and tc_track.latlon_list[0][1] < -90: # south of 15N and west of 90W
					print("west of Central America")
					TC_tracks_list.remove(tc_track)
					continue

				# if the tc_track object has not had any lat/lon points that are close to new TC lat/lon points for over 2 days (48 hours divided by dt to get
				# the number of time steps), add the track object to finished_TC_tracks_list and remove it from TC_tracks_list
				number_of_time_steps = int(48/common_object.dt) # 48/dt is the number of time steps in 2 days
				if tc_track.counter > number_of_time_steps: 
					print("track finished - fading TC candidates")
					# remove the track object from TC_tracks_list
					TC_tracks_list.remove(tc_track)
					# remove the last 8 (which is number_of_time_steps) times and lat/lon pairs from the track object
					remove_latlon_time(tc_track,number_of_time_steps)
					# add track object to finished_TC_tracks_list
					finished_TC_tracks_list.append(tc_track)
					continue

				# if the last latitude (most recent time) in the tc_track object goes too far north (> 58N), add the track object to 
				# finished_TC_tracks_list and remove it from TC_tracks_list. The TC track is finished because it's going too far north.
				if tc_track.latlon_list[-1][0] > 58: # 58N
					print("track finished - too far north")
					finished_TC_tracks_list.append(tc_track)
					TC_tracks_list.remove(tc_track)
					continue

				# if the last latitude or longitude (most recent time) in the tc_track object goes outside the boundaries of the dataset (e.g. the track lat > 
				# the maximum lat of the dataset, add the track object to finished_TC_tracks_list and remove it from TC_tracks_list. 
				# The TC track is finished because it has reached the boundaries of the dataset.
				if tc_track.latlon_list[-1][0] > np.amax(common_object.lat) or tc_track.latlon_list[-1][0] < np.amin(common_object.lat) \
					or tc_track.latlon_list[-1][1] > np.amax(common_object.lon) or tc_track.latlon_list[-1][1] < np.amin(common_object.lon): 
					print("track finished - outside of boundary")
					finished_TC_tracks_list.append(tc_track)
					TC_tracks_list.remove(tc_track)
					continue

				# advect tracks
				print("advecting...")
				advect_tracks(common_object, u_3d, v_3d, tc_track, times, time_index)

				print(tc_track.latlon_list)


	print("Total number of TC tracks =", len(finished_TC_tracks_list))

	# More filtering to check for tracks that weren't long enough
	for tc_track in list(finished_TC_tracks_list):
		# check for tracks that haven't lasted long enough. If the track hasn't lasted for two days (which is < 48/dt + 1 time steps ), get rid of it
		if len(tc_track.latlon_list) < ((48/common_object.dt)+1):
			print("not enough times")
#			print(tc_track.latlon_list)
#			print(tc_track.time_list)
			finished_TC_tracks_list.remove(tc_track)
			continue

#	print("length of TC list =", len(final_TC_tracks_list))
	# for tc_track in TC_tracks_list:
	# 	print(tc_track.time_list)
	# 	print(tc_track.latlon_list)
	# 	print(tc_track.counter)

	for tc_track in finished_TC_tracks_list:
		print(tc_track.time_list)
		print(tc_track.latlon_list)
		print(tc_track.counter)

	print("Total number of TC tracks =", len(finished_TC_tracks_list))
	print(year)
	print(model_type)

	# save tracks to file
	tracks_file = open(model_type + '_' + scenario_type + '_TC_tracks_' + year + '_E9_0506_Jun-Nov.obj', 'wb') # 'wb' means write binary, if just 'w' is used, a string is expected
	pickle.dump(finished_TC_tracks_list, tracks_file)

if __name__ == '__main__':
	main()
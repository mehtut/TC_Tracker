from __future__ import division  # makes division not round with integers 
import os
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import wrf as wrf
from datetime import datetime, timedelta
import math as math
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from scipy import inf
from collections.abc import Iterable


# This function gets the starting latitude and longitude values for AEW tracks.
# The function takes the common_object and smoothed curvature vorticity as parameters and
# returns a list of lat/lon pairs as tuples that are the locations of unique vorticity maxima.
def get_rel_vort_targets(common_object,rel_vort): 
	# get a list of indices (lat/lon index pairs) of local maxima in the relative vorticity field
	max_indices = peak_local_max(rel_vort[0,common_object.lat_index_south:common_object.lat_index_north+1,common_object.lon_index_west:common_object.lon_index_east+1], min_distance=1, threshold_abs=common_object.rel_vort_threshold) # indices come out ordered lat, lon
	
	# get a list of weighted averaged lat/lon index pairs
	weighted_max_indices = get_mass_center(common_object,rel_vort[0,:,:],max_indices)  # the index 0 in rel_vort is to take the 850 hPa level

	# Remove duplicate locations. This checks to see if new starting targets actually belong to existing tracks.
	# The 99999999 is the starting value for unique_loc_number; it just needs to be way bigger than the 
	# possible number of local maxima in weighted_max_indices. This function recursively calls itself until
	# the value for unique_loc_number doesn't decrease anymore.
	# Only need to check for duplicate locations if there is more than one location (meaning len(weighted_max_indices) > 1).
	# If there is only one member of the the list, then unique_max_locs = weighted_max_indices.
	if len(weighted_max_indices) > 1:
		unique_max_locs = unique_locations(weighted_max_indices,common_object.radius,99999999) 
	else:
		unique_max_locs = weighted_max_indices

	return unique_max_locs

def get_slp_targets(common_object,slp): 
	# we want local minima of the slp field, but peak_local_max doesn't have a minimum counterpart. To still use peak_local_max, flip the signs of slp and use 
	# peak_local_max. Eg originally 1050 > 900 hPa, but -900 > -1050, so with the signs flipped we find the minima. the slp_threshold in the common_object
	# also needs its sign flipped to match the flipped signs of the slp array.
	# get a list of indices (lat/lon index pairs) of local maxima in the -1*slp field 
	max_indices = peak_local_max(-1.0*slp[common_object.lat_index_south:common_object.lat_index_north+1,common_object.lon_index_west:common_object.lon_index_east+1], min_distance=1, threshold_abs=common_object.slp_threshold*-1.0) # indices come out ordered lat, lon
#	print(max_indices)

#	# get a list of weighted averaged lat/lon index pairs
	weighted_max_indices = get_mass_center(common_object,-1.0*slp[:,:],max_indices)  
#	print(weighted_max_indices)

	# Remove duplicate locations. This checks to see if new starting targets actually belong to existing tracks.
	# The 99999999 is the starting value for unique_loc_number; it just needs to be way bigger than the 
	# possible number of local maxima in weighted_max_indices. This function recursively calls itself until
	# the value for unique_loc_number doesn't decrease anymore.
	# Only need to check for duplicate locations if there is more than one location (meaning len(weighted_max_indices) > 1).
	# If there is only one member of the the list, then unique_max_locs = weighted_max_indices.
	if len(weighted_max_indices) > 1:
		unique_max_locs = unique_locations(weighted_max_indices,common_object.radius,99999999) 
	else:
		unique_max_locs = weighted_max_indices

	return unique_max_locs


# This function takes the lat/lon indices of vorticity maxima and uses a weighted average to adjust the location.
# The function takes the common_object, a variable (vorticity), and a list of lat/lon indices as parameters.
# The function returns a list of lat/lon values.
def get_mass_center(common_object,var,max_indices): #,lat_index_south,lon_index_west):
 	# set a minimum radius in km
	min_radius_km = 100. # km

	# lists to store the weighted lats and lons
	weight_lat_list = []
	weight_lon_list = []

	# get differences between neighboring lat and lon points
	dlon = common_object.lon[0,1] - common_object.lon[0,0]
	dlat = common_object.lat[1,0] - common_object.lat[0,0]
#	print("dlon =", dlon)
#	print("dlat =", dlat)

	# this is the approximate number of degrees covered by the common_object radius + 4 to have a buffer 
	delta = int(math.ceil(common_object.radius/111.)+4) # ceil rounds up to the nearest int

	# loop through all the max indices and calculate weighted lat and lon values for the max locations
	for max_index in max_indices:
		# to get the max lat and lon indices, we need to add lat_index_south and lon_index_west because the max indices 
		# were found using a dataset that was cropped over the region of interest in get_starting_targets
		max_lat_index = max_index[0] + common_object.lat_index_south # this is an index
		max_lon_index = max_index[1] + common_object.lon_index_west # this is an index

		# get the max lat and lon values from the max lat and lon indices and then add dlat/2 and dlon/2 to nudge the points
		max_lat = common_object.lat[max_lat_index,max_lon_index] + (dlat/2.) # this is the actual latitude value
		max_lon = common_object.lon[max_lat_index,max_lon_index] + (dlon/2.) # this is the actual longitude value
	
		# get new max lat and lon indices using the adjusted max_lat and max_lon valus above and adding or subtracting delta
		max_lat_index_plus_delta = (np.abs(common_object.lat[:,0] - (max_lat+delta))).argmin()
		max_lat_index_minus_delta = (np.abs(common_object.lat[:,0] - (max_lat-delta))).argmin()
		max_lon_index_plus_delta = (np.abs(common_object.lon[0,:] - (max_lon+delta))).argmin()
		max_lon_index_minus_delta = (np.abs(common_object.lon[0,:] - (max_lon-delta))).argmin()
		
		# create a cropped version of the variable array, lat and lon arrays using the delta modified lat/lon indices above
		var_crop = var[max_lat_index_minus_delta:max_lat_index_plus_delta,max_lon_index_minus_delta:max_lon_index_plus_delta]
		lat_crop = common_object.lat[max_lat_index_minus_delta:max_lat_index_plus_delta,max_lon_index_minus_delta:max_lon_index_plus_delta]
		lon_crop = common_object.lon[max_lat_index_minus_delta:max_lat_index_plus_delta,max_lon_index_minus_delta:max_lon_index_plus_delta]
#		print(lat_crop)
#		print(lon_crop)
		
		# Find mass center over large area first, and then progressively make the radius smaller to hone in on the center 
		# using a weighted average.
		weight_lat, weight_lon = subgrid_location_km(var_crop, max_lat, max_lon, lat_crop, lon_crop, common_object.radius)  
#		print("weight_lat =", weight_lat)
#		print("weight_lon =", weight_lon)
		# now find mass center over a smaller area by dividing common_object.radius by n (defined below). Keep doing this while 
		# common_object.radius/n is greater than the min radius defined at the beginning of this function.
		n=2 
		while common_object.radius/float(n) > min_radius_km:
			weight_lat, weight_lon = subgrid_location_km(var_crop, weight_lat, weight_lon, lat_crop, lon_crop, common_object.radius/float(n))
			n += 1
		# one last round of taking a weighted average, this time using the minimum radius.
		weight_lat, weight_lon = subgrid_location_km(var_crop, weight_lat, weight_lon, lat_crop, lon_crop, min_radius_km)
#		print("weight_lat =", weight_lat)
#		print("weight_lon =", weight_lon)

		weight_lat_list.append(weight_lat)
		weight_lon_list.append(weight_lon)
		del weight_lat
		del weight_lon

	# zip the weighted lat and lon lists together to get a list of ordered pairs [(lat1,lon1), (lat2,lon2), etc.]
	weight_lat_lon_list = list(zip(weight_lat_list,weight_lon_list))

	return weight_lat_lon_list

# This function finds the weighted maxima lat/lon point from a subset of data. 
# The function takes the cropped variable (cropping comes from the common_object), the max latitude and longitude
# values (NOT the indices), and the radius to use for the weights.
# The function returns a weighted average lat and lon value.
def subgrid_location_km(var_crop,max_lat,max_lon,lat,lon,radius):
	# replace all values less than zero with zero
	var_crop[var_crop<0] = 0.0
	# calculate the great circle distance in km between the max lat/lon point and all of
	# the lat and lon values from the cropped lat and lon arrays
	gc_dist = great_circle_dist_km(max_lon, max_lat, lon, lat)
	# calculate the weights using a generous barnes-like weighting function (from Albany)
	# and then use the weights on var_crop
	weights = np.exp( -1 * ((gc_dist**2)/(radius**2))) 
	var_crop = (var_crop**2)*weights
	# flatten the var_crop array
	var_crop_1d = var_crop.flatten()
	# set any values in the flattened var_crop array less than zero equal to zero
	var_crop_1d[var_crop_1d<0] = 0.0 
	# check to see if all the values in var_crop_1d are equal to zero. If they are, then return
	# the original max_lat and max_lon that were passed in as parameters. If not, then use a 
	# weighted average with var_crop_1d on the lat and lon arrays to get new lat and lon values.
	if var_crop_1d.all() == 0:
		return max_lat, max_lon
	else:
		weight_lat = np.average(lat.flatten(), weights=var_crop_1d)
		weight_lon = np.average(lon.flatten(), weights=var_crop_1d)
#	print(weight_lat)
#	print(weight_lon)

	return weight_lat, weight_lon 

# This function calculates the great circle distance between two lat/lon points. 
# The function takes the two sets lat/lon points as parameters and returns the distance in km. 
def great_circle_dist_km(lon1, lat1, lon2, lat2):
	# switch degrees to radians
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2 # note that everything was already switched to radians a few lines above
	c = 2 * np.arcsin(np.sqrt(a))
	dist_km = 6371. * c # 6371 is the radius of the Earth in km

	return dist_km

# Average locations that are within radius in km of each other.
# This function takes a list of lat/lon locations as tuples in a list. These are locations of local
# maxima. In this function, locations that are within the radius of each other are averaged. To make sure
# that no duplicate locations are returned, the function is recursively called until the number of unique locations
# no longer decreases. The function takes the list of lat/lon locations, the radius used to check for duplicates,
# and a unique_loc_number as parameters and returns a list of lat/lon tuples of the unique locations. The unique_loc_number
# is originally set to a very large value (99999999) when this function is called, and then this is what is compared with for 
# recursive call (eg. is the new number of unique locations, say 20, less than 99999999? If yes, then go through unique_locations again, but this
# time unique_loc_number is set to 20. Only when the new number of unique locations equals the previous unique_loc_number, is the recursion over).
def unique_locations(max_locations,radius,unique_loc_number):
	# sort the list by the latitudes, so the lat/lon paris go from south to north
	max_locations.sort(key=lambda x:x[0])

	# convert the radius to its approximate degree equivalent by dividing the radius in km by
	# 111 km. This is because 1 degree is approximately 111 km (at the equator).
	max_distance = radius/111. # convert to degrees
#	print("max_distance =", max_distance)

	# make a tree out of the lat/lon points in the list
	tree = cKDTree(max_locations)
	# make a list to hold the average of closely neighboring points 
	point_neighbors_list_avg = [] 
	first_time = True
	# loop through all the lat/lon max locations and find the points that are within
	# the radius/max_distance of each other
	for max_loc in max_locations:
		# check to see if it's the first time 
		if first_time:
			first_time = False
		else:
			# skip points that have just been found as near neighbors
			# so if a point was just in point_neighbors, move on to the next point
			if max_loc in point_neighbors: 
#				print("continue")
				continue 

		# calculate the distance and indices between the max location lat/lon points
		distances, indices = tree.query(max_loc, len(max_locations), p=10, distance_upper_bound=max_distance) # p=10 seems to be a good choice
		# the following conditional catches the cases where distances and indices are a float and integer, respectively.
		# in this case, the zip a few lines below won't work because floats and ints aren't interable. Check and see if 
		# distance and indices are not iterable and if they're not, make them lists so that they are iterable. 
		if not isinstance(distances, Iterable) and not isinstance(indices, Iterable):
			distances = [distances]
			indices = [indices]
		# store the neighboring points that are close to each in a list
		point_neighbors = []
		for index, distance in zip(indices, distances):
			# points that aren't close are listed at a distance of inf so ignore those
			if distance == inf:
				break
			point_neighbors.append(max_locations[index])

		# the points_neighbors list has all the lat/lon points that are close to each other
		# average those points, keep them as a tuple (so a new averaged lat/lon point), and 
		# append that to point_neighbors_list_avg
		point_neighbors_list_avg.append(tuple(np.mean(point_neighbors,axis=0)))
#		print(point_neighbors_list_avg)

	# the number of unique locations is the length of the point_neighbors_list_avg list
	new_unique_loc_number = len(list(set(point_neighbors_list_avg))) # use the list set option here to make sure any duplicates aren't counted

	# check to see if the number of unique locations is the same as it was the last time through
	# the function. If it is the same, return point_neighbors_list_avg, which is the unique lat/lon locations.
	# If it is not the same, recursively go back through unique_locations to weed out any more locations that are
	# within the radius/max_distance of each other.
	if new_unique_loc_number == unique_loc_number:
		return list(set(point_neighbors_list_avg)) # use the list set option here to make sure any duplicates aren't counted
	else:
		return unique_locations(point_neighbors_list_avg,radius,new_unique_loc_number)

def adjust_tc_locs(common_object, rel_vort_locs, slp_locs):
	# new list for the tc locs that have both a relative vorticity maximum and a slp minimum
	adjusted_tc_locs = []
	# loop through the relative vorticity maxima in rel_vort_locs
	# and loop through the slp minimum in slp_locs
	for max_loc in rel_vort_locs:
		for min_loc in slp_locs: 
			# get the distance between max_loc and min_loc
			dist_km = great_circle_dist_km(max_loc[1], max_loc[0], min_loc[1], min_loc[0])
			# if the distance is less than the common_object radius, then add the slp lat/lon point to the list adjusted_tc_locs
			if dist_km < common_object.radius:
				adjusted_tc_locs.append(min_loc)
	
	# Remove duplicate locations. 
	# The 99999999 is the starting value for unique_loc_number; it just needs to be way bigger than the 
	# possible number of lat/lon points in adjusted_tc_locs. This function recursively calls itself until
	# the value for unique_loc_number doesn't decrease anymore.
	# Only need to check for duplicate locations if there is more than one location (meaning len(adjusted_tc_locs) > 1).
	# If there is only one member of the the list, then unique_adjusted_tc_locs = adjusted_tc_locs.
	if len(adjusted_tc_locs) > 1:
		unique_adjusted_tc_locs = unique_locations(adjusted_tc_locs,common_object.radius,99999999) 
	else:
		unique_adjusted_tc_locs = adjusted_tc_locs

	return unique_adjusted_tc_locs


#def adjust_max_locs(common_object,unique_max_locs,rel_vort_3d):
	# adjust_max_locs may have lat/lon pairs removed and new ones added to it. We don't want to use unique_max_locs list because
	# that's in the loop and it may mess up the loop.
#	adjust_max_locs = list(unique_max_locs) # use list here to avoid a pointere where changes to one list affect the other

	# check for vorticity max within 2.5 degrees on each side of TC center when vorticity threshold is met and assign as TC center location
#	for max_loc in unique_max_locs:
		# get the indices for the max_loc lat/lon point
#		max_loc_lat_index = (np.abs(common_object.lat[:,0] - max_loc[0])).argmin()
#		max_loc_lon_index = (np.abs(common_object.lon[0,:] - max_loc[1])).argmin()
		# get the indices for the lat/lon point +/- 2.5 degrees north, south, east, and west of max_loc
#		lat_index_north = (np.abs(common_object.lat[:,0] - (max_loc[0]+2.5))).argmin()
#		lat_index_south = (np.abs(common_object.lat[:,0] - (max_loc[0]-2.5))).argmin()
#		lon_index_east = (np.abs(common_object.lon[0,:] - (max_loc[1]+2.5))).argmin()
#		lon_index_west = (np.abs(common_object.lon[0,:] - (max_loc[1]-2.5))).argmin()

		# create a temporary max value for comparison purposes 
#		temp_max = rel_vort_3d[0,max_loc_lat_index,max_loc_lon_index]
		# check to see if any points within 2.5 degrees of the max_loc point have a larger relative vorticity value
#		for lat_index in range(lat_index_south,lat_index_north+1): # latitude loop
#			for lon_index in range(lon_index_west,lon_index_east+1): # longitude loop
				# check to see if the current lat/lon point has a larger relative vorticity at 850 hPa (that's the 0 index) than temp_max
#				if rel_vort_3d[0,lat_index,lon_index] > temp_max:
#					temp_max = rel_vort_3d[0,lat_index,lon_index]
#					max_lat_index = lat_index
#					max_lon_index = lon_index
		# check to see if temp max is bigger the relative vorticity at max_loc
		# if it is bigger, remove max_loc from unique_max_locs list and add the new lat/lon point
#		if temp_max > rel_vort_3d[0,max_loc_lat_index,max_loc_lon_index]:
#			print(max_loc)
#			adjust_max_locs.remove(max_loc)
#			print((common_object.lat[max_lat_index,0],common_object.lon[0,max_lon_index]))
#			adjust_max_locs.append((common_object.lat[max_lat_index,0],common_object.lon[0,max_lon_index]))
#			del max_lat_index
#			del max_lon_index
#		del temp_max

#	return adjust_max_locs

def windspeed_check(common_object, adjust_tc_locs_rel_vort, wspd_10m_2d):
	# adjust_locs_wspd may have lat/lon pairs removed and new ones added to it. We don't want to use adjust_tc_locs_rel_vort list because
	# that's in the loop and it may mess up the loop.
	adjust_locs_wspd = list(adjust_tc_locs_rel_vort) # use list here to avoid a pointere where changes to one list affect the other

	# loop through the potential TC location lat/lon pairs in adjust_tc_locs_rel_vort and check to see if the 10m windspeed is greater than or equal to the threshold  
	# that was set in the common_object. If the 10m speed is above the threshold, that lat/lon pair stays a TC candidate. Otherwise, the lat/lon pair is removed.
	# check for 10m windspeed within 2 degrees on each side of TC center.
	for max_loc in adjust_tc_locs_rel_vort:
		# get the indices for the lat/lon point +/- 2 degrees north, south, east, and west of max_loc
		lat_index_north = (np.abs(common_object.lat[:,0] - (max_loc[0]+2))).argmin()
		lat_index_south = (np.abs(common_object.lat[:,0] - (max_loc[0]-2))).argmin()
		lon_index_east = (np.abs(common_object.lon[0,:] - (max_loc[1]+2))).argmin()
		lon_index_west = (np.abs(common_object.lon[0,:] - (max_loc[1]-2))).argmin()
		# check to see if any points within 2 degrees of the max_loc point have 10m windspeed values less than the threshold in common_object.
		# If a max_loc lat/lon point has a 10m windspeed less than the threshold, remove it from adjust_locs_wspd.
		# As long as the 10m windspeed is greater than or equal to the threshold somewhere within 2 degrees of the TC center, we keep the TC center point
		windspeed_exceeded = False
		for lat_index in range(lat_index_south,lat_index_north+1): # latitude loop
			for lon_index in range(lon_index_west,lon_index_east+1): # longitude loop
				# if the 10m windspeed threshold is met, then we switch windspeed_exceeded to True
				if wspd_10m_2d[lat_index,lon_index] >= common_object.wspd_10m_threshold:
					windspeed_exceeded = True
		# after running through all points 2 degrees from the TC center, if no 10m windspeed was greater than or equal to the threshold, remove max_loc				
		if not windspeed_exceeded:
			adjust_locs_wspd.remove(max_loc)

	return adjust_locs_wspd

def warm_core_check(common_object, adjust_tc_locs_wspd, t_3d):
	# adjust_locs_warm_core may have lat/lon pairs removed and new ones added to it. We don't want to use adjust_tc_locs_wspd list because
	# that's in the loop and it may mess up the loop.
	adjust_locs_warm_core = list(adjust_tc_locs_wspd) # use list here to avoid a pointere where changes to one list affect the other
	# TTOT (midtropospheric temperature anomaly) was determined by ﬁrst calculating a mean reference temperature at 700, 500, and 300 hPa over a band 2 grid points
	# north and south of the pressure minimum and 13 grid points east and west. Then TTOT was taken to be the sum of the temperature anomalies relative 
	# to the reference temperature at each levels, when calculated at the TC lat/lon point from adjust_tc_locs_wspd.  
	for max_loc in adjust_tc_locs_wspd:
		# get the indices for the max_loc lat/lon point
		max_loc_lat_index = (np.abs(common_object.lat[:,0] - max_loc[0])).argmin()
		max_loc_lon_index = (np.abs(common_object.lon[0,:] - max_loc[1])).argmin()
		# get the indices for the lat/lon point +/- 2.5 degrees north, south, east, and west of max_loc
		lat_index_north = max_loc_lat_index + 2
		lat_index_south = max_loc_lat_index - 2
		lon_index_east = max_loc_lon_index + 13
		lon_index_west = max_loc_lon_index - 13
		# calculate the average temperature at 850, 700, 500, and 300 hPa
		t_850_avg = np.mean(t_3d[0,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) # index 0 is 850 hPa
		t_700_avg = np.mean(t_3d[1,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) # index 1 is 700 hPa
		t_500_avg = np.mean(t_3d[2,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) # index 2 is 500 hPa
		t_300_avg = np.mean(t_3d[3,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) # index 3 is 300 hPa
		# calculate the temperature anomalies, which are the temperature at the TC center minus the reference (or average) temperature
		# at the corresponding level that is calculated directly above this comment
		t_850_anomaly = t_3d[0,max_loc_lat_index,max_loc_lon_index] - t_850_avg
		t_700_anomaly = t_3d[1,max_loc_lat_index,max_loc_lon_index] - t_700_avg
		t_500_anomaly = t_3d[2,max_loc_lat_index,max_loc_lon_index] - t_500_avg
		t_300_anomaly = t_3d[3,max_loc_lat_index,max_loc_lon_index] - t_300_avg
		# calculate TTOT as follows
		TTOT = t_700_anomaly + t_500_anomaly + t_300_anomaly
		temp_exceeded = False
		# check to see if TTOT is greater than the warm core threshold and also check and see if the 300 hPa temperature 
		# anomaly is greater than the 850 hPa temperature anomaly. If they are, keep this TC point. Otherwise remove the TC point.
		if TTOT > common_object.warm_core_threshold and t_300_anomaly > t_850_anomaly:
			temp_exceeded = True
		if not temp_exceeded:
			adjust_locs_warm_core.remove(max_loc)

	return adjust_locs_warm_core


def shear_check(common_object, adjust_tc_locs_warm_core, u_3d, v_3d):
	# adjust_locs_shear may have lat/lon pairs removed and new ones added to it. We don't want to use adjust_tc_locs_warm_core list because
	# that's in the loop and it may mess up the loop.
	adjust_locs_shear = list(adjust_tc_locs_warm_core) # use list here to avoid a pointere where changes to one list affect the other

	# loop through the potential TC location lat/lon pairs in adjust_tc_locs_warm_core and check to see if the mean windspeed around the center of 
	# the storm at 850 hPa is greater than at 300 hPa. Calculate average windspeeds at 850 hPa and 300 hPa within a 2.5 degree 
	# box on each side of TC center (max_loc). If the mean windspeed at 850 hPa is greater than at 300 hPa, that lat/lon pair stays a TC candidate. 
	# Otherwise, the lat/lon pair is removed.
	for max_loc in adjust_tc_locs_warm_core:
#		print(max_loc)
		# get the indices for the lat/lon point +/- 2 degrees north, south, east, and west of max_loc
		lat_index_north = (np.abs(common_object.lat[:,0] - (max_loc[0]+2.5))).argmin()
		lat_index_south = (np.abs(common_object.lat[:,0] - (max_loc[0]-2.5))).argmin()
		lon_index_east = (np.abs(common_object.lon[0,:] - (max_loc[1]+2.5))).argmin()
		lon_index_west = (np.abs(common_object.lon[0,:] - (max_loc[1]-2.5))).argmin()
		# calculate the average windspeed at 850 hPa and 300 hPa. Windspeed = sqrt(u^2 + v^2)
		# in the first dimension of u_3d and v_3d, index 0 is 850 hPa and index 2 is 300 hPa	
		wspd_850 = np.mean(np.sqrt(np.square(u_3d[0,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) \
			+ np.square(v_3d[0,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1])))
		wspd_300 = np.mean(np.sqrt(np.square(u_3d[2,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1]) \
			+ np.square(v_3d[2,lat_index_south:lat_index_north+1,lon_index_west:lon_index_east+1])))
		# boolean for whether or not the 850 hPa windspeed is greater than at 300 hPa
		windspeed_850_greater = False
		if wspd_850 > wspd_300:
			windspeed_850_greater = True
		# if the 850 hPa mean windspeed was not greater than at 300 hPa, remove max_loc from the potential TC list
		if not windspeed_850_greater:
			adjust_locs_shear.remove(max_loc)

	return adjust_locs_shear

# This function calculates the great circle distance between two lat/lon points. 
# The function takes the two sets lat/lon points as parameters and returns the distance in km. 
def great_circle_dist_km(lon1, lat1, lon2, lat2):
	# switch degrees to radians
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2 # note that everything was already switched to radians a few lines above
	c = 2 * np.arcsin(np.sqrt(a))
	dist_km = 6371. * c # 6371 is the radius of the Earth in km

	return dist_km


def remove_latlon_time(track_object,time_number):

	latlon_list = track_object.latlon_list[-(time_number-4):]
	time_list = track_object.time_list[-(time_number-4):]

	for latlon_pair in latlon_list:
		track_object.remove_latlon(latlon_pair)
	for time in time_list:
		track_object.remove_time(time)

	return


# This is a function that takes a track object, the list of vorticity maximum lat/lon locations, the current time index,
# and the radius in km. The lat/lon location of the track object at the current time is found. That lat/lon is then
# compared with all of the lat/lon pairs in the the list. If any of the lat/lons in the list are within the radius of the 
# track object's lat/lon, this lat/lon pair replaces the lat/lon pair in the track object by averaging the old and new lat/lon pairs.
# The lat/lon pair in the list that was too close to the track object is then removed. The whole point of this function is to 
# make sure no duplicate track objects are created by weeding out possible track locations that are already represented in 
# existing tracks. This function doesn't return anything, it just modifies the existing track object and lat/lon list.
def unique_track_locations(track_object,combined_unique_max_locs,current_time_index,radius):
	# get the lat/lon pair for the track object at the current time using current_time_index
	current_latlon_pair_list = []
	for time_index in range(current_time_index,min(current_time_index+1,len(track_object.latlon_list))):
		current_latlon_pair_list.append(track_object.latlon_list[time_index])
	found_matching_latlon = False
	# loop through the lat/lon locations in the combined_unique_max_locs list
#	print(combined_unique_max_locs)
	for latlon_loc in list(combined_unique_max_locs):
		for current_latlon_pair in current_latlon_pair_list:
			# get the distance between the track object's lat/lon and the lat/lon pair from the list
			dist_km = great_circle_dist_km(current_latlon_pair[1], current_latlon_pair[0], latlon_loc[1], latlon_loc[0])
			# check and see if the distance is less than the radius. If it is, replace the track_object lat/lon pair with
			# the average of the existing track_object lat/lon pair and the newly found lat/lon pair, remove the new pair
			# from the combined_unique_max_locs list, and continue to the next pair.
			if dist_km < radius:
#				print(latlon_loc)
#				print(track_object.latlon_list)
#				track_object.latlon_list = [((latlon_loc[0]+current_latlon_pair[0])/2,(latlon_loc[1]+current_latlon_pair[1])/2) if x==current_latlon_pair else x for x in track_object.latlon_list]
				track_object.latlon_list = [(latlon_loc[0],latlon_loc[1]) if x==current_latlon_pair else x for x in track_object.latlon_list]
#				print(track_object.latlon_list)
				combined_unique_max_locs.remove(latlon_loc) # remove the lat/lon pair from the list
				found_matching_latlon = True
				continue		
	if not found_matching_latlon:
		# increase the counter of the track object to keep track of how many times a lat/lon pair that is close is NOT found
		track_object.counter += 1	
	return


# This function takes a circle average at a specific point. The function takes the common_object,
# the var (u or v), and a lat/lon pair from an AEW track and returns the circle smoothed variable 
# that has been smoothed at the lat/lon point of interest. This is different than the c smoothing because
# the c smoothing smooths a larger domain, while this just smooths focused on the lat/lon point.
def circle_avg_m_point(common_object,var,lat_lon_pair): 
	# Take cos of lat in radians
	cos_lat = np.cos(np.radians(common_object.lat))  

	R=6371. # Earth radius in km
	# Get the number of gridpoints equivalent to the radius being used for the smoothing.
	# To convert the smoothing radius in km to gridpoints, multiply the radius (in km) by the total number of 
	# longitude gridpoints = var.shape[2] for the whole domain divided by the degrees of longitude in the domain
	# divided by 360 times the circumference of the Earth = 2piR. The degrees of longitude/360 * circumference is to
	# scale the circumference to account for non-global data. This is also a rough approximation since we're not quite at the equator.
	# So radius_gridpts = radius (in km) * (longitude gridpoints / scaled circumference of Earth (in km))
	# Make radius_gridpts an int so it can be used as a loop index later.  
	radius_gridpts = int(common_object.radius*(common_object.lat.shape[1]/((common_object.total_lon_degrees/360)*2*np.pi*R)))

	# create a copy of the var array
	smoothed_var = np.copy(var)

	# get the indices for the lat/lon pairs of the maxima
	lat_index_maxima = (np.abs(common_object.lat[:,0] - lat_lon_pair[0])).argmin()
	lon_index_maxima = (np.abs(common_object.lon[0,:] - lat_lon_pair[1])).argmin()
	# take circle average 
	tempv = 0.0
	divider = 0.0
	for radius_index in range(-radius_gridpts,radius_gridpts+1): # work way up circle
#		print("radius_index =", radius_index)
		# make sure we're not goint out of bounds, and if we are go to the next iteration of the loop
		if (lat_index_maxima+radius_index) < 0 or (lat_index_maxima+radius_index) > (common_object.lat.shape[1]-1):
			continue

		lat1 = common_object.lat[lat_index_maxima,lon_index_maxima]  # center of circle
		lat2 = common_object.lat[lat_index_maxima+radius_index,lon_index_maxima] # vertical distance from circle center
		# make sure that lat2, which has the radius added, doesn't go off the grid (either off the top or the bottom) 
		
		# need to switch all angles from degrees to radians
		angle_rad = np.arccos(-((np.sin(np.radians(lat1))*np.sin(np.radians(lat2)))-np.cos(common_object.radius/R))/(np.cos(np.radians(lat1))*np.cos(np.radians(lat2))))  # haversine trig

		# convert angle from radians to degrees and then from degrees to gridpoints
		# divide angle_rad by pi/180 = 0.0174533 to convert radians to degrees
		# multiply by lat.shape[1]/360 which is the lon gridpoints over the total 360 degrees around the globe
		# the conversion to gridpoints comes from (degrees)*(gridpoints/degrees) = gridpoints
		# lon_gridpts defines the longitudinal grid points for each lat
		lon_gridpts = int((angle_rad/0.0174533)*(common_object.lat.shape[1]/360.))

		for lon_circle_index in range(lon_index_maxima-lon_gridpts, lon_index_maxima+lon_gridpts+1):  # work across circle
			# the following conditionals handle the cases where the longitude index is out of bounds (from the Albany code that had global data)
			cyclic_lon_index = lon_circle_index
			if cyclic_lon_index<0: 
				cyclic_lon_index = cyclic_lon_index+common_object.lat.shape[1]
			if cyclic_lon_index>common_object.lat.shape[1]-1:
				cyclic_lon_index = cyclic_lon_index-common_object.lat.shape[1]

			tempv = tempv + (cos_lat[lat_index_maxima+radius_index,lon_index_maxima]*var[(lat_index_maxima+radius_index),cyclic_lon_index])
			divider = divider + cos_lat[lat_index_maxima+radius_index,lon_index_maxima]
			
	smoothed_var[lat_index_maxima,lon_index_maxima] =  tempv/divider	

	return smoothed_var

# This function uses the average of the u and v wind between 850 and 600 hPa to advect
# a track object's last lat/lon point to get the next lat/lon point in time. 
# The function takes the common_object, the zonal wind u, the meridional wind v, the track_object,
# the times array, and the current time_index. The function doesn't return anything, but rather 
# adds a new lat/lon point to the end of the track_object's lat/lon list and also adds a 
# new time to the end of the track_object's times list. 
def advect_tracks(common_object, u_3d, v_3d, track_object, times, time_index):
	# get the last lat/lon tuple in the track object's latlon_list
	# We want the last lat/lon tuple because that will be the last one in time
	# and we want to advect the last lat/lon point to get the next point in time.
	lat_lon_pair = track_object.latlon_list[-1] 

	# calculate the u and v wind averaged over the two steering levels; the steering levels are 850 and 600 hPa
	u_2d = (u_3d[0,:,:]+u_3d[1,:,:])/2. # take average of the 850 (index 0) and 600 (index 1) hPa levels
	v_2d = (v_3d[0,:,:]+v_3d[1,:,:])/2. # take average of the 850 (index 0) and 600 (index 1) hPa levels

	# smooth the u and v arrays at the lat/lon points that have been identified as unique maxima
	# the lat/lon pairs come in through track_locations as tuples in a list. The smoothing is 
	# done using the python circle_avg_m_point function.
	u_2d_smooth = circle_avg_m_point(common_object,u_2d,lat_lon_pair)
	v_2d_smooth = circle_avg_m_point(common_object,v_2d,lat_lon_pair)

	# find the indices for the lat/lon pairs
	lat_index = (np.abs(common_object.lat[:,0] - lat_lon_pair[0])).argmin()
	lon_index = (np.abs(common_object.lon[0,:] - lat_lon_pair[1])).argmin()

	# get new lat/lon values for the next time step by advecting the existing point using u and v
	# multiply dt (in hours) by 60*60 to get seconds; 111120. is the approximate meters in one degree on Earth
	# the *60*60*dt / 111120 converts the m/s from u and v into degrees
	new_lat_value = lat_lon_pair[0] + ((v_2d_smooth[lat_index,lon_index]*60.*60.*common_object.dt) /111120.) 
	new_lon_value = lat_lon_pair[1] + ((u_2d_smooth[lat_index,lon_index]*60.*60.*common_object.dt) /111120.)*np.cos(np.radians(lat_lon_pair[0])) # to switch to radians

	# make sure that the next time step is actually within our timeframe by checking to see if time_index+1 (the next time) 
	# is less than times.shape[0], otherwise there wil be an index error
	if time_index+1 < times.shape[0]:
		# add the new lat/lon pair to the end of the track_object's latlon_list	
		track_object.add_latlon((new_lat_value,new_lon_value))
		# add the next time to the end of the track_objects time list
		track_object.add_time(times[time_index+1])

	return




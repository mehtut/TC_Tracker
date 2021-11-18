# TC_Tracker
Tropical cyclone tracking algorithm

## Readme
The tropical cyclone (TC) tracking algorithm has multiple components. The main program that is executed is `TC_Tracks.py`. The arguments model type (WRF), climate scenario type (Historical, late_century), and the year are parsed from the command line. To run the program, for example, type:
```
python TC_Tracks.py --model 'WRF' --scenario 'Historical' --year '2001'
```
Before running this program, check and make sure that all modules are available. To run the tracker, update the time range on line 124 of `TC_Tracks.py`. The locations of the data in `Pull_TC_data.py` need to be udpated on lines 27 and 122. 

Tracks are saved as track objects (e.g. .obj files). To open the track objects in another program, include the following line of code in the module imports:
```
from TC_Tracks import *  
```
To open track object files, use the following code to load the list of all track objects:
```
track_file = open('track_file_name.obj', "rb")
TC_track_list = pickle.load(track_file) 
```
Each track object will have the attributes defined in TC_Tracks.py.

To iterate through all the track objects, use the following code:
```
for tc_track in TC_track_list:
  print(tc_track.latlon_list)	
```

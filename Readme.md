# This project is still in progress, but the preprocessing is now finished. 
This means you can extract all custom mapped songs to analyze or predict on it, 
including notes, obstacles and show effects.
Map versions with custom modded data (values out of normal boundaries) are excluded,
so the data is as smooth as possible. 

### If you are looking for a working automapper, search for BeatSage.
As of now this upload is made for easy analyzation of the custom maps.

### Author: Frederic Brenner
frederic.brenner@tum.de

## To adjust paths go to:
tools/config/paths.py

## folder structure needed for this version:
.../Automapper_data/training/fail_list/

.../Automapper_data/training/maps/

.../Automapper_data/training/maps_dict_all/

.../Automapper_data/training/songs_diff/

.../Automapper_data/training/songs_egg/

Create them on a disk of your choice and link the main folder in the paths.py file,
the program can't create them currently.

## starting oder for extraction of beatsaber data:
01/preprocessing/shift.py (whole preprocessing)

prediction may be coming later

## passing the beatsaber songs into a different directory (e.g. for PowerBeats VR)
tools/PowerBeats_extension/PowerBeats_shift.py

The beatsaber path is taken from the paths.py file, 
the destination folder needs to be set inside the file

The song name detection is quite simple,
for better naming extract the files from the preprocessing algorithm (not implemented)

## map prediction is in progress
but several AI methods I tried couldn't reach the performance of a human mapper

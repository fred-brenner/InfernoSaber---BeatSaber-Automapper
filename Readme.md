# Automapper for Beatsaber made for expert+ levels 
Extract maps from Beatsaber/Bsaber to feed them into AI models
Map versions with custom modded data (values out of normal boundaries) are excluded,
so the data is as smooth as possible.

Automapper is trained on expert+ maps for average 5 notes-per-second in prediction

The automapper consists of 4 consecutive AI models:
1. Deep convolutional autoencoder - to encode the music/simplify all other models
2. Temporal Convolutional Network (TCN) - to generate the beat
3. Deep Neural Network (Classification) - mapping the notes/bombs
4. Deep Neural Network (Classification) - mapping the events/lights

An overview over the current status of map generation (including BSMapper and BeatSage) can be found at:
https://youtu.be/2bP9YcAgG-E

### Author: Frederic Brenner
frederic.brenner@tum.de

## To install suitable python environment import in Anaconda:
anaconda_environment.yaml

## To run automapper go to:
Download models from GDrive link in model_data/Data/model/link_to_model.txt
run main.py

## To adjust paths go to:
tools/config/paths.py

## To adjust difficulty go to:
tools/config/config.py


## folder structure needed for this version:
can be automatically created with:
tools/config/check_folder_structure.py

## starting extraction of beatsaber data:
01/preprocessing/shift.py (whole preprocessing)


## passing the beatsaber songs into a different directory (e.g. for PowerBeats VR)
tools/PowerBeats_extension/PowerBeats_shift.py

The beatsaber path is taken from the paths.py file, 
the destination folder needs to be set inside the file

The song name detection is quite simple,
for better naming extract the files from the preprocessing algorithm (not implemented)

## Automapper is finally there. Have fun!

## Flexible Automapper for Beatsaber made for any difficulty
    
    Automapper with fully adjustable difficulty (inpsired by star difficulty) ranging from easy maps (1) to Expert++ maps (10+)

    New: Get maps from the discord bot (if online): https://discord.gg/cdV6HhpufY

    Recommendation: Generate BeatSaber maps using AI with the convenience of Google Drive storage.
    Use the Google Colab template included in the repository without the need of hardware.
    
    Alternative: Install the project locally with anaconda (not recommended)


## Roadmap 2023
Already Finished:

    Publish InfernoSaber
    Add obstacles in unused spaces
    Simple cardio obstacle model

Mid of 2023:

    Create InfernoSaber website/independent server (?)
    Rework AI model to create "impossible" note speeds

End of 2023:

    Create Cardio Obstacle AI Model
    Check out Reinforcement Models
    Add bombs in unused spaces (?)


## Automapper for Beatsaber made for expert+ levels

Extract maps from Beatsaber/Bsaber to feed them 
into AI models.
Map versions with custom modded data (values out of normal boundaries) are excluded,
so that the data is as smooth as possible.

Automapper is trained on expert+ maps for 
average 6 notes-per-second in prediction

The automapper consists of 4 consecutive AI models:
1. Deep convolutional autoencoder - to encode the music/simplify all other models
2. Temporal Convolutional Network (TCN) - to generate the beat
3. Deep Neural Network (Classification) - mapping the notes/bombs
4. Deep Neural Network (Classification) - mapping the events/lights

An overview over the current status of map generation (and past ones) can be found at:
https://youtu.be/2uK22jXeNLw

### Author: Frederic Brenner
frederic.brenner@tum.de

## Alternatively you can run the project local:

## To install suitable python environment import in Anaconda:
anaconda_environment.yaml

## To run automapper go to:
[Outdated] Download models from GDrive link in model_data/Data/model/link_to_model.txt

[Currently] Create Colab notebook and download model data from the created GDrive repository.

run main.py

## To adjust paths go to:
tools/config/paths.py

## To adjust difficulty go to:
tools/config/config.py

especially max_speed,
change the rest with caution!


## folder structure needed for this version:
can be automatically created with:
tools/config/check_folder_structure.py

## starting extraction of beatsaber data:
01/preprocessing/shift.py (whole preprocessing)

see end of main.py file for more information on training order


## passing the beatsaber songs into a different directory (e.g. for PowerBeats VR)
tools/PowerBeats_extension/PowerBeats_shift.py

The beatsaber path is taken from the paths.py file, 
the destination folder needs to be set inside the file

The song name detection is quite simple,
for better naming extract the files from the preprocessing algorithm (not implemented)

## Automapper is finally there. Have fun!

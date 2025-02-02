## Flexible Automapper for Beatsaber made for any difficulty
    
    Automapper with fully adjustable difficulty (inpsired by star difficulty) ranging from easy maps (1) to Expert++ maps (10+)

    Update 2025: App is finally available via Pinokio: https://program.pinokio.computer/#/
    Just got to "Discover" and then "Download from URL": https://github.com/fred-brenner/InfernoSaber-App

    This installs all dependencies in the capsulated environment of Pinokio and loads the application from (this) main repository:
    https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/main_app

    Alternatively:

    Join the Discord and let the bot generate single difficulty maps for you (not always available):
    https://discord.com/invite/cdV6HhpufY

    ... Or clone the repo yourself (Note: Use a conda environment to install audio packages on windows machines)


## Roadmap 2025

    Increase number of models to improve accuracy and enable more options
    
    Support new features for InfernoSaber Pinokio App    

## 2023 Notes: Automapper for Beatsaber made for expert+ levels

You can also train your own models on your favorite maps and difficulty.
This can only be done locally with cloning the repo and using GPU (one better consumer GPU is enough)
A guide to train the 4 models is included in the repo: How_to_Train_InfernoSaber.docx

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
[Automatically] download model data from the HuggingFace repository (already implementated in code).

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

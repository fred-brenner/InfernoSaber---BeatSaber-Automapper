
<br/>
<div align="center">
<a href="https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/edit/main_app">
<img src="https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/blob/main_app/app_helper/cover.jpg" alt="Logo" width="80" height="80">
</a>
<h3 align="center">InfernoSaber</h3>
<p align="center">
Flexible Automapper for Beatsaber made for any difficulty
<br/>
<br/>
<a href="https://www.youtube.com/watch?v=GpdHE6puDng"><strong>Installation walkthrough »</strong></a>
<br/>
<br/>
<a href="https://www.youtube.com/watch?v=wJSOBuKs42Q">View Demo .</a>  
<a href="https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/issues">Report Bug .</a>
<a href="https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/discussions">Request Feature</a>
</p>
</div>

## About The Project

![Screenshot played by RamenBot](https://i.imgur.com/ECXMxY5.jpeg)

Automapper with fully adjustable difficulty (inpsired by star difficulty) ranging from easy maps (< 1) to Expert+++ maps (10+)

Update Jan 2025: App is finally available via Pinokio: https://program.pinokio.computer/#/
Just got to "Discover" and then "Download from URL": https://github.com/fred-brenner/InfernoSaber-App

This installs all dependencies in the capsulated environment of Pinokio and loads the application from (this) main repository:
https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/main_app

Alternatively:

Join the Discord and let the bot generate single difficulty maps for you (currently not available):
https://discord.com/invite/cdV6HhpufY

... Or clone the repo yourself (Note: Use a conda environment to install audio packages on windows machines)

### Built With

The automapper currently consists of 4 consecutive AI models:

1. Deep convolutional autoencoder - to encode the music/simplify all other models
2. Temporal Convolutional Network (TCN) - to generate the beat
3. Deep Neural Network (Classification) - mapping the notes/bombs
4. Deep Neural Network (Classification) - mapping the events/lights

## Getting Started

Install via Pinokio. A walkthrough is given in: https://www.youtube.com/watch?v=GpdHE6puDng

This project is open-source, free-to-use and will remain so. Enjoy :)

### Prerequisites

Current pinokio version from: https://github.com/pinokiocomputer/pinokio/releases

### Installation

(Not recommended) You can also clone the repo yourself. Note: Conda environment works best to install audio packages on windows machines

## Usage

The inference usage is simplified with the included app in branch 'main_app'. The AI models will be automatically downloaded during runtime from [Hugging Face](https://huggingface.co/BierHerr/InfernoSaber), if not yet available.

You can also train your own models on your favorite maps and difficulty. This can only be done locally with cloning the repo and using GPU (one better consumer GPU is enough) A guide to train the 4 models is included in the repo: 'How_to_Train_InfernoSaber.docx'

Extract maps from Beatsaber/Bsaber to feed them into AI models. Map versions with custom modded data (values out of normal boundaries) are excluded, so that the data is as smooth as possible.

## Roadmap

[ ] Increase number of models to improve accuracy and enable more options

[x] Support new features for InfernoSaber Pinokio App

[ ] Get the discord bot back online (yt blocks the bot)

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star and join the [Discord community](https://discord.com/invite/cdV6HhpufY)! Thanks again!

1. Fork the Project and checkout the 'main_app' branch
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [MIT License](https://opensource.org/licenses/MIT) for more information.

## Contact

I've been working on this app since the release of BeatSaber and am happy to share the progress here!

Author: Frederic Brenner
frederic.brenner@tum.de

## Acknowledgments

Thanks for the many contributions on Discord and Github so far. Here, I want to thank the code contributors
- [aCzenderCa](https://github.com/aCzenderCa) - App enhancement and fixes
- [tjoen](https://github.com/tjoen) - Prototype for Pinokio install script

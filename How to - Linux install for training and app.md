# How to Train Your BeatSaber Automapper with InfernoSaber

## System Requirements

**Recommended setup for training:**
- **GPU:** NVIDIA with ≥ 8GB VRAM (NOTE: Last tested May 2025, current NVIDIA GPUs (5000 series) are not supported)
- **RAM:** ≥ 24GB
- **OS:** Linux (or WSL2 on Windows)
- *(This spec supports ~50–150 songs, depending on variety)*

**Recommended setup for inference/app execution:**
- **GPU:** None
- **RAM:** ≥ 8GB
- **OS:** Windows, Linux

---

## Using WSL2 on Windows

You will need Linux for training (and only for that).
If you’re not using Linux natively or via dual boot, follow guides online to set up WSL2 and increase its memory allocation (reserving less for Windows). Tested with NVIDIA 30-series GPUs. Failed with NVIDIA 50-series GPUs due to driver issues.

---

## Installation

1. **Clone the repository**:  
   [InfernoSaber GitHub](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper)
Use the `main` branch for the latest stable version for training. Use the `main_app` branch for inference.

2. **Editor Recommendation**:  
   Use **PyCharm** or **VSCode** for project editing.

3. **Install WSL2 and Python 3.10**:  
   [Guide](https://learn.microsoft.com/en-us/windows/python/web-frameworks#install-windows-subsystem-for-linux)

4. **Update and upgrade packages**:
   ```bash
   sudo apt update && sudo apt upgrade

5. **Create your preferred Python env**:
   go to InfernoSaber folder with cd, ls:
   ```bash
   cd mnt/c/Users/YourUsername/Desktop/BS_Automapper/InfernoSaber---BeatSaber-Automapper
   ```
   ```bash
   sudo apt install -y software-properties-common
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt install -y python3.10 python3.10-venv python3.10-dev
   python3.10 --version
   python3.10 -m venv ubuntu_venv
   ```
   Always after startup:
   ```bash
   source ubuntu_venv/bin/activate
   ```

6. **Install TensorFlow with CUDA (for NVIDIA)**:
   ```bash
   pip install tensorflow[and-cuda]==2.15
   ```
   Tested with TensorFlow 2.15
   For app install only (no training), you don't need CUDA:
   ```bash
   pip install tensorflow==2.15
   ```

7. **Install required dependencies**:
   ```bash
   sudo apt install libswresample-dev libsamplerate-dev libsndfile-dev txt2man doxygen
   sudo apt install python3-aubio aubio-tools ffmpeg libavcodec-extra
   sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
   ```

8. **In case of `aubio` issues (skip else)**:
   ```bash
   pip uninstall -yv aubio
   pip install --force-reinstall --no-cache-dir --verbose aubio
   ```

9. **Install Python requirements**:
   ```bash
   pip install git+https://git.aubio.org/aubio/aubio/
   ```
   Aubio tends to make problems. Alternative is to install via Conda or try to use pip install aubio.
   
   Make sure all already installed versions are removed from the requirements.txt (tensorflow, keras, aubio)
   ```bash
   pip install -r requirements.txt
   ```

10. **Configure your paths**:  
    Edit `/tools/config/paths.py` and set desired folders.

---

## Configuration

Edit `config.py` for training setup:

```python
use_mapper_selection = "your_model_name"
use_bpm_selection = True  # Set to False for advanced sorting
min_bps_limit = 1         # Songs with less will be excluded
max_bps_limit = 500       # Optional upper bound
training_songs_diff = "Expert"
training_songs_diff2 = "Hard"
allow_training_diff2 = True
vram_limit = 8            # Adjust based on GPU
autoenc_song_limit = 100  # Reduce if needed
mapper_song_limit = 200   # Approx. 200 per 30GB RAM
beat_song_limit = 200     # Same as above
```

Songs exceeding these limits will be randomly discarded.

---

## Folder Structure

Before starting, review the folder architecture from the **Pinokio app** and replicate it in your working "Data" directory. Configure the `config.py` to match.
When using the app from `main_app`, a script will always be started to create the necessary folder structure.

---

## Preparing Input Songs

1. **Back up your Beat Saber songs**  
   Copy them to a separate folder and set this path in `bs_input_path`.

2. **Select your favorite maps**:
   - Run `hashtest.py` after adapting paths.
   - Run `copyfavorites.py` to copy selected songs.

3. **Clean and format songs**:
   ```bash
   cd InfernoSaber  # Must run from the project root
   python3 bs_shift/cleanup_n_format.py
   ```

   This formats `.dat` files and prepares songs for training.

---

## Training

Run the training script:
```bash
python3 main_training.py
```

You’ll be prompted five times—type `yyyyy` to run all stages consecutively.

If interrupted or one stage fails, restart from where you left off (e.g., `nnnyy`).

---

## Testing the Environment

Check if TensorFlow detects the GPU:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:
```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Check memory available to Linux:
```bash
free
```

---

## Troubleshooting

For questions or issues, reach out on **Discord**.  
Check the **improvements channel** for successful implementations and community tips.

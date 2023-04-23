import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from tools.config import config, paths
from map_creation.sanity_check import add_breaks
from tools.utils.load_and_save import load_pkl

data = load_pkl(paths.temp_path)
[notes_l, notes_r, timings] = data

notes_l = add_breaks(notes_l, timings)
notes_r = add_breaks(notes_r, timings)

notes_l

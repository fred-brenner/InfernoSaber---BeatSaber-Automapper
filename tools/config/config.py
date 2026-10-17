import numpy as np
from typing import Optional

# from map_creation.sanity_check import improve_timings

########################################
# config file for all important values
# used in multiple codes
########################################
InfernoSaber_version = "1.7.2.app1-reinstall"  # coded into the info.dat file


def get_clean_version(version: Optional[str] = None) -> str:
    """Return ``version`` without any reinstall flag suffix."""

    suffix = "-reinstall"
    if version is None:
        version = InfernoSaber_version
    if version.endswith(suffix):
        return version[: -len(suffix)]
    return version
bs_mapping_version = "v3"  # allows to generate advanced features like arcs
# bs_mapping_version = "v2"  # legacy mode, may be deprecated in future

"""Do change"""
# select mapper style
# use_mapper_selection = "pp3_15"
# use_mapper_selection = "easy_15"
# use_mapper_selection = "hard_15"
# use_mapper_selection = "expert_15"
use_mapper_selection = "fav_15"

# select song metadata search mode
enable_auto_metadata = True  # if True, tries to get song metadata from song name
# allowed patterns for metadata: artist, title
metadata_naming_convention = "{artist} - {title}"  # naming convention for auto metadata

"""Change only for new training"""
use_mapper_selection = use_mapper_selection.lower()
use_bpm_selection = True   # use number of beats for selection of maps in training
min_bps_limit = 2  # minimum beats_per_second value for training
max_bps_limit = 12  # maximum beats_per_second value for training

"""Change for application"""
# Map creation model configuration
max_speed = 4 * 5.0  # set around 5-40 (normal-expert++)
add_beat_intensity = 95  # try to match bps by x% [80, 120]
gimme_more_notes_flag = True   # try to always use notes on both sides
gimme_more_notes_prob = 0.25     # probability to activate [0.0-1.0]
cdf = 1.2  # cut director factor (to calculate speed, [0.5, 1.5])
cdf_lr = 1.15  # speed addition factor for left right movement
expert_fact = 0.63  # expert plus to expert factor [0.6, 0.7]
create_expert_flag = True  # create second expert map
thresh_beat = 0.45  # minimum beat response required to trigger generator [0.3, 0.6]
thresh_onbeat = 0.09
# thresh_pitch = 0.90  # minimum beat for pitch check (0.8,low-1.5,high)
threshold_start = 1.1  # factor for start and end threshold [0.8, 1.2]
threshold_end = 0.7    # factor for start and end threshold (squared) [0.6, 1.1]
factor_pitch_certainty = 0.5  # select emphasis on first (>1) or second pitch method
factor_pitch_meanmax = 3    # select pitch certainty for mean (>=3) or max (<3)
random_note_map_factor = 0.3  # stick note map to random song/center (set to 0 to disable)
random_note_map_change = 3  # change frequency for center (1-5)
# quick_start = 0     # map quick start mode (0 off, 1-3 on)    # TODO: rework
t_diff_bomb = 1.5   # minimum time between notes to add bomb
t_diff_bomb_react = 0.3  # minimum time between finished added bombs
allow_mismatch_flag = False  # if True, wrong turned notes won't be removed
flow_model_flag = True  # use improved direction flow
# furious_lighting_flag = False  # increase frequency of light effects
normalize_song_flag = True  # normalize song volume
increase_volume_flag = True  # increase song volume (only used in combination with normalize flag)
audio_rms_goal = 0.60
allow_dot_notes = False  # if False, all notes must have a cut direction
jump_speed_offset = -0.2
map_filler_iters = 10  # max iterations for map filler
add_dot_notes = 2  # add dot notes for fastest patterns in percent [0-10]
add_breaks_flag = True  # add breaks after strong patterns
silence_threshold = 0.20   # silence threshold quantile value [0.0, 0.3]
silence_thresh_hard = 0.25  # add fixed threshold to dynamic value [0-2]
add_silence_flag = True  # whether to apply silence threshold
emphasize_beats_flag = True  # emphasize beats into double notes
single_notes_only_flag = False  # no more double notes
single_notes_only_strict_flag = True  # if previous flag applied, restricts to one note at a time
single_notes_remove_lr = 0.4   # >0.5 removes left, <0.5 removes right more often
add_obstacle_flag = True  # add obstacles in free areas
obstacle_time_gap = [0.3, 0.8]  # time gap before [0.2-1] after [0.5-2]
obstacle_min_duration = 0.1  # minimum duration for each obstacle [0.1-2]
obstacle_max_count = 2  # maximum appearance count for obstacles
sporty_obstacles = False
add_slider_flag = True  # add arcs between notes in free areas
slider_time_gap = [0.5, 12.0]    # time gap in seconds
slider_probability = 0.8    # [0.1-1.0] with 1 meaning all on
slider_movement_minimum = 3   # minimum movement between notes [0-5]
slider_radius_multiplier = 1.0  # [0.5-2.0]
slider_turbo_start = True   # start slider towards first note
auto_move_song_afterwards = False  # Cut out song after generation and move to y_done folder

check_all_first_notes = True  # if False only change dot notes
first_note_layer_threshold = 1  # Layer index from where first note should face up [0(all up)-3(all down)]
allow_double_first_notes = False  # if False remove second note if necessary for first occurrence
# improve timings
# improve_timings_mcfactor = 2.5  # max change bandwidth (2 wide, 4+ narrow)
# improve_timings_mcchange = 1.2   # max change time in seconds
# improve_timings_act_time = 0.35  # min time gap to activate

add_waveform_pattern_flag = 0   # [0: off, 1: on, 2: double on]
waveform_pattern = [
    [0, 1, 2, 3, 2, 1],
    [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1],
    [0, 1, 2, 1, 2, 3, 2, 1, 2, 1, 0],
    [0, 1, 2, 1],
    [1, 2, 3, 2],
    [0, 2, 1, 3, 1, 2],
    # [0, 1],
    # [2, 3],
    # [0, 3, 0, 1, 2, 3, 2, 1, 0, 3],
]
waveform_apply_dir = [0, 4, 5, 1, 6, 7]     # either [0, 1] or [0, 4, 5, 1, 6, 7]
waveform_pattern_length = 25   # pattern length in sampling rate [10-200]
waveform_threshold = 4  # minimum number of notes applicable for waveform to start

"""Pinokio app only"""
num_workers = 1
silence_threshold_percentage = 100
difficulty_1 = 5
difficulty_2 = 0
difficulty_3 = 0
difficulty_4 = 0
difficulty_5 = 0

"""Caution on changes"""
verbose_level = 1       # verbose level from 0 (only ETA) to 5 (all messages)
obstacle_crouch_width = 4
obstacle_width = 1
max_obstacle_height = 5     # <= 5
# normal obstacles
norm_obstacle_allowed_types = [0, 1, 2]  # 0wall, 1ceiling, 2jump, 3onesaber
norm_obstacle_positions = [[0], [3]]  # outside position of notes
# sporty obstacles
sport_obstacle_allowed_types = [0, 1]  # (ceiling walls for crouch are fixed)
sport_obstacle_positions = [[0, 1, 1], [2, 2, 3]]  # inside position of notes

check_silence_flag = True  # check for extremely silent songs
check_silence_value = -12.6  # value in dB [-15 (low filter), -11 (high filter)]
jump_speed_expert_factor = 0.91     # factor from expert+ to expert
jsb_offset = [0.21, 0.15]  # note jump speed offset for Expert, Expert+ (range [-0.5, 0.5])
jsb_offset_min = [-0.2, -0.4]  # minimum allowed values (expert, expert+)
jsb_offset_factor = 0.011  # note jump factor for high difficulties
use_fixed_bpm = 0   # use fixed bpm or set to None for the song bpm
max_njs = 26  # maximum Note Jump Speed allowed
decr_speed_range = 30  # range for start and end (n first and last notes)
decr_speed_val = 0.35  # decrease max speed at start
reaction_time = 1.15   # reaction time (0.5-2)
reaction_time_fact = 0.013  # factor including max_speed
jump_speed = 12.4  # jump speed from beat saber (10-15)
jump_speed_fact = 0.310  # factor including max_speed
min_beat_time = 1 / 16  # in seconds (first sanity check)
beat_spacing = 5587 / 196  # 5587/196s = 28.5051 steps/s
# favor_last_class = 0.15     # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 25  # set maximum speed difference between double notes (10-30)
emphasize_beats_3 = 0.010  # fraction beats to triple
emphasize_beats_3_fact = 0.001  # factor incl max_speed
emphasize_beats_2 = 0.40  # fraction beats to double
emphasize_beats_2_fact = 0.002  # factor incl max_speed
emphasize_beats_quantile = 0.65     # disengage quantile of fast patterns
shift_beats_fact = 0.30  # fraction beats to shift in cut direction
# add_beat_fact = 0.90        # fraction add beats (beat_generator)
add_beat_max_bounds = [0.1, 0.5, 0.8, 1.6]
# pitches_allowed = [40, 50]  # percentage of pitches to be over threshold
add_start_end_beats = True


"""Do not change unless you change the DNN models"""
# Data Processing configuration
training_songs_diff = 'ExpertPlus'
allow_training_diff2 = True
training_songs_diff2 = 'Expert'  # second try if first one is not available

general_diff = 'ExpertPlus'
exclude_requirements = False    # exclude all maps which have any kind of requirement noted
random_seed = 3
min_time_diff = 0.02  # minimum time between cuts, otherwise synchronized
samplerate_music = 14800  # samplerate for the music import
hop_size = 512
window = 2.0  # window in seconds for each song to spectrum picture (from wav_to_pic)
specgram_res = 24  # y resolution of the spectrogram (frequency subdivisions)
# ram_limit = 24      # free RAM in GB (unused currently)
vram_limit = 20     # free VRAM in GB (needed for lighting training)

# Model versions
enc_version = 'tf_model_enc_'
autoenc_version = 'tf_model_autoenc_'
mapper_version = 'tf_model_mapper_'
beat_gen_version = 'tf_beat_gen_'
event_gen_version = 'tf_event_gen_'

# Autoencoder model configuration
learning_rate = 3e-4  # model learning rate
n_epochs = 50  # number of total epochs
batch_size = 128  # batch size
test_samples = 10  # number of test files to plot (excluded from training)
bottleneck_len = 16  # size of bottleneck distribution (1D array)
autoenc_song_limit = 120    # maximum number of songs used for training, to not overload RAM

# Mapper model configuration
map_learning_rate = 4e-4  # model learning rate
map_n_epochs = 180  # number of total epochs
map_batch_size = 128  # batch size
map_test_samples = 10  # number of test files to plot (excluded from training)
lstm_len = 16
remove_double_notes = False
mapper_song_limit = 240    # maximum number of songs used for training, to not overload RAM

# Beat prediction model configuration
beat_learning_rate = 5e-4
beat_n_epochs = 80
beat_batch_size = 128
tcn_len = 24
tcn_test_samples = 350
delete_offbeats = 0.6  # < 1 delete non-beats to free ram
# tcn_skip = 10
beat_song_limit = 240    # maximum number of songs used for training, to not overload RAM

# New beat models: shared audio input (separate from legacy samplerate_music/hop_size)
# Changing these inputs requires preparing a new dataset; dimensions are recorded in its manifest.
beat_models_samplerate_music = 16000
beat_models_hop_size = 160
beat_models_fft_size = 512
beat_models_mel_bins = 80
beat_models_validation_fraction = 0.1
beat_models_test_fraction = 0.1

# Intensity model configuration (new whole-song normalized note distribution)
intensity_optimizer = 'Adam'
intensity_learning_rate = 5e-4
intensity_n_epochs = 80
intensity_batch_size = 1  # whole songs
intensity_clipnorm = 1.0
intensity_early_stopping_patience = 10
intensity_early_stopping_min_delta = 0.0
intensity_monitor = 'val_loss'
intensity_monitor_mode = 'min'  # min / max / auto
intensity_random_seed = random_seed
intensity_shuffle = True
intensity_verbose = 2
intensity_channels = 48
intensity_kernel_size = 3
intensity_convolutions_per_block = 2
intensity_dilations = (1, 2, 4, 8)
intensity_dropout = 0.1
intensity_activation = 'relu'

# Timing model configuration (new zero/one/two-note frame counts)
timing_optimizer = 'Adam'
timing_learning_rate = 5e-4
timing_n_epochs =80
timing_batch_size = 8  # context windows
timing_clipnorm = 1.0
timing_early_stopping_patience = 10
timing_early_stopping_min_delta = 0.0
timing_monitor = 'val_loss'
timing_monitor_mode = 'min'  # min / max / auto
timing_random_seed = random_seed
timing_shuffle = True
timing_verbose = 2
timing_conditioning = 'noisy-reference'  # reference / noisy-reference / predicted
timing_intensity_noise_std = 0.1
timing_channels = 64
timing_kernel_size = 3
timing_convolutions_per_block = 2
timing_dilations = (1, 2, 4, 8, 16, 32, 64)
timing_dropout = 0.1
timing_activation = 'relu'
timing_chunk_frames = 512  # context margins are derived from the receptive field
timing_count_loss_weight = 0.1
timing_count_huber_delta = 1.0
timing_count_rate_scale = 10.0  # NPS unit for the auxiliary count loss

# Notes model configuration (new per-hand position/direction categories)
notes_optimizer = 'Adam'
notes_learning_rate = 5e-4
notes_n_epochs = 80
notes_batch_size = 128  # note events
notes_clipnorm = 1.0
notes_early_stopping_patience = 10
notes_early_stopping_min_delta = 0.0
notes_monitor = 'val_loss'
notes_monitor_mode = 'min'  # min / max / auto
notes_random_seed = random_seed
notes_shuffle = True
notes_verbose = 2
notes_conditioning = 'noisy-reference'  # reference / noisy-reference / predicted
notes_intensity_noise_std = 0.1
notes_patch_frames = 200  # centered audio patch; must be even
notes_history_length = 32
notes_cnn_filters = (16, 32)
notes_cnn_kernel_size = 3
notes_pool_sizes = ((4, 4), (5, 4))
notes_audio_dense_units = 64
notes_embedding_dim = 16
notes_gru_units = 64
notes_shared_dense_units = 128
notes_history_dropout = 0.1
notes_activation = 'relu'
notes_red_loss_weight = 0.5
notes_blue_loss_weight = 0.5

# V2 concept pipeline configuration (independent downstream stages)
# Intensity/timing and their existing checkpoints retain their current contracts.
pipeline_v2_encoder_grid_seconds = 0.5
pipeline_v2_encoder_batch_size = 128
pipeline_v2_max_multiplier_templates = 256
pipeline_v2_max_wall_templates = 64
pipeline_v2_max_light_classes = 2048
pipeline_v2_max_copies = 11
pipeline_v2_max_chain_slices = 32
pipeline_v2_max_chain_span_seconds = 2.0
pipeline_v2_expansion_time_quantum = 0.005
pipeline_v2_chain_span_quantum = 0.02
pipeline_v2_lighting_match_tolerance = 0.02
pipeline_v2_wall_min_gap_seconds = 0.8
pipeline_v2_wall_margin_before_seconds = 0.3
pipeline_v2_wall_margin_after_seconds = 0.3
pipeline_v2_wall_strobe_grid_beats = 1.0  # zero disables candidate proposals
pipeline_v2_wall_strobe_duration_seconds = 0.1
pipeline_v2_wall_strobe_min_gap_seconds = 0.3
pipeline_v2_clearance_seconds = 0.1  # occupancy context; gameplay clearance still requires review
pipeline_v2_bomb_threshold = 0.5
pipeline_v2_lighting_threshold = 0.5
pipeline_v2_bomb_grid_beats = 0.5
pipeline_v2_bomb_min_gap_seconds = 0.3
pipeline_v2_lighting_grid_beats = 0.5
pipeline_v2_lighting_onset_threshold = 1.0
pipeline_v2_arc_min_gap_seconds = 0.15
pipeline_v2_arc_max_gap_seconds = 0.8
pipeline_v2_max_candidates = 10000

# V2 positions: training and network parameters
positions_optimizer = 'Adam'
positions_learning_rate = 0.0005
positions_n_epochs = 80
positions_batch_size = 128
positions_clipnorm = 1.0
positions_early_stopping_patience = 10
positions_early_stopping_min_delta = 0.0
positions_monitor = 'val_loss'
positions_monitor_mode = 'min'
positions_shuffle = True
positions_verbose = 2
positions_random_seed = random_seed
positions_history_length = 32
positions_embedding_dim = 16
positions_audio_dense_units = 64
positions_gru_units = 64
positions_shared_dense_units = 128
positions_activation = 'relu'
positions_history_dropout = 0.1
positions_loss_weights = {'red': 1.0, 'blue': 1.0}

# V2 arrows: training and network parameters
arrows_optimizer = 'Adam'
arrows_learning_rate = 0.0005
arrows_n_epochs = 80
arrows_batch_size = 128
arrows_clipnorm = 1.0
arrows_early_stopping_patience = 10
arrows_early_stopping_min_delta = 0.0
arrows_monitor = 'val_loss'
arrows_monitor_mode = 'min'
arrows_shuffle = True
arrows_verbose = 2
arrows_random_seed = random_seed
arrows_history_length = 32
arrows_embedding_dim = 16
arrows_audio_dense_units = 64
arrows_gru_units = 64
arrows_shared_dense_units = 128
arrows_activation = 'relu'
arrows_history_dropout = 0.1
arrows_loss_weights = {'red': 1.0, 'blue': 1.0}

# V2 multiplier: training and network parameters
multiplier_optimizer = 'Adam'
multiplier_learning_rate = 0.0005
multiplier_n_epochs = 80
multiplier_batch_size = 128
multiplier_clipnorm = 1.0
multiplier_early_stopping_patience = 10
multiplier_early_stopping_min_delta = 0.0
multiplier_monitor = 'val_loss'
multiplier_monitor_mode = 'min'
multiplier_shuffle = True
multiplier_verbose = 2
multiplier_random_seed = random_seed
multiplier_history_length = 32
multiplier_embedding_dim = 16
multiplier_audio_dense_units = 64
multiplier_gru_units = 64
multiplier_shared_dense_units = 128
multiplier_activation = 'relu'
multiplier_history_dropout = 0.1
multiplier_loss_weights = {'red': 1.0, 'blue': 1.0}

# V2 hazards: training and network parameters
hazards_optimizer = 'Adam'
hazards_learning_rate = 0.0005
hazards_n_epochs = 80
hazards_batch_size = 128
hazards_clipnorm = 1.0
hazards_early_stopping_patience = 10
hazards_early_stopping_min_delta = 0.0
hazards_monitor = 'val_loss'
hazards_monitor_mode = 'min'
hazards_shuffle = True
hazards_verbose = 2
hazards_random_seed = random_seed
hazards_history_length = 32
hazards_embedding_dim = 16
hazards_audio_dense_units = 64
hazards_gru_units = 64
hazards_shared_dense_units = 128
hazards_activation = 'relu'
hazards_history_dropout = 0.1
hazards_loss_weights = {'wall': 1.0, 'bombs': 1.0}

# V2 lighting: training and network parameters
lighting_optimizer = 'Adam'
lighting_learning_rate = 0.0005
lighting_n_epochs = 80
lighting_batch_size = 8
lighting_clipnorm = 1.0
lighting_early_stopping_patience = 10
lighting_early_stopping_min_delta = 0.0
lighting_monitor = 'val_loss'
lighting_monitor_mode = 'min'
lighting_shuffle = True
lighting_verbose = 2
lighting_random_seed = random_seed
lighting_block_length = 16
lighting_conv_filters = 32
lighting_time_lstm_units = 32
lighting_history_lstm_units = 64
lighting_combined_lstm_units = 64
lighting_dense_units = 1024
lighting_dropout = 0.05
lighting_loss_weights = {'events': 1.0}

# V2 arcs: training and network parameters
arcs_optimizer = 'Adam'
arcs_learning_rate = 0.0005
arcs_n_epochs = 80
arcs_batch_size = 128
arcs_clipnorm = 1.0
arcs_early_stopping_patience = 10
arcs_early_stopping_min_delta = 0.0
arcs_monitor = 'val_loss'
arcs_monitor_mode = 'min'
arcs_shuffle = True
arcs_verbose = 2
arcs_random_seed = random_seed
arcs_history_length = 32
arcs_embedding_dim = 16
arcs_audio_dense_units = 64
arcs_gru_units = 64
arcs_shared_dense_units = 128
arcs_activation = 'relu'
arcs_history_dropout = 0.1
arcs_loss_weights = {'attach': 1.0, 'shape': 1.0, 'anchor': 1.0}
arcs_shape_bounds = (0.0, 2.0)  # mu/tmu; out-of-range source controls are masked and reported

# Event prediction model configuration
event_learning_rate = 1e-3
event_n_epochs = 180
event_lstm_len = 16
event_batch_size = 128
# event_song_limit covered by vram_limit

# needed for reset
max_speed_orig = max_speed
add_beat_intensity_orig = add_beat_intensity
silence_threshold_orig = silence_threshold
jump_speed_offset_orig = jump_speed_offset
obstacle_time_gap = np.asarray(obstacle_time_gap)
obstacle_time_gap_orig = obstacle_time_gap
thresh_beat_orig = thresh_beat
thresh_onbeat_orig = thresh_onbeat

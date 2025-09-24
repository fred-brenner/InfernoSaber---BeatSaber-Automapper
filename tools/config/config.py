import numpy as np


class Config:
    """Container for InfernoSaber configuration values."""

    def __init__(self) -> None:
        """Initialize configuration with default values."""
        # General settings
        self.InfernoSaber_version = "1.7.1.app8"  # coded into the info.dat file
        self.bs_mapping_version = "v3"  # allows to generate advanced features like arcs
        # self.bs_mapping_version = "v2"  # legacy mode, may be deprecated in future

        # Mapper selection
        # self.use_mapper_selection = "pp3_15"
        # self.use_mapper_selection = "easy_15"
        # self.use_mapper_selection = "hard_15"
        # self.use_mapper_selection = "expert_15"
        self.use_mapper_selection = "fav_15"

        # Training configuration
        self.use_mapper_selection = self.use_mapper_selection.lower()
        self.use_bpm_selection = True   # use number of beats for selection of maps in training
        self.min_bps_limit = 2  # minimum beats_per_second value for training
        self.max_bps_limit = 12  # maximum beats_per_second value for training

        # Map creation model configuration
        self.max_speed = 4 * 8.0  # set around 5-40 (normal-expert++)
        self.add_beat_intensity = 95  # try to match bps by x% [80, 120]
        self.gimme_more_notes_flag = True   # try to always use notes on both sides
        self.gimme_more_notes_prob = 0.25     # probability to activate [0.0-1.0]
        self.cdf = 1.2  # cut director factor (to calculate speed, [0.5, 1.5])
        self.cdf_lr = 1.15  # speed addition factor for left right movement
        self.expert_fact = 0.63  # expert plus to expert factor [0.6, 0.7]
        self.create_expert_flag = True  # create second expert map
        self.thresh_beat = 0.45  # minimum beat response required to trigger generator [0.3, 0.6]
        self.thresh_onbeat = 0.09
        # self.thresh_pitch = 0.90  # minimum beat for pitch check (0.8,low-1.5,high)
        self.threshold_start = 1.1  # factor for start and end threshold [0.8, 1.2]
        self.threshold_end = 0.7    # factor for start and end threshold (squared) [0.6, 1.1]
        self.factor_pitch_certainty = 0.5  # select emphasis on first (>1) or second pitch method
        self.factor_pitch_meanmax = 3    # select pitch certainty for mean (>=3) or max (<3)
        self.random_note_map_factor = 0.3  # stick note map to random song/center (set to 0 to disable)
        self.random_note_map_change = 3  # change frequency for center (1-5)
        # self.quick_start = 0     # map quick start mode (0 off, 1-3 on)    # TODO: rework
        self.t_diff_bomb = 1.5   # minimum time between notes to add bomb
        self.t_diff_bomb_react = 0.3  # minimum time between finished added bombs
        self.allow_mismatch_flag = False  # if True, wrong turned notes won't be removed
        self.flow_model_flag = True  # use improved direction flow
        # self.furious_lighting_flag = False  # increase frequency of light effects
        self.normalize_song_flag = True  # normalize song volume
        self.increase_volume_flag = True  # increase song volume (only used in combination with normalize flag)
        self.audio_rms_goal = 0.60
        self.allow_dot_notes = False  # if False, all notes must have a cut direction
        self.jump_speed_offset = -0.2
        self.map_filler_iters = 10  # max iterations for map filler
        self.add_dot_notes = 2  # add dot notes for fastest patterns in percent [0-10]
        self.add_breaks_flag = True  # add breaks after strong patterns
        self.silence_threshold = 0.20   # silence threshold quantile value [0.0, 0.3]
        self.silence_thresh_hard = 0.25  # add fixed threshold to dynamic value [0-2]
        self.add_silence_flag = True  # whether to apply silence threshold
        self.emphasize_beats_flag = True  # emphasize beats into double notes
        self.single_notes_only_flag = False  # no more double notes
        self.single_notes_only_strict_flag = True  # if previous flag applied, restricts to one note at a time
        self.single_notes_remove_lr = 0.4   # >0.5 removes left, <0.5 removes right more often
        self.add_obstacle_flag = True  # add obstacles in free areas
        self.obstacle_time_gap = np.asarray([0.3, 0.8])  # time gap before [0.2-1] after [0.5-2]
        self.obstacle_min_duration = 0.1  # minimum duration for each obstacle [0.1-2]
        self.obstacle_max_count = 2  # maximum appearance count for obstacles
        self.sporty_obstacles = False
        self.add_slider_flag = True  # add arcs between notes in free areas
        self.slider_time_gap = [0.5, 12.0]    # time gap in seconds
        self.slider_probability = 0.8    # [0.1-1.0] with 1 meaning all on
        self.slider_movement_minimum = 3   # minimum movement between notes [0-5]
        self.slider_radius_multiplier = 1.0  # [0.5-2.0]
        self.slider_turbo_start = True   # start slider towards first note
        self.auto_move_song_afterwards = False  # Cut out song after generation and move to y_done folder

        self.check_all_first_notes = True  # if False only change dot notes
        self.first_note_layer_threshold = 1  # Layer index from where first note should face up [0(all up)-3(all down)]
        self.allow_double_first_notes = False  # if False remove second note if necessary for first occurrence
        # improve timings
        # self.improve_timings_mcfactor = 2.5  # max change bandwidth (2 wide, 4+ narrow)
        # self.improve_timings_mcchange = 1.2   # max change time in seconds
        # self.improve_timings_act_time = 0.35  # min time gap to activate

        self.add_waveform_pattern_flag = 0   # [0: off, 1: on, 2: double on]
        self.waveform_pattern = [
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
        self.waveform_apply_dir = [0, 4, 5, 1, 6, 7]     # either [0, 1] or [0, 4, 5, 1, 6, 7]
        self.waveform_pattern_length = 25   # pattern length in sampling rate [10-200]
        self.waveform_threshold = 4  # minimum number of notes applicable for waveform to start

        # Pinokio app settings
        self.num_workers = 4
        self.silence_threshold_percentage = 100
        self.difficulty_1 = 4
        self.difficulty_2 = 5
        self.difficulty_3 = 6
        self.difficulty_4 = 7
        self.difficulty_5 = 8

        # Advanced configuration
        self.verbose_level = 1       # verbose level from 0 (only ETA) to 5 (all messages)
        self.obstacle_crouch_width = 4
        self.obstacle_width = 1
        self.max_obstacle_height = 5     # <= 5
        # normal obstacles
        self.norm_obstacle_allowed_types = [0, 1, 2]  # 0wall, 1ceiling, 2jump, 3onesaber
        self.norm_obstacle_positions = [[0], [3]]  # outside position of notes
        # sporty obstacles
        self.sport_obstacle_allowed_types = [0, 1]  # (ceiling walls for crouch are fixed)
        self.sport_obstacle_positions = [[0, 1, 1], [2, 2, 3]]  # inside position of notes

        self.check_silence_flag = True  # check for extremely silent songs
        self.check_silence_value = -12.6  # value in dB [-15 (low filter), -11 (high filter)]
        self.jump_speed_expert_factor = 0.91     # factor from expert+ to expert
        self.jsb_offset = [0.21, 0.15]  # note jump speed offset for Expert, Expert+ (range [-0.5, 0.5])
        self.jsb_offset_min = [-0.2, -0.4]  # minimum allowed values (expert, expert+)
        self.jsb_offset_factor = 0.011  # note jump factor for high difficulties
        self.use_fixed_bpm = 0   # use fixed bpm or set to None for the song bpm
        self.max_njs = 26  # maximum Note Jump Speed allowed
        self.decr_speed_range = 30  # range for start and end (n first and last notes)
        self.decr_speed_val = 0.35  # decrease max speed at start
        self.reaction_time = 1.15   # reaction time (0.5-2)
        self.reaction_time_fact = 0.013  # factor including max_speed
        self.jump_speed = 12.4  # jump speed from beat saber (10-15)
        self.jump_speed_fact = 0.310  # factor including max_speed
        self.min_beat_time = 1 / 16  # in seconds (first sanity check)
        self.beat_spacing = 5587 / 196  # 5587/196s = 28.5051 steps/s
        # self.favor_last_class = 0.15     # set factor to favor the next beat class (0.0-0.3)
        self.max_double_note_speed = 25  # set maximum speed difference between double notes (10-30)
        self.emphasize_beats_3 = 0.010  # fraction beats to triple
        self.emphasize_beats_3_fact = 0.001  # factor incl max_speed
        self.emphasize_beats_2 = 0.40  # fraction beats to double
        self.emphasize_beats_2_fact = 0.002  # factor incl max_speed
        self.emphasize_beats_quantile = 0.65     # disengage quantile of fast patterns
        self.shift_beats_fact = 0.30  # fraction beats to shift in cut direction
        # self.add_beat_fact = 0.90        # fraction add beats (beat_generator)
        self.add_beat_max_bounds = [0.1, 0.5, 0.8, 1.6]
        # self.pitches_allowed = [40, 50]  # percentage of pitches to be over threshold
        self.add_start_end_beats = True

        # Data processing configuration
        self.training_songs_diff = 'ExpertPlus'
        self.allow_training_diff2 = True
        self.training_songs_diff2 = 'Expert'  # second try if first one is not available

        self.general_diff = 'ExpertPlus'
        self.exclude_requirements = False    # exclude all maps which have any kind of requirement noted
        self.random_seed = 3
        self.min_time_diff = 0.01  # minimum time between cuts, otherwise synchronized
        self.samplerate_music = 14800  # samplerate for the music import
        self.hop_size = 512
        self.window = 2.0  # window in seconds for each song to spectrum picture (from wav_to_pic)
        self.specgram_res = 24  # y resolution of the spectrogram (frequency subdivisions)
        # self.ram_limit = 24      # free RAM in GB (unused currently)
        self.vram_limit = 20     # free VRAM in GB (needed for lighting training)

        # Model versions
        self.enc_version = 'tf_model_enc_'
        self.autoenc_version = 'tf_model_autoenc_'
        self.mapper_version = 'tf_model_mapper_'
        self.beat_gen_version = 'tf_beat_gen_'
        self.event_gen_version = 'tf_event_gen_'

        # Autoencoder model configuration
        self.learning_rate = 3e-4  # model learning rate
        self.n_epochs = 50  # number of total epochs
        self.batch_size = 128  # batch size
        self.test_samples = 10  # number of test files to plot (excluded from training)
        self.bottleneck_len = 16  # size of bottleneck distribution (1D array)
        self.autoenc_song_limit = 120    # maximum number of songs used for training, to not overload RAM

        # Mapper model configuration
        self.map_learning_rate = 4e-4  # model learning rate
        self.map_n_epochs = 180  # number of total epochs
        self.map_batch_size = 128  # batch size
        self.map_test_samples = 10  # number of test files to plot (excluded from training)
        self.lstm_len = 16
        self.remove_double_notes = False
        self.mapper_song_limit = 240    # maximum number of songs used for training, to not overload RAM

        # Beat prediction model configuration
        self.beat_learning_rate = 5e-4
        self.beat_n_epochs = 80
        self.beat_batch_size = 128
        self.tcn_len = 24
        self.tcn_test_samples = 350
        self.delete_offbeats = 0.6  # < 1 delete non-beats to free ram
        # self.tcn_skip = 10
        self.beat_song_limit = 240    # maximum number of songs used for training, to not overload RAM

        # Event prediction model configuration
        self.event_learning_rate = 1e-3
        self.event_n_epochs = 180
        self.event_lstm_len = 16
        self.event_batch_size = 128
        # event_song_limit covered by vram_limit

        # Needed for reset
        self.max_speed_orig = self.max_speed
        self.add_beat_intensity_orig = self.add_beat_intensity
        self.silence_threshold_orig = self.silence_threshold
        self.jump_speed_offset_orig = self.jump_speed_offset
        self.obstacle_time_gap_orig = self.obstacle_time_gap
        self.thresh_beat_orig = self.thresh_beat
        self.thresh_onbeat_orig = self.thresh_onbeat


_CONFIG_INSTANCE = Config()


def get_config() -> Config:
    """Return the shared configuration instance."""
    return _CONFIG_INSTANCE


__all__ = ["Config", "get_config"]

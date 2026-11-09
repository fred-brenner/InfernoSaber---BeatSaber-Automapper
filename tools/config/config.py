import numpy as np
from typing import Optional

# from map_creation.sanity_check import improve_timings

########################################
# config file for all important values
# used in multiple codes
########################################
InfernoSaber_version = "2.0.1.app1"  # coded into the info.dat file


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
use_mapper_selection = "v2_expert"

# V2 generation bundle is independent of prediction/library folder settings.
v2_model_bundle_path = "../Data/model/v2_expert"
# Retained v1 lighting assets ship inside the v2 bundle so one branch covers all.
v1_lighting_model_path = "../Data/model/v2_expert/v1_lighting/tf_event_gen_2_12_5_5__22_18.h5"
v1_lighting_vocabulary_path = "../Data/model/v2_expert/v1_lighting/onehot_encoder_events.pkl"
v2_inference_threads = 2
v2_timing_batch_size = 8
v2_compile_inference = True
# Experimental: mask hazard decisions to feasible placements before argmax/threshold.
v2_hazard_feasibility_decode = False
# With feasibility decode on: sample among legal options at this temperature (0 = argmax).
v2_hazard_sample_temperature = 0.0
# Walls-only acceptance (2026-10-30): bomb placement disabled at decode — no
# head/head-combination produced a usable autoregressive operating point yet.
# Revisit entry lives in TODO.md; set True to re-enable bomb firing.
v2_hazard_bombs_enabled = False
# Experimental: rate-matching bomb decode — fire top-ranked eligible cells while the
# trailing-window placed count lags the model's own expected bomb mass.
v2_hazard_bomb_rate_decode = False
v2_hazard_bomb_rate_seconds = 2.0   # trailing window for expected-vs-placed density
v2_hazard_bomb_score_floor = 0.05   # minimum eligible-cell score to place at all
v2_hazard_bomb_rate_factor = 0.0    # scales predicted mass; 0 = use calibrated threshold
# Walls reverted to the pre-model implementation (2026-10-31): the legacy
# gen_obstacles rule generator fills note-free lane spans and validate_hazards
# applies the legacy margins/clearance — the model placed continuous walls at
# fixed spots interfering with notes. The whole hazards model stage is skipped
# (bombs already off); set False to decode walls with the model again.
v2_walls_heuristic = True
# Arcs reverted to the pre-model implementation (2026-10-31): adjacent same-hand
# pairs gated by slider_time_gap/slider_probability/slider_movement_minimum in
# validate_arcs. The arcs model had no usable autoregressive operating point
# (see docs/baseline.md Arm E/G/H); set False to decode with the model again.
v2_arcs_heuristic = True
# Sample positions/arrows categories at this temperature (0 = argmax). Sampling
# replaces the legacy random_note_map_factor mixture for note decoding.
v2_note_sample_temperature = 0.9

# V2 intensity postprocessing (generation only; training shares stay unchanged).
v2_intensity_low_boost = 0.3  # fraction of the deficit below uniform density [0, 1];
                              # experiment value (was 0.1) — more notes in quiet parts
v2_intensity_high_reduction = 0.1  # fraction of the excess above uniform density [0, 1]
v2_intensity_scale = 1.0  # overall target gain; nonnegative, 1 preserves average NPS
v2_intensity_adjustment_strength = 0.8  # fraction of the target change applied [0, 1]

# Timing event bias (tuning experiment 2026-11-04): flat offset added to the
# timing model's event classes (single/double) before the per-frame argmax,
# lifting borderline frames into events. Must stay < 1 so a fully silent
# frame can never flip; 0 restores pure argmax.
# timing model's event classes (single/double) before the per-frame argmax,
# lifting borderline frames into events. Must stay < 1 so a fully silent
# frame can never flip; 0 restores pure argmax. 2026-11-06: raised 0.1 -> 0.3
# for recall — the model stays silent on many audible hits in loud sections
# (measured ~2x raw event rate on under-triggering songs); downstream stages
# (silence removal, spacing floor, speed cap) absorb the accuracy cost.
v2_timing_event_bias = 0.3

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
max_speed = 4 * 8.0  # set around 5-40 (normal-expert++)
# Beat Intensity (%) is the app's difficulty dial: it scales the density-fill
# budget (v1 + v2) and, since 2026-11-06, the v2 speed cap via
# v2_speed_cap_scale (100% = the configured reference, others scale linearly).
add_beat_intensity = 90  # try to match bps by x% [40, 125]
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
add_bombs_flag = True     # add gameplay bombs (master switch incl. calm-effect bombs)
v2_bombs_prob = 0.25      # [0-1] probability a note gets a bomb proposal
v2_bomb_note_time_gap = 0.8   # seconds around each bomb kept clear of note swings
v2_bomb_note_cell_gap = 1     # cell rings around each note swing kept clear of bombs
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

shift_blocks_left_right_flag = True  # shift notes off the previous note's column
shift_blocks_middle_flag = True      # shift notes off the two center-middle cells
shift_blocks_up_down_flag = True     # shift notes along the previous note's cut direction
correct_notes_all_flag = True        # cross-hand static-position adjustment
sanity_direction_recut_flag = True   # rewrite cut directions from movement reversal
# V2 note cleanup relaxations applied only around the v2 legacy flow pass:
# sampled model directions/positions are trusted more than the homogenizers.
v2_cleanup_allow_dots = True       # effective allow_dot_notes during v2 cleanup
v2_cleanup_flow_recut = False      # effective flow_model_flag during v2 cleanup
v2_cleanup_lane_shift = False      # effective shift_blocks_left_right_flag during v2 cleanup
v2_cleanup_sanity_recut = False    # effective sanity_direction_recut_flag during v2 cleanup
v2_cleanup_add_dots = 0            # effective add_dot_notes percent during v2 cleanup
v2_cleanup_middle_shift = True     # effective shift_blocks_middle_flag during v2 cleanup
v2_cleanup_same_dir_recut = 0.5    # enforce >=135° direction changes between separated same-hand
                                   # notes (see v2_flow_min_turn_steps); 0 disables
v2_cleanup_same_dir_gap = 0.1      # same-stroke exemption (s): closer notes may repeat a direction
v2_flow_min_turn_steps = 3         # minimum turn between separated same-hand cuts in 45° steps
                                   # (3 = 135°, no travel exemption); <=2 restores the legacy
                                   # 90°-with->=2-cell-travel rule; 4 = only 180° reversals
                                   # legal, >=5 makes every separated pair illegal (undocumented
                                   # territory)
v2_cleanup_updown_shift = False    # effective shift_blocks_up_down_flag during v2 cleanup
v2_cleanup_cross_hand_adjust = False  # effective correct_notes_all_flag during v2 cleanup
v2_cleanup_edge_recut = False      # edge cuts are legal (swing may exit the grid); kept off
v2_cleanup_cell_clash = True       # simultaneous notes must never share a cell (any hands)
v2_cleanup_swing_clash = True      # no other-hand note in a cut's swept cells (approach+cut+exit)
v2_cleanup_swing_window = .15      # forward horizon (s): a later-standing block still in reach
v2_diagonal_max_fraction = 0.10    # cap diagonal cuts (d 4-7) at this fraction of all notes
                                   # (dots counted); excess diagonals are recut to the cheapest
                                   # flow-legal adjacent cardinal, vertical wins ties, a diagonal
                                   # stays when no cardinal is legal; >= 1 disables
v2_speed_cap_scale = 2.0           # multiplier on the v1 movement-speed cap
                                   # (max_speed * get_factor_from_max_speed distance/s between
                                   # consecutive same-hand notes); <= 0 disables the pass.
                                   # Reference value at add_beat_intensity 100 — the app field
                                   # scales the effective cap linearly (default 90% -> 1.8x,
                                   # 95% -> 1.9x, 125% -> 2.5x).
                                   # 2026-11-06: playtested at 2.0 for NPS 5
v2_speed_cap_start_notes = 30      # first N combined chronological note positions ramp
                                   # max_speed in (v1 decr_speed_range); 0 disables the ramp
v2_speed_cap_start_factor = .35    # max_speed multiplier at the first note position,
                                   # rising linearly to 1 (v1 decr_speed_val); the cap's
                                   # get_factor_from_max_speed is evaluated on the ramped
                                   # value, like v1
v2_fast_gap_seconds = .2           # same-hand gap below which consecutive notes form a
                                   # fast run: each note's direction is restricted to
                                   # its predecessor's, the 180° reverse and (admitted
                                   # at min(1, expected/local pair speed)) the ±45°
                                   # family — the reverse's own ±45° neighbours (the
                                   # 135°/225° off-reversals, the only ±45° cuts legal
                                   # between separated strokes) plus the predecessor's
                                   # ±45° neighbours for same-stroke pairs;
                                   # <= 0 disables both fast rules
v2_fast_counter_direction = "recut"  # inside fast runs, recut cuts opposing their row
                                   # first (top-row down-family 1/6/7, bottom-row
                                   # up-family 0/4/5), preferring the allowed pool;
                                   # "off"/"keep" disables
# Final v2 policy. Turning the master switch off restores historical app1.
v2_playability_enabled = True
v2_playability_priority = True  # final reach/path/axis and local visibility policy
v2_playability_reach = True
v2_playability_late_turns = True
v2_playability_m1 = False
v2_playability_isolated = False
v2_playability_m2 = False
v2_playability_m3 = False
v2_playability_m4 = False
v2_playability_slow90_seconds = .5
v2_playability_slow45_seconds = .65
v2_playability_dwell = False
v2_playability_recovery_ratio = 0.
v2_playability_recovery_balanced = False  # complete policy: bounded local hand recovery
v2_cleanup_note_heuristics = False  # run the legacy v1 note mutators (mirroring + flow pass
                                   # incl. emphasis/break/dot additions); False keeps model
                                   # notes — structural dedup, same-direction recut and the
                                   # clearance fixpoint still apply (docs/generation-v2.md)
v2_notes_onset_snap = True          # snap each note onto its nearest detected onset peak after
                                   # the clearance fixpoint; bounded per note/hand, no model
v2_notes_onset_snap_ioi_ratio = 0.15  # max shift as fraction of the note's own same-hand IOI
                                   # (min of prev/next gap) — a note can never reach another
                                   # note's clearance zone; 0 disables the snap effectively
v2_notes_onset_snap_wait = 4         # onset-detect refractory in envelope frames (~46 ms at
                                   # hop 512/44.1 kHz); the legacy snap uses 8 — shorter wins
                                   # coverage at equal peak quality (docs/baseline.md)
v2_arrow_approach_prior = 0.6      # override arrow with deterministic approach direction when the
                                   # hand approaches axis-aligned and model top prob is below this; 0 disables
v2_arrows_variety = 0.5            # blend share of the reference bundle's arrows model at decode:
                                   # 0 = production arrows only, 1 = reference arrows only
v2_arrows_reference_bundle = "../Data/model/v2_expert_noterepair-c50"  # variety (history-corrupted) arrows

# Calm-area decoration (2026-10-31): note-free stretches keep zero notes but get
# beat-synced visual pulses — short edge-lane walls, bottom-corner bombs and
# alternating side-laser flashes — like human maps that decorate silent parts
# without forcing movement. Master switch plus per-element toggles.
v2_calm_effects = True             # decorative pulses inside note-free gaps
v2_calm_effects_walls = True       # short alternating edge-lane wall flashes (add_obstacle_flag still applies)
v2_calm_effects_bombs = True       # alternating bottom-corner bombs
v2_calm_effects_lighting = True    # alternating left/right laser flash events
v2_calm_effects_min_gap_seconds = 4.0   # note-free span required before pulses start
v2_calm_effects_edge_seconds = 1.0      # clearance kept around boundary notes/chains
v2_calm_effects_start_seconds = 1.5     # no pulses before this song time (quiet map open)
v2_calm_effects_division = 1            # pulses per beat (1 = every beat, 2 = eighths)
v2_calm_effects_wall_beats = 0.25       # flash wall duration as a beat fraction
v2_calm_effects_variety = True          # rotate placement themes per gap (False = alternate only)

# Start-of-map free zones (2026-11-04): keep the opening seconds readable like
# human maps — notes and hazards (incl. calm-effect pulses) only appear past
# these elapsed-time floors; 0 disables each zone.
v2_map_intro_no_notes_seconds = 2.0    # drop notes before this time (s)
v2_map_intro_no_hazards_seconds = 1.0  # drop walls/bombs starting before this (s)
v2_suppress_start_fill = True          # start_end_fill drops its pre-first-event inserts
                                       # while the note floor is active; the end segment
                                       # is unchanged (False = legacy opening burst)

# Fill entry ramp (2026-11-04): after map start and after every note-free pause
# of at least v2_fill_entry_pause_beats beats, subdivision fills are gated for
# v2_fill_entry_delay_seconds — only whole-beat model events enter there.
# (2026-11-05) Each blackout also runs until the v2_fill_entry_min_notes-th kept
# candidate after its entry (union of both windows).
v2_fill_entry_delay_seconds = 1.0  # subdivision-fill blackout after each entry (s); 0 disables
v2_fill_entry_pause_beats = 4.0    # note-free gap counting as a new section entry (beats);
                                   # 0 = only the map start counts as an entry
v2_fill_entry_min_notes = 4        # each entry's fill blackout also covers the span up to
                                   # the k-th kept candidate after it (fewer remaining =
                                   # gated to map end); 0 disables the count gate
v2_fill_respect_silence = True     # keep subdivision fills out of detected silent spans

# Onset-snap unmatched policy (2026-11-04): the timing onset snap zeroes model
# events with no detected onset inside its window. "always" is the legacy
# drop-all, which gutted quiet sections; "non_silent" restores unmatched events
# unless they sit inside a detected silent span; "never" restores them all.
v2_onset_snap_unmatched_policy = "non_silent"

# Max-gap safety net (2026-11-04): after all timing fills/removals, any
# remaining note-free gap beyond v2_max_note_gap_seconds gets candidates
# inserted from v2_gap_fill_source, inset v2_gap_fill_edge_seconds from the
# gap's bounding events and still gated by detected silent spans (always —
# independent of v2_fill_respect_silence, which scopes to the subdivision
# fills); inserted times flow through the normal note decode like model events.
v2_max_note_gap_seconds = 4.0    # largest tolerated note-free gap (s); 0 disables
v2_gap_fill_edge_seconds = 0.5   # clearance kept next to the events bounding a gap
                                 # (values below min_beat_time clamp up to it)
v2_gap_fill_source = "onsets"    # candidate pool: "onsets" | "bpm_grid" | "v1_beat";
                                 # "onsets" is the proven default, "v1_beat" experimental
v2_gap_fill_v1_bundle = "fav_15" # experimental: v1 beat-TCN bundle supplying the
                                 # "v1_beat" source — its K.clear_session() inside a
                                 # cached-predictor worker was never exercised live

# Detected-tempo octave repair (2026-11-06, v2 only): the whole-song pulse
# tracker favours subdivisions, so detections far above the genre's beat
# range (e.g. ~310 BPM for a ~155 track) arrive doubled and every
# beat-relative stage — entry pauses, bpm alignment/grid fills, calm pulses,
# the exported BPM field — runs at half real-time. An auto-detected BPM at or
# above v2_bpm_octave_min_bpm is halved while the onset autocorrelation
# supports it (ac(2T) > ac(T)) and no energy sits at the doubled rate
# (ac(T/2) below v2_bpm_octave_subbeat_max — the veto that keeps genuinely
# fast songs like 191 BPM intact). Manual BPM input is never corrected.
v2_bpm_octave_correct = True      # halve clearly-doubled auto-detected tempos
v2_bpm_octave_min_bpm = 180.      # only check detections at/above this BPM
v2_bpm_octave_subbeat_max = 0.15  # veto: halving requires ac(T/2) below this
v2_bpm_octave_min_result = 90.    # never correct below this BPM

# Fill interval floor (2026-11-06): inserted candidates (subdivision,
# start/end, gap-net and onset-coverage fills) keep this minimum spacing from
# existing and already-accepted times, so fills cannot create unplayable
# streams the speed cap then has to cull in bulk. The floor scales down with
# difficulty — effective = base * ref_nps / nps clamped to [min_seconds,
# base]: nps 5 -> 0.09 s, nps 10 -> 0.045 s, nps 20 -> min_seconds.
v2_fill_min_interval_seconds = 0.09     # fill spacing floor at the reference NPS; 0 disables
v2_fill_min_interval_nps = 5.0          # NPS the base floor is calibrated for
v2_fill_min_interval_min_seconds = .035 # hard lower bound for high-NPS maps

# Missed-onset coverage (2026-11-06): the timing model sometimes stays silent
# on audible hits. Detected onsets stronger than the quantile share of all
# detected onsets and farther than v2_onset_fill_window_seconds from every
# candidate are appended as events (count from the model's own frame
# probabilities), respecting detected silence, the fill floor and the
# post-entry delay — but not the N-note entry runway, which only throttles
# synthetic subdivision fills (onset hits are real audio events, and a sparse
# stream would otherwise stretch the runway over exactly the loud sections
# coverage exists to repair). Runs after the final bpm alignment so inserts
# stay on the onset.
v2_onset_fill_flag = True             # insert candidates at strong uncovered onsets
v2_onset_fill_window_seconds = .12    # onset is "covered" by an event inside +- this (s)
v2_onset_fill_strength_quantile = .5  # keep onsets at/above this envelope-strength quantile
# Loudness tier (2026-11-06): in sections loud enough for their onsets to sit
# above v2_onset_fill_loudness_quantile of all onset loudness (RMS smoothed
# over the window below), coverage also admits onsets down to the lower
# strength quantile — busy loud sections under-trigger the timing model more
# often than quiet ones, where the strict bar still applies. 0 disables it.
v2_onset_fill_loudness_quantile = .6          # onset loudness bar for the second tier
v2_onset_fill_loudness_strength_quantile = .25  # relaxed strength bar in loud sections
v2_onset_fill_loudness_smooth_seconds = 1.    # RMS averaging window = "section is loud"

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

# Intensity/timing training execution (model and target contracts stay unchanged)
beat_models_training_prefetch_batches = 2  # zero disables the explicit input buffer
beat_models_training_compile_steps = True
beat_models_training_steps_per_execution = 16  # sequential updates per graph call, not gradient accumulation
beat_models_training_log_every_batches = 100  # progress rounds to graph-call boundaries

# Intensity model configuration (new whole-song normalized note distribution)
intensity_optimizer = 'Adam'
intensity_learning_rate = 5e-4
intensity_n_epochs = 80
intensity_batch_size = 1  # whole songs
intensity_clipnorm = 1.0
intensity_early_stopping_patience = 5
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
timing_early_stopping_patience = 5
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
notes_early_stopping_patience = 5
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
pipeline_v2_tolerance_seconds = 0.25  # sparse-stage window-matched evaluation
pipeline_v2_hazard_context_seconds = 2.0  # note-space lookahead for hazard candidates
pipeline_v2_hazard_context_back_seconds = 0.5  # note-space lookback for hazard candidates
pipeline_v2_hazard_context_bins = 10  # equal bins spanning [now-back, now+forward)
pipeline_v2_hazard_feasibility_input = True  # gated: legal wall/bomb masks as hazard model inputs
# Experimental Arm F: factorized hazard bomb head — a query-level presence sigmoid
# multiplied into cell scores at decode. Evaluated 2026-10-30: improved TF signal
# but no usable autoregressive operating point and walls regressed — keep off.
hazards_bomb_presence = False

# Repaired sparse-stage targets; recorded in each new dataset, never applied to old artifacts.
pipeline_v2_repair_sparse_stages = True
pipeline_v2_hazard_section_beats = 0.25
pipeline_v2_sparse_max_candidates = 50000
pipeline_v2_lighting_coverage_target = 0.95
pipeline_v2_arc_use_legacy_gap = False
pipeline_v2_positive_weight_cap = 100.0
pipeline_v2_metric_thresholds = 512

# V2 training execution (does not alter model/data contracts)
pipeline_v2_training_ram_cache_mib = 16384  # active stage only; zero streams every song
pipeline_v2_training_prefetch_batches = 8  # bounded CPU producer; zero disables prefetch
pipeline_v2_training_compile_steps = True
pipeline_v2_training_log_every_batches = 100  # zero keeps epoch summaries only

# V2 positions: training and network parameters
positions_optimizer = 'Adam'
positions_learning_rate = 0.001
positions_n_epochs = 80
positions_batch_size = 128
positions_clipnorm = 1.0
positions_early_stopping_patience = 5
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
positions_history_corrupt_rate = 0.5
positions_history_corrupt_drop = 0.5

# V2 arrows: training and network parameters
arrows_optimizer = 'Adam'
arrows_learning_rate = 0.0005
arrows_n_epochs = 80
arrows_batch_size = 128
arrows_clipnorm = 1.0
arrows_early_stopping_patience = 5
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
arrows_history_corrupt_rate = 0.5
arrows_history_corrupt_drop = 0.5

# V2 multiplier: training and network parameters
multiplier_optimizer = 'Adam'
multiplier_learning_rate = 0.0005
multiplier_n_epochs = 80
multiplier_batch_size = 128
multiplier_clipnorm = 1.0
multiplier_early_stopping_patience = 5
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
multiplier_history_corrupt_rate = 0.0
multiplier_history_corrupt_drop = 0.5

# V2 hazards: training and network parameters
hazards_optimizer = 'Adam'
hazards_learning_rate = 0.0005
hazards_n_epochs = 80
hazards_batch_size = 128
hazards_clipnorm = 1.0
hazards_early_stopping_patience = 5
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
hazards_enabled_song_supervision = True
hazards_history_corrupt_rate = 0.5
hazards_history_corrupt_drop = 0.5
hazards_negative_supervision_weight = 0.5

# Experimental V2 lighting: not yet implemented for production (optional/deferred)
lighting_optimizer = 'Adam'
lighting_learning_rate = 0.0005
lighting_n_epochs = 80
lighting_batch_size = 8
lighting_clipnorm = 1.0
lighting_early_stopping_patience = 5
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
arcs_early_stopping_patience = 5
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
# Arm H-arcs: supervise attach negatives on arc-free songs too — ~72% of Arm E
# false attaches land on songs with zero supervised attach gradient (2026-10-30).
arcs_enabled_song_supervision = False
arcs_history_corrupt_rate = 0.5
arcs_history_corrupt_drop = 0.5
# Arm G evaluated keep_guard=False and regressed (2026-10-30); restored.
arcs_history_corrupt_keep_guard = True
arcs_negative_supervision_weight = 0.5

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

# ---------------------------------------------------------
# Persisted app/user overrides live in the untracked user_settings.py and are
# applied here so tracked defaults stay clean for `git pull --ff-only`.
# Derived settings are re-evaluated from the effective inputs unless the
# derived name was overridden explicitly.
try:
    from tools.config import user_settings as _user_settings
except Exception as _error:
    if not isinstance(_error, ImportError):
        print(f"Warning: could not load tools/config/user_settings.py ({_error}); using defaults.")
    _user_settings = None
if _user_settings is not None:
    _overridden = {name for name in vars(_user_settings) if not name.startswith('_')}
    globals().update({name: getattr(_user_settings, name) for name in _overridden})
    use_mapper_selection = use_mapper_selection.lower()
    obstacle_time_gap = np.asarray(obstacle_time_gap)
    for _derived, _source in {
        'intensity_random_seed': 'random_seed',
        'timing_random_seed': 'random_seed',
        'notes_random_seed': 'random_seed',
        'positions_random_seed': 'random_seed',
        'arrows_random_seed': 'random_seed',
        'multiplier_random_seed': 'random_seed',
        'hazards_random_seed': 'random_seed',
        'lighting_random_seed': 'random_seed',
        'arcs_random_seed': 'random_seed',
        'max_speed_orig': 'max_speed',
        'add_beat_intensity_orig': 'add_beat_intensity',
        'silence_threshold_orig': 'silence_threshold',
        'jump_speed_offset_orig': 'jump_speed_offset',
        'obstacle_time_gap_orig': 'obstacle_time_gap',
        'thresh_beat_orig': 'thresh_beat',
        'thresh_onbeat_orig': 'thresh_onbeat',
    }.items():
        if _derived not in _overridden:
            globals()[_derived] = globals()[_source]
    del _derived, _source, _overridden
del _user_settings

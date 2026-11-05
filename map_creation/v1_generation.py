"""Legacy v1 song generation behind the v2-shaped per-song interface.

Imports are inert: TensorFlow only arrives through the lazy legacy_generator
import inside generate_song_v1, keeping v2 worker startup spawn-safe.
"""

import os
from pathlib import Path

from tools.config import config, paths
from tools.config.mapper_selection import get_full_model_path, update_model_file_paths
from tools.utils.huggingface import model_download

from map_creation.v2_generation import _emit, scoped_settings


V1_MAPPER_SELECTIONS = ("fav_15", "pp3_15", "easy_15", "hard_15", "expert_15")


def _missing_v1_assets():
    """Model prefixes and pickle paths absent from the current selection."""
    missing = []
    for prefix in (config.beat_gen_version, config.enc_version,
                   config.mapper_version, config.event_gen_version):
        try:
            get_full_model_path(prefix)
        except FileNotFoundError:
            missing.append(prefix)
    for pkl in (paths.notes_classify_dict_file, paths.beats_classify_encoder_file,
                paths.events_classify_encoder_file):
        if not os.path.isfile(pkl):
            missing.append(pkl)
    return missing


def ensure_v1_bundle(selection: str) -> None:
    """Download the v1 preset's model folder on demand, then validate it."""
    selection = str(selection)
    if selection not in V1_MAPPER_SELECTIONS:
        raise ValueError(f"Unknown v1 mapper selection: {selection}")
    config.use_mapper_selection = selection
    update_model_file_paths(check_model_exists=False)
    missing = _missing_v1_assets()
    if missing:
        model_download(model_branch=selection)
        update_model_file_paths(check_model_exists=False)
        missing = _missing_v1_assets()
    if missing:
        raise FileNotFoundError(
            f"v1 bundle '{selection}' is missing required assets: {', '.join(missing)}")


def generate_song_v1(name: str, internal_difficulty: float, logger=None) -> dict:
    """Run the untouched v1 generator for one song under scoped settings."""
    from evaluation.legacy_generator import main as legacy_main  # lazy: TF at module top

    name = str(name)
    if name.lower().endswith(".egg"):
        name = name[:-4]
    difficulty = float(internal_difficulty)
    with scoped_settings():
        update_model_file_paths(check_model_exists=False)
        config.max_speed = config.max_speed_orig = difficulty
        _emit(f"{name}: v1 model '{config.use_mapper_selection}' at {difficulty / 4:g} NPS", logger)
        status = legacy_main([name])  # single name only: main() exit()s on longer lists
    folder = Path(paths.new_map_path) / f"1234_{difficulty:.1f}_{name}"
    if status is not False or not folder.is_dir():
        raise RuntimeError(f"v1 generation failed for {name} (status={status})")
    return {"status": "ok", "song": name, "internal_difficulty": difficulty,
            "path": str(folder), "package": {}}

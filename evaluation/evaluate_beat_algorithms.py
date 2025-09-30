"""Utilities for evaluating beat generation parameters.

This module provides a command line script that can be used to benchmark
configuration values for the beat generator.  The original version used an
exhaustive grid search and relied on hard coded paths which made experimentation
slow and platform specific.  The revised script introduces a small genetic
algorithm that searches the configuration space more efficiently while also
supporting portable, argument driven configuration.

Example usage::

    python -m evaluation.evaluate_beat_algorithms \
        --data-root /data/InfernoSaber \
        --generations 8 --population 12 --song-index 0

The script prints the best configuration discovered together with the achieved
metrics.  When the optional ``--plot`` flag is supplied the generated beats are
visualised against the ground truth data.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from beat_prediction.validate_find_beats import plot_beat_vs_real
from tools.config import get_config, paths

GEN_BEATS_IMPORT_ERROR: ModuleNotFoundError | None = None
try:  # noqa: SIM105 - module availability depends on optional dependencies
    from map_creation.gen_beats import main as gen_beats_main
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local setup
    gen_beats_main = None  # type: ignore[assignment]
    GEN_BEATS_IMPORT_ERROR = exc

config = get_config()


# ---------------------------------------------------------------------------
# Configuration helpers


def _ensure_trailing_slash(path: Path) -> str:
    """Return *path* as POSIX string ending with ``/``."""

    return path.as_posix().rstrip("/") + "/"


def override_data_root(base_dir: str | Path) -> None:
    """Override the configured data root directory.

    Allowing the data location to be supplied via command line arguments makes
    the script usable on different machines without editing the source code.
    """

    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Data root '{base_path}' does not exist")

    model_root = base_path / "model"
    if not config.use_mapper_selection:
        model_root = model_root / "general_new"
    else:
        model_root = model_root / config.use_mapper_selection.lower()

    train_root = base_path / "training"
    pred_root = base_path / "prediction"
    temp_root = base_path / "temp"

    paths.dir_path = _ensure_trailing_slash(base_path)
    paths.model_path = _ensure_trailing_slash(model_root)
    paths.pred_path = _ensure_trailing_slash(pred_root)
    paths.train_path = _ensure_trailing_slash(train_root)
    paths.temp_path = _ensure_trailing_slash(temp_root)

    paths.copy_path_song = _ensure_trailing_slash(train_root / "songs_egg")
    paths.copy_path_map = _ensure_trailing_slash(train_root / "maps")
    paths.dict_all_path = _ensure_trailing_slash(train_root / "maps_dict_all")
    paths.songs_pred = _ensure_trailing_slash(pred_root / "songs_predict")
    paths.new_map_path = _ensure_trailing_slash(pred_root / "new_map")
    paths.fail_path = _ensure_trailing_slash(train_root / "fail_list")
    paths.song_data = _ensure_trailing_slash(train_root / "song_data")
    paths.ml_input_path = _ensure_trailing_slash(train_root / "ml_input")

    diff_root = train_root / "songs_diff"
    paths.diff_path = _ensure_trailing_slash(diff_root)
    paths.diff_ar_file = (diff_root / "diff_ar.npy").as_posix()
    paths.name_ar_file = (diff_root / "name_ar.npy").as_posix()

    paths.ml_input_beat_file = paths.ml_input_path + "beat_ar.npy"
    paths.ml_input_song_file = paths.ml_input_path + "song_ar.npy"

    model_root_path = Path(paths.model_path)
    paths.notes_classify_dict_file = (model_root_path / "notes_class_dict.pkl").as_posix()
    paths.beats_classify_encoder_file = (model_root_path / "onehot_encoder_beats.pkl").as_posix()
    paths.events_classify_encoder_file = (model_root_path / "onehot_encoder_events.pkl").as_posix()


def snapshot_config_state(cfg: Any) -> Dict[str, Any]:
    """Create a deep copy of a configuration object's state."""

    return copy.deepcopy(cfg.__dict__)


def restore_config_state(cfg: Any, state: MutableMapping[str, Any]) -> None:
    """Restore a previously captured configuration state."""

    cfg.__dict__.clear()
    cfg.__dict__.update(copy.deepcopy(state))


# ---------------------------------------------------------------------------
# Accuracy metrics


def calculate_beat_accuracy(
    beat_pred: Sequence[float], beat_real: Sequence[float], tolerance: float
) -> Tuple[float, float, float]:
    """Return precision, recall and F1 score for the predicted beats.

    The original implementation compared each prediction with *all* reference
    beats which resulted in quadratic complexity.  The revised version performs
    a greedy matching on the sorted beat arrays.  The runtime is therefore
    linear in the number of beats while still yielding deterministic metrics.
    """

    pred = np.sort(np.asarray(beat_pred, dtype=float).ravel())
    real = np.sort(np.asarray(beat_real, dtype=float).ravel())

    if len(pred) == 0 or len(real) == 0:
        return 0.0, 0.0, 0.0

    matched = 0
    i = j = 0
    while i < len(pred) and j < len(real):
        diff = pred[i] - real[j]
        if abs(diff) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1

    tp = matched
    fp = len(pred) - matched
    fn = len(real) - matched

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f_measure


# ---------------------------------------------------------------------------
# Genetic algorithm primitives


@dataclass
class ParameterDefinition:
    """Metadata describing a configuration value optimised by the GA."""

    name: str
    attr: str
    kind: str
    bounds: Tuple[float, float] | None = None
    step: float | None = None
    choices: Sequence[Any] | None = None
    propagate_to: Sequence[str] = ()

    def sample(self, rng: random.Random) -> Any:
        if self.choices is not None:
            return rng.choice(list(self.choices))
        if self.kind == "bool":
            return bool(rng.random() < 0.5)
        if self.kind == "int":
            assert self.bounds is not None
            lo, hi = map(int, self.bounds)
            return rng.randint(lo, hi)
        if self.kind == "float":
            assert self.bounds is not None
            lo, hi = self.bounds
            value = rng.uniform(lo, hi)
            return self._quantize(value)
        raise ValueError(f"Unsupported parameter kind '{self.kind}'")

    def mutate(self, value: Any, rng: random.Random) -> Any:
        if self.choices is not None:
            choices = [choice for choice in self.choices if choice != value]
            return rng.choice(choices) if choices else value
        if self.kind == "bool":
            return not value
        if self.kind == "int":
            assert self.bounds is not None
            lo, hi = map(int, self.bounds)
            step = max(1, int(round((hi - lo) * 0.1)))
            delta = rng.randint(-step, step)
            mutated = int(value) + delta
            return max(lo, min(hi, mutated))
        if self.kind == "float":
            assert self.bounds is not None
            lo, hi = self.bounds
            span = hi - lo
            scale = span * 0.1 if span else 1.0
            mutated = float(value) + rng.gauss(0.0, scale)
            mutated = max(lo, min(hi, mutated))
            return self._quantize(mutated)
        raise ValueError(f"Unsupported parameter kind '{self.kind}'")

    def apply(self, cfg: Any, value: Any) -> None:
        setattr(cfg, self.attr, value)
        for attr in self.propagate_to:
            setattr(cfg, attr, value)

    def _quantize(self, value: float) -> float:
        if self.step is None:
            return value
        return round(value / self.step) * self.step


@dataclass
class Individual:
    params: Dict[str, Any]
    precision: float | None = None
    recall: float | None = None
    fitness: float | None = None
    beats: np.ndarray | None = None

    def clone(self) -> "Individual":
        return Individual(
            params=dict(self.params),
            precision=self.precision,
            recall=self.recall,
            fitness=self.fitness,
            beats=None if self.beats is None else np.copy(self.beats),
        )


PARAMETERS: Tuple[ParameterDefinition, ...] = (
    ParameterDefinition(
        name="add_silence_flag",
        attr="add_silence_flag",
        kind="bool",
        choices=(True, False),
    ),
    ParameterDefinition(
        name="add_beat_intensity",
        attr="add_beat_intensity_orig",
        kind="float",
        bounds=(40.0, 120.0),
        step=1.0,
        propagate_to=("add_beat_intensity",),
    ),
    ParameterDefinition(
        name="silence_threshold",
        attr="silence_threshold_orig",
        kind="float",
        bounds=(0.02, 0.45),
        step=0.01,
        propagate_to=("silence_threshold",),
    ),
    ParameterDefinition(
        name="thresh_beat",
        attr="thresh_beat_orig",
        kind="float",
        bounds=(0.25, 0.65),
        step=0.01,
        propagate_to=("thresh_beat",),
    ),
    ParameterDefinition(
        name="map_filler_iters",
        attr="map_filler_iters",
        kind="int",
        bounds=(0, 12),
    ),
    ParameterDefinition(
        name="factor_pitch_certainty",
        attr="factor_pitch_certainty",
        kind="float",
        bounds=(0.1, 2.5),
        step=0.05,
    ),
    ParameterDefinition(
        name="factor_pitch_meanmax",
        attr="factor_pitch_meanmax",
        kind="float",
        bounds=(1.0, 5.0),
        step=0.1,
    ),
)


def create_individual(rng: random.Random) -> Individual:
    params = {definition.name: definition.sample(rng) for definition in PARAMETERS}
    return Individual(params=params)


def mutate_individual(individual: Individual, mutation_rate: float, rng: random.Random) -> Individual:
    params = dict(individual.params)
    for definition in PARAMETERS:
        if rng.random() <= mutation_rate:
            params[definition.name] = definition.mutate(params[definition.name], rng)
    return Individual(params=params)


def crossover(parent_a: Individual, parent_b: Individual, rng: random.Random) -> Tuple[Individual, Individual]:
    child_params_a: Dict[str, Any] = {}
    child_params_b: Dict[str, Any] = {}
    for definition in PARAMETERS:
        if rng.random() < 0.5:
            child_params_a[definition.name] = parent_a.params[definition.name]
            child_params_b[definition.name] = parent_b.params[definition.name]
        else:
            child_params_a[definition.name] = parent_b.params[definition.name]
            child_params_b[definition.name] = parent_a.params[definition.name]
    return Individual(params=child_params_a), Individual(params=child_params_b)


def tournament_selection(population: Sequence[Individual], rng: random.Random, k: int = 3) -> Individual:
    competitors = rng.sample(population, k=min(k, len(population)))
    return max(competitors, key=lambda candidate: candidate.fitness or 0.0)


# ---------------------------------------------------------------------------
# Genetic algorithm driver


def evaluate_individual(
    individual: Individual,
    base_config_state: Dict[str, Any],
    song_names: List[str],
    real_beats: np.ndarray,
    tolerance: float,
) -> None:
    """Evaluate *individual* and populate its metrics in-place."""

    if GEN_BEATS_IMPORT_ERROR is not None or gen_beats_main is None:
        raise RuntimeError(
            "Beat generation requires optional dependency 'aubio'. "
            "Install it to run the evaluation script."
        ) from GEN_BEATS_IMPORT_ERROR

    restore_config_state(config, base_config_state)
    for definition in PARAMETERS:
        definition.apply(config, individual.params[definition.name])

    try:
        beats = gen_beats_main(song_names, debug_beats=True)
        beats = np.asarray(beats, dtype=float).ravel()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to generate beats for {song_names[0]}: {exc}")
        individual.precision = individual.recall = individual.fitness = 0.0
        individual.beats = None
        return

    if beats.size == 0 or not np.isfinite(beats).any():
        individual.precision = individual.recall = individual.fitness = 0.0
        individual.beats = None
        return

    precision, recall, f_measure = calculate_beat_accuracy(beats, real_beats, tolerance)
    individual.precision = precision
    individual.recall = recall
    individual.fitness = f_measure
    individual.beats = beats


def run_genetic_algorithm(
    song_names: List[str],
    real_beats: np.ndarray,
    tolerance: float,
    *,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    elite_count: int,
    rng: random.Random,
) -> Tuple[Individual, List[Individual]]:
    """Run the evolutionary search and return the best individual."""

    base_config_state = snapshot_config_state(config)
    population = [create_individual(rng) for _ in range(population_size)]
    history: List[Individual] = []
    best_overall: Individual | None = None

    for generation in range(generations):
        for member in population:
            if member.fitness is None:
                evaluate_individual(member, base_config_state, song_names, real_beats, tolerance)

        population.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)
        best_generation = population[0].clone()
        history.append(best_generation)

        if best_overall is None or (best_generation.fitness or 0.0) > (best_overall.fitness or 0.0):
            best_overall = best_generation.clone()

        print(
            f"Generation {generation + 1}/{generations}: "
            f"F1={best_generation.fitness:.4f} "
            f"(precision={best_generation.precision:.4f}, recall={best_generation.recall:.4f})"
        )

        elites = [population[idx].clone() for idx in range(min(elite_count, len(population)))]

        next_population: List[Individual] = elites
        while len(next_population) < population_size:
            parent_a = tournament_selection(population, rng)
            parent_b = tournament_selection(population, rng)

            if rng.random() <= crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a.clone(), parent_b.clone()

            child_a = mutate_individual(child_a, mutation_rate, rng)
            child_b = mutate_individual(child_b, mutation_rate, rng)

            next_population.append(child_a)
            if len(next_population) < population_size:
                next_population.append(child_b)

        population = next_population

    assert best_overall is not None  # guaranteed when generations > 0
    return best_overall, history


# ---------------------------------------------------------------------------
# Command line interface


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        help=(
            "Override the data root directory (expects the usual training/ and "
            "prediction/ sub folders)."
        ),
    )
    parser.add_argument(
        "--song-index",
        type=int,
        default=0,
        help="Zero-based index of the song to analyse after filtering by BPM.",
    )
    parser.add_argument(
        "--min-bps",
        type=float,
        default=0.1,
        help="Lower BPM bound used when selecting evaluation songs.",
    )
    parser.add_argument(
        "--max-bps",
        type=float,
        default=50.0,
        help="Upper BPM bound used when selecting evaluation songs.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Maximum timing difference in seconds when matching beats.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=12,
        help="Number of individuals in each generation of the genetic algorithm.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to evolve.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.25,
        help="Probability of mutating each parameter during reproduction.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.85,
        help="Probability of performing crossover for two selected parents.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="Number of top performers carried over unchanged to the next generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the beat comparison for the best discovered configuration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.population <= 0:
        raise ValueError("Population size must be positive")
    if args.generations <= 0:
        raise ValueError("Generation count must be positive")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("Mutation rate must be between 0 and 1")
    if not 0.0 <= args.crossover_rate <= 1.0:
        raise ValueError("Crossover rate must be between 0 and 1")
    if args.elite_count < 0:
        raise ValueError("Elite count cannot be negative")

    if args.data_root:
        override_data_root(args.data_root)

    rng = random.Random(args.seed)

    from preprocessing.bs_mapper_pre import load_beat_data
    from training.helpers import filter_by_bps

    name_ar, diff_ar = filter_by_bps(args.min_bps, args.max_bps)
    if not name_ar:
        raise RuntimeError("No songs available for the requested BPM range")

    if args.song_index < 0 or args.song_index >= len(name_ar):
        raise IndexError(
            f"Song index {args.song_index} out of range (available songs: {len(name_ar)})"
        )

    song_name = name_ar[args.song_index]
    _, real_beats_list = load_beat_data([song_name], return_notes=True)
    real_beats = np.asarray(real_beats_list[0], dtype=float)

    config.max_speed = diff_ar[args.song_index] * 4
    config.max_speed_orig = config.max_speed

    print(
        f"Using song '{song_name}' with ≈{diff_ar[args.song_index]:.2f} BPS "
        f"and {len(real_beats)} reference beats."
    )

    best_individual, _ = run_genetic_algorithm(
        [song_name],
        real_beats,
        args.tolerance,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_count=max(1, args.elite_count),
        rng=rng,
    )

    best_precision = best_individual.precision or 0.0
    best_recall = best_individual.recall or 0.0
    best_f1 = best_individual.fitness or 0.0

    print("\nBest configuration found:")
    for name, value in best_individual.params.items():
        print(f"  {name}: {value}")
    print(
        f"Metrics — F1: {best_f1:.4f}, Precision: {best_precision:.4f}, "
        f"Recall: {best_recall:.4f}"
    )

    if args.plot and best_individual.beats is not None:
        plot_beat_vs_real(best_individual.beats, real_beats)
if __name__ == "__main__":
    main()

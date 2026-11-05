"""Generate maps with resident CPU workers and explicit completion records.

v2_expert is the production generator; legacy v1 mapper selections route
each song through the wrapped v1 generator and per-difficulty ZIP export.
"""

import json
import math
import multiprocessing as mp
from pathlib import Path
import os
import queue
import shutil
import sys
import time
from functools import partial

from tools.config import config, paths
from bs_shift.export_map import check_music_files, shutil_copy_maps
from map_creation.v1_generation import V1_MAPPER_SELECTIONS, ensure_v1_bundle
from map_creation.v2_generation import (apply_snapshot, configure_inference, generate_song,
                                       save_report, scoped_settings, settings_snapshot)
from map_creation.v2_runtime import validate_production_bundle
from map_creation.v2_export import combine_packages, write_zip


_PROGRESS = None
_LABELS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")


def _emit(message, callback=None, important=False):
    """CLI output always gets the full log; the app box only sees important
    lines unless verbose progress is enabled (config.verbose_level > 1)."""
    print(message, flush=True)
    if callback is not None and (important or config.verbose_level > 1):
        callback(message)


def _worker_init(snapshot, progress=None):
    global _PROGRESS
    apply_snapshot(snapshot)
    configure_inference()
    _PROGRESS = progress


def _worker_log(message, important=False):
    if _PROGRESS is None:
        print(message, flush=True)
    else:
        _PROGRESS.put((str(message), bool(important)))


def process_song(song_list_worker, total_runs):
    """Keep all difficulties of a recording on one resident worker when possible."""
    results = []
    for song, difficulty in song_list_worker:
        try:
            generator = generate_song
            if config.use_mapper_selection != "v2_expert":
                # Lazy: the module is TF-free, but keeps the worker contract patchable.
                from map_creation.v1_generation import generate_song_v1
                generator = generate_song_v1
            results.append(generator(song, float(difficulty), logger=_worker_log))
        except Exception as error:
            result = {"status": "failed", "song": Path(song).stem,
                      "internal_difficulty": float(difficulty),
                      "error": f"{type(error).__name__}: {error}"}
            results.append(result)
            _worker_log(f"Failed {result['song']} at {difficulty / 4:g} NPS: {result['error']}", important=True)
    return results


def _groups(songs, difficulties, workers):
    groups = [[(song, difficulty) for difficulty in difficulties] for song in songs]
    # Split only when needed to use idle workers; otherwise retain song reuse.
    while len(groups) < workers:
        largest = max(range(len(groups)), key=lambda index: len(groups[index]), default=None)
        if largest is None or len(groups[largest]) < 2:
            break
        group = groups.pop(largest)
        middle = (len(group) + 1) // 2
        groups[largest:largest] = [group[:middle], group[middle:]]
    return groups


def _difficulties(values):
    if not 1 <= len(values) <= 5:
        raise ValueError("Select between one and five difficulties")
    if any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("Difficulties must be finite nonnegative NPS values")
    result = sorted(float(value) * 4 for value in values)
    if len({f"{value:.1f}" for value in result}) != len(result):
        raise ValueError("Difficulties must produce distinct output filenames")
    return result


def combine_maps(song_list_potential, song_list_run, diff_list, export_results_to_bs,
                 logger=None):
    """Combine only successful records supplied by this run, never scan old folders.

    The second argument now contains completion records from ``process_song``.
    The other arguments retain their existing signature and internal speed units.
    """
    return _combine_maps(song_list_potential, song_list_run, diff_list,
                         export_results_to_bs, settings_snapshot(), logger=logger)


def _combine_v1_packages(song_list_potential, song_list_run, diff_list,
                         export_results_to_bs, run_config, run_paths, logger=None):
    """Zip each 1234_<speed>_<name> v1 folder directly, never a combined 12345_* package.

    Same gate as _combine_maps: a song is packaged and archived only when
    every requested difficulty succeeded; partial successes stay in place
    for retry (their records remain in the run report).
    """
    packages = []
    for filename in song_list_potential:
        name = Path(filename).stem
        by_difficulty = {record["internal_difficulty"]: record
                         for record in song_list_run
                         if record["song"] == name and record["status"] == "ok"}
        if any(difficulty not in by_difficulty for difficulty in diff_list):
            continue
        variants = []
        for difficulty in diff_list:
            record = by_difficulty[difficulty]
            zip_path = write_zip(record["path"],
                                 f"{run_paths['new_map_path']}{record['internal_difficulty']:.1f}_{name}.zip")
            variants.append({"path": record["path"], "zip": zip_path})
            if export_results_to_bs and shutil_copy_maps(f"{record['internal_difficulty']:.1f}_{name}"):
                _emit("Copied map(s) to BeatSaber directory.", logger)
        packages.append({"song": name, "variants": variants})
        _emit(f"Packaged {name}: {len(variants)} v1 map archive(s)", logger, important=True)
        if run_config["auto_move_song_afterwards"]:
            done = Path(run_paths["songs_pred"]) / "y_done"
            done.mkdir(exist_ok=True)
            shutil.move(str(Path(run_paths["songs_pred"]) / filename), str(done / filename))
    return packages


def _combine_maps(song_list_potential, song_list_run, diff_list, export_results_to_bs,
                  snapshot, logger=None):
    run_config, run_paths = snapshot["config"], snapshot["paths"]
    if run_config.get("use_mapper_selection", "v2_expert") != "v2_expert":
        return _combine_v1_packages(song_list_potential, song_list_run, diff_list,
                                    export_results_to_bs, run_config, run_paths,
                                    logger=logger)
    packages = []
    for filename in song_list_potential:
        name = Path(filename).stem
        records = [record for record in song_list_run if record["song"] == name and record["status"] == "ok"]
        by_difficulty = {record["internal_difficulty"]: record for record in records}
        if any(difficulty not in by_difficulty for difficulty in diff_list):
            continue
        variants = [{"difficulty": _LABELS[index], "rank": 2 * index + 1,
                     "path": by_difficulty[difficulty]["path"], "filename": f"{run_config['general_diff']}.dat",
                     "report": by_difficulty[difficulty].get("package", {})}
                    for index, difficulty in enumerate(diff_list)]
        if len(diff_list) == 1:
            source = by_difficulty[diff_list[0]]
            automatic = next((item for item in source.get("package", {}).get("variants", [])
                              if item["difficulty"] == "Expert" and item["filename"] != variants[0]["filename"]), None)
            if automatic is not None:
                variants.append({"difficulty": "Expert", "rank": 7, "path": source["path"],
                                 "filename": automatic["filename"], "report": source["package"]})
        folder = Path(run_paths["new_map_path"]) / f"12345_{name}"
        report = combine_packages(name, variants, folder, Path(str(folder) + ".zip"))
        report["song"] = name
        packages.append(report)
        if export_results_to_bs:
            library = run_paths["bs_song_path"]
            report["library_exported"] = bool(library and Path(library).is_dir())
            if report["library_exported"]:
                shutil.copytree(folder, Path(library) / folder.name, dirs_exist_ok=True)
            else:
                _emit("Warning: Beatsaber folder not found, automatic export disabled.")
        for record in records:
            shutil.rmtree(record["path"])
        if run_config["auto_move_song_afterwards"]:
            done = Path(run_paths["songs_pred"]) / "y_done"
            done.mkdir(exist_ok=True)
            shutil.move(str(Path(run_paths["songs_pred"]) / filename), str(done / filename))
    return packages


def _run(n_workers, diff_list, export_results_to_bs, logger_callback, parallel, model_selection=None):
    # Windows consoles use cp1252; replace unmappable chars instead of dying mid-run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if isinstance(n_workers, bool) or int(n_workers) != n_workers or n_workers < 1:
        raise ValueError("Worker count must be a positive integer")
    n_workers = int(n_workers)
    difficulties = _difficulties(diff_list)
    # Invalid bootstrap settings must fail here, not cause endless Pool respawns.
    configure_inference()
    selection = str(model_selection or "v2_expert").lower()
    if selection != "v2_expert" and selection not in V1_MAPPER_SELECTIONS:
        raise ValueError(f"Unknown mapper selection: {selection}")
    if selection == "v2_expert":
        validate_production_bundle(config.v2_model_bundle_path)
    else:
        ensure_v1_bundle(selection)  # calls update_model_file_paths internally
    snapshot = settings_snapshot()  # AFTER ensure — workers inherit paths.model_path
    snapshot["config"]["use_mapper_selection"] = selection
    _emit("Checking and normalizing song files...", logger_callback, important=True)
    input_path = snapshot["paths"]["songs_pred"]
    songs = check_music_files(sorted(os.listdir(input_path)), input_path)
    if len(difficulties) > 1:
        snapshot["config"]["create_expert_flag"] = False
    groups = _groups(songs, difficulties, n_workers)
    records, packages, failures = [], [], []
    started = time.perf_counter()
    workers = min(n_workers, len(groups))
    total_variants = sum(len(group) for group in groups)
    _emit(f"Generating {len(songs)} songs at {len(difficulties)} difficulties with {workers} CPU workers",
          logger_callback, important=True)

    def _eta():
        if not records:
            return
        per_variant = (time.perf_counter() - started) / len(records)
        _emit(f"### ETA: {(total_variants - len(records)) * per_variant / 60:.1f} minutes. "
              f"Time per song: {per_variant:.0f} s ###", logger_callback, important=True)

    if groups and parallel:
        context = mp.get_context("spawn")
        progress = context.Queue()
        try:
            with context.Pool(workers, initializer=_worker_init, initargs=(snapshot, progress)) as pool:
                results = pool.imap_unordered(partial(process_song, total_runs=len(groups)), groups)
                finished = 0
                while finished < len(groups):
                    while True:
                        try:
                            message, important = progress.get_nowait()
                        except queue.Empty:
                            break
                        _emit(message, logger_callback, important=important)
                    try:
                        records.extend(results.next(timeout=.2))
                    except mp.TimeoutError:
                        continue
                    finished += 1
                    _eta()
                    _emit(f"Completed {finished}/{len(groups)} work groups in {time.perf_counter() - started:.1f}s", logger_callback)
                while True:
                    try:
                        message, important = progress.get_nowait()
                    except queue.Empty:
                        break
                    _emit(message, logger_callback, important=important)
        finally:
            progress.close()
            progress.join_thread()
    elif groups:
        with scoped_settings(snapshot):
            configure_inference()
            for group in groups:
                records.extend(process_song(group, len(groups)))
                _eta()
    failures.extend(record for record in records if record["status"] != "ok")
    # Handle packaging failures individually and keep completed variants for retry.
    for song in songs:
        try:
            packages.extend(_combine_maps([song], records, difficulties, export_results_to_bs,
                                          snapshot, logger=logger_callback))
        except Exception as error:
            failures.append({"status": "failed", "song": Path(song).stem,
                             "error": f"Package failed: {type(error).__name__}: {error}"})
    report = {"status": "failed" if failures else "ok", "results": records,
              "packages": packages, "failures": failures, "workers": workers,
              "elapsed_seconds": time.perf_counter() - started}
    report["report_path"] = save_report(report, root=snapshot["paths"]["temp_path"])
    _emit(f"Finished: {len(packages)} complete song packages, {len(failures)} failures",
          logger_callback, important=True)
    return report


def main_multi_par(n_workers: int, diff_list: list, export_results_to_bs=True, logger_callback=None,
                   model_selection=None):
    return _run(n_workers, diff_list, export_results_to_bs, logger_callback, parallel=True,
                model_selection=model_selection)


def main_multi(diff_list: list, export_results_to_bs=True):
    return _run(1, diff_list, export_results_to_bs, None, parallel=False)


if __name__ == "__main__":
    mp.freeze_support()
    diff_list = json.loads(os.environ.get("diff_list", "[6, 8, 10]"))
    n_workers = 7 if paths.main_path.startswith("/mnt/") else 3
    use_model = os.environ.get("inferno_model")  # None routes to v2_expert
    result = main_multi_par(n_workers, diff_list, export_results_to_bs=True,
                            model_selection=use_model)
    raise SystemExit(1 if result["failures"] else 0)

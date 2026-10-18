"""Structured v2-generation export, with seconds converted only at serialization.

Native fields follow https://bsmg.wiki/mapping/map-format/beatmap.html.
The legacy output keeps colored chain heads and reports unsupported objects.
"""

from copy import deepcopy
import filecmp
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile


DIFFICULTY_RANKS = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
FAMILIES = ("notes", "bombs", "walls", "chains", "arcs", "lighting")


def _number(value, label, minimum=0, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{label} is outside its valid bounds")
    return value


def _integer(value, label, minimum=0, maximum=None):
    value = _number(value, label, minimum)
    if not value.is_integer() or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be an integer within its valid bounds")
    return int(value)


def _note(record):
    return {"b": _number(record["time"], "Note time"),
            "x": _integer(record["x"], "Note x", maximum=3),
            "y": _integer(record["y"], "Note y", maximum=2),
            "c": _integer(record["hand"], "Note hand", maximum=1),
            "d": _integer(record["d"], "Note direction", maximum=8), "a": 0}


def _span(record, label):
    start = _number(record["time"], f"{label} time")
    end = _number(record["end_time"], f"{label} end time")
    if end <= start:
        raise ValueError(f"{label} must have a positive duration")
    return start, end


def serialize_beatmap(state, bpm, version="v3"):
    """Return a JSON-ready beatmap and omission counts without changing *state*."""
    factor = _number(bpm, "BPM", positive=True) / 60
    if version not in ("v3", "v2"):
        raise ValueError("Beatmap version must be v3 or v2")
    if not isinstance(state, dict):
        raise ValueError("Generation state must be an object")
    for family in FAMILIES:
        if not isinstance(state.get(family, []), list):
            raise ValueError(f"{family} must be an object array")
    notes, identities = [], {}
    for record in state.get("notes", []):
        note = _note(record)
        identity = record["id"]
        if not isinstance(identity, (str, int)) or isinstance(identity, bool) or identity in identities:
            raise ValueError("Colored note IDs must be unique strings or integers")
        identities[identity] = note
        notes.append(note)
    bombs = [{"b": _number(record["time"], "Bomb time"),
              "x": _integer(record["x"], "Bomb x", maximum=3),
              "y": _integer(record["y"], "Bomb y", maximum=2)} for record in state.get("bombs", [])]
    walls = []
    for record in state.get("walls", []):
        start, end = _span(record, "Wall")
        wall = {"b": start, "d": end - start,
                "x": _integer(record["x"], "Wall x"),
                "y": _integer(record["y"], "Wall y", maximum=4),
                "w": _integer(record["width"], "Wall width", minimum=1),
                "h": _integer(record["height"], "Wall height", minimum=1)}
        if wall["x"] + wall["w"] > 4 or wall["y"] + wall["h"] > 5:
            raise ValueError("Wall geometry must fit the ordinary playfield")
        walls.append(wall)
    chains = []
    for record in state.get("chains", []):
        start, end = _span(record, "Chain")
        head = identities.get(record["anchor_id"])
        if head is None or _note(record) != head:
            raise ValueError("Chain must reference an unchanged existing colored head")
        chains.append({key: head[key] for key in ("b", "x", "y", "c", "d")})
        chains[-1].update(tb=end, tx=_integer(record["tx"], "Chain tail x", maximum=3),
                          ty=_integer(record["ty"], "Chain tail y", maximum=2),
                          sc=_integer(record["slices"], "Chain slice count", minimum=2),
                          s=_number(record["squish"], "Chain squish", positive=True))
    arcs = []
    for record in state.get("arcs", []):
        start, end = _span(record, "Arc")
        endpoints = record.get("endpoint_ids", [])
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            raise ValueError("Arc requires exactly two existing colored endpoints")
        head, tail = (identities.get(identity) for identity in endpoints)
        if (head is None or tail is None or head["c"] != tail["c"]
                or head["c"] != _integer(record["hand"], "Arc hand", maximum=1)
                or head["b"] != start or tail["b"] != end):
            raise ValueError("Arc endpoints must match existing same-hand notes and times")
        arc = {key: head[key] for key in ("b", "x", "y", "c", "d")}
        arc.update(tb=end, tx=tail["x"], ty=tail["y"], tc=tail["d"],
                   mu=_number(record["mu"], "Arc head multiplier"),
                   tmu=_number(record["tmu"], "Arc tail multiplier"),
                   m=_integer(record["anchor_mode"], "Arc anchor mode", maximum=2))
        arcs.append(arc)
    lighting = []
    for record in state.get("lighting", []):
        event_type = _integer(record["type"], "Lighting event type")
        if event_type == 100:
            raise ValueError("Tempo changes cannot be serialized as lighting")
        lighting.append({"b": _number(record["time"], "Lighting time"), "et": event_type,
                         "i": _integer(record["value"], "Lighting value"),
                         "f": _number(record.get("float_value", 1), "Lighting float value")})
    omissions = {"arcs": 0, "chain_bodies": 0, "walls": 0}
    for objects in (notes, bombs, walls, chains, arcs, lighting):
        objects.sort(key=lambda item: item["b"])
        for record in objects:
            for key in ("b", "tb", "d") if objects is walls else ("b", "tb"):
                if key in record:
                    record[key] *= factor
                    if not math.isfinite(record[key]):
                        raise ValueError("Serialized beat times must be finite")
    if version == "v3":
        return {"version": "3.2.0", "bpmEvents": [], "rotationEvents": [],
                "colorNotes": notes, "bombNotes": bombs, "obstacles": walls,
                "sliders": arcs, "burstSliders": chains, "waypoints": [],
                "basicBeatmapEvents": lighting, "colorBoostBeatmapEvents": [],
                "lightColorEventBoxGroups": [], "lightRotationEventBoxGroups": [],
                "lightTranslationEventBoxGroups": [], "basicEventTypesWithKeywords": {},
                "useNormalEventsAsCompatibleEvents": False}, omissions
    legacy_notes = [{"_time": note["b"], "_lineIndex": note["x"], "_lineLayer": note["y"],
                     "_type": note["c"], "_cutDirection": note["d"]} for note in notes]
    legacy_notes.extend({"_time": bomb["b"], "_lineIndex": bomb["x"], "_lineLayer": bomb["y"],
                         "_type": 3, "_cutDirection": 8} for bomb in bombs)
    legacy_notes.sort(key=lambda note: note["_time"])
    legacy_walls = []
    for wall in walls:
        if (wall["y"], wall["h"]) not in ((0, 5), (2, 3)):
            omissions["walls"] += 1
            continue
        legacy_walls.append({"_time": wall["b"], "_lineIndex": wall["x"],
                             "_type": 0 if wall["y"] == 0 else 1,
                             "_duration": wall["d"], "_width": wall["w"]})
    omissions.update(arcs=len(arcs), chain_bodies=len(chains))
    return {"_version": "2.0.0", "_notes": legacy_notes, "_obstacles": legacy_walls,
            "_events": [{"_time": event["b"], "_type": event["et"], "_value": event["i"]}
                        for event in lighting]}, omissions


def _filename(value):
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in "/\\\0"):
        raise ValueError("Package names must be plain filenames")
    return value


def _json(path, document):
    def ordinary(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False,
                                     default=ordinary) + "\n", encoding="utf-8")


def _publish(staging, target):
    """Expose only complete directories; restore an old package on rename failure."""
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Package destination must be an ordinary directory")
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    existing = target.exists()
    if existing:
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if existing:
            os.replace(backup, target)
        raise
    if existing:
        shutil.rmtree(backup)


def write_package(output_folder, name, bpm, variants, audio_path, cover_path=None, version=None):
    """Write all variants and publish one complete package; input assets stay intact."""
    from map_creation.map_creator import get_info_map_string
    from tools.config import config, paths

    name = _filename(name)
    _number(bpm, "BPM", positive=True)
    version = config.bs_mapping_version if version is None else version
    if not variants:
        raise ValueError("A package requires at least one difficulty")
    audio = Path(audio_path)
    cover = Path(cover_path) if cover_path is not None else Path(paths.new_map_path) / "cover.jpg"
    if cover_path is None and not cover.is_file():
        cover = Path(__file__).resolve().parents[1] / "app_helper" / "cover.jpg"
    if not audio.is_file() or not cover.is_file():
        raise FileNotFoundError("Package audio and cover must exist")
    target = Path(output_folder).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        info, entries, reports = None, [], []
        for variant in variants:
            difficulty = variant["difficulty"]
            if difficulty not in DIFFICULTY_RANKS or difficulty in {item["difficulty"] for item in reports}:
                raise ValueError("Package difficulties must be unique standard labels")
            internal = _number(variant["internal_difficulty"], "Internal difficulty")
            document, omissions = serialize_beatmap(variant["state"], bpm, version)
            offset = variant.get("jump_speed_offset")
            if offset is not None:
                offset = _number(offset, "Jump speed offset", minimum=-math.inf)
            automatic_expert = variant.get("automatic_expert", False)
            if automatic_expert and difficulty != "Expert":
                raise ValueError("Automatic Expert metadata requires the Expert difficulty")
            metadata_mode = "Expert" if automatic_expert else "ExpertPlus"
            metadata = json.loads(get_info_map_string(name, bpm, metadata_mode, max_speed=internal,
                                                       jump_speed_offset=offset))
            if info is None:
                info = metadata
            metadata_entries = metadata["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
            entry = next(item for item in metadata_entries if item["_difficulty"] == metadata_mode)
            entry.update(_difficulty=difficulty, _difficultyRank=DIFFICULTY_RANKS[difficulty],
                         _beatmapFilename=f"{difficulty}.dat")
            entries.append(entry)
            _json(staging / entry["_beatmapFilename"], document)
            reports.append({"difficulty": difficulty, "filename": entry["_beatmapFilename"],
                            "internal_difficulty": internal, "omissions": omissions,
                            "diagnostics": deepcopy(variant.get("diagnostics", {}))})
        info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"] = entries
        info["_songFilename"] = f"{name}.egg"
        _json(staging / "info.dat", info)
        shutil.copyfile(audio, staging / info["_songFilename"])
        shutil.copyfile(cover, staging / "cover.jpg")
        report = {"path": str(target), "info_path": str(target / "info.dat"), "version": version,
                  "variants": reports,
                  "omissions": {key: sum(item["omissions"][key] for item in reports)
                                for key in ("arcs", "chain_bodies", "walls")}}
        _publish(staging, target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_zip(folder, zip_path):
    """Atomically replace a ZIP only after every package file was archived."""
    folder, target = Path(folder).absolute(), Path(zip_path).absolute()
    if not (folder / "info.dat").is_file():
        raise FileNotFoundError("A complete map package is required for ZIP export")
    if folder == target or folder in target.parents:
        raise ValueError("ZIP destination must be outside the package")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(folder.rglob("*")):
                if source.is_file():
                    archive.write(source, source.relative_to(folder).as_posix())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)


def combine_packages(name, packages, output_folder, zip_path=None):
    """Combine explicitly successful packages, preserving each variant's NJS."""
    _filename(name)
    if not packages:
        raise ValueError("No completed packages were supplied")
    target = Path(output_folder).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        info, entries, reports, first_audio = None, [], [], None
        for package in packages:
            difficulty = package["difficulty"]
            if difficulty not in DIFFICULTY_RANKS or difficulty in {item["_difficulty"] for item in entries}:
                raise ValueError("Combined difficulties must be unique standard labels")
            rank = _integer(package["rank"], "Difficulty rank")
            if rank != DIFFICULTY_RANKS[difficulty]:
                raise ValueError("Difficulty rank does not match its label")
            source = Path(package["path"])
            metadata = json.loads((source / "info.dat").read_text(encoding="utf-8-sig"))
            filename = _filename(package.get("filename", "ExpertPlus.dat"))
            candidates = [entry for group in metadata["_difficultyBeatmapSets"]
                          if group["_beatmapCharacteristicName"] == "Standard"
                          for entry in group["_difficultyBeatmaps"] if entry["_beatmapFilename"] == filename]
            if len(candidates) != 1:
                raise ValueError("Completed package must identify exactly one Standard difficulty")
            audio = source / _filename(metadata["_songFilename"])
            if info is None:
                info, first_audio = deepcopy(metadata), audio
                shutil.copyfile(audio, staging / metadata["_songFilename"])
                cover = _filename(metadata["_coverImageFilename"])
                shutil.copyfile(source / cover, staging / cover)
            elif (metadata["_beatsPerMinute"] != info["_beatsPerMinute"]
                  or metadata["_songFilename"] != info["_songFilename"]
                  or not filecmp.cmp(first_audio, audio, shallow=False)):
                raise ValueError("Combined packages must use the same BPM and audio")
            entry = deepcopy(candidates[0])
            entry.update(_difficulty=difficulty, _difficultyRank=rank, _beatmapFilename=f"{difficulty}.dat")
            shutil.copyfile(source / filename, staging / entry["_beatmapFilename"])
            entries.append(entry)
            reports.append({"difficulty": difficulty, "source": str(source),
                            "report": deepcopy(package.get("report", {}))})
        info["_difficultyBeatmapSets"] = [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": entries}]
        _json(staging / "info.dat", info)
        report = {"path": str(target), "info_path": str(target / "info.dat"), "variants": reports}
        _publish(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if zip_path is not None:
        report["zip_path"] = write_zip(target, zip_path)
    return report

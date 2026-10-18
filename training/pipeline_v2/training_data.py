"""Read validated examples for one v2 stage without loading other song arrays."""

from numbers import Real

import numpy as np

from training.beat_models.artifacts import file_hash, read_json

from .artifacts import _Songs, _local
from .contracts import NEW_STAGES, StageExamples, validate_examples
from .arc_policy import stage_split
from .supervision import support, add_support, describe_support
from .policies import SPARSE_STAGES


SPLITS = ("train", "validation", "test")


def _selection(stage, split=None):
    if stage not in NEW_STAGES:
        raise ValueError(f"Unknown v2 stage: {stage}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown v2 split: {split}")


def _metadata(metadata, song_id, identities):
    split, group = metadata.get("split"), metadata.get("group_id")
    if split not in SPLITS or not group:
        raise ValueError("V2 training requires recording groups and explicit dataset splits")
    duration, bpm = metadata.get("duration", 0), metadata.get("bpm", 0)
    if not song_id or not np.isfinite([duration, bpm]).all() or duration <= 0 or bpm <= 0:
        raise ValueError("Song requires an identifier, positive duration and BPM")
    for key in ("group_id", "audio_sha256", "decoded_sha256"):
        identity = metadata.get(key)
        if identity:
            token = ("group" if key == "group_id" else "audio", identity)
            if identities.get(token, split) != split:
                raise ValueError("Recording identity leaks between v2 training splits")
            identities[token] = split
    for source in metadata.get("sources", []):
        path = str(source.get("path", "")).replace("\\", "/").lower()
        if "/evaluate_beats/bs_map_input/" in path and split != "test":
            raise ValueError("Curated benchmark recordings cannot enter training/validation")
    return split


def _stage_arrays(path, stage):
    """NPZ central-directory sizes bound cache storage without decoding other stages."""
    with np.load(path, allow_pickle=False) as archive:
        prefix = f"{stage}::"
        keys = [key for key in archive.files if key.startswith(prefix)]
        estimated = sum(info.file_size for info in archive.zip.infolist()
                        if info.filename.startswith(prefix))
        arrays = {key: archive[key] for key in keys}
    try:
        families = [{key.split("::", 2)[2]: value for key, value in arrays.items()
                     if key.startswith(f"{stage}::{family}::")}
                    for family in ("inputs", "targets", "weights")]
        examples = StageExamples(*families, arrays[f"{stage}::times"])
    except (KeyError, IndexError) as error:
        raise ValueError(f"Missing or malformed {stage} examples") from error
    expected = {f"{stage}::times"}
    expected.update(f"{stage}::{name}::{key}" for name, family in
                    zip(("inputs", "targets", "weights"), families) for key in family)
    if set(arrays) != expected:
        raise ValueError(f"Unexpected {stage} array keys")
    if stage == "arcs":
        from .arc_policy import canonical_history_inputs
        validate_examples(examples, stage)
        examples.inputs = canonical_history_inputs(examples.inputs)
    return examples, estimated


def _disk_song(songs, record, stage, identities):
    path = _local(songs.root, record["file"], ".npz")
    metadata_path = _local(songs.root, record["metadata_file"], ".json")
    if file_hash(path) != record["sha256"] or file_hash(metadata_path) != record["metadata_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {record['song_id']}")
    meta = read_json(metadata_path)
    metadata = meta["metadata"]
    if (meta["song_id"] != record["song_id"] or metadata.get("split") != record["split"]
            or metadata.get("group_id") != record["group_id"]):
        raise ValueError("Dataset song identity/split mismatch")
    for key in ("audio_sha256", "decoded_sha256"):
        if record.get(key) and metadata.get(key) != record[key]:
            raise ValueError("Dataset recording alias disagrees with indexed identity")
    _metadata(metadata, meta["song_id"], identities)
    stages = meta.get("stages", [])
    if not isinstance(stages, list) or len(set(stages)) != len(stages) or any(s not in NEW_STAGES for s in stages):
        raise ValueError("Invalid dataset stages")
    examples, estimated = _stage_arrays(path, stage) if stage in stages else (None, 0)
    if examples is not None:
        validate_examples(examples, stage)
    return examples, estimated, metadata


def iter_stage_examples(dataset, stage, split=None):
    """Validate/load only each requested song and stage as the iterator advances."""
    _selection(stage, split)
    if isinstance(dataset, TrainingData):
        if dataset.stage != stage:
            raise ValueError("Training data belongs to another v2 stage")
        yield from dataset.examples(split)
        return
    songs = dataset.songs
    identities = {}
    if isinstance(songs, _Songs):
        for record in songs.records:
            if split is None or stage_split(getattr(dataset, "manifest", {}), stage, record) == split:
                examples, estimated, metadata = _disk_song(songs, record, stage, identities)
                del estimated, metadata
                if examples is not None:
                    yield examples
                del examples
        return
    records = getattr(songs, "records", None)
    selected = (songs[index] for index, record in enumerate(records)
                if split is None or stage_split(getattr(dataset, "manifest", {}), stage, record) == split) if records is not None else iter(songs)
    for song in selected:
        if split is None or stage_split(getattr(dataset, "manifest", {}), stage, song.base.metadata) == split:
            _metadata(song.base.metadata, song.base.song_id, identities)
            examples = song.examples.get(stage)
            if examples is not None:
                validate_examples(examples, stage)
                yield examples


class TrainingData:
    """A stage reader; construction will scan once before epochs reuse its checks."""

    def __init__(self, dataset, stage, ram_cache_mib=1024, verbose=0, require_real=False):
        _selection(stage)
        from .arc_policy import validate_split_overlay
        validate_split_overlay(dataset.manifest)
        if (isinstance(ram_cache_mib, bool) or not isinstance(ram_cache_mib, Real)
                or not np.isfinite(ram_cache_mib) or ram_cache_mib < 0):
            raise ValueError("RAM cache budget must be a finite nonnegative MiB value")
        if require_real:
            encoder = dataset.manifest.get("encoder_identity", {})
            if (dataset.manifest.get("status") != "prepared" or encoder.get("status") != "retained"
                    or not encoder.get("checkpoint_fingerprint")):
                raise ValueError("Training requires real prepared examples from the explicit retained encoder")
        self.stage = stage
        self.support = {split: {} for split in SPLITS}
        effective_identities = {}
        self.counts = {split: {"songs": 0, "examples": 0, "supervised_values": 0} for split in SPLITS}
        self._entries = []
        self._cache = []
        self._songs = dataset.songs
        disk = isinstance(self._songs, _Songs)
        self._disk = disk
        records = getattr(self._songs, "records", None)
        total = len(records) if records is not None else len(self._songs)
        budget = int(ram_cache_mib * 1024 ** 2)
        caching = budget > 0
        identities, song_ids = {}, set()
        stage_bytes = estimated_bytes = retained = scanned = 0
        for index in range(total):
            if disk:
                record = self._songs.records[index]
                path = _local(self._songs.root, record["file"], ".npz")
                metadata_path = _local(self._songs.root, record["metadata_file"], ".json")
                source = (path, metadata_path)
                before = tuple(_stat(item) for item in source)
                examples, estimated, metadata = _disk_song(self._songs, record, stage, identities)
                _unchanged(source, before)
                song_id = record["song_id"]
            else:
                song = self._songs[index]
                song_id, metadata = song.base.song_id, song.base.metadata
                _metadata(metadata, song_id, identities)
                examples = song.examples.get(stage)
                if examples is not None:
                    validate_examples(examples, stage)
                estimated = _nbytes(examples) if examples is not None else 0
                source, before = index, None
            if song_id in song_ids:
                raise ValueError("Dataset contains duplicate song IDs")
            song_ids.add(song_id)
            if require_real:
                adapter = str(metadata.get("provenance", {}).get("adapter", ""))
                if "dummy" in adapter.lower() or adapter == "v2_synthetic_fixture":
                    raise ValueError("Dummy examples cannot be used for training")
            scanned += 1
            if examples is not None:
                split = stage_split(getattr(dataset, "manifest", {}), stage, {**metadata, "song_id": song_id})
                _metadata({**metadata, "split": split}, song_id, effective_identities)
                if stage in SPARSE_STAGES:
                    add_support(self.support[split], support(examples, stage), metadata["group_id"])
                count = self.counts[split]
                count["songs"] += 1
                count["examples"] += len(examples.times)
                count["supervised_values"] += sum(int(np.count_nonzero(w)) for w in examples.weights.values())
                size = _nbytes(examples)
                stage_bytes += size
                estimated_bytes += estimated
                if caching and estimated_bytes > budget:
                    caching = False
                    self._cache.clear()
                    retained = 0
                if caching:
                    self._cache.append(examples)
                    retained += size
                self._entries.append((split, source, before))
            if verbose and (scanned == 1 or scanned % 25 == 0 or scanned == total):
                mode = "ram" if caching else "stream"
                print(f"{stage}: scanned {scanned}/{total} songs; "
                      f"cache={mode}, retained={retained} bytes, budget={budget} bytes", flush=True)
            # Metadata contains source objects and reduction diagnostics. It is
            # deliberately absent from the reader's retained per-song index.
            del metadata, examples
            if not disk:
                del song
        if disk or caching:
            self._songs = None
        self.support = {split: describe_support(value) for split, value in self.support.items()}
        self.report = {"supervision": self.support, "stage": stage, "mode": "ram" if caching else "stream",
                       "source": "npz" if disk else "memory", "ram_cache_mib": float(ram_cache_mib),
                       "budget_bytes": budget, "retained_bytes": retained, "stage_bytes": stage_bytes,
                       "estimated_stage_bytes": estimated_bytes, "songs_scanned": scanned,
                       "integrity_passes": 1}

    def examples(self, split=None, rng=None):
        """Yield original song order, reusing checks from construction."""
        _selection(self.stage, split)
        order = np.arange(len(self._entries))
        if rng is not None:
            rng.shuffle(order)
        for index in order:
            song_split, source, snapshot = self._entries[index]
            if split is not None and split != song_split:
                continue
            if self.report["mode"] == "ram":
                yield self._cache[index]
            elif self._disk:
                _unchanged(source, snapshot)
                examples, _ = _stage_arrays(source[0], self.stage)
                _unchanged(source, snapshot)
                yield examples
                del examples
            else:
                yield self._songs[source].examples[self.stage]


def _nbytes(examples):
    return examples.times.nbytes + sum(array.nbytes for family in
                                      (examples.inputs, examples.targets, examples.weights)
                                      for array in family.values())


def _stat(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino


def _unchanged(paths, expected):
    try:
        actual = tuple(_stat(path) for path in paths)
    except OSError as error:
        raise ValueError("Dataset source changed after training input validation") from error
    if actual != expected:
        raise ValueError("Dataset source changed after training input validation")

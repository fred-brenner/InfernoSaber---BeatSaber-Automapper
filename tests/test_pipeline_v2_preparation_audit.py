"""Compact training metadata preserves separately hashed preparation evidence."""

from copy import deepcopy
import json

import pytest

from training.pipeline_v2 import artifacts
from training.pipeline_v2.data import make_dummy_dataset


@pytest.fixture(scope="module")
def dataset():
    return make_dummy_dataset()


def _detailed_audit():
    return {"lighting": {"status": "ready", "policy": {"version": 2},
                         "selected": {"coverage": .97, "queries": 100, "songs": [{"song_id": "song"}],
                                      "collisions": [{"source_ids": ["a", "b"]}], "collision_queries": 1},
                         "candidate_schedules": [{"queries": 100, "songs": [{"song_id": "song"}]}],
                         "unsupported_records": [{"id": "event", "reason": "custom_data_dependent"}]},
            "support": {"lighting": {"train": {"events": {"positive": [1.], "negative": [9.],
                                                           "groups": ["group"], "positive_groups": ["group"]}}}},
            "songs": [{"song_id": "song", "source_exclusions": [{"id": "event", "reason": "custom_data_dependent"}]}]}


def test_pure_compaction_retains_summaries_and_does_not_change_evidence():
    audit = _detailed_audit()
    manifest = {"preparation_audit": audit,
                "preparation_songs": [{"song_id": "song", "group_id": "group", "positive_count": 1,
                                       "source_exclusions": audit["songs"][0]["source_exclusions"],
                                       "unsupported_reasons": {"custom_data_dependent": 1}}]}
    before = deepcopy(manifest)
    reference = {"file": "preparation-audit.json", "sha256": "a" * 64}
    compact = artifacts.compact_preparation_metadata(manifest, reference)
    assert manifest == before
    assert compact["preparation_audit"]["file"] == reference["file"]
    assert compact["preparation_audit"]["sha256"] == reference["sha256"]
    assert compact["preparation_audit"]["support"] == audit["support"]
    assert compact["preparation_audit"]["source_exclusion_count"] == 1
    lighting = compact["preparation_audit"]["lighting"]
    assert lighting["selected"] == {"coverage": .97, "queries": 100, "collision_queries": 1}
    assert lighting["candidate_schedules"] == [{"queries": 100}]
    assert lighting["unsupported_record_count"] == 1
    assert lighting["unsupported_reasons"] == {"custom_data_dependent": 1}
    assert compact["preparation_songs"] == [{key: value for key, value in manifest["preparation_songs"][0].items()
                                             if key != "source_exclusions"}]
    assert artifacts.compact_preparation_metadata(compact, reference) == compact
    with pytest.raises(ValueError, match="different preparation audit"):
        artifacts.compact_preparation_metadata(compact, {**reference, "sha256": "b" * 64})


def test_writer_preserves_full_audit_and_normal_loading_does_not_parse_it(tmp_path, dataset, monkeypatch):
    original = deepcopy(dataset)
    original.manifest["preparation_songs"] = original.manifest["songs"]
    before = deepcopy(original.manifest)
    root = tmp_path / "dataset"
    report = artifacts.write_dataset(root, original)
    assert original.manifest == before
    sidecar = root / "preparation-audit.json"
    assert artifacts.read_json(sidecar) == before["preparation_audit"]
    assert report["preparation_audit"]["sha256"] == artifacts.file_hash(sidecar)
    assert all("source_exclusions" not in record for record in report["preparation_songs"])
    read = artifacts.read_json
    opened = []

    def read_without_audit(path):
        opened.append(path.name)
        assert path.name != "preparation-audit.json"
        return read(path)

    monkeypatch.setattr(artifacts, "read_json", read_without_audit)
    restored = artifacts.load_dataset(root)
    assert len(restored.songs) == len(dataset.songs)
    assert opened == ["manifest.json"]
    monkeypatch.setattr(artifacts, "read_json", read)
    assert artifacts.load_preparation_audit(root, report) == before["preparation_audit"]
    assert artifacts.load_preparation_audit(root) == before["preparation_audit"]


def test_explicit_audit_loading_checks_hash_and_supports_legacy_inline(tmp_path):
    audit = _detailed_audit()
    path = tmp_path / "preparation-audit.json"
    artifacts.write_json(path, audit)
    compact = artifacts.compact_preparation_metadata({"preparation_audit": audit},
                                                    {"file": path.name, "sha256": artifacts.file_hash(path)})
    assert artifacts.load_preparation_audit(tmp_path, {"preparation_audit": audit}) == audit
    assert artifacts.load_preparation_audit(tmp_path, {}) == {}
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.load_preparation_audit(tmp_path, compact)
    with pytest.raises(ValueError, match="Invalid preparation audit"):
        artifacts.load_preparation_audit(tmp_path, {"preparation_audit": {"file": "../audit.json", "sha256": "a" * 64}})


def test_already_compact_writer_verifies_sidecar_without_overwriting(tmp_path, dataset):
    audit = dataset.manifest["preparation_audit"]
    writer = artifacts.DatasetWriter(tmp_path / "dataset")
    for song in dataset.songs:
        writer.add(song)
    path = writer.root / "preparation-audit.json"
    # Noncanonical whitespace is intentional: a valid existing full sidecar is
    # an immutable byte sequence once referenced by its hash.
    content = json.dumps(audit).encode("utf-8")
    path.write_bytes(content)
    compact = artifacts.compact_preparation_metadata(dataset.manifest, {"file": path.name, "sha256": artifacts.file_hash(path)})
    report = writer.finish(dataset.vocabularies, compact)
    assert path.read_bytes() == content
    assert artifacts.load_preparation_audit(writer.root, report) == audit


def test_writer_rejects_a_dangling_compact_reference(tmp_path, dataset):
    writer = artifacts.DatasetWriter(tmp_path / "dataset")
    writer.add(dataset.songs[0])
    compact = artifacts.compact_preparation_metadata(dataset.manifest, {"file": "preparation-audit.json", "sha256": "a" * 64})
    with pytest.raises(ValueError, match="Missing detailed preparation audit"):
        writer.finish(dataset.vocabularies, compact)
    assert not (writer.root / "manifest.json").exists()


def test_writer_does_not_replace_a_different_existing_detailed_audit(tmp_path, dataset):
    writer = artifacts.DatasetWriter(tmp_path / "dataset")
    writer.add(dataset.songs[0])
    path = writer.root / "preparation-audit.json"
    content = b'{"existing": "evidence"}\n'
    path.write_bytes(content)
    with pytest.raises(ValueError, match="not overwritten"):
        writer.finish(dataset.vocabularies, dataset.manifest)
    assert path.read_bytes() == content
    assert not (writer.root / "manifest.json").exists()


def test_partial_or_unknown_version_references_do_not_become_inline_audits(tmp_path):
    with pytest.raises(ValueError, match="Invalid preparation audit"):
        artifacts.load_preparation_audit(tmp_path, {"preparation_audit": {"file": "preparation-audit.json"}})
    with pytest.raises(ValueError, match="reference version"):
        artifacts.load_preparation_audit(tmp_path, {"preparation_audit": {
            "version": 2, "file": "preparation-audit.json", "sha256": "a" * 64}})

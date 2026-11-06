import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.playability_report import (
    DIFFICULTY_ORDER,
    band_label,
    decode_package_notes,
    list_standard_difficulties,
    score_rows,
    verdict,
)


def write_package(tmp_path, *, info_changes=None, beatmaps=None):
    """Write a synthetic package; ``beatmaps`` maps difficulty -> beatmap dict."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    beatmaps = beatmaps or {"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [],
        "colorNotes": [{"b": 2, "x": 1, "y": 1, "c": 0, "d": 0}],
    }}
    info = {
        "_version": "2.0.0", "_songName": "Human Song", "_songAuthorName": "Artist",
        "_levelAuthorName": "Mapper", "_beatsPerMinute": 120, "_songFilename": "song.ogg",
        "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [
                {"_difficulty": name, "_beatmapFilename": f"{name}.dat"}
                for name in beatmaps],
        }],
    }
    info.update(info_changes or {})
    (tmp_path / "INFO.DAT").write_text(json.dumps(info), encoding="utf-8-sig")
    for name, beatmap in beatmaps.items():
        (tmp_path / f"{name}.dat").write_text(json.dumps(beatmap), encoding="utf-8-sig")
    (tmp_path / "song.ogg").write_bytes(b"audio")
    return tmp_path


def test_decode_v3_colored_notes(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [],
        "colorNotes": [
            {"b": 4, "x": 2, "y": 1, "c": 1, "d": 1},
            {"b": 2, "x": 1, "y": 0, "c": 0, "d": 0},
            {"b": 3, "x": 0, "y": 2, "c": 0, "d": 8},
        ],
        "bombNotes": [{"b": 5, "x": 3, "y": 1}],
        "sliders": [{"b": 6, "x": 0, "y": 0, "c": 0, "d": 0}],
        "obstacles": [{"b": 7, "x": 0, "y": 0, "d": 1, "w": 1, "h": 1}],
    }})
    decoded = decode_package_notes(tmp_path)
    assert decoded["bpm"] == 120
    assert decoded["difficulty"] == "ExpertPlus"
    assert decoded["notes"] == [
        (1.0, 0, 1, 0, 0),
        (1.5, 0, 0, 2, 8),
        (2.0, 1, 2, 1, 1),
    ]


def test_decode_v2_notes(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "_version": "2.2.0", "_BPMChanges": [],
        "_notes": [
            {"_time": 4, "_lineIndex": 2, "_lineLayer": 1, "_type": 1, "_cutDirection": 3},
            {"_time": 2, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
            {"_time": 3, "_lineIndex": 0, "_lineLayer": 2, "_type": 3, "_cutDirection": 0},
        ],
        "_obstacles": [{"_time": 5, "_type": 0, "_lineIndex": 0}],
        "_events": [{"_time": 6, "_type": 1, "_value": 1}],
    }})
    decoded = decode_package_notes(tmp_path)
    assert decoded["notes"] == [
        (1.0, 0, 1, 0, 1),
        (2.0, 1, 2, 1, 3),
    ]


def test_decode_highest_difficulty_default(tmp_path):
    write_package(tmp_path, beatmaps={
        "Hard": {"version": "3.2.0", "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}]},
        "ExpertPlus": {"version": "3.2.0", "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1}]},
    })
    decoded = decode_package_notes(tmp_path)
    assert decoded["difficulty"] == "ExpertPlus"
    assert decoded["notes"] == [(2.0, 1, 1, 1, 1)]


def test_decode_explicit_difficulty(tmp_path):
    write_package(tmp_path, beatmaps={
        "Hard": {"version": "3.2.0", "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}]},
        "ExpertPlus": {"version": "3.2.0", "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1}]},
    })
    decoded = decode_package_notes(tmp_path, difficulty="Hard")
    assert decoded["difficulty"] == "Hard"
    assert decoded["notes"] == [(1.0, 0, 0, 0, 0)]
    with pytest.raises(ValueError, match="Standard/Easy"):
        decode_package_notes(tmp_path, difficulty="Easy")


def test_list_standard_difficulties_canonical_order(tmp_path):
    assert DIFFICULTY_ORDER == ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    write_package(tmp_path, beatmaps={
        "ExpertPlus": {"version": "3.2.0", "colorNotes": []},
        "Easy": {"version": "3.2.0", "colorNotes": []},
        "Hard": {"version": "3.2.0", "colorNotes": []},
    })
    assert list_standard_difficulties(tmp_path) == ["Easy", "Hard", "ExpertPlus"]


def test_decode_nonstandard_skipped(tmp_path):
    groups = [{"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"},
    ]}]
    write_package(tmp_path, info_changes={"_difficultyBeatmapSets": groups})
    assert list_standard_difficulties(tmp_path) == []
    with pytest.raises(ValueError, match="Standard"):
        decode_package_notes(tmp_path)


@pytest.mark.parametrize("filename", [None, "", 4])
def test_rejects_missing_beatmap_filename(tmp_path, filename):
    entry = {"_difficulty": "ExpertPlus"}
    if filename is not None:
        entry["_beatmapFilename"] = filename
    groups = [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [entry]}]
    write_package(tmp_path, info_changes={"_difficultyBeatmapSets": groups})
    with pytest.raises(ValueError, match="filename"):
        decode_package_notes(tmp_path)
    with pytest.raises(ValueError, match="filename"):
        list_standard_difficulties(tmp_path)


def test_decode_tempo_change_rejected(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [{"b": 4, "m": 150}],
        "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}],
    }})
    with pytest.raises(ValueError, match="tempo changes"):
        decode_package_notes(tmp_path)


def fake_scores(nps=4.0, span=60.0, **metrics):
    """Minimal ``score_map``-shaped dict for verdict tests."""
    scores = {"notes": 0, "span_seconds": span, "nps": nps}
    for name in ("m1", "m2", "m3", "m4"):
        metric = metrics.get(name, {})
        scores[name] = {"value": metric.get("value", 0.0),
                        "units": np.asarray(metric.get("units", []), dtype=float)}
    return scores


def human_row(name, nps=4.0, span=60.0, **metrics):
    return {"package": name, "difficulty": "ExpertPlus",
            "scores": fake_scores(nps=nps, span=span, **metrics)}


def test_band_labels():
    labels = [band_label(nps) for nps in (0, 2.99, 3, 4.99, 5, 6.99, 7, 20)]
    assert labels == ["<3", "<3", "3-5", "3-5", "5-7", "5-7", "7+", "7+"]


def test_percentile_and_band_fallback():
    human_rows = [human_row("low", nps=8.0, m1={"value": 0.1}),
                  human_row("mid", nps=4.0, m1={"value": 0.2}),
                  human_row("high", nps=1.0, m1={"value": 0.3})]
    gen = fake_scores(nps=4.0, m1={"value": 0.25})
    result = verdict(gen, human_rows)["m1"]
    assert result["band"] == "3-5"
    assert result["band_n"] == 1          # below the 5-map in-band floor
    # band_n < 5 → percentile falls back to all three human scalars:
    # 0.1 and 0.2 are <= 0.25 → 2/3.
    assert result["percentile"] == pytest.approx(200 / 3)
    assert not result["flag"]


def test_incident_rate():
    units = list(range(1, 21))
    human_rows = [human_row("h1", nps=6.0, m4={"value": 0.5, "units": units}),
                  human_row("h2", nps=6.0, m4={"value": 0.5, "units": units})]
    # pooled p95 over the 40 human units is 19.05 → only the two 20s count
    # → human rate = 2 incidents / 2 span-minutes = 1.0.
    gen = fake_scores(nps=6.0, m4={"value": 0.05, "units": [20.0, 20.0, 20.0]})
    result = verdict(gen, human_rows)["m4"]
    assert result["human_incident_rate"] == pytest.approx(1.0)
    assert result["incident_rate"] == pytest.approx(3.0)
    assert result["flag"]                 # exactly 3x the human rate flags

    calm = fake_scores(nps=6.0, m4={"value": 0.05, "units": [20.0]})
    calm_result = verdict(calm, human_rows)["m4"]
    assert calm_result["incident_rate"] == pytest.approx(1.0)
    assert not calm_result["flag"]

    # percentile >= 95 flags even with zero incidents.
    hot = fake_scores(nps=6.0, m4={"value": 9.9, "units": []})
    hot_result = verdict(hot, human_rows)["m4"]
    assert hot_result["percentile"] == 100.0
    assert hot_result["incident_rate"] == 0.0
    assert hot_result["flag"]

    # an empty human unit pool gives no rate verdict — no flag.
    pooled_empty = [human_row("h", nps=6.0, m4={"value": 0.5, "units": []})]
    assert not verdict(gen, pooled_empty)["m4"]["flag"]


def test_score_rows_skip(tmp_path):
    good = write_package(tmp_path / "good_pkg", beatmaps={
        "Hard": {"version": "3.2.0",
                 "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0},
                                {"b": 8, "x": 2, "y": 1, "c": 1, "d": 1}]},
        "ExpertPlus": {"version": "3.2.0",
                       "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1},
                                      {"b": 12, "x": 2, "y": 1, "c": 0, "d": 0}]},
    })
    bad = tmp_path / "bad_pkg"
    bad.mkdir()                           # no Info.dat at all
    rows, skipped = score_rows([bad, good], per_difficulty=False)
    assert [row["package"] for row in rows] == ["good_pkg"]
    assert rows[0]["difficulty"] == "ExpertPlus"   # highest by default
    assert rows[0]["scores"]["notes"] == 2
    assert len(skipped) == 1
    assert skipped[0]["package"] == "bad_pkg"
    assert skipped[0]["error"]

    rows, skipped = score_rows([good], per_difficulty=True)
    assert [row["difficulty"] for row in rows] == ["Hard", "ExpertPlus"]
    assert skipped == []


def test_score_rows_skips_tiny_maps(tmp_path):
    tiny = write_package(tmp_path / "tiny_pkg", beatmaps={"ExpertPlus": {
        "version": "3.2.0",
        "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}],
    }})
    rows, skipped = score_rows([tiny], per_difficulty=False)
    assert rows == []
    assert len(skipped) == 1 and "2 playable notes" in skipped[0]["error"]

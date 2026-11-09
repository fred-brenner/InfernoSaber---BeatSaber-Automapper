import json
from pathlib import Path
import subprocess
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
                        "units": np.asarray(metric.get("units", []), dtype=float),
                        "incidents": []}
    if "m5" in metrics:
        scores["m5"] = metrics["m5"]
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

    # a partially broken package still scores the healthy difficulty, and
    # the skip error names the failed one.
    partial = write_package(tmp_path / "partial_pkg", beatmaps={
        "Hard": {"version": "3.2.0",
                 "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0},
                                {"b": 8, "x": 2, "y": 1, "c": 1, "d": 1}]},
        "ExpertPlus": {"version": "3.2.0",
                       "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1},
                                      {"b": 12, "x": 2, "y": 1, "c": 0, "d": 0}]},
    })
    (partial / "ExpertPlus.dat").unlink()
    rows, skipped = score_rows([partial], per_difficulty=True)
    assert [row["difficulty"] for row in rows] == ["Hard"]
    assert len(skipped) == 1
    assert skipped[0]["package"] == "partial_pkg"
    assert skipped[0]["error"].startswith("ExpertPlus: ")


def test_score_rows_skips_tiny_maps(tmp_path):
    tiny = write_package(tmp_path / "tiny_pkg", beatmaps={"ExpertPlus": {
        "version": "3.2.0",
        "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}],
    }})
    rows, skipped = score_rows([tiny], per_difficulty=False)
    assert rows == []
    assert len(skipped) == 1 and "2 playable notes" in skipped[0]["error"]


def m5_scores(share=0.5, fast=1, span=60.0, transitions=10):
    """Minimal direction-variety summary for corpus comparison tests."""
    return {"transitions": transitions, "non_reversal_share": share,
            "fast_reversals": fast,
            "fast_reversals_per_minute": fast * 60 / span if span else None}


def test_m5_verdict_uses_low_share_and_pooled_fast_event_rate():
    humans = [human_row(f"h{i}", m5=m5_scores(share=share))
              for i, share in enumerate((0.2, 0.3, 0.4, 0.5, 0.6))]
    # Five eligible in-band rows exclude this high-rate, out-of-band map.
    humans.append(human_row("outside", nps=8, m5=m5_scores(fast=100)))
    result = verdict(fake_scores(m5=m5_scores(share=0.1, fast=3)), humans)["m5"]
    assert result["percentile"] == 0.0
    assert result["percentile_direction"] == "lower"
    assert result["incident_rate_direction"] == "higher"
    assert result["band_n"] == result["reference_n"] == 5
    assert result["incident_rate"] == 3.0
    assert result["human_incident_rate"] == 1.0
    assert result["non_reversal_share_flag"]
    assert result["fast_reversal_rate_flag"]
    assert result["flag"]


def test_m5_rate_fallback_uses_counts_and_durations_not_mean_map_rates():
    humans = [human_row("short", span=30, m5=m5_scores(fast=1, span=30)),
              human_row("long", nps=8, span=90, m5=m5_scores(fast=1, span=90)),
              human_row("old_cached")]
    # Two events / two minutes = one per minute, not mean(2, 2/3).
    result = verdict(fake_scores(m5=m5_scores(fast=3)), humans)["m5"]
    assert result["band_n"] == 1
    assert result["reference_n"] == 2
    assert result["human_incident_rate"] == 1.0
    assert result["incident_rate"] == 3.0
    assert not result["non_reversal_share_flag"]
    assert result["fast_reversal_rate_flag"]
    assert result["flag"]
    calm = verdict(fake_scores(span=61, m5=m5_scores(fast=3, span=61)), humans)["m5"]
    assert not calm["flag"]


@pytest.mark.parametrize("share,fast", [(0.0, 0), (0.5, 0), (0.5, 1)])
def test_m5_ties_do_not_flag(share, fast):
    humans = [human_row("same", m5=m5_scores(share=share, fast=fast))]
    result = verdict(fake_scores(m5=m5_scores(share=share, fast=fast)), humans)["m5"]
    assert result["percentile"] == 100.0
    assert not result["flag"]


@pytest.mark.parametrize("generated,human", [
    (None, m5_scores()),
    (m5_scores(), None),
    (m5_scores(transitions=0, share=None), m5_scores()),
    (m5_scores(), m5_scores(transitions=0, share=None)),
])
def test_m5_unavailable_scores_do_not_flag(generated, human):
    gen = fake_scores(**({"m5": generated} if generated is not None else {}))
    row = human_row("h", **({"m5": human} if human is not None else {}))
    result = verdict(gen, [row])["m5"]
    assert result["percentile"] is None
    assert result["incident_rate"] is None
    assert result["human_incident_rate"] is None
    assert not result["flag"]


def test_m5_zero_human_fast_rate_flags_positive_generated_events():
    result = verdict(fake_scores(m5=m5_scores(fast=1)),
                     [human_row("h", m5=m5_scores(fast=0))])["m5"]
    assert result["percentile"] == 100.0
    assert result["human_incident_rate"] == 0.0
    assert result["fast_reversal_rate_flag"]
    assert result["flag"]


def run_report_cli(tmp_path, human_root, generated_root):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.playability_report",
         "--human-root", str(human_root), "--generated", str(generated_root),
         "--json", str(output)],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    return json.loads(output.read_text()), result.stdout


def direction_package(root, directions):
    return write_package(root, beatmaps={"ExpertPlus": {
        "version": "3.2.0", "colorNotes": [
            {"b": 2 + i * 0.4, "x": 1, "y": 1, "c": 0, "d": direction}
            for i, direction in enumerate(directions)],
    }})


def test_cli_m5_json_round_trip_and_direction_labels(tmp_path):
    humans, generated = tmp_path / "humans", tmp_path / "generated"
    for i in range(5):
        direction_package(humans / f"human{i}", [0, 2, 0, 2])
    direction_package(generated / "repetitive", [0, 1, 0, 1])
    report, console = run_report_cli(tmp_path, humans, generated)
    row = report["generated"][0]
    assert row["scores"]["m5"]["transitions"] == 3
    assert row["scores"]["m5"]["non_reversal_share"] == 0.0
    assert row["scores"]["m5"]["fast_reversals"] == 3
    assert row["scores"]["m5"]["fast_reversals_per_minute"] == pytest.approx(300)
    assert row["verdict"]["m5"]["flag"]
    diagnostics = report["human"]["rows"][0]["scores"]["m5"]["diagnostics"]["overall"]
    assert {angle: summary["count"] for angle, summary in diagnostics["angles"].items()} == {
        "0": 0, "45": 0, "90": 3, ">=135": 0}
    assert diagnostics["non_reversal_intervals"]["seconds"]["median"] == pytest.approx(0.2)
    assert row["flagged"].count("M5") == 1
    quantiles = report["human"]["quantiles"]["5-7"]["m5"]
    assert quantiles["maps"] == 5
    assert quantiles["non_reversal_share"]["p5"] == 1.0
    assert quantiles["fast_reversals_per_minute"]["p95"] == 0.0
    assert report["human"]["incident_p95"]["m5"] is None
    assert report["human"]["incident_rates"]["m5"] == 0.0
    assert "non-reversal share (low) 0.00%" in console
    assert "fast reversals/min (high)" in console
    assert "Cut turns (0/45/90/>=135 deg): 0/0/0/3" in console


@pytest.mark.parametrize("baseline", ["empty", "sparse"])
def test_cli_m5_empty_or_sparse_baseline_is_na(tmp_path, baseline):
    humans, generated = tmp_path / "humans", tmp_path / "generated"
    humans.mkdir()
    if baseline == "sparse":
        # Both notes score for legacy metrics but neither hand has a transition.
        write_package(humans / "sparse", beatmaps={"ExpertPlus": {
            "version": "3.2.0", "colorNotes": [
                {"b": 2, "x": 0, "y": 1, "c": 0, "d": 0},
                {"b": 4, "x": 3, "y": 1, "c": 1, "d": 1}],
        }})
    direction_package(generated / "repetitive", [0, 1, 0, 1])
    report, console = run_report_cli(tmp_path, humans, generated)
    m5 = report["generated"][0]["verdict"]["m5"]
    assert m5["percentile"] is None
    assert m5["incident_rate"] is None
    assert not m5["flag"]
    assert "M5" not in report["generated"][0]["flagged"]
    assert "M5" in console and "n/a" in console
    assert "pn/a" not in console


@pytest.mark.parametrize("with_m5", [False, True])
def test_cli_cached_scores_keep_existing_flags(tmp_path, monkeypatch, capsys, with_m5):
    from evaluation import playability_report

    humans, generated = tmp_path / "humans", tmp_path / "generated"
    humans.mkdir()
    generated.mkdir()
    human = human_row("old_human", **{name: {"value": 1.0}
                                      for name in ("m1", "m2", "m3", "m4")})
    gen = human_row("old_generated", m4={"value": 2.0})
    if with_m5:
        human["scores"]["m5"] = m5_scores(fast=0)
        gen["scores"]["m5"] = m5_scores(share=0.0, fast=1)
    monkeypatch.setattr(playability_report, "score_rows",
                        lambda packages, per_difficulty: ([gen if per_difficulty else human], []))
    output = tmp_path / "cached.json"
    playability_report.main(["--human-root", str(humans), "--generated", str(generated),
                             "--json", str(output)])
    report = json.loads(output.read_text())
    assert report["generated"][0]["flagged"] == (["M4", "M5"] if with_m5 else ["M4"])
    assert report["generated"][0]["verdict"]["m5"]["percentile"] == (0.0 if with_m5 else None)
    console = capsys.readouterr().out
    assert ("FLAG M4,M5" if with_m5 else "FLAG M4") in console


def test_m5_low_share_flag_has_lower_tail_orientation():
    humans = [human_row("human", m5=m5_scores(share=0.5, fast=0))]
    low = verdict(fake_scores(m5=m5_scores(share=0.0, fast=0)), humans)["m5"]
    high = verdict(fake_scores(m5=m5_scores(share=1.0, fast=0)), humans)["m5"]
    assert low["percentile"] == 0.0
    assert low["non_reversal_share_flag"] and low["flag"]
    assert not low["fast_reversal_rate_flag"]
    assert high["percentile"] == 100.0
    assert not high["flag"]


def test_cli_m5_no_generated_transitions_is_na(tmp_path):
    humans, generated = tmp_path / "humans", tmp_path / "generated"
    direction_package(humans / "human", [0, 2, 0, 2])
    write_package(generated / "sparse", beatmaps={"ExpertPlus": {
        "version": "3.2.0", "colorNotes": [
            {"b": 2, "x": 0, "y": 1, "c": 0, "d": 0},
            {"b": 4, "x": 3, "y": 1, "c": 1, "d": 1}],
    }})
    report, console = run_report_cli(tmp_path, humans, generated)
    row = report["generated"][0]
    assert row["scores"]["m5"]["transitions"] == 0
    assert row["scores"]["m5"]["non_reversal_share"] is None
    assert row["verdict"]["m5"]["percentile"] is None
    assert row["verdict"]["m5"]["incident_rate"] is None
    assert "M5" not in row["flagged"]
    assert "M5 non-reversal share (low) n/a" in console


def test_m5_shared_low_minimum_does_not_flag():
    humans = [human_row(f"h{i}", m5=m5_scores(share=0.0 if i < 2 else 0.5, fast=0))
              for i in range(20)]
    result = verdict(fake_scores(m5=m5_scores(share=0.0, fast=0)), humans)["m5"]
    assert result["percentile"] == 10.0
    assert not result["flag"]

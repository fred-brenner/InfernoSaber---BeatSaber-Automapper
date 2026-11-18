"""Summary time-estimate line and duration formatting for the Run tab."""

from app_helper.check_input import format_time_estimate, get_summary
from tools.config import config, paths


def _setup_songs(monkeypatch, tmp_path, count, workers):
    for i in range(count):
        (tmp_path / f"song{i}.mp3").write_text("x")
    monkeypatch.setattr(paths, "songs_pred", str(tmp_path) + "/")
    monkeypatch.setattr(paths, "bs_song_path", str(tmp_path) + "/")
    monkeypatch.setattr(config, "num_workers", workers)


def test_format_time_estimate_seconds():
    assert format_time_estimate(0) == "0 s"
    assert format_time_estimate(45) == "45 s"
    assert format_time_estimate(59) == "59 s"


def test_format_time_estimate_minutes():
    assert format_time_estimate(60) == "1 min"
    assert format_time_estimate(77) == "1.3 min"
    assert format_time_estimate(600) == "10 min"


def test_format_time_estimate_hours():
    assert format_time_estimate(3600) == "1 h"
    assert format_time_estimate(5400) == "1.5 h"
    assert format_time_estimate(7200) == "2 h"


def test_summary_reports_estimated_time(monkeypatch, tmp_path):
    # 4 songs x 2 difficulties = 8 runs
    # 20 s + 4 * 1 s song startup + ceil(8 * 15 s * 1.5 / 3 workers) = 24 + 60 = 84 s
    _setup_songs(monkeypatch, tmp_path, count=4, workers=3)
    summary = get_summary(5, 7, 0, 0, 0)
    assert "Total runs: 8" in summary
    assert "Info: Estimated time: ~1.4 min (3 worker(s))" in summary


def test_summary_estimate_scales_with_workers(monkeypatch, tmp_path):
    # 2 songs, 2 runs; 20 s + 2 * 1 s + ceil(2 * 15 s * 1.5 / 1) = 22 + 45 = 67 s
    _setup_songs(monkeypatch, tmp_path, count=2, workers=1)
    assert "~1.1 min" in get_summary(5, 0, 0, 0, 0)
    # 20 s + 2 * 1 s + ceil(2 * 15 s * 1.5 / 4) = 22 + 12 = 34 s
    _setup_songs(monkeypatch, tmp_path, count=2, workers=4)
    assert "~34 s" in get_summary(5, 0, 0, 0, 0)


def test_summary_omits_estimate_without_runs(monkeypatch, tmp_path):
    _setup_songs(monkeypatch, tmp_path, count=3, workers=2)
    summary = get_summary(0, 0, 0, 0, 0)
    assert "Total runs: 0" in summary
    assert "Estimated time" not in summary

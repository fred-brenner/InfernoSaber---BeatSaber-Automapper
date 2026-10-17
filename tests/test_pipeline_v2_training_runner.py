"""Training input and execution plumbing without starting training."""

import numpy as np
import pytest
from threading import Event
from types import SimpleNamespace

from training.pipeline_v2.artifacts import load_dataset, write_dataset
from training.pipeline_v2.data import make_dummy_dataset
from training.pipeline_v2.runner import _epoch, batches, prefetch_batches
from training.pipeline_v2.settings import performance_settings


def test_training_batches_do_not_decompress_other_models(tmp_path, monkeypatch):
    write_dataset(tmp_path / "dataset", make_dummy_dataset())
    dataset = load_dataset(tmp_path / "dataset")
    original = np.lib.npyio.NpzFile.__getitem__
    reads = []

    def track(archive, key):
        reads.append(key)
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", track)
    batch = next(batches(dataset, "positions", "train", batch_size=2))
    assert len(batch.times) == 2
    assert reads and all(key.startswith("positions::") for key in reads)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_prefetch_preserves_order_and_final_item(count):
    assert list(prefetch_batches(iter(range(7)), count)) == list(range(7))


def test_prefetch_bounds_lookahead_and_closes_the_producer():
    reached, closed = Event(), Event()
    produced = []

    def producer():
        try:
            for index in range(20):
                produced.append(index)
                if len(produced) == 3:
                    reached.set()
                yield index
        finally:
            closed.set()

    stream = prefetch_batches(producer(), 2)
    assert next(stream) == 0
    assert reached.wait(5)
    assert len(produced) == 3  # one consumed batch and two lookahead batches
    stream.close()
    assert closed.is_set()
    assert len(produced) == 3


def test_prefetch_propagates_read_failure():
    def broken():
        yield 0
        raise ValueError("bad archive")

    stream = prefetch_batches(broken(), 2)
    assert next(stream) == 0
    with pytest.raises(ValueError, match="bad archive"):
        next(stream)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_prefetch_count(count):
    with pytest.raises(ValueError, match="Prefetch"):
        list(prefetch_batches(iter([]), count))


def test_epoch_reports_progress_and_retains_tail_without_training(monkeypatch, capsys):
    from training.pipeline_v2 import runner

    source = [SimpleNamespace(times=np.arange(size)) for size in (3, 3, 1)]
    monkeypatch.setattr(runner, "batches", lambda *args: iter(source))
    data = SimpleNamespace(counts={"validation": {"examples": 7}})

    class ValidationOnly:
        def reset(self):
            self.sizes = []

        def step(self, examples, training=False):
            assert not training
            self.sizes.append(len(examples.times))

        def loss(self):
            return .125

    execution = ValidationOnly()
    loss, report = _epoch(data, "positions", execution, 3, prefetch=2, log_every=2, verbose=1)
    assert execution.sizes == [3, 3, 1]
    assert loss == .125
    assert report["examples"] == 7 and report["batches"] == 3
    assert report["seconds"] > 0 and report["examples_per_second"] > 0
    assert "batch 2/3" in capsys.readouterr().out


def test_performance_settings_are_execution_only(monkeypatch):
    from tools.config import config

    assert performance_settings() == {
        "ram_cache_mib": 1024, "prefetch_batches": 2,
        "compile_steps": True, "log_every_batches": 100}
    monkeypatch.setattr(config, "pipeline_v2_training_prefetch_batches", -1)
    with pytest.raises(ValueError, match="prefetch_batches"):
        performance_settings()


def test_compiled_step_setting_requires_boolean(monkeypatch):
    from tools.config import config

    monkeypatch.setattr(config, "pipeline_v2_training_compile_steps", 1)
    with pytest.raises(ValueError, match="Compiled training"):
        performance_settings()

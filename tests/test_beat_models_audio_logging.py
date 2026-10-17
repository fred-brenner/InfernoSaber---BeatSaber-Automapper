"""Native decoder logging keeps all output except one complete Vorbis warning."""

import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from training.beat_models import audio_logging
from training.beat_models.features import decode_audio


WARNING = b"[vorbis @ 0x7fABC123] Could not update timestamps for discarded samples.\n"


@pytest.mark.parametrize("warning", [WARNING, WARNING.replace(b"\n", b"\r\n")])
def test_only_exact_warning_is_filtered(capfdbinary, warning):
    preserved = [
        WARNING.replace(b"vorbis", b"opus"),
        WARNING.replace(b"discarded", b"skipped"),
        WARNING.replace(b"0x7fABC123", b"0xGG"),
        WARNING.replace(b"0x7fABC123", b"(nil)"),
        WARNING.replace(b"0x7fABC123", b"0X7fABC123"),
        b"prefix " + WARNING,
        WARNING.replace(b".\n", b". More details\n"),
        b"A different decoder error\n",
        b"Binary stderr: \xff\x00\n",
    ]
    with audio_logging.filter_vorbis_timestamp_warning():
        os.write(2, b"before\n" + warning + b"".join(preserved) + warning + b"after\n")
    assert capfdbinary.readouterr().err == b"before\n" + b"".join(preserved) + b"after\n"


def test_split_writes_and_unterminated_messages_are_preserved(capfdbinary):
    with audio_logging.filter_vorbis_timestamp_warning():
        for byte in WARNING:
            os.write(2, bytes([byte]))
        os.write(2, b"prefix ")
        os.write(2, WARNING)
        os.write(2, WARNING[:-1])
    assert capfdbinary.readouterr().err == b"prefix " + WARNING + WARNING[:-1]


def test_nonmatching_unterminated_output_is_forwarded_during_decode(monkeypatch, capfdbinary):
    forwarded = threading.Event()
    write_all = audio_logging._write_all

    def observe(fd, data):
        write_all(fd, data)
        if data == b"decoder progress":
            forwarded.set()

    monkeypatch.setattr(audio_logging, "_write_all", observe)
    with audio_logging.filter_vorbis_timestamp_warning():
        os.write(2, b"decoder progress")
        assert forwarded.wait(2)
    assert capfdbinary.readouterr().err == b"decoder progress"


def test_scope_restores_stderr_after_exception_and_repeated_nested_calls(capfdbinary):
    failure = ValueError("decode failed")
    for _ in range(3):
        with pytest.raises(ValueError) as caught:
            with audio_logging.filter_vorbis_timestamp_warning():
                with audio_logging.filter_vorbis_timestamp_warning():
                    os.write(2, WARNING + b"decode diagnostic\n")
                    raise failure
        assert caught.value is failure
        os.write(2, WARNING)
    assert capfdbinary.readouterr().err == (b"decode diagnostic\n" + WARNING) * 3


def test_stderr_descriptor_inheritance_is_preserved(capfdbinary):
    original = os.get_inheritable(2)
    try:
        os.set_inheritable(2, False)
        with audio_logging.filter_vorbis_timestamp_warning():
            assert not os.get_inheritable(2)
            os.write(2, WARNING)
        assert not os.get_inheritable(2)
        os.write(2, b"restored\n")
    finally:
        os.set_inheritable(2, original)
    assert capfdbinary.readouterr().err == b"restored\n"


@pytest.mark.parametrize("failure_stage", ["start", "redirect"])
def test_filter_setup_failure_releases_pipe_and_restores_stderr(
        monkeypatch, capfdbinary, failure_stage):
    failure = RuntimeError("could not start stderr filter")

    def fail(*args, **kwargs):
        raise failure

    if failure_stage == "start":
        monkeypatch.setattr(audio_logging.threading.Thread, "start", fail)
    else:
        monkeypatch.setattr(audio_logging.os, "dup2", fail)
    with pytest.raises(RuntimeError) as caught:
        with audio_logging.filter_vorbis_timestamp_warning():
            pytest.fail("failed filter must not start decoding")
    assert caught.value is failure
    monkeypatch.undo()
    os.write(2, WARNING)
    assert capfdbinary.readouterr().err == WARNING
    assert not any(thread.name == "vorbis-stderr-filter" for thread in threading.enumerate())


@pytest.mark.parametrize("decode_fails", [False, True])
def test_failed_output_sink_does_not_block_decoder_or_replace_decode_exception(
        monkeypatch, capfdbinary, decode_fails):
    output_failure = OSError("output sink failed")
    decode_failure = RuntimeError("decode failed")

    def fail_write(fd, data):
        raise output_failure

    monkeypatch.setattr(audio_logging, "_write_all", fail_write)
    with pytest.raises(RuntimeError if decode_fails else OSError) as caught:
        with audio_logging.filter_vorbis_timestamp_warning():
            # More than a pipe buffer: its reader must keep draining on errors.
            os.write(2, b"decoder diagnostic\n" * 20000)
            if decode_fails:
                raise decode_failure
    assert caught.value is (decode_failure if decode_fails else output_failure)
    os.write(2, b"restored\n")
    assert capfdbinary.readouterr().err == b"restored\n"


def test_concurrent_decode_scopes_are_serialized(capfdbinary):
    attempted = threading.Event()
    entered = threading.Event()

    def other_decode():
        attempted.set()
        with audio_logging.filter_vorbis_timestamp_warning():
            entered.set()
            os.write(2, WARNING + b"second decode\n")

    with audio_logging.filter_vorbis_timestamp_warning():
        worker = threading.Thread(target=other_decode)
        worker.start()
        assert attempted.wait(2)
        assert not entered.wait(.05)
        os.write(2, WARNING + b"first decode\n")
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert entered.is_set()
    assert capfdbinary.readouterr().err == b"first decode\nsecond decode\n"


@pytest.mark.parametrize("failure_stage", [None, "create", "read", "nonfinite"])
def test_decode_scope_includes_source_creation_reads_and_destruction(
        monkeypatch, capfdbinary, failure_stage):
    failure = RuntimeError("native decoder failed")

    def read_source():
        os.write(2, WARNING + b"read diagnostic\n")
        if failure_stage == "read":
            raise failure
        return np.array([np.nan if failure_stage == "nonfinite" else .5, .25, 99]), 2

    class Source:
        hop_size = 4096
        # Native aubio reads do not retain Python self frames in a traceback.
        __call__ = staticmethod(read_source)

        def __del__(self):
            os.write(2, WARNING + b"close diagnostic\n")

    def create_source(path, **kwargs):
        assert path == "song.ogg"
        assert kwargs == {"samplerate": 16000, "hop_size": 4096, "channels": 1}
        os.write(2, WARNING + b"create diagnostic\n")
        if failure_stage == "create":
            raise failure
        return Source()

    monkeypatch.setitem(sys.modules, "aubio", SimpleNamespace(source=create_source))
    if failure_stage in ("create", "read"):
        with pytest.raises(RuntimeError) as caught:
            decode_audio("song.ogg")
        assert caught.value is failure
    elif failure_stage == "nonfinite":
        with pytest.raises(ValueError, match="nonfinite"):
            decode_audio("song.ogg")
    else:
        np.testing.assert_array_equal(decode_audio("song.ogg"), np.array([.5, .25], np.float32))
    os.write(2, b"outside decode\n")
    output = capfdbinary.readouterr().err
    assert b"create diagnostic\n" in output
    if failure_stage == "create":
        assert b"read diagnostic\n" not in output
        assert b"close diagnostic\n" not in output
    else:
        assert b"read diagnostic\n" in output
        assert b"close diagnostic\n" in output
    assert output.endswith(b"outside decode\n")
    assert WARNING not in output

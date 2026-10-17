"""Narrow, temporary filtering of a native decoder's stderr output."""

from contextlib import contextmanager
import os
import re
import threading


_STDERR_LOCK = threading.RLock()
_PREFIX = b"[vorbis @ 0x"
_MESSAGE = b" Could not update timestamps for discarded samples."
_WARNING = re.compile(rb"\[vorbis @ 0x[0-9a-fA-F]+\]" + re.escape(_MESSAGE) + rb"\r?\n")


def _possible_warning(partial):
    """Buffer only text that can still become the exact, complete warning."""
    if len(partial) <= len(_PREFIX):
        return _PREFIX.startswith(partial)
    if not partial.startswith(_PREFIX):
        return False
    address, separator, message = partial[len(_PREFIX):].partition(b"]")
    if not address or address.lstrip(b"0123456789abcdefABCDEF"):
        return False
    return not separator or _MESSAGE.startswith(message) or message == _MESSAGE + b"\r"


def _write_all(fd, data):
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if not written:
            raise OSError("Could not forward decoder stderr")
        remaining = remaining[written:]


def _forward_stderr(read_fd, original_fd, errors):
    pending = b""
    passthrough = False

    def forward(data):
        if not errors:
            try:
                _write_all(original_fd, data)
            except OSError as error:
                # Keep draining even if the original output sink fails, so the
                # decoder cannot block indefinitely on a full internal pipe.
                errors.append(error)

    try:
        while chunk := os.read(read_fd, 4096):
            parts = chunk.split(b"\n")
            for index, part in enumerate(parts):
                complete = index < len(parts) - 1
                data = part + (b"\n" if complete else b"")
                if passthrough:
                    forward(data)
                    if complete:
                        passthrough = False
                    continue
                pending += data
                if complete:
                    if not _WARNING.fullmatch(pending):
                        forward(pending)
                    pending = b""
                elif not _possible_warning(pending):
                    forward(pending)
                    pending = b""
                    passthrough = True
        # An unterminated message is never discarded.
        forward(pending)
    except OSError as error:
        errors.append(error)
    finally:
        os.close(read_fd)


@contextmanager
def filter_vorbis_timestamp_warning():
    """Remove one exact FFmpeg warning during a synchronous aubio decode.

    Native FFmpeg writes bypass Python's stderr object. Descriptor redirection
    is process-wide while this scope is active; unrelated stderr is forwarded
    unchanged. Decode scopes are serialized and may nest in the same thread.
    The reader writes to the saved descriptor, never into its own input pipe.
    """
    with _STDERR_LOCK:
        inheritable = os.get_inheritable(2)
        original_fd = os.dup(2)
        try:
            read_fd, write_fd = os.pipe()
        except BaseException:
            os.close(original_fd)
            raise
        errors = []
        started = False
        redirected = False
        try:
            reader = threading.Thread(target=_forward_stderr,
                                      args=(read_fd, original_fd, errors),
                                      name="vorbis-stderr-filter", daemon=True)
            reader.start()
            started = True
            os.dup2(write_fd, 2, inheritable=inheritable)
            redirected = True
            os.close(write_fd)
            write_fd = None
            yield
        finally:
            if redirected:
                os.dup2(original_fd, 2, inheritable=inheritable)
            if write_fd is not None:
                os.close(write_fd)
            if started:
                reader.join()
            else:
                os.close(read_fd)
            os.close(original_fd)
        # A decoder exception takes precedence over an output-sink failure.
        if errors:
            raise errors[0]

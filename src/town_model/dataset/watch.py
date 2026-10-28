"""Best-effort Linux inotify invalidation for immutable dataset snapshots.

The watcher is deliberately process-local and dependency-free. It observes
writes, metadata changes, replacement, creation, and removal for dataset files;
callers still verify SHA-256 identity for bytes they actually consume.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# linux/inotify.h event bits; IN_ACCESS/OPEN/CLOSE_NOWRITE are intentionally
# omitted so reading a validated dataset does not invalidate its own cache.
_EVENT_MASK = (
    0x00000002  # IN_MODIFY
    | 0x00000004  # IN_ATTRIB
    | 0x00000008  # IN_CLOSE_WRITE
    | 0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
    | 0x00000400  # IN_DELETE_SELF
    | 0x00000800  # IN_MOVE_SELF
    | 0x00004000  # IN_Q_OVERFLOW
)
_REPORT_DIRECTORY = "reports"


class DatasetFileWatcher:
    """Non-blocking watcher; any relevant event invalidates the whole snapshot."""

    def __init__(self, root: Path) -> None:
        if sys.platform != "linux":
            raise NotImplementedError("inotify-backed dataset caching is available on Linux")

        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int

        self._libc = libc
        self._add_watch = add_watch
        self._fd = init(os.O_CLOEXEC | os.O_NONBLOCK)
        if self._fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self._closed = False
        try:
            self._watch_tree(root.resolve())
        except Exception:
            self.close()
            raise

    def _watch(self, path: Path) -> None:
        result = self._add_watch(self._fd, os.fsencode(path), _EVENT_MASK)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))

    def _watch_tree(self, root: Path) -> None:
        if not root.is_dir():
            raise NotADirectoryError(root)
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            relative = current_path.relative_to(root)
            if relative.parts and relative.parts[0] == _REPORT_DIRECTORY:
                directories.clear()
                continue
            directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
            self._watch(current_path)
            for name in files:
                path = current_path / name
                if not path.is_symlink():
                    self._watch(path)

    def consume_changes(self) -> bool:
        """Drain queued events and report whether any dataset state may differ."""

        if self._closed:
            return True
        changed = False
        while True:
            try:
                data = os.read(self._fd, 1024 * 1024)
            except BlockingIOError:
                break
            except OSError:
                return True
            if not data:
                break
            offset = 0
            while offset + 16 <= len(data):
                mask = int.from_bytes(data[offset + 4 : offset + 8], byteorder=sys.byteorder)
                name_length = int.from_bytes(data[offset + 12 : offset + 16], byteorder=sys.byteorder)
                offset += 16 + name_length
                changed = True
        return changed

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            os.close(self._fd)

    def __del__(self) -> None:
        self.close()

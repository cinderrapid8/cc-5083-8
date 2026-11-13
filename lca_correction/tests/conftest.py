"""Pytest bootstrap: headless Qt and local GUI runtime libraries.

In a normal desktop install none of this is needed; it only makes the GUI
test runnable in stripped-down containers where the Qt platform libraries
were unpacked into a user-local sysroot and LD_LIBRARY_PATH cannot be set
before the interpreter starts. Libraries are preloaded with RTLD_GLOBAL so
their transitive dependencies resolve from the same directory.
"""

import glob
import os
import ctypes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_LIBDIRS = [
    "/tmp/syslibs/root/lib/aarch64-linux-gnu",
    "/tmp/syslibs/root/usr/lib/aarch64-linux-gnu",
]


def _preload_sysroot() -> None:
    if not any(os.path.isdir(d) for d in _LIBDIRS):
        return
    # Repeated passes resolve inter-library dependencies regardless of order.
    paths = [p for d in _LIBDIRS for p in sorted(glob.glob(d + "/lib*.so*"))]
    loaded = set()
    for _ in range(4):
        progress = False
        for path in paths:
            if path in loaded or not os.path.isfile(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.add(path)
                progress = True
            except OSError:
                continue
        if not progress:
            break


_preload_sysroot()

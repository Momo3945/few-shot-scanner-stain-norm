#!/usr/bin/env python3
"""
progress.py -- small progress-bar helper shared by the eval scripts.

Registration and baseline scoring both grind through the held-out inventory for
minutes at a time (affine ECC is 5000 iterations per frame), so a silent run is
indistinguishable from a hung one. This wraps tqdm when it is installed and
degrades to a dependency-free bar when it is not, so the eval scripts never gain
a hard requirement just for progress reporting.

Dependencies: none required (tqdm used automatically if present).
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only without tqdm
    _tqdm = None


class _FallbackBar:
    """Minimal stderr progress bar used when tqdm is unavailable."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(0, total)
        self.desc = desc
        self.n = 0
        self.postfix = ""
        self.start = time.monotonic()
        self._draw()

    # tqdm-compatible surface -------------------------------------------------
    def update(self, n: int = 1) -> None:
        self.n += n
        self._draw()

    def set_postfix_str(self, s: str) -> None:
        self.postfix = s
        self._draw()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self) -> "_FallbackBar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------------
    def _draw(self) -> None:
        if not self.total:
            return
        frac = min(1.0, self.n / self.total)
        width = max(10, min(shutil.get_terminal_size((80, 20)).columns - 45, 40))
        filled = int(width * frac)
        elapsed = time.monotonic() - self.start
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        bar = "#" * filled + "-" * (width - filled)
        line = (f"\r{self.desc}: {frac * 100:5.1f}% |{bar}| "
                f"{self.n}/{self.total} [{_fmt(elapsed)}<{_fmt(eta)}]")
        if self.postfix:
            line += f" {self.postfix}"
        sys.stderr.write(line.ljust(shutil.get_terminal_size((80, 20)).columns - 1))
        sys.stderr.flush()


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def progress(iterable: Sequence[T] | Iterable[T], desc: str,
             total: int | None = None, unit: str = "it"):
    """Wrap `iterable` in a progress bar (tqdm if available, fallback if not).

    Returns an object that is both iterable and usable as a context manager,
    exposing tqdm's `update` / `set_postfix_str` either way.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = 0
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, unit=unit,
                     dynamic_ncols=True, leave=True)
    return _FallbackIterator(iterable, desc, total)


class _FallbackIterator:
    """Iterator adapter giving _FallbackBar the same call shape as tqdm."""

    def __init__(self, iterable: Iterable[T], desc: str, total: int) -> None:
        self._iterable = iterable
        self._bar = _FallbackBar(total, desc)

    def __iter__(self) -> Iterator[T]:
        for item in self._iterable:
            yield item
            self._bar.update(1)
        self._bar.close()

    def set_postfix_str(self, s: str) -> None:
        self._bar.set_postfix_str(s)

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def close(self) -> None:
        self._bar.close()

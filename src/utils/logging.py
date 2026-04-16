"""Logging utilities for bias_mi pipeline."""

import time

from tqdm import tqdm


def log(msg: str, flush: bool = True) -> None:
    """Print a message with flush=True for tee piping compatibility."""
    print(msg, flush=flush)


class ProgressLogger:
    """Tracks and logs progress for item-level processing loops.

    Kept for backward compatibility — new code should prefer ``progress_bar``.
    """

    def __init__(self, total: int, prefix: str = ""):
        self.total = total
        self.prefix = prefix
        self.start_time = time.time()
        self.count = 0

    def step(self, extra: str = "") -> None:
        """Log progress for the current item."""
        self.count += 1
        elapsed = time.time() - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0.0
        parts = [f"[{self.count}/{self.total}]"]
        if self.prefix:
            parts.insert(0, self.prefix)
        parts.append(f"{rate:.1f} items/s")
        if extra:
            parts.append(extra)
        log(" ".join(parts))

    def skip(self, reason: str = "exists") -> None:
        """Log a skipped item without affecting rate calculation."""
        self.count += 1
        parts = [f"[{self.count}/{self.total}]"]
        if self.prefix:
            parts.insert(0, self.prefix)
        parts.append(f"skipped ({reason})")
        log(" ".join(parts))


def progress_bar(
    iterable: any = None,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
    **kwargs: any,
) -> tqdm:
    """Create a tqdm progress bar with project-standard settings.

    Works in terminals, Jupyter notebooks, and piped output.  All pipeline
    loops should use this instead of bare ``for`` loops.

    Examples::

        for item in progress_bar(items, desc="Extracting"):
            process(item)

        pbar = progress_bar(total=100, desc="Layer 14")
        for i in range(100):
            pbar.update(1)
        pbar.close()
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        dynamic_ncols=True,
        leave=True,
        **kwargs,
    )

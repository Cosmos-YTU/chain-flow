from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator

import torch


def synchronize_if_needed(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class TimingStats:
    sections: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.sections[name] = self.sections.get(name, 0.0) + seconds

    def get(self, name: str, default: float = 0.0) -> float:
        return self.sections.get(name, default)

    def merge(self, other: "TimingStats", prefix: str | None = None) -> None:
        for name, seconds in other.sections.items():
            self.add(f"{prefix}.{name}" if prefix else name, seconds)


@contextmanager
def timed_section(
    stats: TimingStats,
    name: str,
    device: torch.device | str | None = None,
) -> Iterator[None]:
    synchronize_if_needed(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize_if_needed(device)
        stats.add(name, time.perf_counter() - start)

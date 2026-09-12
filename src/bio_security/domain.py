"""Pure domain contracts for the camera/detection benchmark slice."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space face bounding box."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("bounding-box origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding-box dimensions must be positive")


@dataclass(frozen=True, slots=True)
class TimingSample:
    """Timing information for one captured and processed frame."""

    capture_ms: float
    detection_ms: float
    total_ms: float
    face_count: int

    def __post_init__(self) -> None:
        if min(self.capture_ms, self.detection_ms, self.total_ms) < 0:
            raise ValueError("timings cannot be negative")
        if self.face_count < 0:
            raise ValueError("face_count cannot be negative")


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Aggregate evidence emitted at the end of a local benchmark session."""

    frames: int
    mean_capture_ms: float
    median_detection_ms: float
    p95_detection_ms: float
    median_total_ms: float
    p95_total_ms: float
    mean_fps: float


class TimingAccumulator:
    """Collect timing samples without depending on camera or CV libraries."""

    def __init__(self) -> None:
        self._samples: list[TimingSample] = []

    def add(self, sample: TimingSample) -> None:
        self._samples.append(sample)

    @property
    def count(self) -> int:
        return len(self._samples)

    def summary(self) -> TimingSummary | None:
        if not self._samples:
            return None

        capture = [sample.capture_ms for sample in self._samples]
        detection = [sample.detection_ms for sample in self._samples]
        total = [sample.total_ms for sample in self._samples]
        valid_fps = [1000.0 / value for value in total if value > 0]

        return TimingSummary(
            frames=len(self._samples),
            mean_capture_ms=fmean(capture),
            median_detection_ms=median(detection),
            p95_detection_ms=_percentile(detection, 0.95),
            median_total_ms=median(total),
            p95_total_ms=_percentile(total, 0.95),
            mean_fps=fmean(valid_fps) if valid_fps else 0.0,
        )


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for deterministic benchmark reporting."""

    if not values:
        raise ValueError("cannot calculate percentile of an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

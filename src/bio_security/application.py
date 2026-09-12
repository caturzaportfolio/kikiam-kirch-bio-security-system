"""Application orchestration for a single camera-to-detection vertical slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from bio_security.domain import BoundingBox, TimingAccumulator, TimingSample
from bio_security.ports import FaceDetector, Frame, FrameSource

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class DetectionObservation:
    """One processed frame plus its non-biometric benchmark metadata."""

    frame: Frame
    boxes: tuple[BoundingBox, ...]
    timing: TimingSample


class DetectionSession:
    """Coordinates capture, detection, and timing without owning CV implementation details."""

    def __init__(
        self,
        frame_source: FrameSource,
        detector: FaceDetector,
        metrics: TimingAccumulator | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self._frame_source = frame_source
        self._detector = detector
        self.metrics = metrics or TimingAccumulator()
        self._clock = clock

    def process_next_frame(self) -> DetectionObservation:
        total_started = self._clock()

        capture_started = self._clock()
        frame = self._frame_source.read()
        capture_finished = self._clock()

        detection_started = self._clock()
        boxes = tuple(self._detector.detect(frame))
        detection_finished = self._clock()

        total_finished = self._clock()
        sample = TimingSample(
            capture_ms=(capture_finished - capture_started) * 1000.0,
            detection_ms=(detection_finished - detection_started) * 1000.0,
            total_ms=(total_finished - total_started) * 1000.0,
            face_count=len(boxes),
        )
        self.metrics.add(sample)
        return DetectionObservation(frame=frame, boxes=boxes, timing=sample)

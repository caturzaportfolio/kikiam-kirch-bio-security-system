import pytest

from bio_security.application import DetectionSession
from bio_security.domain import BoundingBox


class FakeFrameSource:
    description = "fake camera"

    def __init__(self) -> None:
        self.frame = object()

    def open(self) -> None:
        pass

    def read(self) -> object:
        return self.frame

    def close(self) -> None:
        pass


class FakeDetector:
    description = "fake detector"

    def detect(self, frame: object) -> tuple[BoundingBox, ...]:
        return (BoundingBox(x=10, y=20, width=30, height=40),)


class StepClock:
    def __init__(self, step_seconds: float = 0.01) -> None:
        self._value = -step_seconds
        self._step = step_seconds

    def __call__(self) -> float:
        self._value += self._step
        return self._value


def test_detection_session_coordinates_capture_detection_and_metrics() -> None:
    frame_source = FakeFrameSource()
    session = DetectionSession(
        frame_source=frame_source,
        detector=FakeDetector(),
        clock=StepClock(),
    )

    observation = session.process_next_frame()

    assert observation.frame is frame_source.frame
    assert observation.boxes == (BoundingBox(x=10, y=20, width=30, height=40),)
    assert observation.timing.capture_ms == pytest.approx(10.0)
    assert observation.timing.detection_ms == pytest.approx(10.0)
    assert observation.timing.total_ms == pytest.approx(50.0)
    assert observation.timing.face_count == 1
    assert session.metrics.count == 1

"""OpenCV adapters for BIO-001 camera capture, face detection, and local display."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from bio_security.domain import BoundingBox, TimingSample
from bio_security.ports import Frame


class CameraOpenError(RuntimeError):
    """Raised when the configured camera cannot be opened."""


class CameraReadError(RuntimeError):
    """Raised when a previously opened camera cannot provide a usable frame."""


class DetectorLoadError(RuntimeError):
    """Raised when the configured detector asset cannot be loaded."""


class OpenCVCamera:
    """Local OpenCV VideoCapture adapter with explicit lifecycle management."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        if index < 0:
            raise ValueError("camera index must be non-negative")
        self._index = index
        self._width = width
        self._height = height
        self._capture: Any | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(f"unable to open camera index {self._index}")

        if self._width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._capture = capture

    def read(self) -> Frame:
        if self._capture is None:
            raise CameraReadError("camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraReadError(f"camera index {self._index} failed to provide a frame")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def description(self) -> str:
        if self._capture is None:
            return f"OpenCV camera index={self._index} (closed)"
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        return f"OpenCV camera index={self._index} {width}x{height} reported_fps={fps:.2f}"


class OpenCVHaarFaceDetector:
    """Replaceable baseline detector using OpenCV's packaged frontal-face cascade."""

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: int = 40,
        cascade_path: Path | None = None,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ValueError("min_neighbors cannot be negative")
        if min_face_size <= 0:
            raise ValueError("min_face_size must be positive")

        default_cascade = (
            Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
        )
        path = cascade_path or default_cascade
        classifier = cv2.CascadeClassifier(str(path))
        if classifier.empty():
            raise DetectorLoadError(f"unable to load Haar cascade: {path}")

        self._classifier = classifier
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_size = min_face_size
        self._cascade_path = path

    @property
    def description(self) -> str:
        return f"OpenCV Haar frontal-face cascade ({self._cascade_path.name})"

    def detect(self, frame: Frame) -> tuple[BoundingBox, ...]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_face_size, self._min_face_size),
        )
        return tuple(
            BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
            for x, y, width, height in detections
        )


class OpenCVWindow:
    """Local visualization that deliberately does not save captured frames."""

    def __init__(self, title: str = "Kikiam Bio Security — BIO-001") -> None:
        self._title = title

    def show(
        self,
        frame: Frame,
        boxes: tuple[BoundingBox, ...],
        timing: TimingSample,
    ) -> None:
        for box in boxes:
            cv2.rectangle(
                frame,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 255, 0),
                2,
            )

        label = (
            f"faces={timing.face_count} detection={timing.detection_ms:.1f}ms "
            f"total={timing.total_ms:.1f}ms"
        )
        cv2.putText(
            frame,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._title, frame)

    def exit_requested(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def close(self) -> None:
        cv2.destroyWindow(self._title)

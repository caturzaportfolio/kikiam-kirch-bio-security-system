"""YuNet detection and SFace embedding adapters for BIO-003."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from bio_security.domain import BoundingBox
from bio_security.identity import FeatureVector, RecognitionResult
from bio_security.ports import Frame

FaceRow = npt.NDArray[np.float32]


class DNNModelLoadError(RuntimeError):
    """Raised when an OpenCV DNN face model cannot be initialized."""


@dataclass(frozen=True, slots=True)
class DetectedFace:
    box: BoundingBox
    confidence: float
    row: FaceRow


class YuNetFaceDetector:
    """OpenCV YuNet detector with landmark rows cached for SFace alignment."""

    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = 0.90,
        nms_threshold: float = 0.30,
        top_k: int = 100,
    ) -> None:
        if not 0.0 < score_threshold <= 1.0:
            raise ValueError("score_threshold must be in (0, 1]")
        try:
            self._detector: Any = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),
                score_threshold,
                nms_threshold,
                top_k,
            )
        except cv2.error as exc:
            raise DNNModelLoadError(f"failed to load YuNet model: {exc}") from exc
        self._rows: dict[tuple[int, int, int, int], FaceRow] = {}
        self._score_threshold = score_threshold

    @property
    def description(self) -> str:
        return f"OpenCV YuNet (score>={self._score_threshold:.2f})"

    def detect(self, frame: Frame) -> tuple[BoundingBox, ...]:
        height, width = frame.shape[:2]
        self._detector.setInputSize((int(width), int(height)))
        _retval, faces = self._detector.detect(frame)
        self._rows.clear()
        if faces is None:
            return ()

        matrix = np.asarray(faces, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        boxes: list[BoundingBox] = []
        for raw in matrix:
            if raw.size < 15:
                continue
            x = max(0, int(round(float(raw[0]))))
            y = max(0, int(round(float(raw[1]))))
            box_width = min(int(round(float(raw[2]))), int(width) - x)
            box_height = min(int(round(float(raw[3]))), int(height) - y)
            if box_width <= 0 or box_height <= 0:
                continue
            confidence = float(raw[-1])
            if confidence < self._score_threshold:
                continue
            box = BoundingBox(x=x, y=y, width=box_width, height=box_height)
            row = raw.astype(np.float32, copy=True)
            self._rows[(x, y, box_width, box_height)] = cast(FaceRow, row)
            boxes.append(box)
        return tuple(boxes)

    def face_row(self, box: BoundingBox) -> FaceRow:
        key = (box.x, box.y, box.width, box.height)
        try:
            return self._rows[key]
        except KeyError as exc:
            raise ValueError("YuNet landmark row is unavailable for this face") from exc


class SFaceFeatureExtractor:
    """Align a YuNet face and derive an SFace embedding."""

    def __init__(self, model_path: Path, detector: YuNetFaceDetector) -> None:
        try:
            self._recognizer: Any = cv2.FaceRecognizerSF.create(str(model_path), "")
        except cv2.error as exc:
            raise DNNModelLoadError(f"failed to load SFace model: {exc}") from exc
        self._detector = detector

    @property
    def description(self) -> str:
        return "OpenCV SFace 2021dec"

    def extract(self, frame: Frame, box: BoundingBox) -> FeatureVector:
        face_row = self._detector.face_row(box)
        aligned = self._recognizer.alignCrop(frame, face_row)
        feature = self._recognizer.feature(aligned)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1).copy()
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("SFace embedding has zero norm")
        vector /= norm
        return cast(FeatureVector, vector.astype(np.float32, copy=False))


class TemporalMatchStabilizer:
    """Require repeated identity agreement before presenting a stable MATCH."""

    def __init__(self, *, window_size: int = 5, required_matches: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if required_matches < 1 or required_matches > window_size:
            raise ValueError("required_matches must be within the window")
        self._required_matches = required_matches
        self._history: deque[RecognitionResult] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def observe(self, result: RecognitionResult) -> RecognitionResult:
        self._history.append(result)
        matched = [
            item
            for item in self._history
            if item.status == "MATCH" and item.person is not None
        ]
        if not matched:
            return RecognitionResult(status="UNKNOWN", person=None, similarity=result.similarity)

        counts = Counter(item.person.person_id for item in matched if item.person is not None)
        person_id, count = counts.most_common(1)[0]
        if count < self._required_matches:
            return RecognitionResult(status="UNKNOWN", person=None, similarity=result.similarity)

        same_person = [item for item in matched if item.person is not None and item.person.person_id == person_id]
        person = same_person[-1].person
        similarity = sum(item.similarity for item in same_person) / len(same_person)
        return RecognitionResult(status="MATCH", person=person, similarity=similarity)

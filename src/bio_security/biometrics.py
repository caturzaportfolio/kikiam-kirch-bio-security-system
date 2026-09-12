"""Local biometric feature extraction and template matching for BIO-002."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from bio_security.domain import BoundingBox
from bio_security.identity import (
    BiometricTemplate,
    FeatureVector,
    RecognitionResult,
)
from bio_security.ports import Frame


class LBPFeatureExtractor:
    """Extract a compact local-binary-pattern face template.

    This is a research baseline, not a production-grade face-recognition model.
    Raw face images are not persisted by this class.
    """

    def __init__(self, face_size: int = 64, grid_size: int = 8) -> None:
        if face_size < 32:
            raise ValueError("face_size must be at least 32 pixels")
        if grid_size <= 0 or grid_size > face_size - 2:
            raise ValueError("grid_size is invalid for the configured face size")
        self._face_size = face_size
        self._grid_size = grid_size

    @property
    def feature_dimension(self) -> int:
        return self._grid_size * self._grid_size * 256

    def extract(self, frame: Frame, box: BoundingBox) -> FeatureVector:
        face = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        if face is None or face.size == 0:
            raise ValueError("face crop is empty")

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        normalized = cv2.resize(
            gray,
            (self._face_size, self._face_size),
            interpolation=cv2.INTER_AREA,
        )
        codes = self._lbp_codes(normalized)
        feature = self._grid_histogram(codes)
        norm = float(np.linalg.norm(feature))
        if norm <= 0.0:
            raise ValueError("face template has zero norm")
        normalized_feature = feature / norm
        return cast(FeatureVector, normalized_feature.astype(np.float32, copy=False))

    @staticmethod
    def _lbp_codes(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        center = image[1:-1, 1:-1]
        codes = np.zeros(center.shape, dtype=np.uint8)
        neighbors = (
            image[:-2, :-2],
            image[:-2, 1:-1],
            image[:-2, 2:],
            image[1:-1, 2:],
            image[2:, 2:],
            image[2:, 1:-1],
            image[2:, :-2],
            image[1:-1, :-2],
        )
        for bit, neighbor in enumerate(neighbors):
            mask = (neighbor >= center).astype(np.uint8)
            codes = np.bitwise_or(codes, mask * np.uint8(1 << bit))
        return codes

    def _grid_histogram(self, codes: npt.NDArray[np.uint8]) -> FeatureVector:
        height, width = codes.shape
        histograms: list[npt.NDArray[np.float32]] = []
        for row in range(self._grid_size):
            y0 = row * height // self._grid_size
            y1 = (row + 1) * height // self._grid_size
            for column in range(self._grid_size):
                x0 = column * width // self._grid_size
                x1 = (column + 1) * width // self._grid_size
                cell = codes[y0:y1, x0:x1]
                histogram = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
                total = float(histogram.sum())
                if total > 0:
                    histogram /= total
                histograms.append(histogram)
        combined = np.concatenate(histograms).astype(np.float32, copy=False)
        return cast(FeatureVector, combined)


def cosine_similarity(left: FeatureVector, right: FeatureVector) -> float:
    """Return cosine similarity in [-1, 1] for two equal-length vectors."""

    if left.shape != right.shape:
        raise ValueError("feature vectors must have equal dimensions")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return max(-1.0, min(1.0, similarity))


class TemplateMatcher:
    """Aggregate template similarities by person and emit MATCH/UNKNOWN."""

    def __init__(self, threshold: float = 0.86, top_templates: int = 3) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if top_templates <= 0:
            raise ValueError("top_templates must be positive")
        self.threshold = threshold
        self._top_templates = top_templates

    def match(
        self,
        query: FeatureVector,
        templates: Sequence[BiometricTemplate],
    ) -> RecognitionResult:
        scores_by_person: dict[int, list[tuple[float, BiometricTemplate]]] = defaultdict(list)
        for template in templates:
            if template.person.status != "active":
                continue
            score = cosine_similarity(query, template.feature)
            scores_by_person[template.person.person_id].append((score, template))

        if not scores_by_person:
            return RecognitionResult(status="UNKNOWN", person=None, similarity=0.0)

        best_person = None
        best_score = -1.0
        for scored_templates in scores_by_person.values():
            scored_templates.sort(key=lambda item: item[0], reverse=True)
            selected = scored_templates[: self._top_templates]
            aggregate = sum(item[0] for item in selected) / len(selected)
            if aggregate > best_score:
                best_score = aggregate
                best_person = selected[0][1].person

        if best_person is not None and best_score >= self.threshold:
            return RecognitionResult(status="MATCH", person=best_person, similarity=best_score)
        return RecognitionResult(status="UNKNOWN", person=None, similarity=max(0.0, best_score))

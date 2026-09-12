from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from bio_security.biometrics import LBPFeatureExtractor, TemplateMatcher, cosine_similarity
from bio_security.domain import BoundingBox
from bio_security.identity import BiometricTemplate, PersonRecord


def _person(person_id: int, name: str) -> PersonRecord:
    return PersonRecord(
        person_id=person_id,
        employee_id=f"EMP-{person_id}",
        first_name=name,
        last_name="Test",
        department="QA",
        position="Tester",
        status="active",
        created_at=datetime.now(UTC).isoformat(),
    )


def test_lbp_feature_extractor_returns_normalized_vector() -> None:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (100, 100), (200, 200, 200), -1)
    extractor = LBPFeatureExtractor()

    feature = extractor.extract(frame, BoundingBox(x=10, y=10, width=100, height=100))

    assert feature.shape == (extractor.feature_dimension,)
    assert np.linalg.norm(feature) == pytest.approx(1.0, rel=1e-5)


def test_template_matcher_returns_match_and_unknown() -> None:
    alice = _person(1, "Alice")
    bob = _person(2, "Bob")
    alice_feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bob_feature = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    templates = (
        BiometricTemplate(1, alice, alice_feature, alice.created_at),
        BiometricTemplate(2, alice, alice_feature, alice.created_at),
        BiometricTemplate(3, bob, bob_feature, bob.created_at),
    )
    matcher = TemplateMatcher(threshold=0.8)

    matched = matcher.match(
        np.array([0.99, 0.01, 0.0], dtype=np.float32),
        templates,
    )
    unknown = matcher.match(
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        templates,
    )

    assert matched.status == "MATCH"
    assert matched.person == alice
    assert matched.similarity > 0.99
    assert unknown.status == "UNKNOWN"
    assert unknown.person is None


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        cosine_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

from __future__ import annotations

import numpy as np

from bio_security.dnn_face import TemporalMatchStabilizer
from bio_security.identity import PersonRecord, RecognitionResult


def _person(person_id: int = 1) -> PersonRecord:
    return PersonRecord(
        person_id=person_id,
        employee_id=f"EMP-{person_id}",
        first_name="Test",
        last_name="Person",
        department="IT",
        position="Tester",
        status="active",
        created_at="2026-09-13T00:00:00+00:00",
    )


def test_temporal_stabilizer_requires_three_agreeing_matches() -> None:
    person = _person()
    stabilizer = TemporalMatchStabilizer(window_size=5, required_matches=3)
    candidate = RecognitionResult(status="MATCH", person=person, similarity=0.72)

    assert stabilizer.observe(candidate).status == "UNKNOWN"
    assert stabilizer.observe(candidate).status == "UNKNOWN"
    stable = stabilizer.observe(candidate)

    assert stable.status == "MATCH"
    assert stable.person == person
    assert np.isclose(stable.similarity, 0.72)


def test_temporal_stabilizer_does_not_mix_people() -> None:
    stabilizer = TemporalMatchStabilizer(window_size=5, required_matches=3)
    first = RecognitionResult(status="MATCH", person=_person(1), similarity=0.70)
    second = RecognitionResult(status="MATCH", person=_person(2), similarity=0.71)

    stabilizer.observe(first)
    stabilizer.observe(second)
    result = stabilizer.observe(first)

    assert result.status == "UNKNOWN"

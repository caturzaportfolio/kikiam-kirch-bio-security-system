from __future__ import annotations

import numpy as np
import pytest

from bio_security.identity import PersonRegistration
from bio_security.infrastructure.sqlite_identity import (
    DuplicatePersonError,
    SQLiteIdentityRepository,
)


def test_sqlite_repository_enrolls_person_and_templates(tmp_path) -> None:
    repository = SQLiteIdentityRepository(tmp_path / "registry.sqlite3")
    registration = PersonRegistration(
        employee_id="EMP-001",
        first_name="Kirch",
        last_name="Test",
        department="IT",
        position="Engineer",
    )
    features = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
    )

    person = repository.enroll(registration, features)
    people = repository.list_people()
    templates = repository.load_templates()

    assert person.employee_id == "EMP-001"
    assert people == (person,)
    assert len(templates) == 2
    assert templates[0].person == person
    assert np.allclose(templates[0].feature, features[0])


def test_sqlite_repository_rejects_duplicate_employee_id(tmp_path) -> None:
    repository = SQLiteIdentityRepository(tmp_path / "registry.sqlite3")
    registration = PersonRegistration(
        employee_id="EMP-001",
        first_name="One",
        last_name="Person",
    )
    feature = np.array([1.0, 0.0], dtype=np.float32)

    repository.enroll(registration, (feature,))

    with pytest.raises(DuplicatePersonError):
        repository.enroll(registration, (feature,))

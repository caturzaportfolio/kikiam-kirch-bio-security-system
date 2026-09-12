"""SQLite persistence for local person metadata and biometric templates."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from bio_security.identity import (
    BiometricTemplate,
    FeatureVector,
    PersonRecord,
    PersonRegistration,
)


class DuplicatePersonError(ValueError):
    """Raised when an employee/person identifier is already registered."""


def default_registry_path() -> Path:
    """Return a per-user local data path outside the Git repository."""

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    return base / "KikiamBioSecurity" / "registry.sqlite3"


class SQLiteIdentityRepository:
    """Persist person metadata and derived templates in a local SQLite database."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL UNIQUE,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    department TEXT NOT NULL DEFAULT '',
                    position TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS biometric_templates (
                    template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    feature BLOB NOT NULL,
                    feature_dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                );

                CREATE INDEX IF NOT EXISTS idx_templates_person
                    ON biometric_templates(person_id);
                """
            )

    def enroll(
        self,
        registration: PersonRegistration,
        features: Sequence[FeatureVector],
    ) -> PersonRecord:
        if not features:
            raise ValueError("at least one biometric template is required")
        feature_dimension = int(features[0].size)
        if feature_dimension <= 0:
            raise ValueError("biometric templates cannot be empty")
        if any(int(feature.size) != feature_dimension for feature in features):
            raise ValueError("all biometric templates must have the same dimension")

        created_at = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO persons (
                        employee_id, first_name, last_name, department, position, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        registration.employee_id.strip(),
                        registration.first_name.strip(),
                        registration.last_name.strip(),
                        registration.department.strip(),
                        registration.position.strip(),
                        created_at,
                    ),
                )
                lastrowid = cursor.lastrowid
                if lastrowid is None:
                    raise RuntimeError("SQLite did not return a person identifier")
                person_id = lastrowid
                connection.executemany(
                    """
                    INSERT INTO biometric_templates (
                        person_id, feature, feature_dimension, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            person_id,
                            feature.astype(np.float32, copy=False).tobytes(),
                            feature_dimension,
                            created_at,
                        )
                        for feature in features
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicatePersonError(
                f"employee/person ID already registered: {registration.employee_id}"
            ) from exc

        return PersonRecord(
            person_id=person_id,
            employee_id=registration.employee_id.strip(),
            first_name=registration.first_name.strip(),
            last_name=registration.last_name.strip(),
            department=registration.department.strip(),
            position=registration.position.strip(),
            status="active",
            created_at=created_at,
        )

    def list_people(self) -> tuple[PersonRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT person_id, employee_id, first_name, last_name,
                       department, position, status, created_at
                FROM persons
                ORDER BY last_name COLLATE NOCASE, first_name COLLATE NOCASE
                """
            ).fetchall()
        return tuple(self._person_from_row(row) for row in rows)

    def load_templates(self) -> tuple[BiometricTemplate, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.template_id,
                    t.feature,
                    t.feature_dimension,
                    t.created_at AS template_created_at,
                    p.person_id,
                    p.employee_id,
                    p.first_name,
                    p.last_name,
                    p.department,
                    p.position,
                    p.status,
                    p.created_at AS person_created_at
                FROM biometric_templates AS t
                JOIN persons AS p ON p.person_id = t.person_id
                WHERE p.status = 'active'
                ORDER BY p.person_id, t.template_id
                """
            ).fetchall()

        templates: list[BiometricTemplate] = []
        for row in rows:
            blob = cast(bytes, row["feature"])
            dimension = int(row["feature_dimension"])
            feature = np.frombuffer(blob, dtype=np.float32).copy()
            if int(feature.size) != dimension:
                raise ValueError(
                    f"stored template {int(row['template_id'])} has invalid feature dimension"
                )
            person = self._person_from_joined_row(row)
            templates.append(
                BiometricTemplate(
                    template_id=int(row["template_id"]),
                    person=person,
                    feature=cast(FeatureVector, feature),
                    created_at=str(row["template_created_at"]),
                )
            )
        return tuple(templates)

    @staticmethod
    def _person_from_row(row: sqlite3.Row) -> PersonRecord:
        return PersonRecord(
            person_id=int(row["person_id"]),
            employee_id=str(row["employee_id"]),
            first_name=str(row["first_name"]),
            last_name=str(row["last_name"]),
            department=str(row["department"]),
            position=str(row["position"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _person_from_joined_row(row: sqlite3.Row) -> PersonRecord:
        return PersonRecord(
            person_id=int(row["person_id"]),
            employee_id=str(row["employee_id"]),
            first_name=str(row["first_name"]),
            last_name=str(row["last_name"]),
            department=str(row["department"]),
            position=str(row["position"]),
            status=str(row["status"]),
            created_at=str(row["person_created_at"]),
        )

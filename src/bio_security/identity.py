"""Identity and recognition domain contracts for BIO-002."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FeatureVector = npt.NDArray[np.float32]
RecognitionStatus = Literal["MATCH", "UNKNOWN"]


@dataclass(frozen=True, slots=True)
class PersonRegistration:
    """Operator-supplied metadata for one locally enrolled person."""

    employee_id: str
    first_name: str
    last_name: str
    department: str = ""
    position: str = ""

    def __post_init__(self) -> None:
        if not self.employee_id.strip():
            raise ValueError("employee_id is required")
        if not self.first_name.strip():
            raise ValueError("first_name is required")
        if not self.last_name.strip():
            raise ValueError("last_name is required")


@dataclass(frozen=True, slots=True)
class PersonRecord:
    """Persisted person metadata linked to biometric templates."""

    person_id: int
    employee_id: str
    first_name: str
    last_name: str
    department: str
    position: str
    status: str
    created_at: str

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass(frozen=True, slots=True)
class BiometricTemplate:
    """Derived biometric template; this is sensitive biometric data."""

    template_id: int
    person: PersonRecord
    feature: FeatureVector
    created_at: str


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Recognition measurement, deliberately not an access-control decision."""

    status: RecognitionStatus
    person: PersonRecord | None
    similarity: float

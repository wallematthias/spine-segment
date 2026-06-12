from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float
    is_valid: bool = True
    value: float = 0.0

    @property
    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def invalid(cls) -> "Landmark":
        nan = float("nan")
        return cls(nan, nan, nan, is_valid=False, value=0.0)


def landmark_from_iterable(coords, *, value: float = 0.0, is_valid: bool = True) -> Landmark:
    x, y, z = coords
    return Landmark(float(x), float(y), float(z), is_valid=is_valid, value=float(value))


def is_finite_landmark(landmark: Landmark) -> bool:
    return landmark.is_valid and all(math.isfinite(value) for value in landmark.coords)

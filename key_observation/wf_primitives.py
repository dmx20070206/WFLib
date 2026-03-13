from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class WFPrimitive(ABC):
    #: Unique human-readable identifier used in configs and registries.
    name: str = ""

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _mag_to_ratio(
        self,
        magnitude: int,
        low: float = 0.0,
        high: float = 1.0,
    ) -> float:
        return low + (magnitude - 1) / 4.0 * (high - low)

    def _mag_to_steps(
        self,
        magnitude: int,
        levels: list,
    ):
        if len(levels) != 5:
            raise ValueError("levels must have exactly 5 entries")
        return levels[magnitude - 1]

    def _fixed_length(self, trace: np.ndarray, length: int) -> np.ndarray:
        if len(trace) == length:
            return trace
        if len(trace) > length:
            return trace[:length]
        pad = np.zeros(length - len(trace), dtype=trace.dtype)
        return np.concatenate([trace, pad])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PacketInsert(WFPrimitive):
    name = "PacketInsert"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class PacketDrop(WFPrimitive):
    name = "PacketDrop"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class PacketDuplicate(WFPrimitive):
    name = "PacketDuplicate"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class TimeStretch(WFPrimitive):
    name = "TimeStretch"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class SubsequenceCrop(WFPrimitive):
    name = "SubsequenceCrop"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class DirectionFlip(WFPrimitive):
    name = "DirectionFlip"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class GaussianNoise(WFPrimitive):
    name = "GaussianNoise"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


class WindowWarp(WFPrimitive):
    name = "WindowWarp"

    def __call__(self, trace: np.ndarray, magnitude: int) -> np.ndarray:
        return trace
        # TODO: implement
        raise NotImplementedError(f"{self.name} not yet implemented")


#: Mapping from primitive ``name`` → singleton instance.
PRIMITIVES: dict[str, WFPrimitive] = {
    cls.name: cls()
    for cls in [
        PacketInsert,
        PacketDrop,
        PacketDuplicate,
        TimeStretch,
        SubsequenceCrop,
        DirectionFlip,
        GaussianNoise,
        WindowWarp,
    ]
}

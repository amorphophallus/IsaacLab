# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory buffer for one RR-style episode."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .schema import ENV_NAME, CameraInfo, Observation, Trajectory


class EpisodeBuffer:
    """Maintain the RR ``T+1 observations / T transitions`` invariant."""

    def __init__(self):
        self.clear()

    @property
    def started(self) -> bool:
        return bool(self._observations)

    @property
    def num_transitions(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._observations: list[Observation] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []

    def start(self, initial_observation: Observation) -> None:
        """Begin a fresh episode with its pre-action observation."""

        self.clear()
        self._observations.append(deepcopy(initial_observation))

    def append(self, action: np.ndarray, reward: float, next_observation: Observation) -> None:
        """Append one transition and the observation it produced."""

        if not self.started:
            raise RuntimeError("EpisodeBuffer.start() must be called before append().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,):
            raise ValueError(f"RR action must have shape (8,), received {action.shape}.")
        self._actions.append(action.copy())
        self._rewards.append(float(reward))
        self._observations.append(deepcopy(next_observation))
        self._assert_lengths()

    def build(self, *, success: bool, task: str, camera_info: CameraInfo) -> Trajectory:
        """Build a pickle-ready payload without exposing mutable buffer storage."""

        if not self.started or not self._actions:
            raise RuntimeError("Cannot build an empty episode.")
        self._assert_lengths()
        return {
            "observations": deepcopy(self._observations),
            "actions": [action.tolist() for action in self._actions],
            "rewards": list(self._rewards),
            "camera_info": deepcopy(camera_info),
            "success": bool(success),
            "task": str(task),
            "action_type": "delta",
            "env": ENV_NAME,
        }

    def _assert_lengths(self) -> None:
        if len(self._actions) != len(self._rewards) or len(self._observations) != len(self._actions) + 1:
            raise RuntimeError(
                "Episode buffer invariant violated: "
                f"obs={len(self._observations)}, actions={len(self._actions)}, rewards={len(self._rewards)}."
            )

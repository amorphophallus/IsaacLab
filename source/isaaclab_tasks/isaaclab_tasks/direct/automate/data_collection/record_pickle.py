# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level facade for recording one RR-compatible AutoMate episode."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .action_adapter import ActionAdapter
from .buffer import EpisodeBuffer
from .schema import CameraInfo, Trajectory
from .state_adapter import StateAdapter
from .validator import validate_trajectory


class PickleRecorder:
    """Coordinate action preparation, state capture, buffering, and validation."""

    def __init__(
        self,
        *,
        state_adapter: StateAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
        env_idx: int = 0,
    ):
        self.state_adapter = state_adapter or StateAdapter()
        self.action_adapter = action_adapter or ActionAdapter(gripper_command=1.0)
        self.env_idx = env_idx
        self.buffer = EpisodeBuffer()
        self._camera_info: CameraInfo | None = None

    def start_episode(self, env: Any) -> None:
        """Capture calibration and the initial pre-action observation."""

        self._camera_info = self.state_adapter.camera_info(env, self.env_idx)
        self.buffer.start(self.state_adapter.capture(env, self.env_idx))

    def prepare_action(self, env: Any, clipped_policy_action: torch.Tensor) -> np.ndarray:
        """Adapt an action while the environment still contains its pre-step state."""

        if not self.buffer.started:
            raise RuntimeError("start_episode() must be called before prepare_action().")
        return self.action_adapter.adapt(env, clipped_policy_action, self.env_idx)

    def record_step(self, env: Any, action: np.ndarray, succeeded: bool) -> None:
        """Capture the post-step observation and append a sparse transition."""

        reward = 1.0 if succeeded else 0.0
        self.buffer.append(action, reward, self.state_adapter.capture(env, self.env_idx))

    def finish_episode(self, *, success: bool, task: str) -> Trajectory:
        """Build and validate the finished trajectory."""

        if self._camera_info is None:
            raise RuntimeError("start_episode() must be called before finish_episode().")
        trajectory = self.buffer.build(success=success, task=task, camera_info=self._camera_info)
        validate_trajectory(trajectory)
        return trajectory

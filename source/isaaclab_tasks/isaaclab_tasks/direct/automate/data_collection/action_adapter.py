# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert AutoMate policy actions into RR delta-pose actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .transforms import (
    pose_world_to_base,
    relative_quat_wxyz,
    upright_target_quat_wxyz,
    wxyz_to_xyzw,
)


class ActionAdapter:
    """Adapt one clipped AutoMate policy action before the environment step.

    The adapter predicts the first controller target of the upcoming policy
    step. It mirrors AutoMate's EMA, physical scaling, workspace clipping, and
    upright-orientation projection. The same already-clipped policy tensor must
    subsequently be passed to the RL-Games environment wrapper.
    """

    def __init__(self, gripper_command: float = 1.0):
        if not -1.0 <= gripper_command <= 1.0:
            raise ValueError(f"gripper_command must be in [-1, 1], received {gripper_command}.")
        self.gripper_command = float(gripper_command)

    def adapt(self, env: Any, clipped_policy_action: torch.Tensor, env_idx: int = 0) -> np.ndarray:
        """Return ``[dpos, dquat_xyzw, gripper]`` for one transition."""

        if env_idx < 0 or env_idx >= env.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {env.num_envs}).")

        action = torch.as_tensor(clipped_policy_action, dtype=torch.float32, device=env.device)
        if action.ndim == 2:
            if action.shape[0] <= env_idx:
                raise IndexError(f"Action batch has {action.shape[0]} rows; cannot read env_idx={env_idx}.")
            action = action[env_idx]
        if action.shape != (6,):
            raise ValueError(f"AutoMate policy action must have shape (6,) or (N, 6), received {tuple(action.shape)}.")
        if not torch.isfinite(action).all():
            raise ValueError("AutoMate policy action contains non-finite values.")

        previous_action = env.actions[env_idx].detach().clone().to(dtype=torch.float32)
        ema_factor = float(env.cfg.ctrl.ema_factor)
        smoothed_action = ema_factor * action + (1.0 - ema_factor) * previous_action

        current_pos_env = env.fingertip_midpoint_pos[env_idx].detach().clone()
        current_quat_w = env.fingertip_midpoint_quat[env_idx].detach().clone()

        pos_delta_env = smoothed_action[:3] * env.pos_threshold[env_idx]
        target_pos_env = current_pos_env + pos_delta_env
        target_from_fixed = target_pos_env - env.fixed_pos_action_frame[env_idx]
        position_bounds = env.cfg.ctrl.pos_action_bounds
        target_from_fixed = torch.clamp(
            target_from_fixed,
            min=-float(position_bounds[0]),
            max=float(position_bounds[1]),
        )
        target_pos_env = env.fixed_pos_action_frame[env_idx] + target_from_fixed

        rot_axis_angle = smoothed_action[3:6].clone()
        if env.cfg_task.unidirectional_rot:
            rot_axis_angle[2] = -(rot_axis_angle[2] + 1.0) * 0.5
        rot_axis_angle = rot_axis_angle * env.rot_threshold[env_idx]
        target_quat_w = upright_target_quat_wxyz(rot_axis_angle, current_quat_w)

        env_origin_w = env.scene.env_origins[env_idx]
        target_pos_w = target_pos_env + env_origin_w
        base_pos_w = env._robot.data.root_pos_w[env_idx]
        base_quat_w = env._robot.data.root_quat_w[env_idx]
        current_pos_w = env._robot.data.body_pos_w[env_idx, env.fingertip_body_idx]
        current_quat_w = env._robot.data.body_quat_w[env_idx, env.fingertip_body_idx]

        current_pos_b, current_quat_b = pose_world_to_base(
            base_pos_w,
            base_quat_w,
            current_pos_w,
            current_quat_w,
        )
        target_pos_b, target_quat_b = pose_world_to_base(
            base_pos_w,
            base_quat_w,
            target_pos_w,
            target_quat_w,
        )
        delta_pos_b = target_pos_b - current_pos_b
        delta_quat_b = relative_quat_wxyz(current_quat_b, target_quat_b)
        delta_quat_xyzw = wxyz_to_xyzw(delta_quat_b)

        rr_action = torch.cat(
            (
                delta_pos_b,
                delta_quat_xyzw,
                torch.tensor([self.gripper_command], dtype=torch.float32, device=env.device),
            )
        )
        return rr_action.detach().cpu().numpy().astype(np.float32, copy=True)

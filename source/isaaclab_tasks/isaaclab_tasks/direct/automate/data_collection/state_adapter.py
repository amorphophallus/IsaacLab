# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract robust-rearrangement observations from an AutoMate environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .schema import (
    STORED_IMAGE_HEIGHT,
    STORED_IMAGE_WIDTH,
    CameraInfo,
    Observation,
    RobotState,
    observation_placeholders,
)
from .transforms import pose_world_to_base, rr_camera_to_base_matrix, twist_world_to_base, wxyz_to_xyzw


def _float32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=True)


class StateAdapter:
    """Capture strict RR robot, object, image, and camera state."""

    def capture(self, env: Any, env_idx: int = 0) -> Observation:
        """Capture one observation after reset or step."""

        self._validate_env_idx(env, env_idx)
        robot_data = env._robot.data
        base_pos_w = robot_data.root_pos_w[env_idx]
        base_quat_w = robot_data.root_quat_w[env_idx]
        ee_pos_w = robot_data.body_pos_w[env_idx, env.fingertip_body_idx]
        ee_quat_w = robot_data.body_quat_w[env_idx, env.fingertip_body_idx]

        ee_pos_b, ee_quat_b = pose_world_to_base(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
        base_lin_vel_w = self._root_velocity(robot_data, "root_lin_vel_w", base_pos_w)
        base_ang_vel_w = self._root_velocity(robot_data, "root_ang_vel_w", base_pos_w)
        ee_lin_vel_b, ee_ang_vel_b = twist_world_to_base(
            base_pos_w,
            base_quat_w,
            base_lin_vel_w[env_idx],
            base_ang_vel_w[env_idx],
            ee_pos_w,
            robot_data.body_lin_vel_w[env_idx, env.fingertip_body_idx],
            robot_data.body_ang_vel_w[env_idx, env.fingertip_body_idx],
        )

        joint_pos = robot_data.joint_pos[env_idx]
        joint_vel = robot_data.joint_vel[env_idx]
        joint_torques = robot_data.applied_torque
        if joint_torques is None:
            joint_torque = torch.zeros_like(joint_pos)
        else:
            joint_torque = joint_torques[env_idx]
        if joint_pos.shape[-1] != 9 or joint_vel.shape[-1] != 9 or joint_torque.shape[-1] != 9:
            raise ValueError(
                "RR AutoMate collection requires exactly 9 Franka joints "
                f"(7 arm + 2 fingers); received {joint_pos.shape[-1]}/{joint_vel.shape[-1]}/{joint_torque.shape[-1]}."
            )

        finger_1 = joint_pos[7:8]
        finger_2 = joint_pos[8:9]
        ee_quat_xyzw = wxyz_to_xyzw(ee_quat_b)
        robot_state: RobotState = {
            "ee_pos": _float32_numpy(ee_pos_b),
            "ee_quat": _float32_numpy(ee_quat_xyzw),
            "ee_pos_vel": _float32_numpy(ee_lin_vel_b),
            "ee_ori_vel": _float32_numpy(ee_ang_vel_b),
            "gripper_width": _float32_numpy(finger_1 + finger_2),
            "joint_positions": _float32_numpy(joint_pos[:7]),
            "joint_velocities": _float32_numpy(joint_vel[:7]),
            "joint_torques": _float32_numpy(joint_torque),
            "gripper_finger_1_pos": _float32_numpy(finger_1),
            "gripper_finger_2_pos": _float32_numpy(finger_2),
            "ee_pos_sim": _float32_numpy(ee_pos_b),
            "ee_quat_sim": _float32_numpy(ee_quat_xyzw),
        }

        held_pose = self._object_pose_in_base(env._held_asset.data, env_idx, base_pos_w, base_quat_w)
        fixed_pose = self._object_pose_in_base(env._fixed_asset.data, env_idx, base_pos_w, base_quat_w)
        parts_poses = np.concatenate((held_pose, fixed_pose), axis=0).astype(np.float32, copy=False)

        camera_obs = env.get_camera_observations(env_idx)
        color_image1 = self._rgb_image(camera_obs["color_image1"], "color_image1")
        color_image2 = self._rgb_image(camera_obs["color_image2"], "color_image2")
        depth_image1 = self._depth_image(camera_obs["depth_image1"], "depth_image1")
        depth_image2 = self._depth_image(camera_obs["depth_image2"], "depth_image2")

        observation: Observation = {
            "robot_state": robot_state,
            "color_image1": color_image1,
            "color_image2": color_image2,
            "depth_image1": depth_image1,
            "depth_image2": depth_image2,
            "parts_poses": parts_poses,
            **observation_placeholders(),
        }
        return observation

    def camera_info(self, env: Any, env_idx: int = 0) -> CameraInfo:
        """Capture static front-camera calibration in the robot base frame."""

        self._validate_env_idx(env, env_idx)
        robot_data = env._robot.data
        camera_data = env._front_camera.data
        base_pos_w = robot_data.root_pos_w[env_idx]
        base_quat_w = robot_data.root_quat_w[env_idx]
        camera_to_base = rr_camera_to_base_matrix(
            base_pos_w,
            base_quat_w,
            camera_data.pos_w[env_idx],
            camera_data.quat_w_world[env_idx],
        )
        base_to_camera = torch.linalg.inv(camera_to_base)
        source_height, source_width = camera_data.image_shape
        crop_top, crop_left = self._center_crop_offsets(source_height, source_width, "front camera")
        intrinsics = camera_data.intrinsic_matrices[env_idx].detach().clone()
        intrinsics[0, 2] -= crop_left
        intrinsics[1, 2] -= crop_top
        return {
            "front_camera": {
                "image_size": np.array([STORED_IMAGE_WIDTH, STORED_IMAGE_HEIGHT], dtype=np.int32),
                "intrinsics": _float32_numpy(intrinsics),
                "camera_to_sim_local": _float32_numpy(camera_to_base),
                "sim_local_to_camera": _float32_numpy(base_to_camera),
            }
        }

    @staticmethod
    def _validate_env_idx(env: Any, env_idx: int) -> None:
        if not isinstance(env_idx, int):
            raise TypeError(f"env_idx must be int, received {type(env_idx).__name__}.")
        if env_idx < 0 or env_idx >= env.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {env.num_envs}).")

    @staticmethod
    def _root_velocity(robot_data: Any, name: str, reference: torch.Tensor) -> torch.Tensor:
        value = getattr(robot_data, name, None)
        if value is None:
            return torch.zeros((robot_data.root_pos_w.shape[0], 3), dtype=reference.dtype, device=reference.device)
        return value

    @staticmethod
    def _object_pose_in_base(
        object_data: Any,
        env_idx: int,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
    ) -> np.ndarray:
        pos_b, quat_b = pose_world_to_base(
            base_pos_w,
            base_quat_w,
            object_data.root_pos_w[env_idx],
            object_data.root_quat_w[env_idx],
        )
        return _float32_numpy(torch.cat((pos_b, wxyz_to_xyzw(quat_b)), dim=-1))

    @staticmethod
    def _rgb_image(image: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"{name} must have shape (H, W, 3/4), received {image.shape}.")
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = StateAdapter._center_crop(image, name)
        return np.ascontiguousarray(image, dtype=np.uint8)

    @staticmethod
    def _depth_image(image: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"{name} must have shape (H, W), received {image.shape}.")
        image = np.nan_to_num(image, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        image = StateAdapter._center_crop(image, name)
        return np.ascontiguousarray(image, dtype=np.float32)

    @staticmethod
    def _center_crop(image: np.ndarray, name: str) -> np.ndarray:
        crop_top, crop_left = StateAdapter._center_crop_offsets(image.shape[0], image.shape[1], name)
        return image[
            crop_top : crop_top + STORED_IMAGE_HEIGHT,
            crop_left : crop_left + STORED_IMAGE_WIDTH,
            ...,
        ]

    @staticmethod
    def _center_crop_offsets(height: int, width: int, name: str) -> tuple[int, int]:
        if height < STORED_IMAGE_HEIGHT or width < STORED_IMAGE_WIDTH:
            raise ValueError(
                f"{name} is too small for a {STORED_IMAGE_WIDTH}x{STORED_IMAGE_HEIGHT} center crop: "
                f"received width={width}, height={height}."
            )
        return (height - STORED_IMAGE_HEIGHT) // 2, (width - STORED_IMAGE_WIDTH) // 2

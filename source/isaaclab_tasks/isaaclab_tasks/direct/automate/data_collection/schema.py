# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed schema for robust-rearrangement-compatible AutoMate rollouts."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class RobotState(TypedDict):
    """Robot state fields emitted by robust-rearrangement simulator rollouts."""

    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_pos_vel: np.ndarray
    ee_ori_vel: np.ndarray
    gripper_width: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_finger_1_pos: np.ndarray
    gripper_finger_2_pos: np.ndarray
    ee_pos_sim: np.ndarray
    ee_quat_sim: np.ndarray


class Observation(TypedDict):
    """One observation in a raw robust-rearrangement rollout."""

    robot_state: RobotState
    color_image1: np.ndarray
    color_image2: np.ndarray
    depth_image1: np.ndarray
    depth_image2: np.ndarray
    parts_poses: np.ndarray
    point_cloud: None
    skill: None
    guidance_point: None
    guidance_point_clean: None
    guidance_pose: None
    guidance_pose_clean: None
    guidance_gripper_width: None
    guidance_point_2d: None
    grasp_annotation_2d: None


class CameraCalibration(TypedDict):
    """RR camera calibration, with ``sim_local`` interpreted as robot base."""

    image_size: np.ndarray
    intrinsics: np.ndarray
    camera_to_sim_local: np.ndarray
    sim_local_to_camera: np.ndarray


class CameraInfo(TypedDict):
    front_camera: CameraCalibration


class Trajectory(TypedDict):
    """Top-level raw rollout payload."""

    observations: list[Observation]
    actions: list[list[float]]
    rewards: list[float]
    camera_info: CameraInfo
    success: bool
    task: str
    action_type: str
    env: str


TRAJECTORY_KEYS = frozenset(Trajectory.__required_keys__)
OBSERVATION_KEYS = frozenset(Observation.__required_keys__)
ROBOT_STATE_KEYS = frozenset(RobotState.__required_keys__)
CAMERA_INFO_KEYS = frozenset(CameraInfo.__required_keys__)
CAMERA_CALIBRATION_KEYS = frozenset(CameraCalibration.__required_keys__)

ENV_NAME = "AutoMate"
STORED_IMAGE_HEIGHT = 224
STORED_IMAGE_WIDTH = 224

ROBOT_STATE_SHAPES: dict[str, tuple[int, ...]] = {
    "ee_pos": (3,),
    "ee_quat": (4,),
    "ee_pos_vel": (3,),
    "ee_ori_vel": (3,),
    "gripper_width": (1,),
    "joint_positions": (7,),
    "joint_velocities": (7,),
    "joint_torques": (9,),
    "gripper_finger_1_pos": (1,),
    "gripper_finger_2_pos": (1,),
    "ee_pos_sim": (3,),
    "ee_quat_sim": (4,),
}

OPTIONAL_OBSERVATION_FIELDS = frozenset(
    {
        "point_cloud",
        "skill",
        "guidance_point",
        "guidance_point_clean",
        "guidance_pose",
        "guidance_pose_clean",
        "guidance_gripper_width",
        "guidance_point_2d",
        "grasp_annotation_2d",
    }
)


def observation_placeholders() -> dict[str, Any]:
    """Return fresh RR optional-field placeholders for one observation."""

    return {key: None for key in OPTIONAL_OBSERVATION_FIELDS}

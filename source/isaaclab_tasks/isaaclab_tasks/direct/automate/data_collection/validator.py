# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict validation for RR-compatible raw rollout pickles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from .schema import (
    ANNOTATION_SOURCE,
    CAMERA_CALIBRATION_KEYS,
    CAMERA_INFO_KEYS,
    ENV_NAME,
    IMAGE_ANNOTATION_MODE,
    OBSERVATION_KEYS,
    ROBOT_STATE_KEYS,
    ROBOT_STATE_SHAPES,
    STORED_IMAGE_HEIGHT,
    STORED_IMAGE_WIDTH,
    TRAJECTORY_KEYS,
    Trajectory,
)


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory does not satisfy the RR contract."""


_TASK_PATTERN = re.compile(r"automate_insertion_\d{5,}")


def _require_exact_keys(value: Any, expected: frozenset[str], path: str) -> None:
    if not isinstance(value, Mapping):
        raise TrajectoryValidationError(f"{path} must be a mapping, received {type(value).__name__}.")
    actual = frozenset(value.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TrajectoryValidationError(f"{path} keys mismatch; missing={missing}, extra={extra}.")


def _require_array(
    value: Any,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TrajectoryValidationError(f"{path} must be numpy.ndarray, received {type(value).__name__}.")
    if value.shape != shape:
        raise TrajectoryValidationError(f"{path} must have shape {shape}, received {value.shape}.")
    if value.dtype != dtype:
        raise TrajectoryValidationError(f"{path} must have dtype {dtype}, received {value.dtype}.")
    if finite and not np.isfinite(value).all():
        raise TrajectoryValidationError(f"{path} contains non-finite values.")
    return value


def _require_unit_xyzw(quat: np.ndarray, path: str, tolerance: float = 1.0e-4) -> None:
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=tolerance, rtol=0.0):
        raise TrajectoryValidationError(f"{path} must be unit length; norm={norm:.8f}.")
    if quat[-1] < -tolerance:
        raise TrajectoryValidationError(f"{path} must use canonical xyzw sign with w >= 0.")


def validate_trajectory(trajectory: Trajectory) -> None:
    """Validate ``trajectory`` or raise :class:`TrajectoryValidationError`."""

    _require_exact_keys(trajectory, TRAJECTORY_KEYS, "trajectory")
    observations = trajectory["observations"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    if not isinstance(observations, list) or not isinstance(actions, list) or not isinstance(rewards, list):
        raise TrajectoryValidationError("observations, actions, and rewards must be Python lists.")
    if not actions:
        raise TrajectoryValidationError("A trajectory must contain at least one transition.")
    if len(observations) != len(actions) + 1 or len(rewards) != len(actions):
        raise TrajectoryValidationError(
            f"Expected obs=actions+1 and rewards=actions; got {len(observations)}/{len(actions)}/{len(rewards)}."
        )

    if type(trajectory["success"]) is not bool:
        raise TrajectoryValidationError("success must be a Python bool.")
    if trajectory["env"] != ENV_NAME:
        raise TrajectoryValidationError(f"env must be {ENV_NAME!r}, received {trajectory['env']!r}.")
    if trajectory["annotation_source"] != ANNOTATION_SOURCE:
        raise TrajectoryValidationError(
            f"annotation_source must be {ANNOTATION_SOURCE!r}, received {trajectory['annotation_source']!r}."
        )
    if trajectory["image_annotation_mode"] != IMAGE_ANNOTATION_MODE:
        raise TrajectoryValidationError(
            "image_annotation_mode must be "
            f"{IMAGE_ANNOTATION_MODE!r}, received {trajectory['image_annotation_mode']!r}."
        )
    if not isinstance(trajectory["task"], str) or _TASK_PATTERN.fullmatch(trajectory["task"]) is None:
        raise TrajectoryValidationError("task must match 'automate_insertion_<assembly_id>'.")
    if trajectory["action_type"] != "delta":
        raise TrajectoryValidationError(f"action_type must be 'delta', received {trajectory['action_type']!r}.")

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.shape != (len(actions), 8):
        raise TrajectoryValidationError(f"actions must have shape (T, 8), received {action_array.shape}.")
    if not np.isfinite(action_array).all():
        raise TrajectoryValidationError("actions contain non-finite values.")
    for index, quat in enumerate(action_array[:, 3:7]):
        _require_unit_xyzw(quat, f"actions[{index}][3:7]")
    if np.any(action_array[:, 7] < -1.0) or np.any(action_array[:, 7] > 1.0):
        raise TrajectoryValidationError("Absolute gripper commands must stay within [-1, 1].")

    reward_array = np.asarray(rewards, dtype=np.float32)
    if reward_array.shape != (len(actions),) or not np.isfinite(reward_array).all():
        raise TrajectoryValidationError("rewards must be a finite scalar list of length T.")
    if not np.all(np.logical_or(reward_array == 0.0, reward_array == 1.0)):
        raise TrajectoryValidationError("RR collection rewards must be binary 0/1 values.")
    if trajectory["success"]:
        if reward_array[-1] != 1.0 or np.any(reward_array[:-1] != 0.0):
            raise TrajectoryValidationError(
                "A successful first-hit trajectory must have only its final reward equal to 1."
            )
    elif np.any(reward_array != 0.0):
        raise TrajectoryValidationError("A failed trajectory must contain only zero rewards.")

    for index, observation in enumerate(observations):
        _validate_observation(observation, index)
    _validate_camera_info(trajectory["camera_info"], observations[0])
    _validate_guidance_projections(
        observations, trajectory["camera_info"]["front_camera"]
    )


def _validate_observation(observation: Any, index: int) -> None:
    path = f"observations[{index}]"
    _require_exact_keys(observation, OBSERVATION_KEYS, path)
    for field in (
        "point_cloud",
        "guidance_pose",
        "guidance_pose_clean",
        "guidance_gripper_width",
        "grasp_annotation_2d",
    ):
        if observation[field] is not None:
            raise TrajectoryValidationError(f"{path}.{field} must be None for AutoMate collection.")
    if observation["skill"] != "insert":
        raise TrajectoryValidationError(f"{path}.skill must be 'insert'.")
    guidance_point = _require_array(
        observation["guidance_point"],
        path=f"{path}.guidance_point",
        shape=(3,),
        dtype=np.dtype(np.float32),
    )
    guidance_point_clean = _require_array(
        observation["guidance_point_clean"],
        path=f"{path}.guidance_point_clean",
        shape=(3,),
        dtype=np.dtype(np.float32),
    )
    if not np.array_equal(guidance_point, guidance_point_clean):
        raise TrajectoryValidationError(
            f"{path}: noiseless AutoMate guidance point and clean copy must match."
        )
    points_2d = observation["guidance_point_2d"]
    _require_exact_keys(
        points_2d,
        frozenset({"color_image1", "color_image2"}),
        f"{path}.guidance_point_2d",
    )
    if points_2d["color_image1"] is not None:
        raise TrajectoryValidationError(
            f"{path}.guidance_point_2d.color_image1 must be None without wrist calibration."
        )
    front_point = _require_array(
        points_2d["color_image2"],
        path=f"{path}.guidance_point_2d.color_image2",
        shape=(2,),
        dtype=np.dtype(np.float32),
    )
    if np.any(front_point < 0.0) or np.any(front_point >= STORED_IMAGE_WIDTH):
        raise TrajectoryValidationError(
            f"{path}.guidance_point_2d.color_image2 lies outside the stored image."
        )

    robot_state = observation["robot_state"]
    _require_exact_keys(robot_state, ROBOT_STATE_KEYS, f"{path}.robot_state")
    for name, shape in ROBOT_STATE_SHAPES.items():
        _require_array(
            robot_state[name],
            path=f"{path}.robot_state.{name}",
            shape=shape,
            dtype=np.dtype(np.float32),
        )
    _require_unit_xyzw(robot_state["ee_quat"], f"{path}.robot_state.ee_quat")
    _require_unit_xyzw(robot_state["ee_quat_sim"], f"{path}.robot_state.ee_quat_sim")
    if not np.array_equal(robot_state["ee_pos"], robot_state["ee_pos_sim"]):
        raise TrajectoryValidationError(f"{path}: ee_pos and ee_pos_sim must be identical base-frame values.")
    if not np.array_equal(robot_state["ee_quat"], robot_state["ee_quat_sim"]):
        raise TrajectoryValidationError(f"{path}: ee_quat and ee_quat_sim must be identical base-frame values.")
    expected_width = robot_state["gripper_finger_1_pos"] + robot_state["gripper_finger_2_pos"]
    if not np.allclose(robot_state["gripper_width"], expected_width, atol=1.0e-6, rtol=0.0):
        raise TrajectoryValidationError(f"{path}.robot_state.gripper_width does not equal the two finger positions.")

    _require_array(
        observation["color_image1"],
        path=f"{path}.color_image1",
        shape=(STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH, 3),
        dtype=np.dtype(np.uint8),
        finite=False,
    )
    _require_array(
        observation["color_image2"],
        path=f"{path}.color_image2",
        shape=(STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH, 3),
        dtype=np.dtype(np.uint8),
        finite=False,
    )
    _require_array(
        observation["depth_image1"],
        path=f"{path}.depth_image1",
        shape=(STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH),
        dtype=np.dtype(np.float32),
    )
    _require_array(
        observation["depth_image2"],
        path=f"{path}.depth_image2",
        shape=(STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH),
        dtype=np.dtype(np.float32),
    )
    parts = _require_array(
        observation["parts_poses"],
        path=f"{path}.parts_poses",
        shape=(14,),
        dtype=np.dtype(np.float32),
    )
    _require_unit_xyzw(parts[3:7], f"{path}.parts_poses[held].quat")
    _require_unit_xyzw(parts[10:14], f"{path}.parts_poses[fixed].quat")


def _validate_camera_info(camera_info: Any, first_observation: Any) -> None:
    _require_exact_keys(camera_info, CAMERA_INFO_KEYS, "camera_info")
    calibration = camera_info["front_camera"]
    _require_exact_keys(calibration, CAMERA_CALIBRATION_KEYS, "camera_info.front_camera")
    image_size = _require_array(
        calibration["image_size"],
        path="camera_info.front_camera.image_size",
        shape=(2,),
        dtype=np.dtype(np.int32),
    )
    height, width = first_observation["color_image2"].shape[:2]
    if not np.array_equal(image_size, np.array([width, height], dtype=np.int32)):
        raise TrajectoryValidationError("front_camera.image_size does not match color_image2.")
    _require_array(
        calibration["intrinsics"],
        path="camera_info.front_camera.intrinsics",
        shape=(3, 3),
        dtype=np.dtype(np.float32),
    )
    camera_to_base = _require_array(
        calibration["camera_to_sim_local"],
        path="camera_info.front_camera.camera_to_sim_local",
        shape=(4, 4),
        dtype=np.dtype(np.float32),
    )
    base_to_camera = _require_array(
        calibration["sim_local_to_camera"],
        path="camera_info.front_camera.sim_local_to_camera",
        shape=(4, 4),
        dtype=np.dtype(np.float32),
    )
    if not np.allclose(camera_to_base @ base_to_camera, np.eye(4), atol=1.0e-4, rtol=0.0):
        raise TrajectoryValidationError("front camera extrinsic matrices are not mutual inverses.")


def _validate_guidance_projections(observations: list[Any], calibration: Any) -> None:
    """Verify that every stored front pixel is the calibrated 3D target."""

    intrinsics = calibration["intrinsics"]
    base_to_camera = calibration["sim_local_to_camera"]
    for index, observation in enumerate(observations):
        point_h = np.concatenate(
            (observation["guidance_point"], np.ones(1, dtype=np.float32))
        )
        point_camera = base_to_camera @ point_h
        if point_camera[2] <= 1.0e-8:
            raise TrajectoryValidationError(
                f"observations[{index}].guidance_point is behind the front camera."
            )
        point_camera = point_camera[:3].copy()
        point_camera[1] *= -1.0
        pixel_h = intrinsics @ point_camera
        expected = pixel_h[:2] / pixel_h[2]
        actual = observation["guidance_point_2d"]["color_image2"]
        if not np.allclose(actual, expected, atol=1.0e-4, rtol=0.0):
            raise TrajectoryValidationError(
                f"observations[{index}].guidance_point_2d.color_image2 does not "
                "match the calibrated 3D guidance point."
            )

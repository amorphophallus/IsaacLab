# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import lzma
import pickle
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# These are pure unit tests and intentionally do not launch Kit. Import the
# package directly so importing isaaclab_tasks does not require pxr/SimulationApp.
AUTOMATE_SOURCE_DIR = Path(__file__).resolve().parents[3] / "isaaclab_tasks" / "direct" / "automate"
sys.path.insert(0, str(AUTOMATE_SOURCE_DIR))

from data_collection import (  # noqa: E402
    ActionAdapter,
    EpisodeBuffer,
    PickleRecorder,
    StateAdapter,
    TrajectoryValidationError,
    validate_trajectory,
    write_trajectory,
)
from data_collection.schema import (  # noqa: E402
    ROBOT_STATE_KEYS,
    ROBOT_STATE_SHAPES,
    STORED_IMAGE_HEIGHT,
    STORED_IMAGE_WIDTH,
)
from data_collection.transforms import (  # noqa: E402
    canonicalize_quat_wxyz,
    pose_world_to_base,
    quat_apply_wxyz,
    quat_from_euler_xyz_wxyz,
    quat_multiply_wxyz,
    relative_quat_wxyz,
    rr_camera_to_base_matrix,
    twist_world_to_base,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _fake_env() -> SimpleNamespace:
    dtype = torch.float32
    root_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    ee_pos = torch.tensor([[1.09, 2.0, 3.3]], dtype=dtype)
    ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype)
    robot_data = SimpleNamespace(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=torch.zeros((1, 3), dtype=dtype),
        root_ang_vel_w=torch.zeros((1, 3), dtype=dtype),
        body_pos_w=ee_pos[:, None, :],
        body_quat_w=ee_quat[:, None, :],
        body_lin_vel_w=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=dtype),
        body_ang_vel_w=torch.tensor([[[0.4, 0.5, 0.6]]], dtype=dtype),
        joint_pos=torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, 0.02]], dtype=dtype),
        joint_vel=torch.tensor([[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]], dtype=dtype),
        applied_torque=torch.arange(9, dtype=dtype).reshape(1, 9),
    )
    held_data = SimpleNamespace(
        root_pos_w=torch.tensor([[1.1, 2.0, 3.0]], dtype=dtype),
        root_quat_w=root_quat.clone(),
    )
    fixed_data = SimpleNamespace(
        root_pos_w=torch.tensor([[1.2, 2.0, 3.0]], dtype=dtype),
        root_quat_w=root_quat.clone(),
    )
    camera_data = SimpleNamespace(
        pos_w=torch.tensor([[2.0, 2.0, 3.5]], dtype=dtype),
        quat_w_world=root_quat.clone(),
        intrinsic_matrices=torch.tensor([[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]]),
        image_shape=(480, 640),
    )
    color_image1 = np.zeros((480, 640, 4), dtype=np.uint8)
    color_image1[..., 0] = np.arange(640, dtype=np.uint16) % 256
    color_image1[..., 1] = (np.arange(480, dtype=np.uint16) % 256)[:, None]
    depth_image1 = (
        np.arange(480, dtype=np.float32)[:, None] * 1000.0 + np.arange(640, dtype=np.float32)[None, :]
    )
    camera_observation = {
        "color_image1": color_image1,
        "color_image2": np.ones((480, 640, 3), dtype=np.uint8),
        "depth_image1": depth_image1,
        "depth_image2": np.full((480, 640), 0.75, dtype=np.float32),
    }
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(ctrl=SimpleNamespace(ema_factor=0.2, pos_action_bounds=[0.1, 0.1, 0.1])),
        cfg_task=SimpleNamespace(unidirectional_rot=False),
        actions=torch.zeros((1, 6), dtype=dtype),
        pos_threshold=torch.full((1, 3), 0.1, dtype=dtype),
        rot_threshold=torch.full((1, 3), 0.01, dtype=dtype),
        fingertip_midpoint_pos=torch.tensor([[0.09, 0.0, 0.3]], dtype=dtype),
        fingertip_midpoint_quat=ee_quat.clone(),
        fixed_pos_action_frame=torch.tensor([[0.0, 0.0, 0.3]], dtype=dtype),
        fingertip_body_idx=0,
        scene=SimpleNamespace(env_origins=torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)),
        _robot=SimpleNamespace(data=robot_data),
        _held_asset=SimpleNamespace(data=held_data),
        _fixed_asset=SimpleNamespace(data=fixed_data),
        _front_camera=SimpleNamespace(data=camera_data),
    )
    env.get_camera_observations = lambda env_idx=0: deepcopy(camera_observation)
    return env


def test_quaternion_conversion_and_base_pose_round_trip():
    yaw = torch.tensor(np.pi / 2.0, dtype=torch.float32)
    base_quat = quat_from_euler_xyz_wxyz(torch.tensor(0.0), torch.tensor(0.0), yaw)
    local_pos = torch.tensor([0.2, -0.1, 0.3])
    base_pos = torch.tensor([1.0, 2.0, 3.0])
    world_pos = base_pos + quat_apply_wxyz(base_quat, local_pos)
    local_quat = quat_from_euler_xyz_wxyz(torch.tensor(0.1), torch.tensor(-0.2), torch.tensor(0.3))
    world_quat = quat_multiply_wxyz(base_quat, local_quat)

    recovered_pos, recovered_quat = pose_world_to_base(base_pos, base_quat, world_pos, world_quat)
    assert torch.allclose(recovered_pos, local_pos, atol=1.0e-6)
    assert torch.allclose(recovered_quat, canonicalize_quat_wxyz(local_quat), atol=1.0e-6)

    xyzw = wxyz_to_xyzw(recovered_quat)
    assert torch.allclose(xyzw_to_wxyz(xyzw), recovered_quat, atol=1.0e-6)
    assert xyzw[-1] >= 0.0
    assert torch.equal(
        canonicalize_quat_wxyz(torch.tensor([-1.0, 0.0, 0.0, 0.0])),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_camera_extrinsic_uses_rr_right_up_forward_axes():
    camera_to_base = rr_camera_to_base_matrix(
        base_pos_w=torch.tensor([1.0, 2.0, 3.0]),
        base_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        camera_pos_w=torch.tensor([2.0, 2.0, 3.5]),
        camera_quat_w_world=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(camera_to_base, expected, atol=1.0e-6)


def test_relative_quaternion_reconstructs_target_with_rr_right_multiplication():
    source = quat_from_euler_xyz_wxyz(torch.tensor(0.2), torch.tensor(-0.1), torch.tensor(0.3))
    target = quat_from_euler_xyz_wxyz(torch.tensor(-0.4), torch.tensor(0.25), torch.tensor(-0.2))
    delta = relative_quat_wxyz(source, target)
    reconstructed = canonicalize_quat_wxyz(quat_multiply_wxyz(source, delta))
    assert torch.allclose(reconstructed, canonicalize_quat_wxyz(target), atol=1.0e-6)


def test_twist_world_to_rotated_moving_base():
    base_quat = quat_from_euler_xyz_wxyz(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(np.pi / 2.0))
    base_pos = torch.zeros(3)
    base_linear = torch.tensor([1.0, 0.0, 0.0])
    base_angular = torch.tensor([0.0, 0.0, 1.0])
    body_pos = torch.tensor([1.0, 0.0, 0.0])
    body_linear = torch.tensor([1.0, 1.0, 0.0])
    body_angular = torch.tensor([0.0, 0.0, 2.0])

    linear_b, angular_b = twist_world_to_base(
        base_pos,
        base_quat,
        base_linear,
        base_angular,
        body_pos,
        body_linear,
        body_angular,
    )
    assert torch.allclose(linear_b, torch.zeros(3), atol=1.0e-6)
    assert torch.allclose(angular_b, torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)


def test_state_adapter_strict_rr_schema_and_parts_order():
    observation = StateAdapter().capture(_fake_env())
    assert set(observation["robot_state"]) == set(ROBOT_STATE_KEYS)
    for name, shape in ROBOT_STATE_SHAPES.items():
        assert observation["robot_state"][name].shape == shape
        assert observation["robot_state"][name].dtype == np.float32
    assert observation["color_image1"].shape == (STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH, 3)
    assert observation["color_image1"].dtype == np.uint8
    assert observation["color_image1"][0, 0, :2].tolist() == [208, 128]
    assert observation["color_image1"][-1, -1, :2].tolist() == [175, 95]
    assert observation["depth_image1"].shape == (STORED_IMAGE_HEIGHT, STORED_IMAGE_WIDTH)
    assert observation["depth_image1"].dtype == np.float32
    assert observation["depth_image1"][0, 0] == pytest.approx(128208.0)
    assert observation["depth_image1"][-1, -1] == pytest.approx(351431.0)
    assert np.allclose(observation["parts_poses"][:3], [0.1, 0.0, 0.0])
    assert np.allclose(observation["parts_poses"][7:10], [0.2, 0.0, 0.0])
    assert np.allclose(observation["robot_state"]["gripper_width"], [0.03])
    required_fields = {"robot_state", "color_image1", "color_image2", "depth_image1", "depth_image2", "parts_poses"}
    assert all(observation[key] is None for key in observation if key not in required_fields)

    front_camera = StateAdapter().camera_info(_fake_env())["front_camera"]
    assert front_camera["image_size"].tolist() == [STORED_IMAGE_WIDTH, STORED_IMAGE_HEIGHT]
    assert front_camera["intrinsics"] == pytest.approx(
        np.array([[600.0, 0.0, 112.0], [0.0, 600.0, 112.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )


def test_action_adapter_applies_ema_workspace_limit_and_rr_quaternion():
    env = _fake_env()
    policy_action = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    action = ActionAdapter().adapt(env, policy_action)

    # EMA produces +0.02 m, but the workspace permits only 0.01 m from x=0.09.
    assert action.shape == (8,)
    assert action.dtype == np.float32
    assert action[0] == pytest.approx(0.01, abs=1.0e-6)
    assert action[1:3] == pytest.approx([0.0, 0.0], abs=1.0e-6)
    assert np.linalg.norm(action[3:7]) == pytest.approx(1.0, abs=1.0e-6)
    assert action[6] >= 0.0
    assert action[7] == 1.0

    current = env.fingertip_midpoint_quat[0]
    delta = xyzw_to_wxyz(torch.from_numpy(action[3:7]))
    reconstructed = canonicalize_quat_wxyz(quat_multiply_wxyz(current, delta))
    expected = canonicalize_quat_wxyz(
        quat_from_euler_xyz_wxyz(
            torch.tensor(3.14159, dtype=torch.float32),
            torch.tensor(0.0),
            torch.tensor(0.002),
        )
    )
    assert torch.allclose(reconstructed, expected, atol=1.0e-5)


def test_action_adapter_zero_action_and_absolute_gripper():
    action = ActionAdapter().adapt(_fake_env(), torch.zeros((1, 6)))
    assert action[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-6)
    assert action[3:7] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=2.0e-6)
    assert action[7] == 1.0


def _valid_trajectory(success: bool = True):
    env = _fake_env()
    recorder = PickleRecorder()
    recorder.start_episode(env)
    action = recorder.prepare_action(env, torch.zeros((1, 6)))
    recorder.record_step(env, action, success)
    return recorder.finish_episode(success=success, task="automate_insertion_00110")


def test_episode_buffer_enforces_t_plus_one_observations():
    observation = StateAdapter().capture(_fake_env())
    buffer = EpisodeBuffer()
    buffer.start(observation)
    buffer.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32), 1.0, observation)
    trajectory = buffer.build(
        success=True,
        task="automate_insertion_00110",
        camera_info=StateAdapter().camera_info(_fake_env()),
    )
    assert len(trajectory["observations"]) == 2
    assert len(trajectory["actions"]) == len(trajectory["rewards"]) == 1
    validate_trajectory(trajectory)


def test_validator_rejects_extra_robot_state_key():
    trajectory = _valid_trajectory()
    trajectory["observations"][0]["robot_state"]["delta_pos"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(TrajectoryValidationError, match="keys mismatch"):
        validate_trajectory(trajectory)


def test_validator_rejects_non_automate_task_name():
    trajectory = _valid_trajectory()
    trajectory["task"] = "assembly_00110"
    with pytest.raises(TrajectoryValidationError, match="automate_insertion"):
        validate_trajectory(trajectory)


@pytest.mark.parametrize("compress", [False, True])
def test_writer_round_trip(tmp_path: Path, compress: bool):
    trajectory = _valid_trajectory()
    output_path = write_trajectory(
        trajectory,
        tmp_path / "raw" / "automate",
        attempt_idx=3,
        compress=compress,
        timestamp=datetime(2026, 8, 1, 12, 0, 0, 123456, tzinfo=timezone.utc),
    )
    assert output_path.parent.name == "success"
    assert output_path.name.endswith(".pkl.xz" if compress else ".pkl")
    opener = lzma.open if compress else open
    with opener(output_path, "rb") as stream:
        loaded = pickle.load(stream)
    validate_trajectory(loaded)
    with pytest.raises(FileExistsError):
        write_trajectory(
            trajectory,
            tmp_path / "raw" / "automate",
            attempt_idx=3,
            compress=compress,
            timestamp=datetime(2026, 8, 1, 12, 0, 0, 123456, tzinfo=timezone.utc),
        )

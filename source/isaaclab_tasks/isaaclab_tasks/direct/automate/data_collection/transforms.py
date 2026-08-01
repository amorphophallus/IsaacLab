# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate and quaternion conversions used by AutoMate data collection.

Isaac Lab represents quaternions as ``wxyz``. Raw robust-rearrangement
pickles use SciPy's ``xyzw`` convention. All quaternion reordering lives in
this module so adapters cannot silently mix the two conventions.
"""

from __future__ import annotations

import torch


def normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Normalize a ``wxyz`` quaternion tensor."""

    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if torch.any(norm <= eps):
        raise ValueError("Cannot normalize a zero-magnitude quaternion.")
    return quat / norm


def canonicalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize ``quat`` and choose the equivalent representation with ``w >= 0``."""

    quat = normalize_quat_wxyz(quat)
    sign = torch.where(quat[..., :1] < 0.0, -torch.ones_like(quat[..., :1]), torch.ones_like(quat[..., :1]))
    return quat * sign


def wxyz_to_xyzw(quat: torch.Tensor, canonicalize: bool = True) -> torch.Tensor:
    """Convert an Isaac Lab ``wxyz`` quaternion to RR ``xyzw``."""

    if canonicalize:
        quat = canonicalize_quat_wxyz(quat)
    return torch.cat((quat[..., 1:], quat[..., :1]), dim=-1)


def xyzw_to_wxyz(quat: torch.Tensor, canonicalize: bool = True) -> torch.Tensor:
    """Convert an RR ``xyzw`` quaternion to Isaac Lab ``wxyz``."""

    quat = torch.cat((quat[..., -1:], quat[..., :-1]), dim=-1)
    return canonicalize_quat_wxyz(quat) if canonicalize else quat


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a normalized ``wxyz`` quaternion."""

    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_inverse_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a ``wxyz`` quaternion."""

    quat = normalize_quat_wxyz(quat)
    return quat_conjugate_wxyz(quat)


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``wxyz`` quaternions."""

    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``vector`` by ``quat`` without constructing a matrix."""

    quat = normalize_quat_wxyz(quat)
    quat_vector = quat[..., 1:]
    uv = torch.cross(quat_vector, vector, dim=-1)
    uuv = torch.cross(quat_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def quat_apply_inverse_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame vector into the frame represented by ``quat``."""

    return quat_apply_wxyz(quat_inverse_wxyz(quat), vector)


def quat_from_axis_angle_wxyz(axis_angle: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Convert an axis-angle vector to a canonical ``wxyz`` quaternion."""

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(angle > eps, torch.sin(half_angle) / angle, 0.5 - angle.square() / 48.0)
    quat = torch.cat((torch.cos(half_angle), axis_angle * scale), dim=-1)
    return canonicalize_quat_wxyz(quat)


def quat_from_euler_xyz_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create ``wxyz`` quaternions from fixed-axis XYZ Euler angles."""

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    return canonicalize_quat_wxyz(
        torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=-1,
        )
    )


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Extract XYZ Euler yaw from a ``wxyz`` quaternion."""

    quat = normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def pose_world_to_base(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    pose_pos_w: torch.Tensor,
    pose_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a world-frame pose in the robot base frame."""

    base_quat_w = normalize_quat_wxyz(base_quat_w)
    base_inv = quat_inverse_wxyz(base_quat_w)
    pos_b = quat_apply_wxyz(base_inv, pose_pos_w - base_pos_w)
    quat_b = quat_multiply_wxyz(base_inv, normalize_quat_wxyz(pose_quat_w))
    return pos_b, canonicalize_quat_wxyz(quat_b)


def twist_world_to_base(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    base_lin_vel_w: torch.Tensor,
    base_ang_vel_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a body's twist relative to a possibly moving base in base axes."""

    position_offset_w = body_pos_w - base_pos_w
    relative_lin_vel_w = body_lin_vel_w - base_lin_vel_w - torch.cross(base_ang_vel_w, position_offset_w, dim=-1)
    relative_ang_vel_w = body_ang_vel_w - base_ang_vel_w
    return (
        quat_apply_inverse_wxyz(base_quat_w, relative_lin_vel_w),
        quat_apply_inverse_wxyz(base_quat_w, relative_ang_vel_w),
    )


def relative_quat_wxyz(source_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Return the right-multiplicative RR delta from ``source`` to ``target``."""

    delta = quat_multiply_wxyz(quat_inverse_wxyz(source_quat), normalize_quat_wxyz(target_quat))
    return canonicalize_quat_wxyz(delta)


def quat_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``wxyz`` quaternions to rotation matrices."""

    quat = normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y.square() + z.square()),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x.square() + z.square()),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def rr_camera_to_base_matrix(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_world: torch.Tensor,
) -> torch.Tensor:
    """Build RR's camera-to-local matrix, using the robot base as local.

    Isaac CameraData's ``world`` convention uses +X forward and +Z up. RR's
    calibration matrix stores columns as camera right, up, and forward.
    """

    camera_pos_b, _ = pose_world_to_base(base_pos_w, base_quat_w, camera_pos_w, camera_quat_w_world)
    world_from_camera = quat_to_matrix_wxyz(camera_quat_w_world)
    world_right = -world_from_camera[..., :, 1]
    world_up = world_from_camera[..., :, 2]
    world_forward = world_from_camera[..., :, 0]
    world_from_rr_camera = torch.stack((world_right, world_up, world_forward), dim=-1)
    base_from_world = quat_to_matrix_wxyz(quat_inverse_wxyz(base_quat_w))
    base_from_rr_camera = base_from_world @ world_from_rr_camera

    matrix = torch.eye(4, dtype=camera_pos_b.dtype, device=camera_pos_b.device)
    matrix[:3, :3] = base_from_rr_camera
    matrix[:3, 3] = camera_pos_b
    return matrix


def upright_target_quat_wxyz(delta_axis_angle: torch.Tensor, current_quat_w: torch.Tensor) -> torch.Tensor:
    """Mirror AutoMate's rotation target and upright roll/pitch projection."""

    delta_quat = quat_from_axis_angle_wxyz(delta_axis_angle)
    unconstrained_target = quat_multiply_wxyz(delta_quat, current_quat_w)
    yaw = yaw_from_quat_wxyz(unconstrained_target)
    # Keep this literal aligned with AssemblyEnv._apply_action().
    pi = torch.full_like(yaw, 3.14159)
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz_wxyz(pi, zero, yaw)

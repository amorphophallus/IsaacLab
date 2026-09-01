# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve official Isaac assets for offline AutoMate collection."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


LOCAL_ISAAC_ASSET_ROOT_ENV = "RR_ISAAC_ASSET_ROOT"


def resolve_isaac_asset_path(relative_path: str, *, default_root: str) -> str:
    """Resolve an Isaac-relative asset path, optionally from a verified local root.

    ``RR_ISAAC_ASSET_ROOT`` must point at the local ``Isaac`` directory. When it
    is unset, this preserves Isaac Lab's configured cloud/Nucleus behavior.
    """
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a safe Isaac-relative asset path, got: {relative_path!r}")

    local_root = os.environ.get(LOCAL_ISAAC_ASSET_ROOT_ENV)
    if not local_root:
        return f"{default_root.rstrip('/')}/{relative.as_posix()}"

    root = Path(local_root).expanduser().resolve()
    asset_path = root.joinpath(*relative.parts).resolve()
    try:
        asset_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Asset path escapes {LOCAL_ISAAC_ASSET_ROOT_ENV}: {relative_path!r}") from exc
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Missing official Isaac asset under {LOCAL_ISAAC_ASSET_ROOT_ENV}={root}: {relative.as_posix()}"
        )
    return str(asset_path)


def resolve_isaac_asset_directory(relative_path: str, *, default_root: str) -> str:
    """Resolve an Isaac-relative asset directory from the same explicit root."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a safe Isaac-relative asset path, got: {relative_path!r}")

    local_root = os.environ.get(LOCAL_ISAAC_ASSET_ROOT_ENV)
    if not local_root:
        return f"{default_root.rstrip('/')}/{relative.as_posix()}"

    root = Path(local_root).expanduser().resolve()
    asset_directory = root.joinpath(*relative.parts).resolve()
    try:
        asset_directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Asset path escapes {LOCAL_ISAAC_ASSET_ROOT_ENV}: {relative_path!r}") from exc
    if not asset_directory.is_dir():
        raise FileNotFoundError(
            f"Missing official Isaac asset directory under {LOCAL_ISAAC_ASSET_ROOT_ENV}={root}: "
            f"{relative.as_posix()}"
        )
    return str(asset_directory)

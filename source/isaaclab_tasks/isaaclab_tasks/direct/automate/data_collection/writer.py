# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic pickle writer for RR-compatible trajectories."""

from __future__ import annotations

import lzma
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from .schema import Trajectory
from .validator import validate_trajectory


def write_trajectory(
    trajectory: Trajectory,
    output_root: str | Path,
    *,
    attempt_idx: int,
    compress: bool = False,
    timestamp: datetime | None = None,
) -> Path:
    """Validate and atomically write one trajectory without overwriting files."""

    validate_trajectory(trajectory)
    if attempt_idx < 1:
        raise ValueError(f"attempt_idx must be positive, received {attempt_idx}.")

    output_root = Path(output_root).expanduser()
    status_dir = output_root / ("success" if trajectory["success"] else "failure")
    status_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().astimezone()
    timestamp_text = timestamp.strftime("%Y-%m-%dT%H-%M-%S.%f")
    extension = ".pkl.xz" if compress else ".pkl"
    output_path = status_dir / f"{timestamp_text}_attempt-{attempt_idx:06d}{extension}"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing rollout: {output_path}")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=status_dir)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as raw_stream:
            if compress:
                with lzma.LZMAFile(raw_stream, "wb") as compressed_stream:
                    _dump(trajectory, compressed_stream)
            else:
                _dump(trajectory, raw_stream)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())

        # A hard link publishes the completed temporary file atomically and,
        # unlike os.replace(), fails if another collector won the same name.
        os.link(temporary_path, output_path)
        directory_fd = os.open(status_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _dump(trajectory: Trajectory, stream: BinaryIO) -> None:
    pickle.dump(trajectory, stream, protocol=pickle.HIGHEST_PROTOCOL)

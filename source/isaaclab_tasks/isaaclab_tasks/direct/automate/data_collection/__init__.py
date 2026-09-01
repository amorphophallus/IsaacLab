"""RR-compatible data collection helpers for AutoMate."""

from .action_adapter import ActionAdapter
from .buffer import EpisodeBuffer
from .record_pickle import PickleRecorder
from .scheduling import classify_batch_results
from .schema import CameraInfo, Observation, RobotState, Trajectory
from .state_adapter import StateAdapter
from .validator import TrajectoryValidationError, validate_trajectory
from .writer import write_trajectory

__all__ = [
    "ActionAdapter",
    "CameraInfo",
    "EpisodeBuffer",
    "Observation",
    "PickleRecorder",
    "RobotState",
    "StateAdapter",
    "Trajectory",
    "TrajectoryValidationError",
    "classify_batch_results",
    "validate_trajectory",
    "write_trajectory",
]

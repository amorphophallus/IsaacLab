# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate robust-rearrangement-compatible pickles from an AutoMate policy."""

from __future__ import annotations

import argparse
import sys
import types

from isaaclab.app import AppLauncher


EXCLUDED_ASSEMBLY_IDS = frozenset({"00755"})


def _normalize_assembly_id(value: str) -> str:
    if not value.isdigit():
        raise argparse.ArgumentTypeError(f"assembly ID must be numeric, received {value!r}.")
    normalized = value.zfill(5) if len(value) < 5 else value
    if normalized in EXCLUDED_ASSEMBLY_IDS:
        raise argparse.ArgumentTypeError(
            f"assembly {normalized} is excluded from the 99-task production campaign"
        )
    return normalized


parser = argparse.ArgumentParser(description="Generate RR-compatible AutoMate rollout pickles.")
parser.add_argument("--task", type=str, default="Isaac-AutoMate-Assembly-Direct-v0", help="AutoMate Gym task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="RL-Games agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="RL-Games checkpoint path.")
parser.add_argument("--assembly-id", type=_normalize_assembly_id, required=True, help="AutoMate assembly asset ID.")
parser.add_argument(
    "--annotation-source",
    choices=("scripted",),
    required=True,
    help="Required provenance gate; AutoMate insertion targets are geometry GT.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Output root; success/failure folders are created below it.",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized rollout environments.")
parser.add_argument("--num-successes", type=int, default=50, help="Number of successful trajectories to save.")
parser.add_argument(
    "--max-attempts",
    type=int,
    default=None,
    help="Maximum attempted episodes. Defaults to 10 times --num-successes.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum policy steps per attempt. Defaults to the configured AutoMate horizon.",
)
parser.add_argument("--save-failures", action="store_true", help="Also write max-step failure trajectories.")
parser.add_argument(
    "--skip-dense-reward",
    action="store_true",
    help="Skip collection-irrelevant SDF/SoftDTW reward while preserving success checks.",
)
parser.add_argument("--compress", action="store_true", help="Write .pkl.xz instead of uncompressed .pkl files.")
parser.add_argument(
    "--writer-workers",
    type=int,
    default=2,
    help="Background workers used for strict validation and pickle writes.",
)
parser.add_argument(
    "--max-pending-writes",
    type=int,
    default=None,
    help="Bounded validation/write queue size. Defaults to max(num_envs, 2*writer_workers).",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Required production mode: use the specialist policy mean.",
)
parser.add_argument(
    "--enable-sbc",
    "--sbc",
    dest="enable_sbc",
    action="store_true",
    help="Rejected by the production collector; retained only for an explicit fail-closed error.",
)
parser.add_argument("--seed", type=int, default=0, help="Environment and RL-Games seed.")
parser.add_argument(
    "--disassembly-path",
    type=str,
    default=None,
    help="Optional disassemble_traj.json override for the selected assembly.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Simulation and learning imports follow app launch.

import hashlib
import json
import math
import os
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.automate.data_collection import (
    PickleRecorder,
    classify_batch_results,
    write_trajectory,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


def _policy_observation(observation):
    return observation["obs"] if isinstance(observation, dict) else observation


def _reset_policy_batch(env: RlGamesVecEnvWrapper, agent: BasePlayer):
    observation = _policy_observation(env.reset())
    _ = agent.get_batch_size(observation, 1)
    if agent.is_rnn:
        agent.init_rnn()
    return observation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finalize_attempt(
    recorder: PickleRecorder,
    *,
    success: bool,
    classification: str,
    task: str,
    output_dir: Path,
    attempt_idx: int,
    compress: bool,
    save_failures: bool,
) -> dict:
    """Strictly validate one finished attempt and optionally publish its pickle."""

    started_at = time.perf_counter()
    trajectory = recorder.finish_episode(success=success, task=task)
    should_write = classification == "selected" or (classification == "failure" and save_failures)
    output_path = None
    output_bytes = 0
    output_sha256 = None
    if should_write:
        output_path = write_trajectory(
            trajectory,
            output_dir,
            attempt_idx=attempt_idx,
            compress=compress,
        )
        output_bytes = output_path.stat().st_size
        output_sha256 = _sha256_file(output_path)
    return {
        "classification": classification,
        "success": success,
        "saved": output_path is not None,
        "relative_path": str(output_path.relative_to(output_dir)) if output_path is not None else None,
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "num_transitions": len(trajectory["actions"]),
        "validation_write_seconds": time.perf_counter() - started_at,
    }


def _drain_attempts(
    pending: dict[Future, dict],
    records: list[dict],
    *,
    all_completed: bool,
) -> None:
    if not pending:
        return
    done, _not_done = wait(
        pending,
        return_when=ALL_COMPLETED if all_completed else FIRST_COMPLETED,
    )
    for future in done:
        base_record = pending.pop(future)
        records.append({**base_record, **future.result()})


def _atomic_write_json(path: Path, payload) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda item: item["global_attempt_idx"]):
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Load an AutoMate checkpoint and collect RR-compatible rollouts."""

    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args_cli.num_envs > 32:
        raise ValueError("--num-envs above 32 has not passed the production camera/schema gate.")
    if args_cli.num_successes <= 0:
        raise ValueError("--num-successes must be positive.")
    if not args_cli.deterministic:
        raise ValueError("Production AutoMate collection requires --deterministic.")
    if args_cli.enable_sbc:
        raise ValueError("Production AutoMate collection forbids --enable-sbc; use hardest initialization.")
    if args_cli.writer_workers <= 0:
        raise ValueError("--writer-workers must be positive.")
    max_pending_writes = args_cli.max_pending_writes
    if max_pending_writes is None:
        max_pending_writes = max(args_cli.num_envs, 2 * args_cli.writer_workers)
    if max_pending_writes <= 0:
        raise ValueError("--max-pending-writes must be positive.")
    if max_pending_writes < args_cli.writer_workers:
        raise ValueError("--max-pending-writes must be at least --writer-workers.")
    max_attempts = args_cli.max_attempts
    if max_attempts is None:
        max_attempts = args_cli.num_successes * 10
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.camera.enabled = True
    env_cfg.action_noise_model = None

    task_cfg = env_cfg.tasks[env_cfg.task_name]
    task_cfg.assembly_id = args_cli.assembly_id
    task_cfg.if_sbc = args_cli.enable_sbc
    task_cfg.if_logging_eval = False
    if args_cli.disassembly_path is not None:
        task_cfg.disassembly_path_json = args_cli.disassembly_path

    configured_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    configured_max_steps = math.ceil(float(env_cfg.episode_length_s) / configured_step_dt)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else configured_max_steps
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    # DirectRLEnv times out at max_episode_length - 1 and resets before it
    # returns the terminal observation. Add one complete safety step beyond
    # the collector boundary so the final RGB-D frame cannot be a reset frame.
    env_cfg.episode_length_s = (max_steps + 2) * configured_step_dt

    agent_cfg["params"]["seed"] = args_cli.seed
    if args_cli.device is not None:
        # Keep RL-Games on the selected simulation device. This mirrors the
        # standard Isaac Lab play/train launchers and avoids device copies.
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    if args_cli.disassembly_path is None:
        cached_disassembly_path = os.path.join(log_dir, "assets", "disassemble_traj.json")
        if os.path.isfile(cached_disassembly_path):
            task_cfg.disassembly_path_json = cached_disassembly_path
            print(
                f"[INFO] Using checkpoint-cached disassembly trajectory: {cached_disassembly_path}"
            )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    incomplete_marker = output_dir / ".collection-incomplete"
    incomplete_marker.write_text(
        f"started_epoch={time.time()}\nannotation_source={args_cli.annotation_source}\n",
        encoding="utf-8",
    )

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped

    raw_env._collection_reset_step_counter = 0

    def step_reset_sim_with_periodic_render(self):
        """Advance reset physics with the same render cadence as normal steps."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self._collection_reset_step_counter += 1
        if self._collection_reset_step_counter % self.cfg.sim.render_interval == 0:
            self.sim.render()
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    # Install this only after construction, when RTX sensors and their render
    # pipeline are initialized. Rendering every IK/gripper reset substep changes
    # the grasp dynamics, while never rendering can desynchronize PhysX/RTX GPU
    # streams. Match DirectRLEnv.step(): render once per configured interval.
    raw_env.step_sim_no_action = types.MethodType(step_reset_sim_with_periodic_render, raw_env)
    if args_cli.skip_dense_reward:
        # Dense SDF/SoftDTW rewards are training signals, not policy inputs or
        # recorded rewards. AssemblyEnv._get_rewards still computes and latches
        # the official insertion success before calling this function.
        raw_env._update_rew_buf = lambda curr_successes: curr_successes.to(dtype=torch.float32)
        print("[INFO] Skipping collection-irrelevant SDF/SoftDTW dense reward.")

    env = RlGamesVecEnvWrapper(
        gym_env,
        rl_device,
        clip_obs,
        clip_actions,
        obs_groups,
        concatenate_obs_groups,
    )

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    task_label = f"automate_insertion_{args_cli.assembly_id}"
    selected_episodes = 0
    observed_successes = 0
    excluded_successes = 0
    attempts_completed = 0
    batch_index = 0
    pending: dict[Future, dict] = {}
    attempt_records: list[dict] = []
    collection_start = time.perf_counter()
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Task label: {task_label}")
    print(f"[INFO] Annotation source: {args_cli.annotation_source}")
    print(
        f"[INFO] Environments: {args_cli.num_envs}; target successes: {args_cli.num_successes}; "
        f"max attempts: {max_attempts}; max steps: {max_steps}"
    )
    print(
        f"[INFO] Writer workers: {args_cli.writer_workers}; "
        f"max pending validations/writes: {max_pending_writes}"
    )
    print(f"[INFO] Policy mode: {'deterministic' if args_cli.deterministic else 'stochastic'}")
    print(f"[INFO] Sampling-based curriculum: {'enabled' if args_cli.enable_sbc else 'disabled'}")

    executor = ThreadPoolExecutor(
        max_workers=args_cli.writer_workers,
        thread_name_prefix="automate-pickle-writer",
    )
    try:
        while (
            simulation_app.is_running()
            and attempts_completed < max_attempts
            and selected_episodes < args_cli.num_successes
        ):
            batch_index += 1
            active_envs = min(args_cli.num_envs, max_attempts - attempts_completed)
            observation = _reset_policy_batch(env, agent)
            # Manual resets do not pass through AssemblyEnv._pre_physics_step(),
            # so the previous episode's latched first-hit status must be cleared.
            # Otherwise every rollout after the first success terminates in one step.
            raw_env.ep_succeeded.zero_()
            recorders = [PickleRecorder(env_idx=env_idx) for env_idx in range(active_envs)]
            for recorder in recorders:
                recorder.start_episode(raw_env)
            completed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=raw_env.device)
            if active_envs < args_cli.num_envs:
                completed[active_envs:] = True
            succeeded = torch.zeros_like(completed)

            for _step_idx in range(max_steps):
                with torch.inference_mode():
                    policy_input = agent.obs_to_torch(observation)
                    policy_action = agent.get_action(policy_input, is_deterministic=args_cli.deterministic)
                # Keep Isaac's step outside inference mode: its persistent
                # simulation buffers must remain mutable across manual resets.
                clipped_action = policy_action.detach().clone()
                clipped_action.clamp_(-clip_actions, clip_actions)
                if bool(torch.any(completed).item()):
                    clipped_action[completed] = 0.0
                rr_actions = {
                    env_idx: recorders[env_idx].prepare_action(raw_env, clipped_action)
                    for env_idx in range(active_envs)
                    if not bool(completed[env_idx].item())
                }
                next_observation, _dense_reward, dones, _extras = env.step(clipped_action)
                observation = _policy_observation(next_observation)
                for env_idx, rr_action in rr_actions.items():
                    success = bool(raw_env.ep_succeeded[env_idx].item())
                    recorders[env_idx].record_step(raw_env, rr_action, success)
                    if success:
                        succeeded[env_idx] = True
                        completed[env_idx] = True
                unexpected_done = torch.as_tensor(dones, device=raw_env.device, dtype=torch.bool) & ~completed
                if bool(torch.any(unexpected_done).item()):
                    raise RuntimeError(
                        "AutoMate auto-reset before the collector boundary; increase the configured safety horizon."
                    )
                if bool(torch.all(completed).item()):
                    break

            success_flags = [bool(succeeded[env_idx].item()) for env_idx in range(active_envs)]
            classifications = classify_batch_results(
                success_flags,
                args_cli.num_successes - selected_episodes,
            )
            batch_selected = classifications.count("selected")
            batch_excluded = classifications.count("excluded")
            selected_episodes += batch_selected
            observed_successes += sum(success_flags)
            excluded_successes += batch_excluded

            for env_idx, (recorder, success, classification) in enumerate(
                zip(recorders, success_flags, classifications, strict=True)
            ):
                attempts_completed += 1
                attempt_idx = attempts_completed
                while len(pending) >= max_pending_writes:
                    _drain_attempts(pending, attempt_records, all_completed=False)
                future = executor.submit(
                    _finalize_attempt,
                    recorder,
                    success=success,
                    classification=classification,
                    task=task_label,
                    output_dir=output_dir,
                    attempt_idx=attempt_idx,
                    compress=args_cli.compress,
                    save_failures=args_cli.save_failures,
                )
                pending[future] = {
                    "assembly_id": args_cli.assembly_id,
                    "task": task_label,
                    "process_seed": args_cli.seed,
                    "batch_index": batch_index,
                    "env_idx": env_idx,
                    "global_attempt_idx": attempt_idx,
                    "annotation_source": args_cli.annotation_source,
                    "image_annotation_mode": "none",
                    "randomness_semantics": "hardest_init",
                    "init_mode": "sbc" if args_cli.enable_sbc else "hardest",
                    "policy_mode": "deterministic" if args_cli.deterministic else "stochastic",
                }
            _drain_attempts(pending, attempt_records, all_completed=False)
            elapsed = time.perf_counter() - collection_start
            attempts_per_hour = attempts_completed * 3600.0 / elapsed
            print(
                f"[INFO] Batch {batch_index}: attempts={attempts_completed}/{max_attempts}, "
                f"selected={selected_episodes}/{args_cli.num_successes}, "
                f"observed_successes={observed_successes}, excluded_successes={excluded_successes}, "
                f"pending_writes={len(pending)}, attempted_rollouts_per_hour={attempts_per_hour:.2f}"
            )

        _drain_attempts(pending, attempt_records, all_completed=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        env.close()

    collection_seconds = time.perf_counter() - collection_start
    attempt_records.sort(key=lambda item: item["global_attempt_idx"])
    if len(attempt_records) != attempts_completed:
        raise RuntimeError(
            f"Attempt manifest count mismatch: records={len(attempt_records)}, attempts={attempts_completed}."
        )
    saved_selected = sum(
        record["classification"] == "selected" and record["saved"] for record in attempt_records
    )
    if saved_selected != selected_episodes:
        raise RuntimeError(
            f"Selected output count mismatch: files={saved_selected}, selected={selected_episodes}."
        )
    saved_failures = sum(
        record["classification"] == "failure" and record["saved"] for record in attempt_records
    )
    selected_transitions = sum(
        record["num_transitions"] for record in attempt_records if record["classification"] == "selected"
    )
    selected_bytes = sum(
        record["output_bytes"] for record in attempt_records if record["classification"] == "selected"
    )
    complete = selected_episodes == args_cli.num_successes
    disassembly_path = Path(task_cfg.disassembly_path_json).expanduser()
    summary = {
        "schema": "rr-automate-production-collection-v1",
        "complete": complete,
        "assembly_id": args_cli.assembly_id,
        "task": task_label,
        "annotation_source": args_cli.annotation_source,
        "image_annotation_mode": "none",
        "randomness_semantics": "hardest_init",
        "init_mode": "sbc" if args_cli.enable_sbc else "hardest",
        "policy_mode": "deterministic" if args_cli.deterministic else "stochastic",
        "process_seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "target_successes": args_cli.num_successes,
        "max_attempts": max_attempts,
        "max_steps": max_steps,
        "attempts_completed": attempts_completed,
        "selected_successes": selected_episodes,
        "observed_successes": observed_successes,
        "excluded_successes": excluded_successes,
        "saved_failures": saved_failures,
        "selected_transitions": selected_transitions,
        "selected_output_bytes": selected_bytes,
        "collection_seconds": collection_seconds,
        "attempted_rollouts_per_hour": attempts_completed * 3600.0 / collection_seconds,
        "writer_workers": args_cli.writer_workers,
        "max_pending_writes": max_pending_writes,
        "checkpoint": str(resume_path),
        "checkpoint_sha256": _sha256_file(Path(resume_path)),
        "disassembly_path": str(task_cfg.disassembly_path_json),
        "disassembly_sha256": _sha256_file(disassembly_path) if disassembly_path.is_file() else None,
    }
    _atomic_write_jsonl(output_dir / "attempt_manifest.jsonl", attempt_records)
    _atomic_write_json(output_dir / "collection_summary.json", summary)
    print(
        f"[INFO] Collection summary: attempts={attempts_completed}, selected={selected_episodes}, "
        f"observed_successes={observed_successes}, excluded_successes={excluded_successes}, "
        f"saved_failures={saved_failures}, attempted_rollouts_per_hour="
        f"{summary['attempted_rollouts_per_hour']:.2f}."
    )
    if not complete:
        raise RuntimeError(
            f"Collected only {selected_episodes}/{args_cli.num_successes} successes within {max_attempts} attempts."
        )
    (output_dir / ".collection-complete").write_text(
        f"selected_successes={selected_episodes}\nattempts={attempts_completed}\n",
        encoding="utf-8",
    )
    incomplete_marker.unlink()
    _fsync_directory(output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

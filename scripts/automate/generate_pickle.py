# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate robust-rearrangement-compatible pickles from an AutoMate policy."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


def _normalize_assembly_id(value: str) -> str:
    if not value.isdigit():
        raise argparse.ArgumentTypeError(f"assembly ID must be numeric, received {value!r}.")
    return value.zfill(5) if len(value) < 5 else value


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
    "--output-dir",
    type=str,
    required=True,
    help="Output root; success/failure folders are created below it.",
)
parser.add_argument("--num-successes", type=int, default=100, help="Number of successful trajectories to save.")
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
parser.add_argument("--compress", action="store_true", help="Write .pkl.xz instead of uncompressed .pkl files.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use the policy mean. The default samples from the PPO action distribution.",
)
parser.add_argument(
    "--enable-sbc",
    "--sbc",
    dest="enable_sbc",
    action="store_true",
    help="Enable the sampling-based curriculum. The default evaluates at the hardest curriculum stage.",
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

import math
import os

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.automate.data_collection import PickleRecorder, write_trajectory
from isaaclab_tasks.utils.hydra import hydra_task_config


def _policy_observation(observation):
    return observation["obs"] if isinstance(observation, dict) else observation


def _reset_policy_episode(env: RlGamesVecEnvWrapper, agent: BasePlayer):
    observation = _policy_observation(env.reset())
    _ = agent.get_batch_size(observation, 1)
    if agent.is_rnn:
        agent.init_rnn()
    return observation


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Load an AutoMate checkpoint and collect RR-compatible rollouts."""

    if args_cli.num_successes <= 0:
        raise ValueError("--num-successes must be positive.")
    max_attempts = args_cli.max_attempts
    if max_attempts is None:
        max_attempts = args_cli.num_successes * 10
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")

    env_cfg.scene.num_envs = 1
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

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped
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
    agent_cfg["params"]["config"]["num_actors"] = 1
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    task_label = f"automate_insertion_{args_cli.assembly_id}"
    successful_episodes = 0
    saved_failures = 0
    attempts_completed = 0
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Task label: {task_label}")
    print(f"[INFO] Target successes: {args_cli.num_successes}; max attempts: {max_attempts}; max steps: {max_steps}")
    print(f"[INFO] Policy mode: {'deterministic' if args_cli.deterministic else 'stochastic'}")
    print(f"[INFO] Sampling-based curriculum: {'enabled' if args_cli.enable_sbc else 'disabled'}")

    try:
        for attempt_idx in range(1, max_attempts + 1):
            if not simulation_app.is_running() or successful_episodes >= args_cli.num_successes:
                break
            attempts_completed = attempt_idx
            observation = _reset_policy_episode(env, agent)
            # Manual resets do not pass through AssemblyEnv._pre_physics_step(),
            # so the previous episode's latched first-hit status must be cleared.
            # Otherwise every rollout after the first success terminates in one step.
            raw_env.ep_succeeded.zero_()
            recorder = PickleRecorder(env_idx=0)
            recorder.start_episode(raw_env)
            success = False

            for _step_idx in range(max_steps):
                with torch.inference_mode():
                    policy_input = agent.obs_to_torch(observation)
                    policy_action = agent.get_action(policy_input, is_deterministic=args_cli.deterministic)
                # Keep Isaac's step outside inference mode: its persistent
                # simulation buffers must remain mutable across manual resets.
                clipped_action = policy_action.detach().clone()
                clipped_action.clamp_(-clip_actions, clip_actions)
                rr_action = recorder.prepare_action(raw_env, clipped_action)
                next_observation, _dense_reward, dones, _extras = env.step(clipped_action)
                observation = _policy_observation(next_observation)
                success = bool(raw_env.ep_succeeded[0].item())
                recorder.record_step(raw_env, rr_action, success)
                if success:
                    break
                if bool(torch.any(dones).item()):
                    raise RuntimeError(
                        "AutoMate auto-reset before the collector boundary; increase the configured safety horizon."
                    )

            trajectory = recorder.finish_episode(success=success, task=task_label)
            if success:
                output_path = write_trajectory(
                    trajectory,
                    args_cli.output_dir,
                    attempt_idx=attempt_idx,
                    compress=args_cli.compress,
                )
                successful_episodes += 1
                print(
                    f"[INFO] Saved success {successful_episodes}/{args_cli.num_successes} "
                    f"from attempt {attempt_idx}: {output_path}"
                )
            elif args_cli.save_failures:
                output_path = write_trajectory(
                    trajectory,
                    args_cli.output_dir,
                    attempt_idx=attempt_idx,
                    compress=args_cli.compress,
                )
                saved_failures += 1
                print(f"[INFO] Saved failure from attempt {attempt_idx}: {output_path}")
            else:
                print(f"[INFO] Discarded failed attempt {attempt_idx}.")
    finally:
        env.close()

    print(
        f"[INFO] Collection summary: attempts={attempts_completed}, successes={successful_episodes}, "
        f"saved_failures={saved_failures}."
    )
    if successful_episodes < args_cli.num_successes:
        raise RuntimeError(
            f"Collected only {successful_episodes}/{args_cli.num_successes} successes within {max_attempts} attempts."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

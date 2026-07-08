# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_name", type=str, nargs="+", default=None, help="Motion artifact name(s). Use multiple for multi-motion training.")
parser.add_argument("--motion_dir", type=str, default=None, help="Directory of training NPZ files for large-scale multi-motion training.")
parser.add_argument("--no_unwanted_contact_penalty", action="store_true", default=False, help="Disable unwanted contact distance penalty.")
parser.add_argument("--mix_contact_groups", type=str, nargs="+", default=None,
                    help="Contact label mix groups as colon-separated motion names, e.g. 'motionA:motionB' 'motionC:motionD:motionE'.")
parser.add_argument("--mix_contact_prob", type=float, default=0.0, help="Probability of mixing contact labels between group members.")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="WandB entity/username (leave empty to use the wandb-CLI default).")
parser.add_argument("--terrain_mode", type=str, default=None, choices=["flat", "rough", "mixed"], help="Override terrain mode (flat/rough/mixed).")
parser.add_argument("--terrain_noise", type=float, default=None, help="Half-range of terrain height noise in meters (e.g. 0.05 for ±5cm).")
parser.add_argument("--stand_ratio", type=float, default=None, help="Fraction of envs frozen at frame 0 (stand-still) for curriculum.")
parser.add_argument("--start_from_zero_ratio", type=float, default=None, help="Per-reset probability that an env starts at frame 0 (for sim2real startup stability).")
parser.add_argument("--wandb_artifact_project", type=str, default="beyondmimic", help="WandB project where artifacts are stored.")
parser.add_argument("--no_contact", action="store_true", default=False, help="Disable all contact observations and rewards.")
parser.add_argument("--chair_offset_cfg", type=str, default=None, help="Path to YAML file with per-motion chair XYZ offsets, e.g. motion_name: {x: 0.1, y: 0.0, z: 0.05}.")
parser.add_argument("--random_chair_y", type=float, default=None, help="Std (meters) of Gaussian noise on chair Y position at each reset (e.g. 0.05 for ~±5 cm).")
parser.add_argument("--remove_object_prob", type=float, default=None, help="Per-reset probability of removing the motion object (sending it underground) for each env (e.g. 0.2).")
parser.add_argument("--mask_contact_prob", type=float, default=None, help="Per-part probability of randomly masking each contact part (seat/backrest/legs) at each reset (e.g. 0.3).")
parser.add_argument("--random_mask_contact_parts", type=int, nargs="+", default=None,
                    help="Explicit part IDs eligible for random masking (e.g. 1 for unsegmented objects). "
                         "Overrides --mask_contact_prob's hardcoded [1,2,3].")
parser.add_argument("--random_mask_contact_prob", type=float, default=None,
                    help="Probability used with --random_mask_contact_parts (whole=False).")
parser.add_argument("--contact_tp_fp_mode", action="store_true", default=False, help="Use TP-FP mode for contact_label_match: reward=TPR - fp_penalty*FPR. Better for sparse-contact motions.")
parser.add_argument("--contact_fp_penalty", type=float, default=None, help="False positive penalty weight for --contact_tp_fp_mode (default: 1.0).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HUMOTO_BASE = os.environ.get(
    "HUMOTO_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "HUMOTO")),
)


def split_motion_urdf(urdf_path: str) -> tuple[str, list[str]]:
    """Split a multi-object URDF into a chair-only URDF and extra-object URDFs.

    Classification: links whose name contains "chair" are chair links.
    If no link contains "chair", all non-world links are treated as the chair
    (backward compat for multi_boxes_link, low_chair_link, etc.).

    Generated files are placed next to the original URDF (lazily, only if absent).

    Returns:
        (chair_urdf_path, extra_urdf_paths)
        chair_urdf_path: path to chair-only URDF (the original if no split needed).
        extra_urdf_paths: list of extra-object URDF paths (empty when no split).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    non_world_links = [lk.get("name") for lk in root.findall("link") if lk.get("name") != "world"]
    chair_link_names = [n for n in non_world_links if "chair" in n.lower()]

    if not chair_link_names:
        # No link with "chair" in name → whole URDF is the chair (backward compat)
        return urdf_path, []

    extra_link_names = [n for n in non_world_links if n not in chair_link_names]
    if not extra_link_names:
        return urdf_path, []

    # Split needed: generate chair-only and per-extra URDFs
    base_dir = os.path.dirname(urdf_path)
    stem = os.path.basename(urdf_path).replace(".urdf", "")
    links_by_name = {lk.get("name"): lk for lk in root.findall("link")}
    all_joints = root.findall("joint")

    def _write_urdf(selected: list[str], out_path: str) -> None:
        new_root = ET.Element("robot", name="multi_boxes")
        ET.SubElement(new_root, "link", name="world")
        for lname in selected:
            new_root.append(links_by_name[lname])
        for jt in all_joints:
            child_elem = jt.find("child")
            if child_elem is not None and child_elem.get("link") in selected:
                new_root.append(jt)
        new_tree = ET.ElementTree(new_root)
        try:
            ET.indent(new_tree, space="  ")
        except AttributeError:
            pass  # ET.indent requires Python 3.9+
        new_tree.write(out_path, xml_declaration=True, encoding="utf-8")

    chair_path = os.path.join(base_dir, f"{stem}_chair_only.urdf")
    if not os.path.exists(chair_path):
        _write_urdf(chair_link_names, chair_path)
        print(f"[INFO]: Generated chair-only URDF: {chair_path}")

    extra_paths: list[str] = []
    for lname in extra_link_names:
        obj_name = lname.replace("_link", "")
        extra_path = os.path.join(base_dir, f"{stem}_{obj_name}_only.urdf")
        if not os.path.exists(extra_path):
            _write_urdf([lname], extra_path)
            print(f"[INFO]: Generated extra-object URDF: {extra_path}")
        extra_paths.append(extra_path)

    return chair_path, extra_paths


def discover_object_urdfs(motion_names: list[str]) -> tuple[list[str | None], list[list[str] | None]]:
    """Auto-discover per-motion chair URDFs and extra-object URDFs from HUMOTO.

    For each motion name, strips suffixes and looks for
    ``multi_boxes_scaled_0.88_0.88_0.88.urdf`` in the matching HUMOTO subdir.
    URDFs with multiple distinct object types (e.g. chair + table) are split:
    the chair URDF is returned as the contact-sensed object, extra objects
    (table, desk, …) as separate non-contact assets.

    Returns:
        chair_urdfs: list parallel to *motion_names*. ``None`` for motions without
            an object.
        extra_urdfs: list parallel to *motion_names*. Each entry is either ``None``
            (no extra objects) or a list of extra-object URDF paths.
    """
    import re
    chair_urdfs: list[str | None] = []
    extra_urdfs: list[list[str] | None] = []
    for name in motion_names:
        m = re.search(r"([a-zA-Z_]+-\d{3})", name)
        base = m.group(1) if m else name
        urdf_path = os.path.join(_HUMOTO_BASE, base, "multi_boxes_scaled_0.88_0.88_0.88.urdf")
        if os.path.exists(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
            chair_urdf, extras = split_motion_urdf(urdf_path)
            chair_urdfs.append(chair_urdf)
            extra_urdfs.append(extras if extras else None)
            suffix = (f" + extras: {[os.path.basename(e) for e in extras]}" if extras else "")
            print(f"[INFO]: Object URDF for '{name}': {os.path.basename(chair_urdf)}{suffix}")
        else:
            chair_urdfs.append(None)
            extra_urdfs.append(None)
            print(f"[INFO]: No object URDF for '{name}' (no HUMOTO dir)")
    return chair_urdfs, extra_urdfs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load motion file(s) from wandb, local path, or directory
    if args_cli.motion_dir is not None:
        # Large-scale training: load all NPZ files from directory
        env_cfg.commands.motion.motion_dir = os.path.abspath(args_cli.motion_dir)
        import glob as _glob
        n_files = len(_glob.glob(os.path.join(args_cli.motion_dir, "*.npz")))
        print(f"[INFO]: Large-scale training with {n_files} motions from {args_cli.motion_dir}")
        motion_names = [os.path.basename(args_cli.motion_dir)]
    elif args_cli.motion_name is not None:
        motion_names = args_cli.motion_name
        import pathlib

        import wandb

        os.environ["WANDB_ENTITY"] = args_cli.wandb_entity
        api = wandb.Api()

        motion_files = []
        for name in motion_names:
            if os.path.exists(name):
                motion_files.append(name)
            else:
                version = "latest" if ":" not in name else name.split(":")[-1]
                base_name = name.split(":")[0] if ":" in name else name
                artifact_path = f"{args_cli.wandb_entity}/{args_cli.wandb_artifact_project}/{base_name}:{version}"
                print(f"[INFO]: Loading motion artifact: {artifact_path}")
                artifact = api.artifact(artifact_path)
                motion_files.append(str(pathlib.Path(artifact.download()) / "motion.npz"))

        if len(motion_files) == 1:
            env_cfg.commands.motion.motion_file = motion_files[0]
        else:
            env_cfg.commands.motion.motion_file = motion_files[0]  # fallback
            env_cfg.commands.motion.motion_files = motion_files
            env_cfg.commands.motion.motion_names = motion_names
            print(f"[INFO]: Multi-motion training with {len(motion_files)} motions: {motion_names}")
    else:
        raise ValueError("Must provide either --motion_name or --motion_dir")

    # Discover per-motion object URDFs and configure scene objects
    object_urdfs, extra_object_urdfs = discover_object_urdfs(motion_names)
    env_cfg.commands.motion.object_urdfs = object_urdfs
    if hasattr(env_cfg, "configure_motion_objects") and any(u is not None for u in object_urdfs):
        env_cfg.configure_motion_objects(object_urdfs, extra_object_urdfs)
    if args_cli.terrain_mode is not None and hasattr(env_cfg, "terrain"):
        env_cfg.terrain.mode = args_cli.terrain_mode
    if args_cli.terrain_noise is not None and hasattr(env_cfg, "terrain"):
        env_cfg.terrain.noise_range = (-args_cli.terrain_noise, args_cli.terrain_noise)
        env_cfg.terrain.noise_step = args_cli.terrain_noise / 4
    if args_cli.stand_ratio is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.stand_ratio = args_cli.stand_ratio
    if args_cli.start_from_zero_ratio is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.start_from_zero_ratio = args_cli.start_from_zero_ratio
    if args_cli.chair_offset_cfg is not None and hasattr(env_cfg.commands, "motion"):
        import yaml
        with open(args_cli.chair_offset_cfg) as _f:
            env_cfg.commands.motion.chair_offset_per_motion = yaml.safe_load(_f)
    if args_cli.random_chair_y is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.chair_y_rand_std = args_cli.random_chair_y
    if args_cli.remove_object_prob is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.object_remove_prob = args_cli.remove_object_prob
    if args_cli.mask_contact_prob is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.random_mask_contact_parts = [1, 2, 3]
        env_cfg.commands.motion.random_mask_contact_prob = args_cli.mask_contact_prob
        env_cfg.commands.motion.random_mask_contact_whole = True
    if args_cli.random_mask_contact_parts is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.random_mask_contact_parts = list(args_cli.random_mask_contact_parts)
        env_cfg.commands.motion.random_mask_contact_whole = False
        if args_cli.random_mask_contact_prob is not None:
            env_cfg.commands.motion.random_mask_contact_prob = args_cli.random_mask_contact_prob
    if args_cli.contact_tp_fp_mode and hasattr(env_cfg, "contact_label_tp_fp_mode"):
        env_cfg.contact_label_tp_fp_mode = True
        if hasattr(env_cfg.rewards, "contact_label_match"):
            env_cfg.rewards.contact_label_match.params["tp_fp_mode"] = True
    if args_cli.contact_fp_penalty is not None and hasattr(env_cfg, "contact_label_fp_penalty_weight"):
        env_cfg.contact_label_fp_penalty_weight = args_cli.contact_fp_penalty
        if hasattr(env_cfg.rewards, "contact_label_match"):
            env_cfg.rewards.contact_label_match.params["fp_penalty_weight"] = args_cli.contact_fp_penalty
    if hasattr(env_cfg, "configure_terrain"):
        env_cfg.configure_terrain()

    # Post-configure flags
    if args_cli.no_contact:
        _contact_obs_terms = [
            "contact_labels_ref", "contact_labels_ref_binary",
            "contact_body_chair_distance", "contact_labels_real", "contact_labels_real_binary",
        ]
        for term in _contact_obs_terms:
            if hasattr(env_cfg.observations.policy, term):
                setattr(env_cfg.observations.policy, term, None)
            if hasattr(env_cfg.observations.critic, term):
                setattr(env_cfg.observations.critic, term, None)
        if hasattr(env_cfg.rewards, "contact_label_match"):
            env_cfg.rewards.contact_label_match = None
        if hasattr(env_cfg.rewards, "contact_distance"):
            env_cfg.rewards.contact_distance = None
    if args_cli.no_unwanted_contact_penalty and hasattr(env_cfg, "enable_unwanted_contact_penalty"):
        env_cfg.enable_unwanted_contact_penalty = False
    if args_cli.mix_contact_groups and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.mix_contact_labels = [g.split(":") for g in args_cli.mix_contact_groups]
        env_cfg.commands.motion.mix_contact_label_prob = args_cli.mix_contact_prob

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
        registry_name=f"{args_cli.wandb_entity}/{args_cli.wandb_artifact_project}/{motion_names[0]}:latest" if not os.path.exists(motion_names[0]) else None
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

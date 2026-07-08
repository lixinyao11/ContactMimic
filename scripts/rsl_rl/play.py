"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--motion_name", type=str, nargs="+", default=None, help="Motion artifact name(s) from wandb.")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="WandB entity/username (leave empty to use the wandb-CLI default).")
parser.add_argument("--wandb_artifact_project", type=str, default="beyondmimic", help="WandB project where artifacts are stored.")
parser.add_argument("--start_from_zero", action="store_true", default=True, help="Always start motion from frame 0.")
parser.add_argument("--no_termi", action="store_true", default=False, help="Disable position/orientation terminations during play.")
parser.add_argument("--mask_back", action="store_true", default=False, help="Mask backrest contact (ignore_contact_parts=[2]).")
parser.add_argument("--mix_contact_groups", type=str, nargs="+", default=None,
                    help="Contact label mix groups as colon-separated motion names, e.g. 'motionA:motionB'.")
parser.add_argument("--mix_contact_prob", type=float, default=0.0, help="Probability of mixing contact labels between group members.")
parser.add_argument("--terrain_mode", type=str, default=None, choices=["flat", "rough", "mixed"], help="Override terrain mode (flat/rough/mixed).")
parser.add_argument("--terrain_noise", type=float, default=None, help="Half-range of terrain height noise in meters (e.g. 0.05 for ±5cm).")
parser.add_argument("--no_contact", action="store_true", default=False, help="Disable all contact observations and rewards.")
parser.add_argument("--contact_peer", type=str, default=None,
                    help="Motion name whose contact labels override the primary motion. "
                         "All envs follow --motion_name keypoints but use this peer's contact labels.")
parser.add_argument("--compute_end", type=float, default=None,
                    help="If set, metrics are computed only from the last N seconds of each episode.")
parser.add_argument("--stand_ratio", type=float, default=None, help="Fraction of envs frozen at frame 0 (stand-still) for curriculum.")
parser.add_argument("--start_from_zero_ratio", type=float, default=None, help="Per-reset probability that an env starts at frame 0.")
parser.add_argument("--no_randomize", action="store_true", default=True, help="Disable all domain randomization (DOF offsets, friction, COM, push). Default: True.")
parser.add_argument("--randomize", dest="no_randomize", action="store_false", help="Enable domain randomization during play.")
parser.add_argument("--chair_offset_cfg", type=str, default=None, help="Path to YAML file with per-motion chair XYZ offsets, e.g. motion_name: {x: 0.1, y: 0.0, z: 0.05}.")
parser.add_argument("--random_chair_y", type=float, default=None, help="Std (meters) of Gaussian noise on chair Y position at each reset (e.g. 0.05 for ~±5 cm).")
parser.add_argument("--remove_object_prob", type=float, default=None, help="Per-reset probability of removing the motion object (sending it underground) for each env (e.g. 0.2).")
parser.add_argument("--mask_contact_prob", type=float, default=None, help="Per-part probability of randomly masking each contact part (seat/backrest/legs) at each reset (e.g. 0.3).")
parser.add_argument("--random_mask_contact_parts", type=int, nargs="+", default=None,
                    help="Explicit part IDs eligible for random masking (e.g. 4 for board/table, 1 for shelf). "
                         "Overrides --mask_contact_prob's hardcoded [1,2,3].")
parser.add_argument("--random_mask_contact_prob", type=float, default=None,
                    help="Probability used with --random_mask_contact_parts (whole=False).")
parser.add_argument("--contact_tp_fp_mode", action="store_true", default=False, help="Use TP-FP mode for contact_label_match: reward=TPR - fp_penalty*FPR. Better for sparse-contact motions.")
parser.add_argument("--contact_fp_penalty", type=float, default=None, help="False positive penalty weight for --contact_tp_fp_mode (default: 1.0).")
parser.add_argument("--video_suffix", type=str, default=None, help="Suffix appended to the video filename prefix (e.g. 'base' → '<motion>_base-episode-0.mp4').")
parser.add_argument("--onnx_suffix", type=str, default=None, help="Suffix for the exported ONNX filename (e.g. 'mixed' → 'policy_mixed.onnx').")
parser.add_argument("--metrics_out", type=str, default=None, help="Path to save metric_series dict as a .pt file for later comparison (e.g. /tmp/base_metrics.pt).")
parser.add_argument("--mask_back_after", type=float, default=None,
                    help="Switch: seconds from sim start after which backrest contact (part 2) is masked. "
                         "When set, also auto-saves pre/post split metrics alongside --metrics_out.")
parser.add_argument("--target_contact_bodies", type=str, nargs="+", default=None,
                    help="Robot body names considered the 'target' contact. The Target contact impulse/body "
                         "count metrics count paired-sensor contacts on these bodies (paired sensor already "
                         "filters to motion_object_0, so this measures body-on-target-object contact).")
parser.add_argument("--video_dir_suffix", type=str, default=None,
                    help="If set, videos go to videos/play_{suffix}/ (and metric plots use the same dir).")
parser.add_argument("--probe_contact", action="store_true", default=False,
                    help="Record actor MLP hidden activations + ground-truth contact for linear probe analysis.")
parser.add_argument("--probe_out", type=str, default=None,
                    help="Path to save probe dataset (.pt). Required when --probe_contact is set.")
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
import pathlib
import re
import torch

from rsl_rl.runners import OnPolicyRunner

_HUMOTO_BASE = os.environ.get(
    "HUMOTO_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "HUMOTO")),
)


def discover_object_urdfs(motion_names: list[str]) -> list[str | None]:
    """Auto-discover per-motion object URDFs from HUMOTO directory by convention."""
    _SPEED_SUFFIX_RE = re.compile(r"[-_](fast|slow|\d+hz|contact|nocontact)$", re.IGNORECASE)
    urdfs: list[str | None] = []
    for name in motion_names:
        # Try HUMOTO/{base}/multi_boxes_scaled_0.88_0.88_0.88.urdf, stripping speed suffixes
        # First try exact match with 3-digit ID pattern (e.g. "some_motion-123")
        m = re.search(r"([a-zA-Z_]+-\d{3})", name)
        base = m.group(1) if m else name
        found = None
        while True:
            urdf_path = os.path.join(_HUMOTO_BASE, base, "multi_boxes_scaled_0.88_0.88_0.88.urdf")
            if os.path.exists(urdf_path):
                found = os.path.abspath(urdf_path)
                break
            stripped = _SPEED_SUFFIX_RE.sub("", base)
            if stripped == base:
                break
            base = stripped
        urdfs.append(found)
    return urdfs

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.tasks.tracking.mdp.rewards import classify_object_part


def _quat_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert wxyz quaternion to roll/pitch (radians)."""
    w, x, y, z = quat.unbind(-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


def _get_joint_effort(robot) -> torch.Tensor | None:
    data = robot.data
    for name in (
        "applied_torque",
        "joint_effort",
        "joint_efforts",
        "joint_torque",
        "joint_torques",
        "joint_applied_torque",
        "applied_joint_torque",
        "joint_applied_effort",
    ):
        if hasattr(data, name):
            return getattr(data, name)
    return None


def _get_com_pos_w(robot) -> torch.Tensor | None:
    data = robot.data
    for name in ("com_pos_w", "root_com_pos_w", "base_com_pos_w", "root_com_pos", "com_pos"):
        if hasattr(data, name):
            return getattr(data, name)
    return None


def _find_joint_indices(robot, preferred_names: list[str], fallback_substr: str | None = None) -> list[int]:
    names = []
    if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
        names = list(robot.data.joint_names)
    elif hasattr(robot, "joint_names"):
        names = list(robot.joint_names)
    if not names:
        return []
    idx = [i for i, n in enumerate(names) if n in preferred_names]
    if not idx and fallback_substr is not None:
        idx = [i for i, n in enumerate(names) if fallback_substr in n]
    return idx


def _find_body_index(body_names: list[str], preferred_names: list[str]) -> int | None:
    for name in preferred_names:
        if name in body_names:
            return body_names.index(name)
    return None


def _mask_failed(metric_list: list[torch.Tensor], fail_mask: torch.Tensor) -> None:
    if not metric_list:
        return
    last = metric_list[-1]
    if last.shape != fail_mask.shape:
        return
    masked = last.clone()
    masked[fail_mask] = float("nan")
    metric_list[-1] = masked


def _apply_compute_end(
    stacked: torch.Tensor,
    episode_bounds: list[list[tuple[int, int]]],
    keep_steps: int,
) -> torch.Tensor:
    """Zero-out (NaN) steps that are NOT in the last `keep_steps` of each episode.

    Args:
        stacked: shape (T, num_envs) or (T, num_envs, ...).
        episode_bounds: per-env list of (start, end) step indices (inclusive).
        keep_steps: number of trailing steps to keep per episode.
    Returns:
        Copy of stacked with early steps set to NaN.
    """
    T = stacked.shape[0]
    num_envs = stacked.shape[1]
    keep_mask = torch.zeros(T, num_envs, dtype=torch.bool)
    for env_id, bounds in enumerate(episode_bounds):
        for start, end in bounds:
            tail_start = max(start, end - keep_steps + 1)
            keep_mask[tail_start : end + 1, env_id] = True
    out = stacked.clone().float()
    # broadcast mask over trailing dims
    while keep_mask.dim() < out.dim():
        keep_mask = keep_mask.unsqueeze(-1)
    out[~keep_mask.expand_as(out)] = float("nan")
    return out


def _nanmean(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    valid = ~torch.isnan(x)
    if dim is None:
        if not valid.any():
            return torch.tensor(float("nan"), device=x.device)
        return x[valid].mean()
    summed = torch.where(valid, x, torch.zeros_like(x)).sum(dim=dim)
    counts = valid.sum(dim=dim)
    mean = summed / torch.clamp(counts, min=1)
    mean = torch.where(counts > 0, mean, torch.full_like(mean, float("nan")))
    return mean


def _nanstd(x: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(x)
    if not valid.any():
        return torch.tensor(float("nan"), device=x.device)
    vals = x[valid]
    mean = vals.mean()
    return torch.sqrt(((vals - mean) ** 2).mean())


def _slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _plot_metric_series(
    name: str,
    data: torch.Tensor,
    episode_bounds: list[list[tuple[int, int]]],
    out_dir: str,
    suffix: str,
    step_dt: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    steps, num_envs = data.shape
    if steps == 0:
        return
    t = (torch.arange(steps).cpu().numpy() * step_dt).tolist()
    fig_h = max(2.2 * num_envs, 2.5)
    fig, axes = plt.subplots(num_envs, 1, figsize=(12, fig_h), sharex=True)
    if num_envs == 1:
        axes = [axes]
    for env_id, ax in enumerate(axes):
        y = data[:, env_id].cpu().numpy()
        ax.plot(t, y, linewidth=1.0)
        for (s, e) in episode_bounds[env_id]:
            ax.axvline(s * step_dt, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.axvline(e * step_dt, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_ylabel(f"env{env_id}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(name)
    legend_elems = [
        Line2D([0], [0], color="green", linestyle="--", label="episode start"),
        Line2D([0], [0], color="red", linestyle="--", label="episode end"),
    ]
    axes[0].legend(handles=legend_elems, loc="upper right", frameon=False)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_slugify(name)}{suffix}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.start_from_zero:
        env_cfg.commands.motion.start_from_zero = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # Load motion from --motion_name if provided
    motion_names = []
    if args_cli.motion_name:
        import pathlib
        import wandb

        os.environ["WANDB_ENTITY"] = args_cli.wandb_entity
        api = wandb.Api()

        motion_files = []
        motion_names = args_cli.motion_name
        for name in motion_names:
            if os.path.exists(name):
                motion_files.append(name)
            else:
                version = "latest" if ":" not in name else name.split(":")[-1]
                base_name = name.split(":")[0] if ":" in name else name
                registry_name = f"{args_cli.wandb_entity}/{args_cli.wandb_artifact_project}/{base_name}:{version}"
                print(f"[INFO]: Loading motion artifact: {registry_name}")
                artifact = api.artifact(registry_name, type="motions")
                motion_files.append(str(pathlib.Path(artifact.download()) / "motion.npz"))

        # Always set motion_names so chair_offset_per_motion (and similar yaml lookups)
        # can resolve the user-supplied motion name even in single-motion runs (where
        # the artifact file's basename is just 'motion').
        env_cfg.commands.motion.motion_names = motion_names
        if len(motion_files) == 1:
            env_cfg.commands.motion.motion_file = motion_files[0]
        else:
            env_cfg.commands.motion.motion_file = motion_files[0]
            env_cfg.commands.motion.motion_files = motion_files
            print(f"[INFO]: Multi-motion play with {len(motion_files)} motions: {motion_names}")

        # --contact_peer: append peer motion so multi_motion=True, pin all envs to primary
        if args_cli.contact_peer is not None:
            peer_name = args_cli.contact_peer
            if os.path.exists(peer_name):
                peer_file = peer_name
            else:
                import wandb as _wandb
                _api = _wandb.Api()
                _version = "latest" if ":" not in peer_name else peer_name.split(":")[-1]
                _base = peer_name.split(":")[0] if ":" in peer_name else peer_name
                _reg = f"{args_cli.wandb_entity}/{args_cli.wandb_artifact_project}/{_base}:{_version}"
                print(f"[INFO]: Loading contact peer artifact: {_reg}")
                _art = _api.artifact(_reg, type="motions")
                peer_file = str(pathlib.Path(_art.download()) / "motion.npz")
            # Primary is always index 0; peer is index 1
            motion_files = [motion_files[0], peer_file]
            motion_names = [motion_names[0], peer_name]
            env_cfg.commands.motion.motion_file = motion_files[0]
            env_cfg.commands.motion.motion_files = motion_files
            env_cfg.commands.motion.motion_names = motion_names
            env_cfg.commands.motion.force_contact_peer_idx = 1
            print(f"[INFO]: Force-peer mode: keypoints from '{motion_names[0]}', "
                  f"contact from '{peer_name}'")

        # Discover per-motion object URDFs and configure scene objects
        object_urdfs = discover_object_urdfs(motion_names)
        env_cfg.commands.motion.object_urdfs = object_urdfs
        if hasattr(env_cfg, "configure_motion_objects") and any(u is not None for u in object_urdfs):
            env_cfg.configure_motion_objects(object_urdfs)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Disable all contact obs and rewards
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

    # Mask backrest contact (part 2)
    if args_cli.mask_back:
        env_cfg.commands.motion.ignore_contact_parts = [2]

    # Mix contact labels between motion groups
    if args_cli.mix_contact_groups and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.mix_contact_labels = [g.split(":") for g in args_cli.mix_contact_groups]
        env_cfg.commands.motion.mix_contact_label_prob = args_cli.mix_contact_prob

    # Disable position/orientation terminations during play
    if args_cli.no_termi:
        env_cfg.terminations.anchor_pos.params["threshold"] = 1e6
        env_cfg.terminations.anchor_ori.params["threshold"] = 1e6
        env_cfg.terminations.ee_body_pos.params["threshold"] = 1e6

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
    # Explicit parts override (per-part mask, whole=False)
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

    # Disable domain randomization during play (default behaviour)
    if args_cli.no_randomize and hasattr(env_cfg, "events"):
        for attr in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            if hasattr(env_cfg.events, attr):
                setattr(env_cfg.events, attr, None)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        # Use motion name(s) as prefix to avoid overwriting
        if args_cli.motion_name:
            if len(args_cli.motion_name) == 1:
                prefix = args_cli.motion_name[0]
            else:
                prefix = f"multi-{len(args_cli.motion_name)}motions"
        else:
            prefix = "rl-video"
        if args_cli.video_suffix:
            prefix = f"{prefix}_{args_cli.video_suffix}"
        _video_subdir = f"play_{args_cli.video_dir_suffix}" if args_cli.video_dir_suffix else "play"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", _video_subdir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": prefix,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ── Probe contact: register forward hooks on actor MLP hidden layers ──
    _probe_activations: dict[str, torch.Tensor] = {}
    _probe_hooks = []
    if args_cli.probe_contact:
        assert args_cli.probe_out, "--probe_out is required when --probe_contact is set"
        actor_mlp = ppo_runner.alg.policy.actor
        for idx, module in enumerate(actor_mlp):
            if isinstance(module, torch.nn.Linear):
                layer_name = f"linear_{idx}"
                def _make_hook(name):
                    def hook(mod, inp, out):
                        _probe_activations[name] = out.detach().cpu()
                    return hook
                _probe_hooks.append(module.register_forward_hook(_make_hook(layer_name)))
        print(f"[PROBE] Registered {len(_probe_hooks)} hooks on actor MLP layers")
    probe_obs_list = []
    probe_hidden_lists: dict[str, list] = {}
    probe_gt_contact_list = []
    probe_ref_contact_list = []

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    onnx_filename = f"policy_{args_cli.onnx_suffix}.onnx" if args_cli.onnx_suffix else "policy.onnx"
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename=onnx_filename,
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir, filename=onnx_filename)
    # setup contact visualization
    from isaaclab.markers import VisualizationMarkersCfg
    from isaaclab.sim.spawners.shapes import SphereCfg
    from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
    from isaaclab.markers import VisualizationMarkers

    contact_sensor = env.unwrapped.scene["contact_forces"]
    num_bodies = contact_sensor.data.net_forces_w.shape[1]
    num_envs = env.unwrapped.num_envs
    max_markers = num_envs * num_bodies

    contact_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactMarkers",
        markers={
            "sphere": SphereCfg(
                radius=0.03,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    contact_markers = VisualizationMarkers(contact_marker_cfg)

    # Setup reference contact point visualization (yellow spheres)
    command = env.unwrapped.command_manager.get_term("motion")
    if command.has_contact:
        # K_max contact points per env
        if command.multi_motion:
            K_max = command.motion_lib.contact_chair_pts.shape[-2]
        else:
            K_max = command.motion.contact_chair_pts.shape[1]
        ref_contact_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/RefContactMarkers",
            markers={
                "sphere": SphereCfg(
                    radius=0.08,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        ref_contact_markers = VisualizationMarkers(ref_contact_marker_cfg)
    else:
        K_max = 0

    # Precompute indices for metrics
    robot = env.unwrapped.scene["robot"]
    waist_joint_names = ["waist_pitch_joint"]
    waist_joint_ids = _find_joint_indices(robot, waist_joint_names, fallback_substr="waist")
    body_names = list(command.cfg.body_names)
    torso_body_idx = _find_body_index(body_names, ["torso_link", "Torso", "torso"])
    pelvis_body_idx = _find_body_index(body_names, ["pelvis", "Pelvis"])
    step_dt = float(getattr(env, "step_dt", getattr(env.unwrapped, "step_dt", 1.0)))

    # reset environment
    obs, _ = env.get_observations()
    step_idx = 0
    episode_start = torch.zeros(num_envs, dtype=torch.long)
    episode_bounds: list[list[tuple[int, int]]] = [[] for _ in range(num_envs)]

    # Contact-label switch state (--mask_back_after)
    _mask_back_active = False
    _switch_step: int | None = None

    # Metric accumulators
    from isaaclab.utils.math import quat_error_magnitude
    metric_anchor_pos_err = []
    metric_anchor_rot_err = []
    metric_body_pos_err = []
    metric_contact_match = []  # per-step contact accuracy
    metric_waist_pitch_tau = []
    metric_waist_power = []
    metric_total_power = []
    metric_per_joint_tau = []   # (steps, num_envs, num_joints) — SIGNED torque per joint
    metric_per_joint_power = [] # (steps, num_envs, num_joints) — abs power per joint
    metric_body_pos_err_per_link = []  # (steps, num_envs, num_bodies) — per-link pos err
    metric_body_pos_err_root_rel = []  # (steps, num_envs) — root-relative MPJPE
    metric_target_impulse = []
    metric_target_contact_time = []
    metric_com_proj_offset = []
    metric_com_x = []
    metric_com_y = []
    metric_torso_osc = []
    metric_torso_pitch = []  # signed mean pitch; negative = lean-back (backrest contact mode)
    metric_contact_label_match_reward = []  # weighted reward value from reward_manager
    metric_contact_tp = []  # true positives per env per step
    metric_contact_tn = []  # true negatives per env per step
    metric_obj_disp_xy = []  # object XY displacement from episode start (m)
    _obj_start_xy: torch.Tensor | None = None  # (num_envs, 2) — reset at episode boundaries
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Contact-label switch: mask backrest once elapsed >= mask_back_after
            if args_cli.mask_back_after is not None and not _mask_back_active:
                elapsed = step_idx * step_dt
                if elapsed >= args_cli.mask_back_after:
                    _mask_back_active = True
                    _switch_step = step_idx
                    if command.has_contact:
                        command.force_mask_contact_parts([2])
                    print(f"[SWITCH] t={elapsed:.1f}s (step {step_idx}) — backrest contact (part 2) masked")

            # agent stepping
            actions = policy(obs)

            # Cache pre-step EE errors (before env.step resets terminated envs)
            ee_names = ["left_ankle_roll_link", "right_ankle_roll_link",
                        "left_wrist_yaw_link", "right_wrist_yaw_link"]
            ee_idxs = [command.cfg.body_names.index(n) for n in ee_names if n in command.cfg.body_names]
            pre_step_ee_err = torch.abs(
                command.body_pos_relative_w[:, ee_idxs, -1]
                - command.robot_body_pos_w[:, ee_idxs, -1]
            ).clone()  # (num_envs, 4)

            # Collect metrics (pre-step, before reset)
            anchor_pos_err = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
            anchor_rot_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
            body_pos_err_per_link = torch.norm(
                command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
            )  # (num_envs, num_bodies)
            body_pos_err = body_pos_err_per_link.mean(dim=-1)
            # Root-relative body pos error (MPJPE): subtract pelvis pos from both ref & sim
            if pelvis_body_idx is not None:
                ref_root = command.body_pos_relative_w[:, pelvis_body_idx:pelvis_body_idx+1, :]
                sim_root = command.robot_body_pos_w[:, pelvis_body_idx:pelvis_body_idx+1, :]
                ref_root_rel = command.body_pos_relative_w - ref_root
                sim_root_rel = command.robot_body_pos_w - sim_root
                mpjpe = torch.norm(ref_root_rel - sim_root_rel, dim=-1).mean(dim=-1)
            else:
                mpjpe = body_pos_err  # fallback
            metric_anchor_pos_err.append(anchor_pos_err.cpu())
            metric_anchor_rot_err.append(anchor_rot_err.cpu())
            metric_body_pos_err.append(body_pos_err.cpu())
            metric_body_pos_err_per_link.append(body_pos_err_per_link.cpu())
            metric_body_pos_err_root_rel.append(mpjpe.cpu())

            # env stepping (triggers reset for terminated envs)
            obs, _, _, _ = env.step(actions)

            # contact_label_match reward value (from reward_manager, post-step)
            reward_mgr = env.unwrapped.reward_manager
            if "contact_label_match" in reward_mgr.active_terms:
                term_idx = reward_mgr._term_names.index("contact_label_match")
                clm_reward = reward_mgr._step_reward[:, term_idx].cpu()  # (num_envs,) weighted value
                metric_contact_label_match_reward.append(clm_reward)

            # Extra metrics (post-step state)
            joint_effort = _get_joint_effort(robot)
            joint_vel = robot.data.joint_vel
            if joint_effort is not None:
                total_power = torch.sum((joint_effort * joint_vel).abs(), dim=-1)
                metric_total_power.append(total_power.cpu())
                metric_per_joint_tau.append(joint_effort.cpu())  # SIGNED
                metric_per_joint_power.append((joint_effort * joint_vel).abs().cpu())
                if waist_joint_ids:
                    waist_tau = joint_effort[:, waist_joint_ids].squeeze(-1)  # (num_envs,)
                    waist_power = (waist_tau * joint_vel[:, waist_joint_ids].squeeze(-1)).abs()
                    metric_waist_pitch_tau.append(waist_tau.cpu())  # signed: negative=extension(lean-back), positive=flexion
                    metric_waist_power.append(waist_power.cpu())

            # COM projection offset (XY) relative to active chair/object origin
            com_pos = _get_com_pos_w(robot)
            if com_pos is None and pelvis_body_idx is not None:
                com_pos = command.robot_body_pos_w[:, pelvis_body_idx]
            if com_pos is not None:
                if command.has_objects and command.active_object_pos_w is not None:
                    obj_pos = command.active_object_pos_w
                else:
                    obj_pos = env.unwrapped.scene.env_origins
                com_xy = com_pos[:, :2]
                obj_xy = obj_pos[:, :2]
                com_offset = torch.norm(com_xy - obj_xy, dim=-1)
                metric_com_proj_offset.append(com_offset.cpu())
                metric_com_x.append(com_pos[:, 0].cpu())
                metric_com_y.append(com_pos[:, 1].cpu())

            # Termination manager (needed by object displacement and later sections)
            term_mgr = env.unwrapped.termination_manager

            # Object XY displacement from episode start (for free-object motions)
            if command.has_objects and command.active_object_pos_w is not None:
                obj_xy = command.active_object_pos_w[:, :2].clone()  # (num_envs, 2)
                if _obj_start_xy is None:
                    _obj_start_xy = obj_xy.clone()
                # Reset start position for envs that just started a new episode
                if term_mgr.terminated.any():
                    _obj_start_xy[term_mgr.terminated] = obj_xy[term_mgr.terminated]
                disp = torch.norm(obj_xy - _obj_start_xy, dim=-1)  # (num_envs,)
                metric_obj_disp_xy.append(disp.cpu())

            # Torso tilt metrics
            if torso_body_idx is not None:
                torso_quat = command.robot_body_quat_w[:, torso_body_idx]
                roll, pitch = _quat_to_roll_pitch(torso_quat)
                # RMS tilt magnitude (sign-free, jitter-sensitive — kept for stability comparison)
                torso_osc = torch.sqrt(0.5 * (roll**2 + pitch**2))
                metric_torso_osc.append(torso_osc.cpu())
                # Signed pitch: negative = lean-back (backrest contact); positive = lean-forward
                # Mean over episode averages out jitter since jitter is zero-mean
                metric_torso_pitch.append(pitch.cpu())

            # Debug: print termination reasons using pre-step cached errors
            failed_envs = term_mgr.terminated & ~term_mgr.time_outs
            if term_mgr.terminated.any():
                for name in term_mgr.active_terms:
                    dones = term_mgr._term_dones[name]
                    if dones.any():
                        ids = torch.where(dones)[0].tolist()
                        if name == "ee_body_pos":
                            for env_id in ids:
                                bad = [ee_names[j] for j, e in enumerate(pre_step_ee_err[env_id]) if e > 0.25]
                                print(f"[TERM] step={step_idx} {name}: env {env_id} -> {bad} "
                                      f"(err={pre_step_ee_err[env_id].tolist()})")
                        else:
                            print(f"[TERM] step={step_idx} {name}: envs {ids}")

            # # Draw red spheres at bodies with contact forces (disabled)
            # forces = contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
            # force_mag = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
            # robot = env.unwrapped.scene["robot"]
            # body_pos = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
            # marker_pos = body_pos.reshape(-1, 3).clone()
            # contact_mask = force_mag.reshape(-1) > 1.0
            # marker_pos[~contact_mask] = torch.tensor([0.0, 0.0, -100.0], device=marker_pos.device)
            # marker_indices = torch.zeros(marker_pos.shape[0], dtype=torch.long, device=marker_pos.device)
            # contact_markers.visualize(marker_pos, marker_indices=marker_indices)

            # Draw yellow spheres at robot body positions that have actual chair contact
            if command.has_contact:
                body_ids = command.contact_body_ids  # (N_valid,)
                robot = env.unwrapped.scene["robot"]
                contact_body_pos = robot.data.body_pos_w[:, body_ids]  # (num_envs, N_valid, 3)
                if command.use_paired_contact:
                    # Paired sensors: chair-only contact forces
                    forces = command.get_paired_chair_contact_forces()  # (num_envs, N_valid, 3)
                else:
                    forces = contact_sensor.data.net_forces_w[:, body_ids]  # (num_envs, N_valid, 3)
                has_contact = torch.norm(forces, dim=-1) > 1.0  # (num_envs, N_valid)
                force_mag = torch.norm(forces, dim=-1)  # (num_envs, N_valid)
                marker_pos = contact_body_pos.reshape(-1, 3).clone()
                marker_pos[~has_contact.reshape(-1)] = torch.tensor([0.0, 0.0, -100.0], device=marker_pos.device)
                marker_indices = torch.zeros(marker_pos.shape[0], dtype=torch.long, device=marker_pos.device)
                ref_contact_markers.visualize(marker_pos, marker_indices=marker_indices)

                # Accumulate contact matching metric
                expected_labels = command.current_contact_labels  # (num_envs, N_valid)
                match = (has_contact == expected_labels).float().mean(dim=-1)  # (num_envs,)
                metric_contact_match.append(match.cpu())

                # Z-based part classification, using per-env geometry from object_geometry.yaml
                # (command.active_obj_*) — replaces the legacy hardcoded chair_z_range=0.58 which
                # was wrong for objects like low_chair (z_range=1.12) and dining_chair (z_range=1.35).
                if command.has_objects and command.active_object_pos_w is not None:
                    obj_pos = command.active_object_pos_w
                else:
                    obj_pos = env.unwrapped.scene.env_origins
                body_z = robot.data.body_pos_w[:, body_ids, 2]  # (num_envs, N)
                obj_z = obj_pos[:, 2:3]
                actual_part = classify_object_part(
                    body_z, obj_z,
                    command.active_obj_z_min, command.active_obj_z_range,
                    command.active_obj_thresholds, command.active_obj_segmented,
                )

                # TP/TN — exactly mirrors contact_label_match reward function
                expected_part = command.current_contact_part_labels  # (num_envs, N) long: 0/1/2/3
                active = expected_part > 0
                actual_part_tp = actual_part.clone()
                actual_part_tp[~has_contact] = 0  # no contact → part 0
                tp = active & (expected_part == actual_part_tp)
                tn = ~active & ~has_contact
                metric_contact_tp.append(tp.float().sum(dim=-1).cpu())  # (num_envs,)
                metric_contact_tn.append(tn.float().sum(dim=-1).cpu())  # (num_envs,)

                # Target contact: sim has_contact at target bodies, on the active object.
                # Paired sensor already filters forces to the primary motion_object_0 (chair/table/board/...).
                # We additionally filter by target_contact_bodies (the "focus pair" — e.g. wrist, torso, foot).
                # NPZ expected_part is intentionally NOT used: the source data sometimes labels contact on
                # bodies (e.g. sphere_hand_link) that aren't in PAIRED_CONTACT_BODIES, causing the
                # expected_part filter to drop legitimate contacts. Pure body-name filtering measures
                # "actual physical contact between target body and target object" regardless of label.
                target_mask = has_contact.clone()
                if args_cli.target_contact_bodies:
                    _robot_body_names = robot.data.body_names
                    contact_body_names = [_robot_body_names[bid] for bid in body_ids.cpu().tolist()]
                    body_keep = torch.zeros(len(contact_body_names), dtype=torch.bool, device=target_mask.device)
                    for j, bn in enumerate(contact_body_names):
                        if bn in args_cli.target_contact_bodies:
                            body_keep[j] = True
                    target_mask = target_mask & body_keep.unsqueeze(0)
                target_force_mag = force_mag * target_mask
                target_impulse = target_force_mag.sum(dim=-1) * step_dt
                target_contact_count = target_mask.sum(dim=-1).float()
                metric_target_impulse.append(target_impulse.cpu())
                metric_target_contact_time.append(target_contact_count.cpu())

                # ── Probe contact: collect hidden activations + ground-truth ──
                if args_cli.probe_contact:
                    probe_obs_list.append(obs.cpu())
                    for name, act in _probe_activations.items():
                        probe_hidden_lists.setdefault(name, []).append(act)
                    probe_gt_contact_list.append(has_contact.cpu())
                    probe_ref_contact_list.append(expected_labels.cpu())

            # Exclude failed episodes from metrics (ignore timeouts)
            if failed_envs.any():
                fail_mask = failed_envs.detach().cpu().bool()
                _mask_failed(metric_anchor_pos_err, fail_mask)
                _mask_failed(metric_anchor_rot_err, fail_mask)
                _mask_failed(metric_body_pos_err, fail_mask)
                _mask_failed(metric_body_pos_err_per_link, fail_mask)
                _mask_failed(metric_body_pos_err_root_rel, fail_mask)
                _mask_failed(metric_contact_match, fail_mask)
                _mask_failed(metric_waist_pitch_tau, fail_mask)
                _mask_failed(metric_waist_power, fail_mask)
                _mask_failed(metric_total_power, fail_mask)
                _mask_failed(metric_per_joint_tau, fail_mask)
                _mask_failed(metric_per_joint_power, fail_mask)
                _mask_failed(metric_target_impulse, fail_mask)
                _mask_failed(metric_target_contact_time, fail_mask)
                _mask_failed(metric_com_proj_offset, fail_mask)
                _mask_failed(metric_com_x, fail_mask)
                _mask_failed(metric_com_y, fail_mask)
                _mask_failed(metric_torso_osc, fail_mask)
                _mask_failed(metric_torso_pitch, fail_mask)
                _mask_failed(metric_contact_label_match_reward, fail_mask)
                _mask_failed(metric_contact_tp, fail_mask)
                _mask_failed(metric_contact_tn, fail_mask)
                _mask_failed(metric_obj_disp_xy, fail_mask)

            # Episode boundaries (after env.step)
            done_envs = (term_mgr.terminated | term_mgr.time_outs).detach().cpu().bool()
            if done_envs.any():
                for env_id in torch.where(done_envs)[0].tolist():
                    episode_bounds[env_id].append((int(episode_start[env_id]), int(step_idx)))
                    episode_start[env_id] = step_idx + 1

        if args_cli.video:
            step_idx += 1
            # Exit the play loop after recording one video
            if step_idx == args_cli.video_length:
                break
        else:
            step_idx += 1

    # Close out last episodes
    last_step = step_idx - 1
    if last_step >= 0:
        for env_id in range(num_envs):
            if episode_start[env_id] <= last_step:
                episode_bounds[env_id].append((int(episode_start[env_id]), int(last_step)))

    # Trim metrics to last N seconds of each episode if --compute_end is set
    if args_cli.compute_end is not None:
        keep_steps = max(1, int(round(args_cli.compute_end / step_dt)))
        print(f"[INFO] --compute_end={args_cli.compute_end}s → keeping last {keep_steps} steps per episode")

        def _trim(lst):
            if not lst:
                return lst
            stacked = torch.stack(lst)
            trimmed = _apply_compute_end(stacked, episode_bounds, keep_steps)
            return list(trimmed.unbind(0))

        metric_anchor_pos_err = _trim(metric_anchor_pos_err)
        metric_anchor_rot_err = _trim(metric_anchor_rot_err)
        metric_body_pos_err = _trim(metric_body_pos_err)
        metric_body_pos_err_per_link = _trim(metric_body_pos_err_per_link)
        metric_body_pos_err_root_rel = _trim(metric_body_pos_err_root_rel)
        metric_contact_match = _trim(metric_contact_match)
        metric_waist_pitch_tau = _trim(metric_waist_pitch_tau)
        metric_waist_power = _trim(metric_waist_power)
        metric_total_power = _trim(metric_total_power)
        metric_per_joint_tau = _trim(metric_per_joint_tau)
        metric_per_joint_power = _trim(metric_per_joint_power)
        metric_target_impulse = _trim(metric_target_impulse)
        metric_target_contact_time = _trim(metric_target_contact_time)
        metric_com_proj_offset = _trim(metric_com_proj_offset)
        metric_com_x = _trim(metric_com_x)
        metric_com_y = _trim(metric_com_y)
        metric_torso_osc = _trim(metric_torso_osc)
        metric_torso_pitch = _trim(metric_torso_pitch)
        metric_contact_label_match_reward = _trim(metric_contact_label_match_reward)
        metric_contact_tp = _trim(metric_contact_tp)
        metric_contact_tn = _trim(metric_contact_tn)

    # Print metrics summary
    print("\n" + "=" * 60)
    print("PLAY METRICS SUMMARY")
    print("=" * 60)
    if metric_anchor_pos_err:
        all_anchor_pos = torch.stack(metric_anchor_pos_err)  # (steps, num_envs)
        all_anchor_rot = torch.stack(metric_anchor_rot_err)
        all_body_pos = torch.stack(metric_body_pos_err)
        print(f"Anchor pos error (m):   mean={_nanmean(all_anchor_pos):.4f}  std={_nanstd(all_anchor_pos):.4f}")
        print(f"Anchor rot error (rad): mean={_nanmean(all_anchor_rot):.4f}  std={_nanstd(all_anchor_rot):.4f}")
        print(f"Body pos error (m):     mean={_nanmean(all_body_pos):.4f}  std={_nanstd(all_body_pos):.4f}")
        if metric_body_pos_err_root_rel:
            all_body_pos_rr = torch.stack(metric_body_pos_err_root_rel)
            print(f"Body pos err root-rel (m): mean={_nanmean(all_body_pos_rr):.4f}  std={_nanstd(all_body_pos_rr):.4f}")
        # Per-env breakdown
        print(f"\nPer-env anchor pos error (m): {_nanmean(all_anchor_pos, dim=0).tolist()}")
        print(f"Per-env body pos error (m):   {_nanmean(all_body_pos, dim=0).tolist()}")
    if metric_contact_match:
        all_contact = torch.stack(metric_contact_match)  # (steps, num_envs)
        print(f"Contact match accuracy:  mean={_nanmean(all_contact):.4f}  std={_nanstd(all_contact):.4f}")
        print(f"Per-env contact match:        {_nanmean(all_contact, dim=0).tolist()}")
    if metric_contact_label_match_reward:
        all_clm_rew = torch.stack(metric_contact_label_match_reward)  # (steps, num_envs)
        print(f"contact_label_match reward:  mean={_nanmean(all_clm_rew):.4f}  std={_nanstd(all_clm_rew):.4f}")
        print(f"Per-env contact_label_match: {_nanmean(all_clm_rew, dim=0).tolist()}")
    if metric_contact_tp:
        all_tp = torch.stack(metric_contact_tp)  # (steps, num_envs)
        all_tn = torch.stack(metric_contact_tn)
        print(f"contact_label_match TP:      mean={_nanmean(all_tp):.4f}  std={_nanstd(all_tp):.4f}")
        print(f"Per-env TP:                  {_nanmean(all_tp, dim=0).tolist()}")
        print(f"contact_label_match TN:      mean={_nanmean(all_tn):.4f}  std={_nanstd(all_tn):.4f}")
        print(f"Per-env TN:                  {_nanmean(all_tn, dim=0).tolist()}")
    if metric_waist_pitch_tau:
        all_waist_pitch_tau = torch.stack(metric_waist_pitch_tau)
        print(f"Waist pitch torque (Nm):  mean={_nanmean(all_waist_pitch_tau):.4f}  std={_nanstd(all_waist_pitch_tau):.4f}")
    if metric_waist_power:
        all_waist_power = torch.stack(metric_waist_power)
        print(f"Waist pitch power |tau*qdot| (W): mean={_nanmean(all_waist_power):.4f}  std={_nanstd(all_waist_power):.4f}")
    if metric_total_power:
        all_total_power = torch.stack(metric_total_power)
        print(f"Total power |tau*qdot| (W): mean={_nanmean(all_total_power):.4f}  std={_nanstd(all_total_power):.4f}")
    if metric_per_joint_tau:
        all_pjt = torch.stack(metric_per_joint_tau)  # (steps, num_envs, num_joints) — SIGNED
        all_pjp = torch.stack(metric_per_joint_power)
        joint_names = list(robot.data.joint_names)
        mean_tau = all_pjt.float().nanmean(dim=(0, 1))   # signed mean
        # Signed peaks: most-positive and most-negative across time/env
        flat = all_pjt.float().reshape(-1, all_pjt.shape[-1])  # (steps*envs, num_joints)
        peak_pos_tau = flat.nan_to_num(nan=float("-inf")).max(dim=0).values
        peak_neg_tau = flat.nan_to_num(nan=float("inf")).min(dim=0).values
        mean_pow = all_pjp.float().nanmean(dim=(0, 1))
        col_w = max(len(n) for n in joint_names)
        print(f"\n{'Joint':<{col_w}}  {'tau mean (Nm)':>16}  {'tau peak+ (Nm)':>16}  {'tau peak- (Nm)':>16}  {'|power| mean (W)':>16}")
        print("-" * (col_w + 76))
        order = mean_pow.argsort(descending=True)
        for i in order.tolist():
            print(f"{joint_names[i]:<{col_w}}  {mean_tau[i].item():>16.3f}  {peak_pos_tau[i].item():>16.3f}  {peak_neg_tau[i].item():>16.3f}  {mean_pow[i].item():>16.3f}")
    if metric_target_impulse:
        all_target_impulse = torch.stack(metric_target_impulse)
        _tb = args_cli.target_contact_bodies or ""
        print(f"Target contact impulse (N*s) [bodies={_tb}]: mean={_nanmean(all_target_impulse):.4f}  std={_nanstd(all_target_impulse):.4f}")
    if metric_target_contact_time:
        all_target_time = torch.stack(metric_target_contact_time)
        print(f"Target contact body count:   mean={_nanmean(all_target_time):.4f}  std={_nanstd(all_target_time):.4f}")
    if metric_com_proj_offset:
        all_com_offset = torch.stack(metric_com_proj_offset)
        print(f"COM proj offset (m):           mean={_nanmean(all_com_offset):.4f}  std={_nanstd(all_com_offset):.4f}")
    if metric_com_x:
        all_com_x = torch.stack(metric_com_x)
        print(f"COM X position (m):            mean={_nanmean(all_com_x):.4f}  std={_nanstd(all_com_x):.4f}")
    if metric_com_y:
        all_com_y = torch.stack(metric_com_y)
        print(f"COM Y position (m):            mean={_nanmean(all_com_y):.4f}  std={_nanstd(all_com_y):.4f}")
    if metric_torso_osc:
        all_torso_osc = torch.stack(metric_torso_osc)
        print(f"Torso oscillation (rad):       mean={_nanmean(all_torso_osc):.4f}  std={_nanstd(all_torso_osc):.4f}")
    if metric_torso_pitch:
        all_torso_pitch = torch.stack(metric_torso_pitch)
        print(f"Torso pitch (rad):             mean={_nanmean(all_torso_pitch):.4f}  std={_nanstd(all_torso_pitch):.4f}  (neg=lean-back)")
    print("=" * 60 + "\n")
    import sys; sys.stdout.flush()

    # Plot per-metric time series with episode boundaries
    if args_cli.mask_back:
        plot_suffix = "_masked"
    elif args_cli.mix_contact_prob > 0.0:
        plot_suffix = "_mixed"
    else:
        plot_suffix = "_base"
    plot_dir = os.path.join(log_dir, "videos", "play")
    metric_series: dict[str, torch.Tensor] = {}
    if metric_anchor_pos_err:
        metric_series["Anchor pos error (m)"] = torch.stack(metric_anchor_pos_err)
    if metric_anchor_rot_err:
        metric_series["Anchor rot error (rad)"] = torch.stack(metric_anchor_rot_err)
    if metric_body_pos_err:
        metric_series["Body pos error (m)"] = torch.stack(metric_body_pos_err)
    if metric_body_pos_err_root_rel:
        metric_series["Body pos err root-rel (m)"] = torch.stack(metric_body_pos_err_root_rel)
    if metric_contact_match:
        metric_series["Contact match accuracy"] = torch.stack(metric_contact_match)
    if metric_waist_pitch_tau:
        metric_series["Waist pitch torque (Nm)"] = torch.stack(metric_waist_pitch_tau)
    if metric_waist_power:
        metric_series["Waist pitch power |tau*qdot| (W)"] = torch.stack(metric_waist_power)
    if metric_total_power:
        metric_series["Total power |tau*qdot| (W)"] = torch.stack(metric_total_power)
    if metric_target_impulse:
        metric_series["Target contact impulse (N*s)"] = torch.stack(metric_target_impulse)
    if metric_target_contact_time:
        metric_series["Target contact body count"] = torch.stack(metric_target_contact_time)
    # Per-link body pos error (one key per body)
    if metric_body_pos_err_per_link:
        all_per_link = torch.stack(metric_body_pos_err_per_link)  # (steps, envs, bodies)
        body_names_for_pos = list(command.cfg.body_names)
        for bi, bn in enumerate(body_names_for_pos):
            metric_series[f"Body pos err / {bn} (m)"] = all_per_link[..., bi]
    # Per-joint signed torque: store full time series per joint
    # compare_metrics._stats(t) computes nanmean = signed mean across all (steps, envs).
    # Peak+/peak- are printed separately in the .log header by play.py itself.
    if metric_per_joint_tau:
        all_pjt = torch.stack(metric_per_joint_tau)  # (steps, envs, joints) — SIGNED
        joint_names = list(robot.data.joint_names)
        for ji, jn in enumerate(joint_names):
            metric_series[f"Torque / {jn} (Nm)"] = all_pjt[..., ji]
    if metric_com_proj_offset:
        metric_series["COM proj offset (m)"] = torch.stack(metric_com_proj_offset)
    if metric_com_x:
        metric_series["COM X position (m)"] = torch.stack(metric_com_x)
    if metric_com_y:
        metric_series["COM Y position (m)"] = torch.stack(metric_com_y)
    if metric_torso_osc:
        metric_series["Torso oscillation (rad)"] = torch.stack(metric_torso_osc)
    if metric_torso_pitch:
        metric_series["Torso pitch (rad)"] = torch.stack(metric_torso_pitch)
    if metric_contact_label_match_reward:
        metric_series["contact_label_match reward"] = torch.stack(metric_contact_label_match_reward)
    if metric_obj_disp_xy:
        metric_series["Object XY displacement (m)"] = torch.stack(metric_obj_disp_xy)

    for name, series in metric_series.items():
        _plot_metric_series(name, series, episode_bounds, plot_dir, plot_suffix, step_dt)

    if args_cli.metrics_out and metric_series:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)), exist_ok=True)
        torch.save(metric_series, args_cli.metrics_out)
        print(f"[INFO]: Metrics saved to {args_cli.metrics_out}")
        # Auto-save pre/post split when a contact-label switch was active
        if _switch_step is not None:
            base, ext = os.path.splitext(args_cli.metrics_out)
            pre_path = f"{base}_pre{ext}"
            post_path = f"{base}_post{ext}"
            pre_series = {k: v[:_switch_step] for k, v in metric_series.items() if v.shape[0] > 0}
            post_series = {k: v[_switch_step:] for k, v in metric_series.items() if v.shape[0] > _switch_step}
            torch.save(pre_series, pre_path)
            torch.save(post_series, post_path)
            print(f"[INFO]: Pre-switch  metrics ({_switch_step} steps) → {pre_path}")
            print(f"[INFO]: Post-switch metrics ({metric_series[next(iter(metric_series))].shape[0] - _switch_step} steps) → {post_path}")

    # ── Probe contact: save dataset + run inline linear probe ──
    if args_cli.probe_contact and probe_gt_contact_list:
        print("\n[PROBE] Assembling probe dataset ...")
        probe_data = {
            "obs": torch.cat(probe_obs_list, dim=0),           # (T*E, obs_dim)
            "gt_contact": torch.cat(probe_gt_contact_list, dim=0),  # (T*E, N_bodies)
            "ref_contact": torch.cat(probe_ref_contact_list, dim=0),
        }
        for name, acts in probe_hidden_lists.items():
            probe_data[name] = torch.cat(acts, dim=0)          # (T*E, hidden_dim)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.probe_out)), exist_ok=True)
        torch.save(probe_data, args_cli.probe_out)
        print(f"[PROBE] Dataset saved to {args_cli.probe_out}")
        print(f"  obs: {probe_data['obs'].shape}")
        for name in probe_hidden_lists:
            print(f"  {name}: {probe_data[name].shape}")
        print(f"  gt_contact: {probe_data['gt_contact'].shape}")
        print(f"  ref_contact: {probe_data['ref_contact'].shape}")

        # Inline linear probe: logistic regression on last hidden layer → per-body contact
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np

        last_hidden_name = sorted(probe_hidden_lists.keys())[-1]
        X_hidden = probe_data[last_hidden_name].numpy()
        X_obs = probe_data["obs"].numpy()
        Y = probe_data["gt_contact"].numpy()  # (N, N_bodies)
        y_any = Y.any(axis=1).astype(int)     # binary: any contact?

        n = len(y_any)
        split = int(0.8 * n)
        idx = np.random.RandomState(42).permutation(n)
        tr, te = idx[:split], idx[split:]

        results = {}
        for feat_name, X in [("obs", X_obs), ("hidden", X_hidden)]:
            if np.isnan(X).any():
                X = np.nan_to_num(X)
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X[tr], y_any[tr])
            pred = clf.predict(X[te])
            acc = accuracy_score(y_any[te], pred)
            f1 = f1_score(y_any[te], pred, zero_division=0)
            results[feat_name] = {"accuracy": acc, "f1": f1}
            print(f"  [PROBE] {feat_name} → any_contact: acc={acc:.3f}  F1={f1:.3f}")

        # Per-body probes on last hidden layer
        n_bodies = Y.shape[1]
        per_body_acc = []
        for bi in range(n_bodies):
            yb = Y[:, bi].astype(int)
            if yb[tr].sum() < 5 or (1 - yb[tr]).sum() < 5:
                continue
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X_hidden[tr], yb[tr])
            pred = clf.predict(X_hidden[te])
            acc = accuracy_score(yb[te], pred)
            f1 = f1_score(yb[te], pred, zero_division=0)
            per_body_acc.append((bi, acc, f1))
        if per_body_acc:
            print(f"  [PROBE] Per-body (hidden → body_i): {len(per_body_acc)} bodies with enough data")
            mean_acc = np.mean([a for _, a, _ in per_body_acc])
            mean_f1 = np.mean([f for _, _, f in per_body_acc])
            print(f"  [PROBE] Mean per-body: acc={mean_acc:.3f}  F1={mean_f1:.3f}")

        # Save results alongside probe data
        probe_data["probe_results"] = results
        probe_data["probe_per_body"] = per_body_acc
        torch.save(probe_data, args_cli.probe_out)
        print(f"[PROBE] Results appended to {args_cli.probe_out}")

        # Clean up hooks
        for h in _probe_hooks:
            h.remove()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

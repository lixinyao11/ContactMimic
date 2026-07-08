from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def body_object_proximity(
    env: ManagerBasedRLEnv,
    asset_name: str,
    object_asset_name: str,
    body_names: list[str],
    target_offsets: list[tuple[float, float, float]],
    std: float = 0.15,
) -> torch.Tensor:
    """General proximity reward: each robot body should be close to its paired target point on an object.

    Args:
        asset_name: robot asset name in scene.
        object_asset_name: object asset name in scene (chair, table, etc.).
        body_names: list of robot body names. len = N.
        target_offsets: list of (x, y, z) offsets from the object's body origin. len = N.
            Each body_names[i] is paired with target_offsets[i].
        std: Gaussian kernel width. Controls how far is "close enough".

    Returns:
        reward = exp(-mean_dist^2 / std^2), shape (num_envs,).
    """
    robot = env.scene[asset_name]
    body_ids = robot.find_bodies(body_names)[0]

    robot_body_pos = robot.data.body_pos_w[:, body_ids, :]  # (num_envs, N, 3)

    # Object origin + per-body offsets → target positions in world frame
    # Use per-motion object position when available, fall back to single scene object
    command: MotionCommand = env.command_manager.get_term("motion")
    if command.has_objects:
        obj_pos = command.active_object_pos_w  # (num_envs, 3)
    else:
        obj = env.scene[object_asset_name]
        obj_pos = obj.data.body_pos_w[:, 0, :]  # (num_envs, 3)
    offsets = torch.tensor(target_offsets, device=obj_pos.device, dtype=obj_pos.dtype)  # (N, 3)
    target_pos = obj_pos[:, None, :] + offsets[None, :, :]  # (num_envs, N, 3)

    dist_sq = torch.sum((robot_body_pos - target_pos) ** 2, dim=-1)  # (num_envs, N)
    return torch.exp(-dist_sq.mean(dim=-1) / std**2)  # (num_envs,)


def classify_object_part(
    body_z: torch.Tensor,
    obj_z: torch.Tensor,
    obj_z_min: torch.Tensor,
    obj_z_range: torch.Tensor,
    thresholds: torch.Tensor,
    segmented: torch.Tensor,
) -> torch.Tensor:
    """Classify which object part each body is at, based on Z height.

    Args:
        body_z: (num_envs, N) body world Z positions.
        obj_z: (num_envs, 1) object origin world Z.
        obj_z_min: (num_envs,) mesh Z minimum in object-local frame.
        obj_z_range: (num_envs,) mesh Z extent.
        thresholds: (num_envs, 2) normalized-Z boundaries [legs|seat, seat|backrest].
        segmented: (num_envs,) bool — True: height-based 3-part split; False: single part.

    Returns:
        (num_envs, N) long tensor: 1=seat/surface, 2=backrest, 3=legs.
        Caller is responsible for zeroing non-contact bodies (setting to 0).
    """
    part = torch.ones_like(body_z, dtype=torch.long)  # default: part 1 (seat / single surface)

    if not segmented.any():
        return part  # all envs are single-part; z geometry is irrelevant

    seg = segmented[:, None]   # (num_envs, 1) broadcast over N
    z_norm = (body_z - obj_z - obj_z_min[:, None]) / obj_z_range[:, None]
    t0 = thresholds[:, 0:1]   # (num_envs, 1)
    t1 = thresholds[:, 1:2]   # (num_envs, 1)
    part = torch.where(seg & (z_norm < t0), torch.full_like(part, 3), part)   # legs
    part = torch.where(seg & (z_norm >= t1), torch.full_like(part, 2), part)  # backrest
    return part


def classify_chair_part(
    body_z: torch.Tensor, obj_z: torch.Tensor, chair_z_min: float, chair_z_range: float
) -> torch.Tensor:
    """Backward-compat wrapper around classify_object_part with scalar geometry params."""
    num_envs = body_z.shape[0]
    device = body_z.device
    z_min = torch.full((num_envs,), chair_z_min, device=device)
    z_range = torch.full((num_envs,), chair_z_range, device=device)
    thresholds = torch.tensor([[0.5, 0.7]], device=device).expand(num_envs, -1)
    segmented = torch.ones(num_envs, dtype=torch.bool, device=device)
    return classify_object_part(body_z, obj_z, z_min, z_range, thresholds, segmented)


def contact_label_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    object_asset_name: str,
    threshold: float = 1.0,
    fp_penalty_weight: float = 1.0,
    tp_fp_mode: bool = False,
    chair_z_min: float = 0.0,
    chair_z_range: float = 0.51,
) -> torch.Tensor:
    """Contact accuracy reward: fraction of bodies with correct contact state.

    Two modes (controlled by ``tp_fp_mode``):

    **tpr_tnr mode** (default, ``tp_fp_mode=False``):
        reward = (TPR + TNR) / 2
        Range [0, 1]. Balanced accuracy; TNR dominates when few bodies need contact.

    **tp_fp mode** (``tp_fp_mode=True``):
        reward = TPR - fp_penalty_weight * FPR
        Range [-fp_penalty_weight, 1]. Baseline=0 when no contact at all; explicit
        penalty for false positives. Much stronger gradient for sparse-contact motions.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, device=env.device)

    expected_part = command.current_contact_part_labels  # (num_envs, N) long: 0/1/2/3
    active = expected_part > 0  # (num_envs, N) bool — bodies that should contact
    N = expected_part.shape[1]

    # Actual contacts from sensor — use paired (chair-only) or aggregate
    if command.use_paired_contact:
        # Paired sensors: force_matrix_w gives chair-only forces per body
        forces = command.get_paired_chair_contact_forces()  # (num_envs, N_valid, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold  # (num_envs, N) bool
    else:
        # Legacy: aggregate sensor (includes ground/self contacts)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        body_ids = command.contact_body_ids  # (N,)
        forces = contact_sensor.data.net_forces_w[:, body_ids]  # (num_envs, N, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold  # (num_envs, N) bool

    # Classify actual object part by body Z height using per-env geometry
    robot = env.scene[command.cfg.asset_name]
    body_ids = command.contact_body_ids
    if command.has_objects:
        obj_pos = command.active_object_pos_w  # (num_envs, 3)
    else:
        obj = env.scene[object_asset_name]
        obj_pos = obj.data.body_pos_w[:, 0, :]  # (num_envs, 3)
    body_z = robot.data.body_pos_w[:, body_ids, 2]  # (num_envs, N)
    obj_z = obj_pos[:, 2:3]  # (num_envs, 1)
    actual_part = classify_object_part(
        body_z, obj_z,
        command.active_obj_z_min, command.active_obj_z_range,
        command.active_obj_thresholds, command.active_obj_segmented,
    )
    actual_part[~has_contact] = 0  # no contact → part 0

    # True positives: should contact AND contacts correct part
    tp = active & (expected_part == actual_part)
    # False positives: should NOT contact, but does
    fp = ~active & has_contact
    # True negatives: should NOT contact AND does NOT have contact
    tn = ~active & ~has_contact

    n_active = active.float().sum(dim=-1)       # N_pos per env
    n_inactive = (~active).float().sum(dim=-1)  # N_neg per env

    if tp_fp_mode:
        # TPR - fp_penalty_weight * FPR
        # Baseline 0 when nothing contacts; reward for TP; penalty for FP.
        # N_pos=0 step (no contact required): return 0.
        tpr = torch.where(n_active > 0, tp.float().sum(dim=-1) / n_active, torch.zeros_like(n_active))
        fpr = torch.where(n_inactive > 0, fp.float().sum(dim=-1) / n_inactive, torch.zeros_like(n_inactive))
        return tpr - fp_penalty_weight * fpr
    else:
        # Balanced accuracy: (TPR + TNR) / 2
        tpr = torch.where(n_active > 0, tp.float().sum(dim=-1) / n_active, torch.ones_like(n_active))
        tnr = torch.where(n_inactive > 0, tn.float().sum(dim=-1) / n_inactive, torch.ones_like(n_inactive))
        return (tpr + tnr) / 2.0


def contact_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_asset_name: str,
    std: float = 0.1,
    unwanted_penalty_threshold: float | None = 0.05,
) -> torch.Tensor:
    """Distance-based contact reward/penalty.

    For bodies that *should* contact the chair: reward closeness to chair surface points
    belonging to the correct chair part (Gaussian kernel).
    For bodies that *should NOT* contact the chair: penalize when their distance to any
    chair surface point is below ``unwanted_penalty_threshold``.

    Returns 0 when no contact data is available for the current motion.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, device=env.device)

    expected_part = command.current_contact_part_labels  # (num_envs, N) long 0/1/2/3
    expected = expected_part > 0                          # (num_envs, N) bool

    body_ids = command.contact_body_ids  # (N,)
    robot = env.scene[command.cfg.asset_name]
    robot_body_pos = robot.data.body_pos_w[:, body_ids]  # (num_envs, N, 3)

    # Chair surface points in world frame
    if command.has_objects:
        obj_pos = command.active_object_pos_w  # (num_envs, 3)
    else:
        obj = env.scene[object_asset_name]
        obj_pos = obj.data.body_pos_w[:, 0, :]  # (num_envs, 3)
    chair_pts = command.current_contact_chair_pts       # (num_envs, K_max, 3)
    chair_pts_w = chair_pts + obj_pos[:, None, :]       # (num_envs, K_max, 3)
    pts_mask = command.current_contact_chair_pts_mask   # (num_envs, K_max) bool
    pts_part = command.current_contact_chair_pts_part   # (num_envs, K_max) long 0/1/2/3

    # Pairwise squared distances: (num_envs, N, K_max)
    diff = robot_body_pos[:, :, None, :] - chair_pts_w[:, None, :, :]
    dist_sq = (diff ** 2).sum(dim=-1)  # (num_envs, N, K_max)

    N = expected_part.shape[1]

    # ---- Per-body Gaussian reward for wanted contacts ----
    # Only consider chair points matching the body's expected part.
    part_match = expected_part[:, :, None] == pts_part[:, None, :]  # (num_envs, N, K_max)
    valid_for_body = pts_mask[:, None, :] & part_match              # (num_envs, N, K_max)

    dist_sq_wanted = dist_sq.masked_fill(~valid_for_body, 1e12)
    min_dist_sq_wanted = dist_sq_wanted.min(dim=-1).values          # (num_envs, N)
    # Per-body Gaussian score: 0 for unwanted bodies (min_dist_sq=0 → score=1 but masked to 0)
    body_wanted_score = torch.exp(-min_dist_sq_wanted / std**2) * expected.float()  # (num_envs, N)

    # ---- Per-body linear penalty for unwanted contacts ----
    if unwanted_penalty_threshold is not None:
        unwanted = ~expected
        dist_sq_all = dist_sq.masked_fill(~pts_mask[:, None, :], 1e12)
        min_dist_unwanted = torch.sqrt(dist_sq_all.min(dim=-1).values.clamp(min=0))  # (num_envs, N)

        body_penalty = (unwanted_penalty_threshold - min_dist_unwanted).clamp(min=0) / unwanted_penalty_threshold
        body_penalty = body_penalty * unwanted.float()  # zero out wanted bodies
        penalty_sum = body_penalty.sum(dim=-1)
    else:
        penalty_sum = torch.zeros(env.num_envs, device=env.device)

    # Both terms averaged over the same N bodies → same scale
    return (body_wanted_score.sum(dim=-1) - penalty_sum) / N


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import classify_object_part

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Contact observations
# ---------------------------------------------------------------------------

def contact_labels_ref(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference contact part labels from motion data. Shape (num_envs, N_valid).
    Values: 0=none, 1=seat, 2=backrest, 3=legs."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, 0, device=env.device)
    return command.current_contact_part_labels.float()


def contact_labels_ref_binary(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Binary reference contact labels from motion data. Shape (num_envs, N_valid).
    Values: 0=no contact, 1=has contact (no chair-part distinction)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, 0, device=env.device)
    return command.current_contact_labels.float()


def contact_body_chair_distance(
    env: ManagerBasedEnv, command_name: str, object_asset_name: str
) -> torch.Tensor:
    """Min distance from each contact body to any chair surface point. Shape (num_envs, N_valid)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, 0, device=env.device)

    body_ids = command.contact_body_ids
    robot = env.scene[command.cfg.asset_name]
    robot_body_pos = robot.data.body_pos_w[:, body_ids]  # (num_envs, N, 3)

    if command.has_objects:
        obj_pos = command.active_object_pos_w  # (num_envs, 3)
    else:
        obj = env.scene[object_asset_name]
        obj_pos = obj.data.body_pos_w[:, 0, :]  # (num_envs, 3)
    chair_pts = command.current_contact_chair_pts  # (num_envs, K_max, 3)
    chair_pts_w = chair_pts + obj_pos[:, None, :]
    mask = command.current_contact_chair_pts_mask  # (num_envs, K_max)

    diff = robot_body_pos[:, :, None, :] - chair_pts_w[:, None, :, :]  # (num_envs, N, K_max, 3)
    dist_sq = (diff ** 2).sum(dim=-1)  # (num_envs, N, K_max)
    dist_sq = dist_sq.masked_fill(~mask[:, None, :], 1e12)
    min_dist = dist_sq.min(dim=-1).values.sqrt()  # (num_envs, N)
    return min_dist


def contact_labels_real(
    env: ManagerBasedEnv,
    command_name: str,
    sensor_cfg: "SceneEntityCfg",
    object_asset_name: str,
    threshold: float = 1.0,
    chair_z_min: float = 0.0,
    chair_z_range: float = 0.51,
) -> torch.Tensor:
    """Actual contact part labels from sensor + height classification. Shape (num_envs, N_valid).
    Values: 0=no contact, 1=seat, 2=backrest, 3=legs."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, 0, device=env.device)

    # Use paired (chair-only) or aggregate contact sensor
    if command.use_paired_contact:
        forces = command.get_paired_chair_contact_forces()  # (num_envs, N_valid, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold
    else:
        body_ids = command.contact_body_ids
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w[:, body_ids]  # (num_envs, N, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold  # (num_envs, N)

    body_ids = command.contact_body_ids
    robot = env.scene[command.cfg.asset_name]
    if command.has_objects:
        obj_pos = command.active_object_pos_w  # (num_envs, 3)
    else:
        obj = env.scene[object_asset_name]
        obj_pos = obj.data.body_pos_w[:, 0, :]  # (num_envs, 3)
    body_z = robot.data.body_pos_w[:, body_ids, 2]  # (num_envs, N)
    obj_z = obj_pos[:, 2:3]  # (num_envs, 1)

    part = classify_object_part(
        body_z, obj_z,
        command.active_obj_z_min, command.active_obj_z_range,
        command.active_obj_thresholds, command.active_obj_segmented,
    )
    part[~has_contact] = 0
    return part.float()


def contact_labels_real_binary(
    env: ManagerBasedEnv,
    command_name: str,
    sensor_cfg: "SceneEntityCfg",
    object_asset_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary actual contact labels from sensor. Shape (num_envs, N_valid).
    Values: 0=no contact, 1=has contact (no chair-part distinction).
    Uses paired sensors (chair-only) when available."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_contact:
        return torch.zeros(env.num_envs, 0, device=env.device)

    if command.use_paired_contact:
        forces = command.get_paired_chair_contact_forces()  # (num_envs, N_valid, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold
    else:
        body_ids = command.contact_body_ids
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w[:, body_ids]  # (num_envs, N, 3)
        has_contact = torch.norm(forces, dim=-1) > threshold  # (num_envs, N)
    return has_contact.float()

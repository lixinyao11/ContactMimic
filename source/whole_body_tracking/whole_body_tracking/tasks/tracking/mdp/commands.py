from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


N_CONTACT_BODIES = 46
"""All motions with contact data must have exactly this many contact bodies in the same order."""

# 16 bodies with >= 0.5% contact rate that are actual robot rigid bodies.
# Excluded: left/right_sphere_hand_link (rubber_hand is a sub-Xform, not a rigid body),
#           left_ankle_roll_sphere_1_link (collision sub-link).
PAIRED_CONTACT_BODIES: list[str] = [
    "pelvis",                        # 63.3% — seat/backrest
    "right_hip_pitch_link",          # 57.6% — seat
    "left_hip_pitch_link",           # 56.2% — seat
    "left_hip_yaw_link",             # 38.5% — seat
    "right_hip_yaw_link",            # 29.3% — seat
    "torso_link",                    # 23.3% — backrest
    "left_ankle_roll_link",          # 19.7% — legs
    "right_hip_roll_link",           # 10.2% — seat
    "right_knee_link",               #  9.9% — seat
    "left_knee_link",                #  6.8% — seat
    "right_wrist_yaw_link",          #  6.7% — backrest
    "right_wrist_pitch_link",        #  6.4% — backrest
    "right_shoulder_yaw_link",       #  6.3% — backrest
    "left_shoulder_yaw_link",        #  2.0% — backrest
    "left_wrist_yaw_link",           #  1.0% — seat
    "left_wrist_pitch_link",         #  0.6% — seat
]
N_PAIRED_CONTACT_BODIES = len(PAIRED_CONTACT_BODIES)  # 16

# Remap data body names → robot body names for paired contact (currently unused)
_CONTACT_BODY_REMAP: dict[str, str] = {}


class ObjectGeometryCfg:
    """Geometry config for a scene object used in contact-part classification.

    z_min / z_range define the vertical extent of the collision mesh in object-local
    frame (after any scale factors). When *segmented* is True, *part_thresholds* (two
    normalized-z values) split the mesh into legs / seat / backrest (part IDs 3/1/2).
    When *segmented* is False the object is treated as a single contact surface (part 1).
    """

    def __init__(
        self,
        z_min: float = 0.0,
        z_range: float = 0.51,
        segmented: bool = True,
        part_thresholds: tuple[float, ...] = (0.5, 0.7),
    ):
        self.z_min = z_min
        self.z_range = z_range
        self.segmented = segmented
        self.part_thresholds = tuple(part_thresholds)

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectGeometryCfg":
        return cls(
            z_min=float(d.get("z_min", 0.0)),
            z_range=float(d.get("z_range", 0.51)),
            segmented=bool(d.get("segmented", True)),
            part_thresholds=tuple(float(t) for t in d.get("part_thresholds", [0.5, 0.7])),
        )

    @classmethod
    def load_yaml(cls, path: str) -> "dict[str, ObjectGeometryCfg]":
        """Load a YAML file mapping keys to ObjectGeometryCfg entries."""
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {k: cls.from_dict(v) for k, v in data.items()}

    @classmethod
    def match_urdf(cls, urdf_path: str, registry: "dict[str, ObjectGeometryCfg]") -> "ObjectGeometryCfg":
        """Return the first registry entry whose key is a substring of *urdf_path*.

        Falls back to the ``_default`` entry, then to a plain ``ObjectGeometryCfg()``.
        Keys that start with ``_`` are skipped during substring search.
        """
        for key, cfg in registry.items():
            if key.startswith("_"):
                continue
            if key in urdf_path:
                return cfg
        return registry.get("_default", cls())


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu",
                 use_paired_contact: bool = False):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        T = self.time_step_total

        if "object_poses" in data:
            # (T, 7): [x, y, z, qw, qx, qy, qz] in MuJoCo/Isaac order
            self.object_poses = torch.tensor(data["object_poses"], dtype=torch.float32, device=device)
        else:
            self.object_poses = None

        n_bodies = N_PAIRED_CONTACT_BODIES if use_paired_contact else N_CONTACT_BODIES

        # Contact fields: present in HUMOTO motions, absent in LAFAN motions.
        # All motions with contact data must have N_CONTACT_BODIES (46) bodies.
        # When use_paired_contact=True, remap to the 19 PAIRED_CONTACT_BODIES subset.
        if "contact_labels" in data:
            full_names: list[str] = list(data["contact_body_names"])
            assert len(full_names) == N_CONTACT_BODIES, (
                f"Expected {N_CONTACT_BODIES} contact bodies, got {len(full_names)} "
                f"in {motion_file}"
            )
            full_labels = torch.tensor(data["contact_labels"], dtype=torch.bool, device=device)  # (T, 46)
            if "contact_part_labels" in data:
                full_parts = torch.tensor(data["contact_part_labels"], dtype=torch.long, device=device)
            else:
                full_parts = full_labels.long()

            if use_paired_contact:
                # Remap 46 → 18: map data body names to robot body names,
                # merging remapped bodies (e.g. sphere_hand → rubber_hand) via OR/max.
                name_to_idx = {n: i for i, n in enumerate(full_names)}
                T = full_labels.shape[0]
                N_p = N_PAIRED_CONTACT_BODIES
                paired_labels = torch.zeros(T, N_p, dtype=torch.bool, device=device)
                paired_parts = torch.zeros(T, N_p, dtype=torch.long, device=device)

                # Build reverse map: robot_name → list of data column indices
                robot_to_data_cols: dict[str, list[int]] = {n: [] for n in PAIRED_CONTACT_BODIES}
                for data_name, data_idx in name_to_idx.items():
                    robot_name = _CONTACT_BODY_REMAP.get(data_name, data_name)
                    if robot_name in robot_to_data_cols:
                        robot_to_data_cols[robot_name].append(data_idx)

                for p_idx, robot_name in enumerate(PAIRED_CONTACT_BODIES):
                    cols = robot_to_data_cols[robot_name]
                    if cols:
                        # OR for labels (any source body in contact → contact)
                        paired_labels[:, p_idx] = full_labels[:, cols].any(dim=1)
                        # Max for part labels (take highest part ID among sources)
                        paired_parts[:, p_idx] = full_parts[:, cols].max(dim=1).values

                self.contact_body_names = list(PAIRED_CONTACT_BODIES)
                self.contact_labels = paired_labels       # (T, 18)
                self.contact_part_labels = paired_parts   # (T, 18)
            else:
                self.contact_body_names = full_names
                self.contact_labels = full_labels
                self.contact_part_labels = full_parts

            # Pad variable-length contact_points_object to (T, K_max, 3) with far sentinel
            pts_obj = data["contact_points_object"]  # object array of (K_t, 3)
            k_max = max(len(pts_obj[t]) for t in range(len(pts_obj))) if len(pts_obj) > 0 else 0
            k_max = max(k_max, 1)  # at least 1 to avoid empty tensors
            chair_pts = torch.full((T, k_max, 3), 1e6, dtype=torch.float32, device=device)
            chair_pts_mask = torch.zeros(T, k_max, dtype=torch.bool, device=device)
            chair_pts_part = torch.zeros(T, k_max, dtype=torch.long, device=device)
            _CHAIR_PART_STR_MAP = {"seat": 1, "backrest": 2, "legs": 3}
            raw_chair_lbl = data["contact_chair_labels"] if "contact_chair_labels" in data else None
            for t in range(T):
                k = len(pts_obj[t])
                if k > 0:
                    chair_pts[t, :k] = torch.tensor(pts_obj[t], dtype=torch.float32, device=device)
                    chair_pts_mask[t, :k] = True
                    if raw_chair_lbl is not None and len(raw_chair_lbl[t]) == k:
                        chair_pts_part[t, :k] = torch.tensor(
                            [_CHAIR_PART_STR_MAP.get(str(s), 0) for s in raw_chair_lbl[t]],
                            dtype=torch.long, device=device,
                        )
            self.contact_chair_pts = chair_pts        # (T, K_max, 3)
            self.contact_chair_pts_mask = chair_pts_mask  # (T, K_max)
            self.contact_chair_pts_part = chair_pts_part  # (T, K_max) long: 0/1/2/3
            print(f"[INFO]: Contact data loaded: {n_bodies} bodies"
                  f"{' (paired, remapped from 46)' if use_paired_contact else ''}, K_max={k_max}")
        else:
            # No contact data (e.g. LAFAN) — fill with zeros so multi-motion stacking works
            self.contact_body_names = None
            self.contact_labels = torch.zeros(T, n_bodies, dtype=torch.bool, device=device)
            self.contact_part_labels = torch.zeros(T, n_bodies, dtype=torch.long, device=device)
            self.contact_chair_pts = torch.full((T, 1, 3), 1e6, dtype=torch.float32, device=device)
            self.contact_chair_pts_mask = torch.zeros(T, 1, dtype=torch.bool, device=device)
            self.contact_chair_pts_part = torch.zeros(T, 1, dtype=torch.long, device=device)
            print(f"[INFO]: No contact data, filled with zeros ({n_bodies} bodies)")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MultiMotionLoader:
    """Loads multiple motions, pads to max length, and stacks into single tensors.

    Suitable for small numbers of motions (< ~100). For large-scale training,
    use PackedMotionLoader instead.
    """

    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu",
                 use_paired_contact: bool = False):
        loaders = [MotionLoader(f, body_indexes, device, use_paired_contact=use_paired_contact)
                    for f in motion_files]
        self.num_motions = len(loaders)
        self.fps = loaders[0].fps
        self.time_steps_per_motion = torch.tensor(
            [m.time_step_total for m in loaders], dtype=torch.long, device=device
        )
        self.time_step_max = int(self.time_steps_per_motion.max().item())

        def _pad(tensors: list[torch.Tensor], max_t: int) -> torch.Tensor:
            """Pad along time dim (0) and stack into [num_motions, max_t, ...]."""
            padded = []
            for t in tensors:
                pad_size = max_t - t.shape[0]
                if pad_size > 0:
                    # repeat last frame for padding
                    padding = t[-1:].expand(pad_size, *t.shape[1:])
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)

        self.joint_pos = _pad([m.joint_pos for m in loaders], self.time_step_max)
        self.joint_vel = _pad([m.joint_vel for m in loaders], self.time_step_max)
        self.body_pos_w = _pad([m.body_pos_w for m in loaders], self.time_step_max)
        self.body_quat_w = _pad([m.body_quat_w for m in loaders], self.time_step_max)
        self.body_lin_vel_w = _pad([m.body_lin_vel_w for m in loaders], self.time_step_max)
        self.body_ang_vel_w = _pad([m.body_ang_vel_w for m in loaders], self.time_step_max)

        # Contact data: all motions have (T, N_CONTACT_BODIES) contact tensors
        # (LAFAN motions are zero-filled by MotionLoader). Use body names from
        # the first motion that has them; otherwise None.
        self.contact_body_names = next(
            (m.contact_body_names for m in loaders if m.contact_body_names is not None), None
        )
        self.contact_labels = _pad([m.contact_labels for m in loaders], self.time_step_max)
        self.contact_part_labels = _pad([m.contact_part_labels for m in loaders], self.time_step_max)

        # Pad contact_chair_pts to global K_max
        K_max_global = max(m.contact_chair_pts.shape[1] for m in loaders)
        padded_pts = []
        padded_mask = []
        padded_part = []
        for m in loaders:
            T_i, K_i = m.contact_chair_pts.shape[:2]
            if K_i < K_max_global:
                pad_pts = torch.full((T_i, K_max_global - K_i, 3), 1e6,
                                     dtype=torch.float32, device=device)
                pad_m = torch.zeros(T_i, K_max_global - K_i, dtype=torch.bool, device=device)
                pad_p = torch.zeros(T_i, K_max_global - K_i, dtype=torch.long, device=device)
                padded_pts.append(torch.cat([m.contact_chair_pts, pad_pts], dim=1))
                padded_mask.append(torch.cat([m.contact_chair_pts_mask, pad_m], dim=1))
                padded_part.append(torch.cat([m.contact_chair_pts_part, pad_p], dim=1))
            else:
                padded_pts.append(m.contact_chair_pts)
                padded_mask.append(m.contact_chair_pts_mask)
                padded_part.append(m.contact_chair_pts_part)
        self.contact_chair_pts = _pad(padded_pts, self.time_step_max)
        self.contact_chair_pts_mask = _pad(padded_mask, self.time_step_max)
        self.contact_chair_pts_part = _pad(padded_part, self.time_step_max)

        if all(m.object_poses is not None for m in loaders):
            self.object_poses = _pad([m.object_poses for m in loaders], self.time_step_max)
        else:
            self.object_poses = None

        print(f"[INFO]: Loaded {self.num_motions} motions, max_steps={self.time_step_max}, "
              f"steps_per_motion={self.time_steps_per_motion.tolist()}")


class PackedMotionLoader:
    """Memory-efficient loader for large numbers of motions.

    Instead of padding all motions to the max length and stacking (which wastes
    memory), this loader concatenates all motions along the time axis and uses
    per-motion offsets to index into the packed data.

    Indexing: data[offset[motion_id] + time_step]
    """

    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu"):
        all_jp, all_jv, all_bp, all_bq, all_blv, all_bav = [], [], [], [], [], []
        all_op = []
        has_object_poses = True
        lengths = []
        fps = None

        for f in motion_files:
            data = np.load(f)
            if fps is None:
                fps = data["fps"]
            bi = body_indexes if isinstance(body_indexes, (list, tuple)) else body_indexes.tolist()
            all_jp.append(data["joint_pos"].astype(np.float32))
            all_jv.append(data["joint_vel"].astype(np.float32))
            all_bp.append(data["body_pos_w"][:, bi].astype(np.float32))
            all_bq.append(data["body_quat_w"][:, bi].astype(np.float32))
            all_blv.append(data["body_lin_vel_w"][:, bi].astype(np.float32))
            all_bav.append(data["body_ang_vel_w"][:, bi].astype(np.float32))
            lengths.append(data["joint_pos"].shape[0])
            if "object_poses" in data:
                all_op.append(data["object_poses"].astype(np.float32))
            else:
                has_object_poses = False

        self.num_motions = len(motion_files)
        self.fps = fps
        self.time_steps_per_motion = torch.tensor(lengths, dtype=torch.long, device=device)
        self.time_step_max = int(self.time_steps_per_motion.max().item())

        # Build offset array: offset[i] = sum of lengths[0..i-1]
        offsets = np.zeros(self.num_motions + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        self._offsets = torch.tensor(offsets, dtype=torch.long, device=device)

        total_frames = int(offsets[-1])
        self.joint_pos = torch.tensor(np.concatenate(all_jp, axis=0), dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(np.concatenate(all_jv, axis=0), dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(np.concatenate(all_bp, axis=0), dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(np.concatenate(all_bq, axis=0), dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(np.concatenate(all_blv, axis=0), dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(np.concatenate(all_bav, axis=0), dtype=torch.float32, device=device)

        if has_object_poses and all_op:
            self.object_poses = torch.tensor(np.concatenate(all_op, axis=0), dtype=torch.float32, device=device)
        else:
            self.object_poses = None

        total_mb = sum(getattr(self, k).nelement() * 4 for k in
                       ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                        "body_lin_vel_w", "body_ang_vel_w"]) / 1e6
        print(f"[INFO]: PackedMotionLoader: {self.num_motions} motions, "
              f"{total_frames} total frames, GPU memory: {total_mb:.0f} MB")

    def index(self, field: str, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Index packed data: data[offset[motion_id] + time_step]."""
        flat_idx = self._offsets[motion_ids] + time_steps
        return getattr(self, field)[flat_idx]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Support single, multiple, or directory-based motion files
        import glob as _glob
        if self.cfg.motion_dir is not None:
            motion_files = sorted(_glob.glob(os.path.join(self.cfg.motion_dir, "*.npz")))
            assert len(motion_files) > 0, f"No .npz files found in {self.cfg.motion_dir}"
            print(f"[INFO]: Loading {len(motion_files)} motions from {self.cfg.motion_dir}")
        elif self.cfg.motion_files:
            motion_files = self.cfg.motion_files
        else:
            motion_files = [self.cfg.motion_file]

        self.multi_motion = len(motion_files) > 1
        self.packed_motion = self.multi_motion and len(motion_files) > 100

        if self.packed_motion:
            self.motion_lib = PackedMotionLoader(motion_files, self.body_indexes, device=self.device)
            self.motion_ids = torch.randint(
                0, self.motion_lib.num_motions, (self.num_envs,), device=self.device
            )
            if self.cfg.motion_z_offset != 0.0:
                self.motion_lib.body_pos_w[:, :, 2] += self.cfg.motion_z_offset
        elif self.multi_motion:
            self.motion_lib = MultiMotionLoader(motion_files, self.body_indexes, device=self.device,
                                                use_paired_contact=self.cfg.use_paired_contact)
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.motion_ids[:] = torch.randint(0, self.motion_lib.num_motions, (self.num_envs,), device=self.device)
            print(f"[INFO]: Initial motion assignment: {self.motion_ids.tolist()}")
            if self.cfg.motion_z_offset != 0.0:
                self.motion_lib.body_pos_w[:, :, :, 2] += self.cfg.motion_z_offset
        else:
            self.motion = MotionLoader(motion_files[0], self.body_indexes, device=self.device,
                                       use_paired_contact=self.cfg.use_paired_contact)
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if self.cfg.motion_z_offset != 0.0:
                self.motion._body_pos_w[:, :, 2] += self.cfg.motion_z_offset

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Stand envs: frozen at frame 0
        self._stand_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.stand_ratio > 0:
            n_stand = int(self.num_envs * self.cfg.stand_ratio)
            self._stand_mask[:n_stand] = True
            print(f"[INFO]: Stand ratio={self.cfg.stand_ratio}, {n_stand}/{self.num_envs} envs frozen at frame 0")

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Resolve contact body names → robot body IDs (once at init)
        # Filter out bodies in the NPZ that don't exist on the Isaac robot model
        src = self.motion_lib if self.multi_motion else self.motion
        if src.contact_body_names is not None:
            robot_body_set = set(self.robot.body_names)
            valid_mask = [name in robot_body_set for name in src.contact_body_names]
            valid_names = [name for name, v in zip(src.contact_body_names, valid_mask) if v]
            skipped = [name for name, v in zip(src.contact_body_names, valid_mask) if not v]
            if skipped:
                print(f"[WARNING]: Skipping contact bodies not on robot: {skipped}")
            self._contact_body_ids = torch.tensor(
                self.robot.find_bodies(valid_names, preserve_order=True)[0],
                dtype=torch.long, device=self.device,
            )
            # Store boolean mask to index contact_labels columns (N_full → N_valid)
            self._contact_valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
            self._has_contact = True

            # Zero out ignored contact parts at data level
            if self.cfg.ignore_contact_parts:
                total_zeroed = 0
                for p in self.cfg.ignore_contact_parts:
                    pmask = src.contact_part_labels == p
                    src.contact_part_labels[pmask] = 0
                    src.contact_labels[pmask] = False
                    total_zeroed += pmask.sum().item()
                print(f"[INFO]: Zeroed contact parts {self.cfg.ignore_contact_parts}, "
                      f"{total_zeroed} entries cleared")
        else:
            self._contact_body_ids = None
            self._contact_valid_mask = None
            self._has_contact = False

        # Random per-env contact part masking
        self._random_mask_parts = self.cfg.random_mask_contact_parts or []
        self._random_mask_prob = self.cfg.random_mask_contact_prob
        if self._random_mask_parts and self._random_mask_prob > 0 and self._has_contact:
            # Per-env mask: which parts are masked (num_envs, num_parts)
            self._masked_parts = torch.zeros(
                self.num_envs, len(self._random_mask_parts), dtype=torch.bool, device=self.device
            )
            self._random_mask_part_ids = torch.tensor(
                self._random_mask_parts, dtype=torch.long, device=self.device
            )
            self._resample_contact_mask(torch.arange(self.num_envs, device=self.device))
            print(f"[INFO]: Random contact masking: parts={self._random_mask_parts}, "
                  f"prob={self._random_mask_prob}")
        else:
            self._masked_parts = None

        # Per-env flag: True when object was randomly removed at last reset (see object_remove_prob)
        self._object_removed: torch.Tensor | None = None

        # Paired contact sensors: gather references for the 19 per-body sensors
        self._use_paired_contact = self.cfg.use_paired_contact and self._has_contact
        if self._use_paired_contact:
            valid_names = [n for n, v in zip(src.contact_body_names, valid_mask) if v] if src.contact_body_names else []
            self._paired_sensors = []
            for name in valid_names:
                sensor_key = f"paired_contact_{name}"
                if sensor_key in env.scene.sensors:
                    self._paired_sensors.append(env.scene.sensors[sensor_key])
                else:
                    print(f"[WARNING]: Paired sensor '{sensor_key}' not found in scene")
                    self._use_paired_contact = False
                    break
            if self._use_paired_contact:
                print(f"[INFO]: Paired contact sensors: {len(self._paired_sensors)} sensors active")

        # Per-motion chair XYZ offset tensor
        self._per_motion_xyz_offsets: torch.Tensor | None = None
        if self.cfg.chair_offset_per_motion:
            _num_motions = self.motion_lib.num_motions if self.multi_motion else 1
            _names = (
                self.cfg.motion_names if self.cfg.motion_names
                else [os.path.splitext(os.path.basename(f))[0] for f in motion_files]
            )
            offsets = torch.zeros(_num_motions, 3, device=self.device)
            for name, xyz in self.cfg.chair_offset_per_motion.items():
                if name in _names:
                    idx = _names.index(name)
                    offsets[idx, 0] = float(xyz.get("x", 0.0))
                    offsets[idx, 1] = float(xyz.get("y", 0.0))
                    offsets[idx, 2] = float(xyz.get("z", 0.0))
                else:
                    print(f"[WARNING]: chair_offset_per_motion: motion '{name}' not found, skipping")
            self._per_motion_xyz_offsets = offsets
            print(f"[INFO]: Per-motion chair offsets loaded for {len(self.cfg.chair_offset_per_motion)} motions")

        # Per-motion object management
        self._has_objects = (
            self.cfg.object_urdfs is not None
            and any(u is not None for u in self.cfg.object_urdfs)
        )
        if self._has_objects:
            self._motion_to_object_idx = torch.tensor(
                env.cfg._motion_to_object_idx, dtype=torch.long, device=self.device
            )
            self._num_objects = env.cfg._num_unique_objects
            prefix = self.cfg.object_asset_prefix
            self._object_assets = [env.scene[f"{prefix}_{i}"] for i in range(self._num_objects)]
            self._underground_pos = torch.tensor([0.0, 0.0, -10.0], device=self.device)
            self._active_object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            self._active_object_pos_w = None

        # Whether per-frame object poses are available in the loaded NPZ data
        _src = self.motion_lib if self.multi_motion else self.motion
        self._has_object_poses = (
            self._has_objects and getattr(_src, "object_poses", None) is not None
        )
        if self._has_object_poses:
            print("[INFO]: Object poses found in NPZ data — objects will be initialized from retargeted poses")

        # Per-env object geometry tensors (updated at reset alongside object positions).
        # Defaults match a standard chair; overridden by object_geometry_yaml when set.
        _MAX_THRESHOLDS = 2
        _def_geom = ObjectGeometryCfg()
        self._env_obj_z_min = torch.full((self.num_envs,), _def_geom.z_min, device=self.device)
        self._env_obj_z_range = torch.full((self.num_envs,), _def_geom.z_range, device=self.device)
        self._env_obj_thresholds = torch.zeros(self.num_envs, _MAX_THRESHOLDS, device=self.device)
        for _j, _t in enumerate(_def_geom.part_thresholds[:_MAX_THRESHOLDS]):
            self._env_obj_thresholds[:, _j] = _t
        self._env_obj_segmented = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-unique-object geometry tensors (None when no YAML / no per-motion objects)
        self._per_object_geom_z_min: torch.Tensor | None = None
        self._per_object_geom_z_range: torch.Tensor | None = None
        self._per_object_geom_thresholds: torch.Tensor | None = None
        self._per_object_geom_segmented: torch.Tensor | None = None

        if self.cfg.object_geometry_yaml:
            _geom_path = self.cfg.object_geometry_yaml
            if os.path.exists(_geom_path):
                _registry = ObjectGeometryCfg.load_yaml(_geom_path)
                if self._has_objects and hasattr(env.cfg, "_object_geometries") and env.cfg._object_geometries:
                    _geoms = env.cfg._object_geometries  # list[ObjectGeometryCfg], indexed by obj_idx
                    _n = len(_geoms)
                    self._per_object_geom_z_min = torch.tensor(
                        [g.z_min for g in _geoms], device=self.device
                    )
                    self._per_object_geom_z_range = torch.tensor(
                        [g.z_range for g in _geoms], device=self.device
                    )
                    _thresh = torch.zeros(_n, _MAX_THRESHOLDS, device=self.device)
                    for _i, _g in enumerate(_geoms):
                        for _j, _t in enumerate(_g.part_thresholds[:_MAX_THRESHOLDS]):
                            _thresh[_i, _j] = _t
                    self._per_object_geom_thresholds = _thresh
                    self._per_object_geom_segmented = torch.tensor(
                        [g.segmented for g in _geoms], dtype=torch.bool, device=self.device
                    )
                    print(f"[INFO]: Loaded per-object geometries ({_n} objects) from {_geom_path}")
                else:
                    # No per-motion objects: apply a single geometry to all envs
                    _key = "_default" if "_default" in _registry else next(iter(_registry), None)
                    if _key:
                        _geom = _registry[_key]
                        self._env_obj_z_min.fill_(_geom.z_min)
                        self._env_obj_z_range.fill_(_geom.z_range)
                        self._env_obj_segmented.fill_(_geom.segmented)
                        for _j, _t in enumerate(_geom.part_thresholds[:_MAX_THRESHOLDS]):
                            self._env_obj_thresholds[:, _j] = _t
                        print(f"[INFO]: Using '{_key}' object geometry from {_geom_path}")
            else:
                print(f"[WARNING]: object_geometry_yaml not found: {_geom_path}")

        # Initial object placement (geometry must be configured before this call)
        if self._has_objects:
            self._update_object_positions(torch.arange(self.num_envs, device=self.device))
            print(f"[INFO]: Per-motion objects: {self._num_objects} objects, "
                  f"mapping: {env.cfg._motion_to_object_idx}")

        if self.multi_motion:
            max_bin = max(
                int(t // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
                for t in self.motion_lib.time_steps_per_motion.tolist()
            )
            self.bin_count = max_bin
        else:
            self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # Per-motion metrics for multi-motion (skip for large-scale to avoid metric bloat)
        self._track_per_motion = self.multi_motion and len(motion_files) <= 100
        if self._track_per_motion:
            self.motion_names = self.cfg.motion_names if self.cfg.motion_names else [os.path.splitext(os.path.basename(f))[0] for f in motion_files]
            for i, name in enumerate(self.motion_names):
                self.metrics[f"per_motion/{name}/success_rate"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[f"per_motion/{name}/episode_length"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[f"per_motion/{name}/error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        if self.multi_motion:
            self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Mix contact labels: build per-motion group membership and peer lists
        self._mix_contact_label_prob = self.cfg.mix_contact_label_prob
        self._mix_group_peers: list[torch.Tensor] | None = None
        if (
            self.cfg.mix_contact_labels is not None
            and self._mix_contact_label_prob > 0.0
            and self._has_contact
            and self.multi_motion
        ):
            # Build motion name → index map (motion_names is set above when _track_per_motion)
            _all_names = (
                self.motion_names if self._track_per_motion
                else [os.path.splitext(os.path.basename(f))[0] for f in motion_files]
            )
            motion_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(_all_names)}
            self._mix_group_peers = [None] * self.motion_lib.num_motions
            for group in self.cfg.mix_contact_labels:
                idxs = []
                for name in group:
                    if name in motion_name_to_idx:
                        idxs.append(motion_name_to_idx[name])
                    else:
                        print(f"[WARNING]: mix_contact_labels: motion '{name}' not found, skipping")
                if len(idxs) < 2:
                    print(f"[WARNING]: mix_contact_labels: group {group} has <2 valid motions, skipping")
                    continue
                idx_tensor = torch.tensor(idxs, dtype=torch.long, device=self.device)
                for mid in idxs:
                    self._mix_group_peers[mid] = idx_tensor[idx_tensor != mid]
            for i in range(self.motion_lib.num_motions):
                if self._mix_group_peers[i] is None:
                    self._mix_group_peers[i] = torch.tensor([], dtype=torch.long, device=self.device)
            self._mix_label_override = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self._resample_mix_contact(torch.arange(self.num_envs, device=self.device))
            print(f"[INFO]: Mix contact labels: {len(self.cfg.mix_contact_labels)} groups, "
                  f"prob={self._mix_contact_label_prob}")
        else:
            self._mix_label_override = None

        # Force-peer mode: pin all envs to motion 0 keypoints, peer idx for contact
        self._force_contact_peer_idx: int | None = self.cfg.force_contact_peer_idx
        if self._force_contact_peer_idx is not None:
            assert self.multi_motion, "force_contact_peer_idx requires at least 2 motions"
            assert self._has_contact, "force_contact_peer_idx requires contact data"
            # Pin all envs to motion 0
            self.motion_ids[:] = 0
            # Override contact for all envs to peer
            self._mix_label_override = torch.full(
                (self.num_envs,), self._force_contact_peer_idx, dtype=torch.long, device=self.device
            )
            print(f"[INFO]: Force contact peer: all envs use motion 0 keypoints, "
                  f"contact from motion {self._force_contact_peer_idx}")

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _get_motion_data(self, field: str) -> torch.Tensor:
        """Index motion data by per-env motion_id and time_step."""
        if self.packed_motion:
            return self.motion_lib.index(field, self.motion_ids, self.time_steps)
        elif self.multi_motion:
            data = getattr(self.motion_lib, field)  # [num_motions, max_t, ...]
            return data[self.motion_ids, self.time_steps]
        else:
            data = getattr(self.motion, field)  # [T, ...]
            return data[self.time_steps]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._get_motion_data("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._get_motion_data("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._get_motion_data("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._get_motion_data("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._get_motion_data("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._get_motion_data("body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._get_motion_data("body_pos_w")[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._get_motion_data("body_quat_w")[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._get_motion_data("body_lin_vel_w")[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._get_motion_data("body_ang_vel_w")[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def use_paired_contact(self) -> bool:
        """True when per-body paired contact sensors are active."""
        return self._use_paired_contact

    def get_paired_chair_contact_forces(self) -> torch.Tensor:
        """Get chair-only contact forces from paired sensors.

        Returns:
            (num_envs, N_valid, 3) — contact force between each body and all chair objects.
            Summed across all filter prims (only the active chair will contribute).
        """
        assert self._use_paired_contact, "Paired contact not configured"
        forces = []
        for sensor in self._paired_sensors:
            # force_matrix_w: (num_envs, 1, M, 3) — M = number of filtered object prims
            fm = sensor.data.force_matrix_w  # (num_envs, 1, M, 3)
            # Sum across M (multiple objects, only active one has nonzero force)
            forces.append(fm[:, 0].sum(dim=1))  # (num_envs, 3)
        return torch.stack(forces, dim=1)  # (num_envs, N_valid, 3)

    @property
    def has_contact(self) -> bool:
        return self._has_contact

    @property
    def contact_body_ids(self) -> torch.Tensor:
        """Robot body IDs for contact_body_names. Shape (N,)."""
        return self._contact_body_ids

    @property
    def current_contact_labels(self) -> torch.Tensor:
        """Per-env contact labels at current timestep. Shape (num_envs, N_valid) bool.
        Applies mix-contact override and random per-env contact part masking if configured."""
        part_labels = self._get_contact_data("contact_part_labels")[:, self._contact_valid_mask]
        part_labels = self._apply_contact_mask(part_labels)
        return part_labels > 0

    @property
    def current_contact_part_labels(self) -> torch.Tensor:
        """Per-env chair part labels at current timestep. Shape (num_envs, N_valid) long.
        0=none, 1=seat, 2=backrest, 3=legs. Applies mix-contact override and random masking."""
        labels = self._get_contact_data("contact_part_labels")[:, self._contact_valid_mask]
        return self._apply_contact_mask(labels)

    @property
    def current_contact_chair_pts(self) -> torch.Tensor:
        """Per-env chair surface points at current timestep. Shape (num_envs, K_max, 3)."""
        return self._get_motion_data("contact_chair_pts")

    @property
    def current_contact_chair_pts_mask(self) -> torch.Tensor:
        """Per-env validity mask for chair surface points. Shape (num_envs, K_max)."""
        return self._get_motion_data("contact_chair_pts_mask")

    @property
    def current_contact_chair_pts_part(self) -> torch.Tensor:
        """Per-env chair part label for each surface point. Shape (num_envs, K_max) long.
        0=unknown, 1=seat, 2=backrest, 3=legs."""
        return self._get_motion_data("contact_chair_pts_part")

    # ------------------------------------------------------------------
    # Per-motion object management
    # ------------------------------------------------------------------

    @property
    def has_objects(self) -> bool:
        """True when per-motion scene objects are configured."""
        return self._has_objects

    @property
    def active_object_pos_w(self) -> torch.Tensor | None:
        """World position of each env's active object. Shape (num_envs, 3).
        Returns underground position for envs with no object.
        ``None`` when no per-motion objects are configured."""
        return self._active_object_pos_w

    def _refresh_active_object_pos_w(self):
        """Update _active_object_pos_w from live asset data (needed for free objects)."""
        if not self._has_objects:
            return
        per_env_obj_idx = self._motion_to_object_idx[self.motion_ids]
        for obj_i in range(self._num_objects):
            active_mask = per_env_obj_idx == obj_i
            if active_mask.any():
                self._active_object_pos_w[active_mask] = (
                    self._object_assets[obj_i].data.root_pos_w[active_mask]
                )

    @property
    def active_obj_z_min(self) -> torch.Tensor:
        """Per-env object mesh Z minimum (local frame). Shape (num_envs,)."""
        return self._env_obj_z_min

    @property
    def active_obj_z_range(self) -> torch.Tensor:
        """Per-env object mesh Z extent. Shape (num_envs,)."""
        return self._env_obj_z_range

    @property
    def active_obj_thresholds(self) -> torch.Tensor:
        """Per-env normalized-Z part boundaries. Shape (num_envs, 2)."""
        return self._env_obj_thresholds

    @property
    def active_obj_segmented(self) -> torch.Tensor:
        """Per-env bool: True → height-based part split; False → single contact region.
        Shape (num_envs,)."""
        return self._env_obj_segmented

    def _resample_contact_mask(self, env_ids: torch.Tensor):
        """Resample which contact parts are randomly masked for given envs."""
        if self._masked_parts is None:
            return
        if self.cfg.random_mask_contact_whole:
            # One coin flip per env: mask all parts together or not at all
            mask = (torch.rand(len(env_ids), device=self.device) < self._random_mask_prob).unsqueeze(1)
            self._masked_parts[env_ids] = mask.expand(-1, len(self._random_mask_parts))
        else:
            # Each part independently masked
            self._masked_parts[env_ids] = (
                torch.rand(len(env_ids), len(self._random_mask_parts), device=self.device)
                < self._random_mask_prob
            )

    def _apply_contact_mask(self, part_labels: torch.Tensor) -> torch.Tensor:
        """Zero out randomly masked contact parts for each env.

        Args:
            part_labels: (num_envs, N) long tensor of part labels.
        Returns:
            Modified part_labels with masked parts set to 0.
        """
        obj_removed = self._object_removed is not None and self._object_removed.any()
        if self._masked_parts is None and not obj_removed:
            return part_labels
        part_labels = part_labels.clone()
        if self._masked_parts is not None:
            for i, part_id in enumerate(self._random_mask_part_ids):
                env_mask = self._masked_parts[:, i]  # (num_envs,) bool
                if env_mask.any():
                    body_mask = part_labels[env_mask] == part_id  # (n_masked_envs, N)
                    part_labels[env_mask] = part_labels[env_mask].masked_fill(body_mask, 0)
        if obj_removed:
            part_labels[self._object_removed] = 0
        return part_labels

    def _resample_mix_contact(self, env_ids: torch.Tensor):
        """For each env in env_ids, decide whether to borrow contact labels from a peer motion."""
        if self._mix_label_override is None or self._mix_group_peers is None:
            return
        self._mix_label_override[env_ids] = -1
        roll = torch.rand(len(env_ids), device=self.device)
        mix_mask = roll < self._mix_contact_label_prob
        if not mix_mask.any():
            return
        mix_env_ids = env_ids[mix_mask]
        for local_i, env_i in enumerate(mix_env_ids.tolist()):
            mid = int(self.motion_ids[env_i].item())
            peers = self._mix_group_peers[mid]
            if peers.numel() == 0:
                continue
            chosen = peers[torch.randint(0, peers.numel(), (1,), device=self.device)].item()
            self._mix_label_override[env_i] = chosen

    def _get_contact_data(self, field: str) -> torch.Tensor:
        """Get contact data, applying per-env mix override where set."""
        if self._mix_label_override is None:
            return self._get_motion_data(field)
        # Default: each env uses its own motion's data
        data = self._get_motion_data(field).clone()
        # Override envs that borrow from a peer
        override_mask = self._mix_label_override >= 0
        if not override_mask.any():
            return data
        # For each unique override motion, fetch all frames and index by each env's time_step
        override_env_ids = torch.where(override_mask)[0]
        src_motion_ids = self._mix_label_override[override_env_ids]  # peer motion ids
        # Temporarily swap motion_ids and fetch
        saved = self.motion_ids[override_env_ids].clone()
        self.motion_ids[override_env_ids] = src_motion_ids
        peer_data = self._get_motion_data(field)[override_env_ids]
        self.motion_ids[override_env_ids] = saved
        data[override_env_ids] = peer_data
        return data

    def _update_object_positions(self, env_ids: torch.Tensor):
        """Teleport active objects to their initial pose, inactive ones underground.

        When object_poses are present in the NPZ data (free-object mode), each active
        object is placed at the retargeted pose (pos + quat) at the current time_step,
        shifted into the env's world frame.  Velocity is zeroed so the object starts
        from rest and is free to interact with the robot from that point on.

        When no object_poses are available, falls back to the legacy behaviour of
        placing at env_origin plus optional per-motion XYZ offsets.

        Called on reset (after ``motion_ids`` are reassigned) and at init.
        """
        if not self._has_objects:
            return

        env_origins = self._env.scene.env_origins  # (num_envs, 3)
        per_env_obj_idx = self._motion_to_object_idx[self.motion_ids[env_ids]]  # (len(env_ids),)

        if self.cfg.object_remove_prob > 0.0:
            remove_mask = (
                (torch.rand(len(env_ids), device=self.device) < self.cfg.object_remove_prob)
                & (per_env_obj_idx >= 0)
            )
            if self._object_removed is None:
                self._object_removed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._object_removed[env_ids] = False
            if remove_mask.any():
                per_env_obj_idx = per_env_obj_idx.clone()
                per_env_obj_idx[remove_mask] = -1
                self._object_removed[env_ids[remove_mask]] = True

        if self._has_object_poses:
            # Per-env pose from retargeted data at the episode-start time_step.
            # object_poses shape: (num_envs, 7) = [x, y, z, qw, qx, qy, qz]
            all_obj_poses = self._get_motion_data("object_poses")   # (num_envs, 7)
            env_obj_pos_local = all_obj_poses[env_ids, :3]           # (n_reset, 3) local frame
            env_obj_quat = all_obj_poses[env_ids, 3:7]               # (n_reset, 4) [qw,qx,qy,qz]
        else:
            env_obj_pos_local = None
            env_obj_quat = None
            # Legacy: per-motion XYZ offsets and optional Y noise
            if self._per_motion_xyz_offsets is not None:
                per_env_offsets = self._per_motion_xyz_offsets[self.motion_ids[env_ids]]
            else:
                per_env_offsets = None
            if self.cfg.chair_y_rand_std > 0.0:
                y_noise = torch.randn(len(env_ids), device=self.device) * self.cfg.chair_y_rand_std
            else:
                y_noise = None

        for obj_i in range(self._num_objects):
            obj = self._object_assets[obj_i]
            active_mask = per_env_obj_idx == obj_i
            root_state = obj.data.default_root_state[env_ids].clone()

            active_local = torch.where(active_mask)[0]
            if active_local.numel() > 0:
                if self._has_object_poses:
                    # World-frame pos = local pose pos + env origin
                    root_state[active_local, :3] = (
                        env_obj_pos_local[active_local] + env_origins[env_ids[active_local]]
                    )
                    root_state[active_local, 3:7] = env_obj_quat[active_local]
                    # Zero velocity so free object starts from rest
                    root_state[active_local, 7:] = 0.0
                else:
                    root_state[active_local, :3] = env_origins[env_ids[active_local]]
                    if per_env_offsets is not None:
                        root_state[active_local, :3] += per_env_offsets[active_local]
                    if y_noise is not None:
                        root_state[active_local, 1] += y_noise[active_local]

            inactive_local = torch.where(~active_mask)[0]
            if inactive_local.numel() > 0:
                root_state[inactive_local, :3] = env_origins[env_ids[inactive_local]] + self._underground_pos

            obj.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Update cached active object position
        has_obj = per_env_obj_idx >= 0
        if self._has_object_poses:
            self._active_object_pos_w[env_ids[has_obj]] = (
                env_obj_pos_local[has_obj] + env_origins[env_ids[has_obj]]
            )
        else:
            self._active_object_pos_w[env_ids[has_obj]] = env_origins[env_ids[has_obj]]
            if per_env_offsets is not None:
                self._active_object_pos_w[env_ids[has_obj]] += per_env_offsets[has_obj]
            if y_noise is not None:
                self._active_object_pos_w[env_ids[has_obj], 1] += y_noise[has_obj]
        no_obj = ~has_obj
        if no_obj.any():
            self._active_object_pos_w[env_ids[no_obj]] = (
                env_origins[env_ids[no_obj]] + self._underground_pos
            )

        # Keep per-env geometry in sync with motion assignment
        self._update_object_geometry(env_ids)

    def _update_object_geometry(self, env_ids: torch.Tensor):
        """Sync per-env geometry tensors with the current motion→object assignment.

        No-op when no per-object geometry has been loaded from YAML.
        """
        if self._per_object_geom_z_min is None:
            return
        per_env_obj_idx = self._motion_to_object_idx[self.motion_ids[env_ids]]
        has_obj = per_env_obj_idx >= 0
        if not has_obj.any():
            return
        valid_env_ids = env_ids[has_obj]
        valid_obj_idx = per_env_obj_idx[has_obj]
        self._env_obj_z_min[valid_env_ids] = self._per_object_geom_z_min[valid_obj_idx]
        self._env_obj_z_range[valid_env_ids] = self._per_object_geom_z_range[valid_obj_idx]
        self._env_obj_thresholds[valid_env_ids] = self._per_object_geom_thresholds[valid_obj_idx]
        self._env_obj_segmented[valid_env_ids] = self._per_object_geom_segmented[valid_obj_idx]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        body_pos_error = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_pos"] = body_pos_error
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        # Per-motion metrics (only for small multi-motion sets)
        if self.multi_motion:
            self._episode_steps += 1
            terminated = self._env.termination_manager.terminated
            timed_out = self._env.termination_manager.time_outs
            done = terminated | timed_out

            if self._track_per_motion and torch.any(done):
                for i, name in enumerate(self.motion_names):
                    mask = (self.motion_ids == i) & done
                    if torch.any(mask):
                        success_mask = (self.motion_ids == i) & timed_out
                        fail_mask = (self.motion_ids == i) & terminated & ~timed_out
                        total = success_mask.sum() + fail_mask.sum()
                        if total > 0:
                            self.metrics[f"per_motion/{name}/success_rate"][:] = success_mask.sum().float() / total
                        self.metrics[f"per_motion/{name}/episode_length"][:] = self._episode_steps[mask].float().mean()
                        self.metrics[f"per_motion/{name}/error_body_pos"][:] = body_pos_error[mask].mean()

            if torch.any(done):
                self._episode_steps[done] = 0

    def _get_time_step_total(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get per-env motion length."""
        if self.multi_motion:
            if env_ids is not None:
                return self.motion_lib.time_steps_per_motion[self.motion_ids[env_ids]]
            return self.motion_lib.time_steps_per_motion[self.motion_ids]
        else:
            return torch.full((len(env_ids) if env_ids is not None else self.num_envs,),
                              self.motion.time_step_total, device=self.device)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        time_step_total = self._get_time_step_total(env_ids)
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps[env_ids] * self.bin_count) // torch.clamp(time_step_total, min=1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # Randomly assign new motion for multi-motion
        if self.multi_motion:
            self.motion_ids[env_ids] = torch.randint(
                0, self.motion_lib.num_motions, (len(env_ids),), device=self.device
            )

        new_time_step_total = self._get_time_step_total(env_ids)
        if self.cfg.start_from_zero:
            self.time_steps[env_ids] = 0
        else:
            self.time_steps[env_ids] = (
                (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
                / self.bin_count
                * (new_time_step_total.float() - 1)
            ).long()
            if self.cfg.start_from_zero_ratio > 0:
                zero_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.start_from_zero_ratio
                if zero_mask.any():
                    self.time_steps[env_ids[zero_mask]] = 0

        # Stand envs always start at frame 0
        stand_in_reset = self._stand_mask[env_ids]
        if stand_in_reset.any():
            self.time_steps[env_ids[stand_in_reset]] = 0

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        # Teleport per-motion objects (active → origin, inactive → underground)
        env_ids_t = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        self._update_object_positions(env_ids_t)

        # Resample random contact part masking
        self._resample_contact_mask(env_ids_t)
        # Resample mix-contact label overrides
        self._resample_mix_contact(env_ids_t)
        # Re-pin motion ids and contact override in force-peer mode
        if self._force_contact_peer_idx is not None:
            self.motion_ids[env_ids_t] = 0
            self._mix_label_override[env_ids_t] = self._force_contact_peer_idx

    def _update_command(self):
        self.time_steps += 1
        self.time_steps[self._stand_mask] = 0
        time_step_totals = self._get_time_step_total()
        env_ids = torch.where(self.time_steps >= time_step_totals)[0]
        self._resample_command(env_ids)
        # Keep cached object positions in sync with live simulation (free-object mode)
        self._refresh_active_object_pos_w()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_files: list[str] | None = None  # For multi-motion: list of NPZ paths
    motion_names: list[str] | None = None  # For multi-motion: human-readable names
    motion_dir: str | None = None  # Directory of NPZ files for large-scale training
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    start_from_zero: bool = False
    start_from_zero_ratio: float = 0.0
    """Per-reset probability that an env starts from frame 0 (covers sim2real startup transition)."""
    motion_z_offset: float = 0.0
    stand_ratio: float = 0.0
    """Fraction of envs that stay frozen at frame 0 (standing pose)."""
    ignore_contact_parts: list[int] | None = None
    """Part IDs to zero out from contact data (e.g. [2] to disable backrest). Applied permanently at init."""

    random_mask_contact_parts: list[int] | None = None
    """Part IDs eligible for random per-env masking (e.g. [1, 2, 3] for seat/backrest/legs).
    Each reset, each eligible part is independently masked with probability ``random_mask_contact_prob``,
    unless ``random_mask_contact_whole`` is True, in which case all parts are masked together."""
    random_mask_contact_prob: float = 0.0
    """Probability of masking at each reset. 0 = disabled."""
    random_mask_contact_whole: bool = False
    """When True, all parts in random_mask_contact_parts are masked together (one coin flip per env).
    When False (default), each part is masked independently."""

    mix_contact_labels: list[list[str]] | None = None
    """Groups of motion names whose contact labels are mixed during training.
    Each sub-list is a group, e.g. [[motionA1, motionA2], [motionB1, motionB2, motionB3]].
    At each reset, an env running motionX from a group will borrow contact labels from
    another motion in the same group (chosen uniformly) with probability ``mix_contact_label_prob``."""
    mix_contact_label_prob: float = 0.0
    """Probability of replacing an env's contact labels with those of another group member."""

    use_paired_contact: bool = False
    """When True, use 19-body paired contact sensors (PAIRED_CONTACT_BODIES) with
    per-object force_matrix_w. When False, use 46-body aggregate contact sensor."""

    object_urdfs: list[str | None] | None = None
    """Per-motion URDF paths for scene objects. Parallel to motion_files.
    None entry = motion has no object. None list = no objects at all."""

    object_geometry_yaml: str | None = None
    """Path to a YAML file mapping object name substrings to ObjectGeometryCfg entries.
    When set, each per-motion object is matched against the YAML to determine its
    z_min / z_range / segmented / part_thresholds for contact-part classification.
    If the env has no per-motion objects, the ``_default`` entry is applied uniformly."""

    object_asset_prefix: str = "motion_object"
    """Prefix for scene asset names: motion_object_0, motion_object_1, ..."""

    chair_y_rand_std: float = 0.0
    """Std (meters) of Gaussian noise applied to the chair Y position at each reset.
    E.g. 0.05 for ±~5 cm variation. Does not affect the reference motion keypoints."""

    object_remove_prob: float = 0.0
    """Per-reset probability of sending the motion object underground (effectively removing it) for each env.
    Useful for training robustness when the object may be absent at test time."""

    chair_offset_per_motion: dict | None = None
    """Per-motion chair XYZ offsets loaded from a YAML file.
    Format: {motion_name: {x: float, y: float, z: float}}.
    When set, overrides the scalar chair_x/y/z_offset for all motions."""

    force_contact_peer_idx: int | None = None
    """If set, all envs are pinned to motion_id=0 for keypoints, but contact labels
    are always borrowed from this motion index. Used for strict mixed-contact play eval."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

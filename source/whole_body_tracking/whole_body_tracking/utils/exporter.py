# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        if cmd.multi_motion:
            # Export ALL motions as (num_motions, T_max, ...)
            self.joint_pos = cmd.motion_lib.joint_pos.to("cpu")
            self.joint_vel = cmd.motion_lib.joint_vel.to("cpu")
            self.body_pos_w = cmd.motion_lib.body_pos_w.to("cpu")
            self.body_quat_w = cmd.motion_lib.body_quat_w.to("cpu")
            self.body_lin_vel_w = cmd.motion_lib.body_lin_vel_w.to("cpu")
            self.body_ang_vel_w = cmd.motion_lib.body_ang_vel_w.to("cpu")
            self.time_steps_per_motion = cmd.motion_lib.time_steps_per_motion.to("cpu").long()
            raw_part_labels = cmd.motion_lib.contact_part_labels  # (N, T, N_full)
        else:
            # Promote single motion to (1, T, ...) for a unified inference API
            self.joint_pos = cmd.motion.joint_pos.unsqueeze(0).to("cpu")
            self.joint_vel = cmd.motion.joint_vel.unsqueeze(0).to("cpu")
            self.body_pos_w = cmd.motion.body_pos_w.unsqueeze(0).to("cpu")
            self.body_quat_w = cmd.motion.body_quat_w.unsqueeze(0).to("cpu")
            self.body_lin_vel_w = cmd.motion.body_lin_vel_w.unsqueeze(0).to("cpu")
            self.body_ang_vel_w = cmd.motion.body_ang_vel_w.unsqueeze(0).to("cpu")
            self.time_steps_per_motion = torch.tensor(
                [cmd.motion.time_step_total], dtype=torch.long
            )
            raw_part_labels = cmd.motion.contact_part_labels.unsqueeze(0) \
                if hasattr(cmd.motion, "contact_part_labels") else None
        self.num_motions = self.joint_pos.shape[0]
        self.time_step_max = self.joint_pos.shape[1]

        # Contact part labels (valid-masked to N_valid columns). Shape (N, T, N_valid) int64.
        # 0=none, 1=seat, 2=backrest, 3=legs. Binary obs is (labels > 0).
        if cmd.has_contact and raw_part_labels is not None:
            valid_mask = cmd._contact_valid_mask.to(raw_part_labels.device)
            self.contact_part_labels = raw_part_labels[..., valid_mask].to("cpu").long()
        else:
            self.contact_part_labels = torch.zeros(self.num_motions, self.time_step_max, 0,
                                                   dtype=torch.long)

        # Apply mix-contact groups: for each motion in a group, replace its contact labels
        # with those from the first peer in the group (deterministic export-time analogue of
        # the stochastic runtime mixing).
        if (
            cmd.multi_motion
            and getattr(cmd, "_mix_group_peers", None) is not None
            and self.contact_part_labels.shape[-1] > 0
        ):
            groups = getattr(cmd.cfg, "mix_contact_labels", None)
            if groups:
                motion_files = list(getattr(cmd.cfg, "motion_files", []) or [])
                motion_names_list = list(getattr(cmd.cfg, "motion_names", []) or [])
                if not motion_names_list:
                    motion_names_list = [
                        os.path.splitext(os.path.basename(f))[0] for f in motion_files
                    ]
                name_to_idx = {n: i for i, n in enumerate(motion_names_list)}
                for group in groups:
                    idxs = [name_to_idx[n] for n in group if n in name_to_idx]
                    if len(idxs) < 2:
                        continue
                    src_idx = idxs[0]
                    src_labels = self.contact_part_labels[src_idx]  # (T, N_valid)
                    for dst_idx in idxs[1:]:
                        T_src = int(cmd.motion_lib.time_steps_per_motion[src_idx].item())
                        T_dst = int(cmd.motion_lib.time_steps_per_motion[dst_idx].item())
                        T_copy = min(T_src, T_dst)
                        self.contact_part_labels[dst_idx, :T_copy] = src_labels[:T_copy]
                        if T_copy < T_dst:
                            self.contact_part_labels[dst_idx, T_copy:T_dst] = 0

        # Handle force_contact_peer_idx: bake peer's contact labels into motion-0 slot.
        # At runtime all envs follow motion-0 keypoints with peer's contact labels,
        # so the exported ONNX should reflect that.
        force_peer = getattr(cmd, "_force_contact_peer_idx", None)
        if (
            force_peer is not None
            and cmd.multi_motion
            and self.contact_part_labels.shape[-1] > 0
        ):
            peer_idx = int(force_peer)
            T_primary = int(cmd.motion_lib.time_steps_per_motion[0].item())
            T_peer = int(cmd.motion_lib.time_steps_per_motion[peer_idx].item())
            T_copy = min(T_primary, T_peer)
            self.contact_part_labels[0, :T_copy] = self.contact_part_labels[peer_idx, :T_copy]
            if T_copy < T_primary:
                self.contact_part_labels[0, T_copy:T_primary] = 0

    def forward(self, x, time_step, motion_id):
        t = torch.clamp(time_step.long().squeeze(-1), min=0, max=self.time_step_max - 1)
        m = torch.clamp(motion_id.long().squeeze(-1), min=0, max=self.num_motions - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[m, t],
            self.joint_vel[m, t],
            self.body_pos_w[m, t],
            self.body_quat_w[m, t],
            self.body_lin_vel_w[m, t],
            self.body_ang_vel_w[m, t],
            self.contact_part_labels[m, t],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        motion_id = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step, motion_id),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step", "motion_id"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "contact_part_labels",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    cmd = env.command_manager.get_term("motion")
    if cmd.multi_motion:
        motion_lengths = cmd.motion_lib.time_steps_per_motion.cpu().tolist()
        motion_names = list(getattr(cmd.cfg, "motion_names", []) or [])
        if not motion_names:
            motion_files = list(getattr(cmd.cfg, "motion_files", []) or [])
            motion_names = [os.path.basename(os.path.dirname(f)) or os.path.basename(f)
                            for f in motion_files]
    else:
        motion_lengths = [int(cmd.motion.time_step_total)]
        motion_names = [os.path.basename(os.path.dirname(cmd.cfg.motion_file))
                        or os.path.basename(cmd.cfg.motion_file)]

    # Contact body names (valid-masked; aligns with ONNX contact_part_labels columns).
    contact_body_names: list[str] = []
    if getattr(cmd, "has_contact", False):
        src = cmd.motion_lib if cmd.multi_motion else cmd.motion
        if src.contact_body_names is not None:
            valid = cmd._contact_valid_mask.cpu().tolist()
            contact_body_names = [n for n, v in zip(src.contact_body_names, valid) if v]

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos[0].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
        "num_motions": len(motion_lengths),
        "motion_lengths": motion_lengths,
        "motion_names": motion_names,
        "contact_body_names": contact_body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)

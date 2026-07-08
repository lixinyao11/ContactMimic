"""Convert HUMOTO retargeted NPZ directly to Isaac-ready motion NPZ with optional contact labels.

Replaces the humoto_to_csv.py → csv_to_npz.py two-step pipeline.
Reads HUMOTO qpos format directly, interpolates from input_fps to output_fps with
lerp/slerp, runs Isaac Sim for body poses, and uploads everything to wandb.

If the input NPZ contains contact fields (contact_body_names, contact_labels, etc.),
they are automatically resampled to match the output_fps and included in the output.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/humoto_to_npz.py \\
        --input HUMOTO/sitting_on_lowchair-928/sitting_on_lowchair-928_contact-label.npz \\
        --output_name sitting_on_lowchair-928-contact \\
        --prepend_seconds 1.5 \\
        --pad_seconds 3.0 \\
        --headless
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert HUMOTO retargeted NPZ to motion NPZ for training.")
parser.add_argument("--input", type=str, required=True,
                    help="Path to HUMOTO .npz (qpos format). Contact fields auto-detected if present.")
parser.add_argument("--output_name", type=str, required=True, help="WandB artifact name")
parser.add_argument("--output_fps", type=int, default=50, help="Output frame rate (default: 50)")
parser.add_argument("--pad_seconds", type=float, default=0.0, help="Seconds to hold final pose after motion ends")
parser.add_argument("--prepend_seconds", type=float, default=0.0, help="Seconds to hold first frame (standing pose) before motion starts")
parser.add_argument("--chair_mesh", type=str, default=None,
                    help="Path to this motion's low_chair.obj. If provided, rotates trajectory to align "
                         "with Isaac Sim's chair mesh orientation.")
parser.add_argument("--no_align", action="store_true", default=False,
                    help="Skip chair alignment. Use when per-motion object URDFs are loaded at training time "
                         "(the original chair mesh is spawned per env, so no alignment needed).")
parser.add_argument("--frame_range", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="Slice input motion to [START, END) frames before processing (qpos + contacts).")
parser.add_argument("--single_part_label", type=int, default=None,
                    help="If set, every contact frame is labelled with this part ID (overrides chair-label "
                         "lookup). Use for single-object non-segmented motions (board, working chair, shelf).")
parser.add_argument("--wandb_project", type=str, default="beyondmimic")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="WandB entity/username (leave empty to use default).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.assets import ASSET_DIR


# ---------------------------------------------------------------------------
# Chair mesh alignment: rigid 2D transform (yaw + XY) to match Isaac Sim chair
# ---------------------------------------------------------------------------

REFERENCE_CHAIR_MESH = f"{ASSET_DIR}/low_chair/meshes/low_chair.obj"


def _load_obj_verts(path: str) -> np.ndarray:
    """Load OBJ file and return all vertices as (N, 3)."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)


def _chair_facing_angle(verts: np.ndarray) -> float:
    """Compute the direction a chair faces using backrest vs seat height.

    The backrest has the highest Z vertices; the seat has the lowest.
    The chair faces from backrest toward seat (the person sits facing away from backrest).
    """
    xy = verts[:, :2]
    z = verts[:, 2]
    top_mask = z > np.percentile(z, 80)
    bottom_mask = z < np.percentile(z, 40)
    backrest_center = xy[top_mask].mean(axis=0)
    seat_center = xy[bottom_mask].mean(axis=0)
    facing = seat_center - backrest_center
    return np.arctan2(facing[1], facing[0])


def compute_chair_transform(motion_chair_mesh: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute full 2D rigid transform to align motion's chair to Isaac Sim's chair.

    Uses the chair facing direction (backrest→seat) from the *_original.obj mesh
    for yaw alignment, and mesh centroid for translation. The original (non-decomposed)
    mesh is preferred because all HUMOTO motions share the same base mesh (2770 verts),
    giving consistent centroids and facing directions. Falls back to the provided mesh
    if no _original variant exists.

    Returns (yaw_delta, motion_center_xy, ref_center_xy).
    """
    import os

    # Prefer original (non-decomposed) mesh for consistent geometry
    motion_original = motion_chair_mesh.replace("low_chair.obj", "low_chair_original.obj")
    if os.path.exists(motion_original):
        motion_verts = _load_obj_verts(motion_original)
    else:
        motion_verts = _load_obj_verts(motion_chair_mesh)

    # Reference: use 928's original mesh (same base geometry as all HUMOTO chairs)
    ref_original = os.path.join(os.path.dirname(REFERENCE_CHAIR_MESH), "low_chair_original.obj")
    if os.path.exists(ref_original):
        ref_verts = _load_obj_verts(ref_original)
    else:
        ref_verts = _load_obj_verts(REFERENCE_CHAIR_MESH)

    motion_center_xy = motion_verts[:, :2].mean(axis=0)
    ref_center_xy = ref_verts[:, :2].mean(axis=0)

    ref_facing = _chair_facing_angle(ref_verts)
    motion_facing = _chair_facing_angle(motion_verts)
    yaw_delta = ref_facing - motion_facing
    # Normalize to [-pi, pi]
    yaw_delta = (yaw_delta + np.pi) % (2 * np.pi) - np.pi

    print(f"[INFO]: Chair alignment: ref_facing={np.degrees(ref_facing):.1f}°, "
          f"motion_facing={np.degrees(motion_facing):.1f}°, yaw_delta={np.degrees(yaw_delta):.1f}°")
    print(f"[INFO]: Chair XY translation: motion_center={motion_center_xy}, "
          f"ref_center={ref_center_xy}")
    return yaw_delta, motion_center_xy.copy(), ref_center_xy.copy()


def align_qpos(qpos: np.ndarray, yaw_delta: float,
               motion_center_xy: np.ndarray, ref_center_xy: np.ndarray) -> np.ndarray:
    """Apply rigid 2D transform to align qpos trajectory from motion chair frame to Isaac Sim chair frame.

    qpos layout: [xyz(3), wxyz(4), joints(29)] = 36 per frame.

    Transform: translate to motion chair center, rotate by yaw_delta, translate to ref chair center.
    This preserves the relative pose between robot and chair.
    """
    qpos = qpos.copy()

    # 1. Translate root XY to center on motion chair
    qpos[:, 0] -= motion_center_xy[0]
    qpos[:, 1] -= motion_center_xy[1]

    # 2. Rotate around origin (now the chair center)
    if abs(yaw_delta) > 1e-6:
        cos_d, sin_d = np.cos(yaw_delta), np.sin(yaw_delta)
        xy = qpos[:, :2].copy()
        qpos[:, 0] = cos_d * xy[:, 0] - sin_d * xy[:, 1]
        qpos[:, 1] = sin_d * xy[:, 0] + cos_d * xy[:, 1]

        # Rotate root quaternion: q_new = q_yaw * q_old (wxyz format)
        half = yaw_delta / 2.0
        qw, qx, qy, qz = np.cos(half), 0.0, 0.0, np.sin(half)
        ow, ox, oy, oz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        qpos[:, 3] = qw * ow - qx * ox - qy * oy - qz * oz
        qpos[:, 4] = qw * ox + qx * ow + qy * oz - qz * oy
        qpos[:, 5] = qw * oy - qx * oz + qy * ow + qz * ox
        qpos[:, 6] = qw * oz + qx * oy - qy * ox + qz * ow

        # Normalize quaternion to prevent drift
        qnorm = np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
        qpos[:, 3:7] /= qnorm

    # 3. Translate to reference chair position
    qpos[:, 0] += ref_center_xy[0]
    qpos[:, 1] += ref_center_xy[1]

    return qpos


def align_contact_points(pts_array: np.ndarray, yaw_delta: float,
                         motion_center_xy: np.ndarray, ref_center_xy: np.ndarray) -> np.ndarray:
    """Apply rigid 2D transform to contact points (object array of (K,3)).

    Same transform as align_qpos: center on motion chair, rotate, translate to ref chair.
    """
    cos_d, sin_d = np.cos(yaw_delta), np.sin(yaw_delta)
    out = np.empty(len(pts_array), dtype=object)
    for t in range(len(pts_array)):
        pts = pts_array[t]
        if len(pts) == 0:
            out[t] = pts
            continue
        pts = pts.copy().astype(np.float64)
        # 1. Translate to center on motion chair
        pts[:, 0] -= motion_center_xy[0]
        pts[:, 1] -= motion_center_xy[1]
        # 2. Rotate around origin
        if abs(yaw_delta) > 1e-6:
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            pts[:, 0] = cos_d * x - sin_d * y
            pts[:, 1] = sin_d * x + cos_d * y
        # 3. Translate to reference chair position
        pts[:, 0] += ref_center_xy[0]
        pts[:, 1] += ref_center_xy[1]
        out[t] = pts.astype(np.float32)
    return out

# Joint order must match the robot's actuated DOF order
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Motion loading and interpolation (pure numpy/torch, no Isaac Sim)
# ---------------------------------------------------------------------------

def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1 - blend) + b * blend


def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = quat_slerp(a[i], b[i], blend[i])
    return out


def _so3_velocity(quats: torch.Tensor, dt: float) -> torch.Tensor:
    """Finite-difference angular velocity from quaternion sequence. Shape (T, 4) → (T, 3)."""
    q_prev, q_next = quats[:-2], quats[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def load_and_interpolate(input_file: str, output_fps: int, pad_seconds: float, prepend_seconds: float, device: torch.device, chair_mesh: str | None = None) -> tuple[dict, dict]:
    """Load HUMOTO NPZ (qpos format) and interpolate to output_fps.

    HUMOTO qpos layout:
      Standard (36 cols): [xyz(3), wxyz(4), joints(29)]
      Extended (43 cols): [xyz(3), wxyz(4), joints(29), obj_freejoint(7)]

    The object freejoint columns (if present) are redundant with the top-level
    ``object_poses`` key that newer files store separately.

    Returns (motion_dict, raw_npz_dict).  raw_npz_dict is the original numpy
    archive so callers can check for optional contact fields.
    """
    raw = dict(np.load(input_file, allow_pickle=True))
    qpos = raw["qpos"]          # (T_in, 36) or (T_in, 43)
    input_fps = int(raw["fps"])

    # Align chair pose (yaw + XY translation) if mesh provided
    if chair_mesh is not None:
        yaw_delta, motion_center_xy, ref_center_xy = compute_chair_transform(chair_mesh)
        qpos = align_qpos(qpos, yaw_delta, motion_center_xy, ref_center_xy)
        # Fine-tune translation: match sitting root XY to reference 928 trajectory.
        # After coarse alignment the person may be slightly off-center due to mesh
        # centroid differences.  Correct by shifting so that the mean sitting root
        # position matches the reference (928) sitting root position.
        # 928 sitting root XY (pre-computed, no alignment needed since its mesh == ref):
        _REF_SITTING_ROOT_XY = np.array([-0.00117761, 0.41526806])
        sitting = slice(len(qpos) // 2, len(qpos))
        current_sitting_xy = qpos[sitting, :2].mean(axis=0)
        correction = _REF_SITTING_ROOT_XY - current_sitting_xy
        qpos[:, 0] += correction[0]
        qpos[:, 1] += correction[1]
        print(f"[INFO]: Sitting root correction: {correction} "
              f"(sitting root {current_sitting_xy} -> {_REF_SITTING_ROOT_XY})")
        raw["qpos"] = qpos
        # Transform contact points with the same rigid 2D transform
        if "contact_points_robot" in raw:
            raw["contact_points_robot"] = align_contact_points(
                raw["contact_points_robot"], yaw_delta, motion_center_xy, ref_center_xy)
        if "contact_points_object" in raw:
            raw["contact_points_object"] = align_contact_points(
                raw["contact_points_object"], yaw_delta, motion_center_xy, ref_center_xy)
        raw["_yaw_delta"] = yaw_delta
    else:
        raw["_yaw_delta"] = 0.0
    T_in = qpos.shape[0]
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (T_in - 1) * input_dt

    print(f"[INFO]: Loaded {input_file}: {T_in} frames @ {input_fps}fps ({duration:.3f}s)")
    print(f"[INFO]: Root Z range: {qpos[:, 2].min():.4f} – {qpos[:, 2].max():.4f}")

    n_joints = len(JOINT_NAMES)  # 29 — always extract exactly this many joints
    pos_in   = torch.tensor(qpos[:, :3],              dtype=torch.float32, device=device)
    quat_in  = torch.tensor(qpos[:, 3:7],             dtype=torch.float32, device=device)  # wxyz
    joint_in = torch.tensor(qpos[:, 7:7 + n_joints],  dtype=torch.float32, device=device)

    # Load object poses from top-level key (present in extended 43-col format).
    obj_poses_in = None
    if "object_poses" in raw:
        obj_poses_in = torch.tensor(raw["object_poses"], dtype=torch.float32, device=device)  # (T_in, 7)
        print(f"[INFO]: object_poses found in input ({obj_poses_in.shape})")

    # --- Interpolate to output_fps ---
    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    T_motion = times.shape[0]   # frames for the actual motion (before padding)

    phase = times / duration
    idx0 = (phase * (T_in - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(T_in - 1, device=device))
    blend = (phase * (T_in - 1) - idx0).unsqueeze(1)  # (T_motion, 1)

    pos   = _lerp(pos_in[idx0], pos_in[idx1], blend)
    quat  = _slerp(quat_in[idx0], quat_in[idx1], blend.squeeze(1))
    joint = _lerp(joint_in[idx0], joint_in[idx1], blend)

    if obj_poses_in is not None:
        obj_pos_interp  = _lerp(obj_poses_in[idx0, :3], obj_poses_in[idx1, :3], blend)
        obj_quat_interp = _slerp(obj_poses_in[idx0, 3:], obj_poses_in[idx1, 3:], blend.squeeze(1))

    print(f"[INFO]: Interpolated {T_in}@{input_fps}fps → {T_motion}@{output_fps}fps")

    # --- Prepend: hold first frame (standing pose) ---
    n_prepend = 0
    if prepend_seconds > 0:
        n_prepend = int(prepend_seconds * output_fps)
        pos   = torch.cat([pos[:1].expand(n_prepend, -1),   pos],   dim=0)
        quat  = torch.cat([quat[:1].expand(n_prepend, -1),  quat],  dim=0)
        joint = torch.cat([joint[:1].expand(n_prepend, -1), joint], dim=0)
        if obj_poses_in is not None:
            obj_pos_interp  = torch.cat([obj_pos_interp[:1].expand(n_prepend, -1),  obj_pos_interp],  dim=0)
            obj_quat_interp = torch.cat([obj_quat_interp[:1].expand(n_prepend, -1), obj_quat_interp], dim=0)
        T_motion = T_motion + n_prepend
        print(f"[INFO]: Prepended {n_prepend} frames ({prepend_seconds}s) of first frame. "
              f"Motion now: {T_motion} frames ({T_motion/output_fps:.2f}s)")

    # --- Velocities ---
    lin_vel   = torch.gradient(pos,   spacing=output_dt, dim=0)[0]
    joint_vel = torch.gradient(joint, spacing=output_dt, dim=0)[0]
    ang_vel   = _so3_velocity(quat, output_dt)

    # --- Padding: freeze final pose ---
    T_out = T_motion
    if pad_seconds > 0:
        n_pad = int(pad_seconds * output_fps)
        zeros3  = torch.zeros(n_pad, 3,              device=device)
        zeros_j = torch.zeros(n_pad, joint.shape[1], device=device)
        pos       = torch.cat([pos,       pos[-1:].expand(n_pad, -1)],  dim=0)
        quat      = torch.cat([quat,      quat[-1:].expand(n_pad, -1)], dim=0)
        joint     = torch.cat([joint,     joint[-1:].expand(n_pad, -1)],dim=0)
        lin_vel   = torch.cat([lin_vel,   zeros3],  dim=0)
        ang_vel   = torch.cat([ang_vel,   zeros3],  dim=0)
        joint_vel = torch.cat([joint_vel, zeros_j], dim=0)
        if obj_poses_in is not None:
            obj_pos_interp  = torch.cat([obj_pos_interp,  obj_pos_interp[-1:].expand(n_pad, -1)],  dim=0)
            obj_quat_interp = torch.cat([obj_quat_interp, obj_quat_interp[-1:].expand(n_pad, -1)], dim=0)
        T_out = T_motion + n_pad
        print(f"[INFO]: Padded {n_pad} frames ({pad_seconds}s). Total: {T_out} frames ({T_out/output_fps:.2f}s)")

    object_poses = torch.cat([obj_pos_interp, obj_quat_interp], dim=1) if obj_poses_in is not None else None

    return dict(
        pos=pos, quat=quat, lin_vel=lin_vel, ang_vel=ang_vel,
        joint=joint, joint_vel=joint_vel,
        T_out=T_out, T_motion=T_motion, input_fps=input_fps,
        n_prepend=n_prepend, object_poses=object_poses,
    ), raw


# ---------------------------------------------------------------------------
# Contact label resampling (pure numpy, no Isaac Sim)
# ---------------------------------------------------------------------------

def resample_contact(raw: dict, T_out: int, output_fps: int, n_prepend: int = 0) -> dict:
    """Resample contact fields from the input NPZ to T_out frames @ output_fps.

    Boolean and object arrays use nearest-neighbor (floor) resampling.
    Frames beyond the contact data duration are clamped to the last contact frame
    (correct for padded frozen-pose frames).
    Prepended frames (standing pose) are filled with no-contact data.
    """
    contact_fps = int(raw["fps"])
    T_contact   = raw["contact_labels"].shape[0]
    N_bodies    = raw["contact_labels"].shape[1]
    contact_dt  = 1.0 / contact_fps

    print(f"[INFO]: Contact data present: {T_contact} frames @ {contact_fps}fps "
          f"({(T_contact - 1) / contact_fps:.3f}s)")

    # Map each output frame index → nearest contact frame index (clamp at end)
    # Offset by n_prepend so prepended frames map to negative time (handled below)
    t_out = (np.arange(T_out) - n_prepend) / output_fps
    ci = np.minimum(np.maximum((t_out / contact_dt).astype(int), 0), T_contact - 1)

    # Fixed-shape: bool label matrix
    contact_labels = raw["contact_labels"][ci]          # (T_out, N) bool
    # Zero out prepended frames (no contact during standing pose)
    if n_prepend > 0:
        contact_labels[:n_prepend] = False

    # Variable-length object arrays: resample by index
    src_pts_robot  = raw["contact_points_robot"]
    src_pts_object = raw["contact_points_object"]
    src_chair_lbl  = raw["contact_chair_labels"]

    empty_pts = np.zeros((0, 3), dtype=np.float32)
    empty_lbl = np.array([], dtype=str)

    pts_robot  = np.empty(T_out, dtype=object)
    pts_object = np.empty(T_out, dtype=object)
    chair_lbl  = np.empty(T_out, dtype=object)
    for i, c in enumerate(ci):
        if i < n_prepend:
            pts_robot[i]  = empty_pts
            pts_object[i] = empty_pts
            chair_lbl[i]  = empty_lbl
        else:
            pts_robot[i]  = src_pts_robot[c].astype(np.float32)
            pts_object[i] = src_pts_object[c].astype(np.float32)
            chair_lbl[i]  = src_chair_lbl[c]

    print(f"[INFO]: Contact data resampled to {T_out} frames @ {output_fps}fps"
          + (f" ({n_prepend} prepended frames with no contact)" if n_prepend > 0 else ""))
    print(f"[INFO]: {contact_labels.sum(axis=1).astype(bool).sum()} / {T_out} frames have active contacts")

    return dict(
        contact_body_names=raw["contact_body_names"],
        contact_labels=contact_labels,
        contact_points_robot=pts_robot,
        contact_points_object=pts_object,
        contact_chair_labels=chair_lbl,
    )


# ---------------------------------------------------------------------------
# Isaac Sim replay loop to extract body poses via FK
# ---------------------------------------------------------------------------

def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion: dict) -> dict:
    """Kinematically pose robot in Isaac Sim each frame to extract body_pos_w etc."""
    robot = scene["robot"]
    joint_ids = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

    T_out = motion["T_out"]
    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}

    for idx in range(T_out):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3]  = motion["pos"][idx:idx+1]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = motion["quat"][idx:idx+1]
        root_state[:, 7:10] = motion["lin_vel"][idx:idx+1]
        root_state[:, 10:]  = motion["ang_vel"][idx:idx+1]
        robot.write_root_state_to_sim(root_state)

        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[:, joint_ids] = motion["joint"][idx:idx+1]
        jv[:, joint_ids] = motion["joint_vel"][idx:idx+1]
        robot.write_joint_state_to_sim(jp, jv)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if not simulation_app.is_running():
            print("[WARNING]: Simulation app stopped early.")
            break

    for k in log:
        log[k] = np.stack(log[k], axis=0)

    return log


# ---------------------------------------------------------------------------
# Per-body chair part label precomputation (requires body_pos_w from Isaac Sim)
# ---------------------------------------------------------------------------

CHAIR_PART_MAP = {"seat": 1, "backrest": 2, "legs": 3}


def compute_contact_part_labels(
    body_pos_w: np.ndarray, contact: dict, robot_body_names: list[str],
    single_part_label: int | None = None,
) -> np.ndarray:
    """For each frame & active body, find closest contact point and record its chair part.

    Args:
        body_pos_w: (T, num_bodies, 3) from Isaac Sim FK.
        contact: dict with contact_body_names, contact_labels, contact_points_robot, contact_chair_labels.
        robot_body_names: body name list in body_pos_w column order.
        single_part_label: if set, all contact frames get this part ID (skip chair-label lookup).

    Returns:
        (T, N) int8 — 0=none, 1=seat, 2=backrest, 3=legs.
    """
    contact_body_names = list(contact["contact_body_names"])
    labels = contact["contact_labels"]      # (T, N) bool
    pts_robot = contact["contact_points_robot"]   # object array of (K_t, 3)
    chair_lbl = contact["contact_chair_labels"]   # object array of (K_t,) str

    T, N = labels.shape
    part_labels = np.zeros((T, N), dtype=np.int8)

    if single_part_label is not None:
        part_labels[labels] = single_part_label
        n_labeled = (part_labels > 0).sum()
        print(f"[INFO]: --single_part_label={single_part_label} → {n_labeled} body-frame entries labeled")
        return part_labels

    name_to_idx = {n: i for i, n in enumerate(robot_body_names)}

    for t in range(T):
        if not labels[t].any():
            continue
        rpts = pts_robot[t]
        clbl = chair_lbl[t]
        if len(rpts) == 0:
            continue
        for i in range(N):
            if not labels[t, i]:
                continue
            body_name = contact_body_names[i]
            if body_name not in name_to_idx:
                continue
            body_pos = body_pos_w[t, name_to_idx[body_name]]
            dists = np.linalg.norm(rpts - body_pos, axis=1)
            closest = int(np.argmin(dists))
            raw_label = str(clbl[closest])
            part_key = raw_label.split(":")[-1] if ":" in raw_label else raw_label
            part_labels[t, i] = CHAIR_PART_MAP.get(part_key, 0)

    n_labeled = (part_labels > 0).sum()
    print(f"[INFO]: Computed contact_part_labels: {n_labeled} body-frame entries labeled")
    return part_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(args_cli.device)

    # 1. Load and interpolate motion
    chair_mesh = None if args_cli.no_align else args_cli.chair_mesh
    if args_cli.no_align:
        print("[INFO]: --no_align: skipping chair alignment (per-motion object loaded at training time)")
    motion, raw = load_and_interpolate(args_cli.input, args_cli.output_fps, args_cli.pad_seconds, args_cli.prepend_seconds, device, chair_mesh)

    # 2. Resample contact labels if present in the input NPZ
    contact = None
    if "contact_labels" in raw:
        contact = resample_contact(raw, motion["T_out"], args_cli.output_fps, motion["n_prepend"])
    else:
        print("[INFO]: No contact fields found in input, skipping.")

    # 3. Run Isaac Sim to extract body poses
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[INFO]: Setup complete, extracting body poses...")

    body_log = run_simulator(sim, scene, motion)

    # 4. Compute per-body chair part labels (needs body_pos_w + robot body names)
    if contact is not None:
        robot = scene["robot"]
        contact["contact_part_labels"] = compute_contact_part_labels(
            body_log["body_pos_w"], contact, list(robot.body_names),
            single_part_label=args_cli.single_part_label,
        )

    # 5. Build output dict
    out = {"fps": np.array([args_cli.output_fps])}
    out.update(body_log)
    if contact is not None:
        out.update(contact)
    if motion["object_poses"] is not None:
        out["object_poses"] = motion["object_poses"].cpu().numpy()
        print(f"[INFO]: object_poses included: {out['object_poses'].shape}")

    T_actual = out["joint_pos"].shape[0]
    print(f"[INFO]: Final NPZ: {T_actual} frames @ {args_cli.output_fps}fps "
          f"({T_actual / args_cli.output_fps:.2f}s)")
    print(f"[INFO]: Keys: {list(out.keys())}")

    # 6. Save and upload to wandb
    np.savez("/tmp/motion.npz", **out)
    print("[INFO]: Saved to /tmp/motion.npz")

    try:
        import wandb
        run = wandb.init(project=args_cli.wandb_project, entity=args_cli.wandb_entity, name=args_cli.output_name)
        artifact = run.log_artifact("/tmp/motion.npz", name=args_cli.output_name, type="motions")
        try:
            run.link_artifact(artifact=artifact, target_path=f"wandb-registry-motions/{args_cli.output_name}")
            print(f"[INFO]: Linked to wandb registry: motions/{args_cli.output_name}")
        except Exception as e:
            print(f"[WARNING]: Registry link failed (artifact still saved): {e}")
        run.finish()
    except Exception as e:
        print(f"[WARNING]: wandb upload failed: {e}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
    import sys
    sys.exit(0)

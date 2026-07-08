import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.mdp.commands import ObjectGeometryCfg, PAIRED_CONTACT_BODIES

_OBJECT_GEOMETRY_YAML = os.path.join(ASSET_DIR, "object_geometry.yaml")


@configclass
class G1SittingEnvCfg(G1FlatEnvCfg):
    """G1 sitting-on-chair environment with distance-based chair rewards."""

    # # Contact observation switches (set before __post_init__ or override after)
    enable_contact_labels_ref: bool = False
    enable_contact_body_chair_distance: bool = False
    enable_contact_labels_real: bool = False
    enable_contact_labels_real_binary: bool = False
    enable_contact_labels_ref_binary: bool = True

    use_paired_contact: bool = True
    """When True, use 19-body paired contact sensors with per-object force_matrix_w.
    When False, use 46-body aggregate contact sensor (legacy)."""

    enable_unwanted_contact_penalty: bool = True

    contact_label_tp_fp_mode: bool = False
    """If True, use TP-FP mode for contact_label_match: reward = TPR - fp_penalty_weight * FPR.
    Stronger gradient for sparse-contact motions (e.g. board-wipe) where TNR saturates."""

    contact_label_fp_penalty_weight: float = 1.0
    """Weight for false positive penalty in tp_fp_mode. Higher = stronger FP suppression."""
    """When True, penalize bodies that should not contact the chair but are within
    unwanted_penalty_threshold distance. Set to False to disable the penalty term."""

    def __post_init__(self):
        super().__post_init__()

        # Single-body chair — free (non-kinematic) so it can react to robot contacts
        self.scene.chair = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Chair",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=f"{ASSET_DIR}/low_chair/low_chair_auto_decomp.urdf",
                fix_base=False,
                activate_contact_sensors=True,
                collider_type="convex_decomposition",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.commands.motion.object_geometry_yaml = _OBJECT_GEOMETRY_YAML
        self.commands.motion.motion_z_offset = 0.0
        # 50% envs frozen at frame 0 (standing pose)
        self.commands.motion.stand_ratio = 0.2
        # Disable backrest contact (part 2)
        self.commands.motion.ignore_contact_parts = []
        # Randomly mask contact parts per env at each reset (data augmentation)
        self.commands.motion.random_mask_contact_parts = []  # 123: seat, backrest, legs
        self.commands.motion.random_mask_contact_prob = 0.0
        self.commands.motion.mix_contact_labels = None
        self.commands.motion.mix_contact_label_prob = 0.0

        # Paired contact sensors: per-body sensors with object filtering
        self.commands.motion.use_paired_contact = self.use_paired_contact
        if self.use_paired_contact:
            # Default filter: single chair rigid body (the "world" link from URDF).
            # configure_motion_objects() updates this for per-motion objects.
            _filter = ["{ENV_REGEX_NS}/Chair/world"]
            for body_name in PAIRED_CONTACT_BODIES:
                setattr(self.scene, f"paired_contact_{body_name}", ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                    filter_prim_paths_expr=list(_filter),
                    update_period=0.0,
                ))

        # ---- Rewards: contact-label-based (replaces heuristic proximity/repulsion) ----
        self.rewards.contact_label_match = RewTerm(
            func=mdp.contact_label_match,
            weight=4.0,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "object_asset_name": "chair",
                "threshold": 1.0,
                "fp_penalty_weight": self.contact_label_fp_penalty_weight,
                "tp_fp_mode": self.contact_label_tp_fp_mode,
            },
        )
        self.rewards.contact_distance = RewTerm(
            func=mdp.contact_distance,
            weight=3.0,
            params={
                "command_name": "motion",
                "object_asset_name": "chair",
                "std": 0.2,
                "unwanted_penalty_threshold": 0.05 if self.enable_unwanted_contact_penalty else None,
            },
        )

        # ---- Contact observations (both policy and critic) ----
        _contact_obs_params = {
            "command_name": "motion",
            "object_asset_name": "chair",
        }
        _contact_real_params = {
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "object_asset_name": "chair",
            "threshold": 1.0,
        }

        if self.enable_contact_labels_ref:
            self.observations.policy.contact_labels_ref = ObsTerm(
                func=mdp.contact_labels_ref,
                params={"command_name": "motion"},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            self.observations.critic.contact_labels_ref = ObsTerm(
                func=mdp.contact_labels_ref,
                params={"command_name": "motion"},
            )

        if self.enable_contact_labels_ref_binary:
            self.observations.policy.contact_labels_ref_binary = ObsTerm(
                func=mdp.contact_labels_ref_binary,
                params={"command_name": "motion"},
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            self.observations.critic.contact_labels_ref_binary = ObsTerm(
                func=mdp.contact_labels_ref_binary,
                params={"command_name": "motion"},
            )

        if self.enable_contact_body_chair_distance:
            self.observations.policy.contact_body_chair_distance = ObsTerm(
                func=mdp.contact_body_chair_distance,
                params=_contact_obs_params,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )
            self.observations.critic.contact_body_chair_distance = ObsTerm(
                func=mdp.contact_body_chair_distance,
                params=_contact_obs_params,
            )

        if self.enable_contact_labels_real:
            self.observations.policy.contact_labels_real = ObsTerm(
                func=mdp.contact_labels_real,
                params=_contact_real_params,
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            self.observations.critic.contact_labels_real = ObsTerm(
                func=mdp.contact_labels_real,
                params=_contact_real_params,
            )

        _contact_real_binary_params = {
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "object_asset_name": "chair",
            "threshold": 1.0,
        }
        if self.enable_contact_labels_real_binary:
            self.observations.policy.contact_labels_real_binary = ObsTerm(
                func=mdp.contact_labels_real_binary,
                params=_contact_real_binary_params,
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            self.observations.critic.contact_labels_real_binary = ObsTerm(
                func=mdp.contact_labels_real_binary,
                params=_contact_real_binary_params,
            )

        # Relax ground undesired_contacts for sitting bodies
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
            r"(?!pelvis$)(?!left_hip_roll_link$)(?!right_hip_roll_link$)"
            r"(?!torso_link$)(?!left_knee_link$)(?!right_knee_link$).+$"
        ]

    def configure_motion_objects(self, object_urdfs: list, extra_object_urdfs: list | None = None):
        """Spawn one RigidObjectCfg per motion that has a URDF. Call before gym.make().

        Each non-None entry in *object_urdfs* produces a separate scene object
        (``motion_object_0``, ``motion_object_1``, …) with the same spawn options
        as the default chair (convex decomposition, kinematic, fixed-base).
        The default ``self.scene.chair`` is removed.

        Object geometry (z_min / z_range / segmented / part_thresholds) is resolved
        per object from ``object_geometry_yaml`` via substring match on the URDF path.

        Args:
            object_urdfs: list parallel to motion_files. ``None`` = no object.
            extra_object_urdfs: list parallel to motion_files of extra (non-contact) object URDF lists. Unused currently.
        """
        # Load geometry registry from YAML
        geom_yaml = self.commands.motion.object_geometry_yaml
        if geom_yaml and os.path.exists(geom_yaml):
            geom_registry = ObjectGeometryCfg.load_yaml(geom_yaml)
        else:
            geom_registry = {}

        # Remove default single chair
        self.scene.chair = None

        obj_idx = 0
        motion_to_object_idx = []
        obj_idx_to_urdf: dict[int, str] = {}
        urdf_to_obj_idx: dict[str, int] = {}  # dedup: same URDF → same physics object
        for urdf_path in object_urdfs:
            if urdf_path is None:
                motion_to_object_idx.append(-1)
                continue
            if urdf_path in urdf_to_obj_idx:
                # Reuse existing physics object for this URDF
                motion_to_object_idx.append(urdf_to_obj_idx[urdf_path])
                continue
            key = f"motion_object_{obj_idx}"
            setattr(self.scene, key, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{key}",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf_path,
                    fix_base=False,
                    activate_contact_sensors=True,
                    collider_type="convex_decomposition",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=False,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        articulation_enabled=False,
                    ),
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=0, damping=0
                        )
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, -10.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ))
            obj_idx_to_urdf[obj_idx] = urdf_path
            urdf_to_obj_idx[urdf_path] = obj_idx
            motion_to_object_idx.append(obj_idx)
            obj_idx += 1

        self._motion_to_object_idx = motion_to_object_idx
        self._num_unique_objects = obj_idx

        # Build per-object geometry configs (indexed by obj_idx) for MotionCommand to consume
        self._object_geometries = [
            ObjectGeometryCfg.match_urdf(obj_idx_to_urdf[i], geom_registry)
            for i in range(obj_idx)
        ]
        for i, geom in enumerate(self._object_geometries):
            segmented_str = f"segmented thresholds={geom.part_thresholds}" if geom.segmented else "single-part"
            print(f"[INFO]: motion_object_{i} geometry: z_min={geom.z_min}, "
                  f"z_range={geom.z_range}, {segmented_str}")

        # Update paired contact sensor filters to match per-motion objects.
        # The URDF converter places the rigid body under the "world" link prim
        # (the fixed base from the URDF). Filter pattern: motion_object_N/world
        if self.use_paired_contact and obj_idx > 0:
            new_filter = [f"{{ENV_REGEX_NS}}/motion_object_{i}/world" for i in range(obj_idx)]
            for body_name in PAIRED_CONTACT_BODIES:
                sensor_cfg = getattr(self.scene, f"paired_contact_{body_name}", None)
                if sensor_cfg is not None:
                    sensor_cfg.filter_prim_paths_expr = new_filter
            print(f"[INFO]: Updated paired sensor filters: {new_filter}")

        # Set low friction on free objects so they slide when pushed (not wobble/tip)
        from isaaclab.managers import EventTermCfg as EventTerm
        for i in range(obj_idx):
            setattr(self.events, f"object_friction_{i}", EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(f"motion_object_{i}", body_names=".*"),
                    "static_friction_range": (0.2, 0.2),
                    "dynamic_friction_range": (0.15, 0.15),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            ))

        print(f"[INFO]: Configured {obj_idx} motion objects, "
              f"mapping: {motion_to_object_idx}")

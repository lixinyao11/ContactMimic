# ContactMimic: Humanoid Object Interaction via Contact Control

[[Website]](https://lixinyao11.github.io/contactmimic-page) | [[arXiv]](https://arxiv.org/abs/2607.08742) | [[Paper]](https://arxiv.org/pdf/2607.08742)

```bibtex
@article{li2026contactmimic,
  title   = {ContactMimic: Humanoid Object Interaction via Contact Control},
  author  = {Li, Xinyao and He, Xialin and Dong, Runpei and Gupta, Saurabh},
  journal = {arXiv preprint arXiv:2607.08742},
  year    = {2026}
}
```

## Installation

- Install [Isaac Lab v2.1.0](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda installation recommended).

- Clone this repository:

```bash
git clone https://github.com/lixinyao11/ContactMimic.git
cd ContactMimic
```

- Pull the Unitree G1 robot description:

```bash
curl -L -o unitree_description.tar.gz \
  https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
  tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
  rm unitree_description.tar.gz
```

- Install the extension package using your Isaac-Lab Python interpreter:

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Data

Our training data is derived from the [HUMOTO](https://humoto.github.io/) dataset, which requires signing a license agreement to access. Please follow HUMOTO's instructions to obtain the raw mocap data.

After obtaining the raw mocap, you need to (1) retarget it to the G1 robot and (2) extract per-frame contact labels. We provide `scripts/humoto_to_npz.py` to convert the retargeted+annotated NPZ into Isaac-ready motion artifacts for training.

### Expected input NPZ format

The input `.npz` to `humoto_to_npz.py` should contain:

| Key | Shape | Description |
| --- | --- | --- |
| `qpos` | `(T, 36)` or `(T, 43)` | Per-frame robot state: `[xyz(3), wxyz(4), joints(29)]`. The 43-col variant appends a 7-DoF object freejoint. |
| `fps` | scalar | Frame rate of the input sequence. |
| `contact_labels` | `(T, N)` bool | Per-frame binary contact flag for each of `N` robot body parts. |
| `contact_body_names` | `(N,)` str | Names of the `N` robot bodies tracked for contact. |
| `contact_points_robot` | `(T,)` object array | Per-frame `(K, 3)` contact points in robot body frame. |
| `contact_points_object` | `(T,)` object array | Per-frame `(K, 3)` contact points on the object surface. |
| `contact_chair_labels` | `(T,)` object array | Per-frame semantic part labels (e.g. seat, backrest, legs) for each contact point. |

The `contact_*` fields are optional — omitting them produces a keypoint-only motion without contact conditioning.

### Convert to Isaac-ready motion artifacts

```bash
python scripts/humoto_to_npz.py \
  --input <retargeted_contact.npz> \
  --output_name <motion_name> \
  --pad_seconds 3.0 \
  --headless
```

The script interpolates qpos to 50 fps, runs FK in Isaac Lab to produce body pose/velocity channels, and resamples contact labels to match.

## Training

Example:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Sitting-G1-v0 \
  --motion_name <motion_contact> <motion_nocontact> \
  --num_envs 4096 --headless \
  --mix_contact_groups "<motion_contact>:<motion_nocontact>" \
  --mix_contact_prob 0.3 \
  --run_name my_run
```

### Contact-related flags

| Flag | Effect |
| --- | --- |
| `--no_contact` | Disable contact obs/rewards entirely (paper's *noContact* ablation). |
| `--mix_contact_groups "A:B"` | Random per-env swap of contact labels between paired motions at training time. |
| `--mix_contact_prob FLOAT` | Per-env probability of performing such a swap each reset. |
| `--random_mask_contact_parts INT [...]` | Contact part IDs eligible for label masking (1=seat, 2=backrest, 3=legs, 4=table-top). |
| `--random_mask_contact_prob FLOAT` | Per-env probability of masking the eligible parts. |
| `--contact_tp_fp_mode` | Use TP-FP contact reward (recommended for sparse-contact / free-object motions). |

### Scene / object flags

| Flag | Effect |
| --- | --- |
| `--chair_offset_cfg PATH` | Per-motion chair pose offset. See `scripts/chair_offset_cfg.yaml`. |
| `--remove_object_prob FLOAT` | Per-env probability of removing the scene object at reset. |
| `--stand_ratio FLOAT` | Fraction of envs frozen at frame 0 (stand-still curriculum). |

## Evaluation / Playback

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Sitting-G1-v0 \
  --num_envs 1 --headless --video --video_length 1000 \
  --load_run <run_dir_name> \
  --motion_name <motion_artifact_name>
```

## Acknowledgements

Built on [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) by Liao et al. (MIT license). Motion data derived from the [HUMOTO](https://humoto.github.io/) dataset.

## License

MIT. See [`LICENCE`](LICENCE).

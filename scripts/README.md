# Scripts

MuJoCo tooling for the Astribot S1 models in this repository.

The MJCF path is resolved relative to the repo root, so these run from any
working directory. The LeRobot dataset lives outside the repo; its location
defaults to `~/Documents/Work/data/astri_vlva` and can be overridden with the
`ASTRIBOT_DATASET` environment variable or `--dataset`.

Requires `mujoco`, `numpy`, `pyarrow`, `pandas`, and `Pillow` for offscreen
rendering.

## `view_astribot.py`

Opens a model in the interactive viewer.

```bash
python3 scripts/view_astribot.py            # default: gripper
python3 scripts/view_astribot.py hand       # gripper-fixed | hand-fixed | aloha
```

## `compute_ee_se3.py`

Computes SE(3) poses of both end-effectors for one episode by forward
kinematics, and writes them to `<dataset>/derived/ee_se3_episode_NNN.parquet`.

```bash
python3 scripts/compute_ee_se3.py --episode 0
python3 scripts/compute_ee_se3.py --episode 0 --torso 0.3 -0.8 1.2 0.0
```

## `compute_ee_se3_all.py`

Same computation over every episode, building the model once. Validates each
result (SE(3) orthonormality, row counts, joint limits) before writing.

```bash
python3 scripts/compute_ee_se3_all.py            # skips existing outputs
python3 scripts/compute_ee_se3_all.py --force
```

## `playback_ee_se3.py`

Replays an episode's recorded joints in the viewer and draws the stored EE
poses as RGB triads with fading trails. The triads are read back from the
derived parquet rather than recomputed, so the overlay doubles as a check that
the stored data is correct. Run `compute_ee_se3.py` for the episode first.

```bash
python3 scripts/playback_ee_se3.py --episode 0 --speed 0.5
python3 scripts/playback_ee_se3.py --episode 0 --action   # commanded poses
```

`SPACE` pause · `←`/`→` step while paused · `R` restart · `T` toggle trails.

## `merge_ee_se3_into_dataset.py`

Merges the derived EE columns into the dataset's own episode parquets,
rewriting them in place. Each file is streamed row-group by row-group into a
temporary file alongside it, verified column-by-column against the original,
and only then atomically renamed over it -- so an interrupted run always
leaves the original intact. Already-merged files are skipped, so the run is
resumable.

The parquet's embedded `huggingface` schema metadata and `meta/info.json` are
both extended so the new columns are visible to the LeRobot loader; the
original info.json is kept as `info.json.bak`.

```bash
python3 scripts/merge_ee_se3_into_dataset.py --dry-run
python3 scripts/merge_ee_se3_into_dataset.py
```

Note that this does not add entries to `meta/stats.json` or the per-episode
stats in `meta/episodes/`.

## Frame conventions

Poses are `T_reference <- toolLink` as 4x4 row-major matrices, plus position
and wxyz quaternion.

**Reference** (`ee.*.torso_frame.*`) is the `astribot_torso_link_4` body — the
body both arms mount on. Axes are +X forward, +Y robot-left, +Z up. At torso
zero it is axis-aligned with `chassis_base`, offset by `(0, 0, 0.89)`.

**Moving frame** is `astribot_arm_{side}_tool_link`, the gripper mounting
flange (coincident with `astribot_gripper_{side}_base`). +Z is the approach
axis, pointing out through the fingers.

This is a flange frame, not a TCP: the fingertip-pair midpoint sits 87.5 mm
along tool +Z, so `T_tcp = T_flange @ translation(0, 0, 0.0875)`.

## The torso caveat

The dataset records only the 14 arm joints and 2 gripper values -- the 4 torso
joints are **not** recorded. EE pose in the chassis base frame is therefore not
recoverable from the dataset alone.

- `ee.*.torso_frame.*` is exact, fully determined by the recorded joints.
- `ee.*.base_frame.*` assumes the torso configuration passed to `--torso`,
  which defaults to all zeros. Note that zero is a boundary pose for
  `torso_joint_1` (range `[0, 1.5]`) and `torso_joint_3` (range `[0, 2.4]`),
  so the default is a placeholder, not the real collection posture. With the
  default it differs from the torso frame by only the constant `(0, 0, 0.89)`.

## Gripper values

The dataset's gripper channel is `0..100` and maps to the master joint by
`g = 0.0093 * value`, derived from the actuator's affine bias
(`4.65*ctrl = 500*L`), reaching exactly the 0.93 rad joint limit at 100. The
remaining jaw joints follow the model's equality constraints:
`L11 = g, L2 = -g, R1 = g, R2 = g, R11 = -g`.

**0 is fully open** (111.5 mm between fingertips) and **100 is closed**
(20.0 mm).

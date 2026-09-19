#!/usr/bin/env python3
"""Compute SE(3) poses of the Astribot S1 left/right end-effectors from a
LeRobot episode's recorded arm joints, via MuJoCo forward kinematics.

The dataset records only the 14 arm joints (7 per arm) plus two gripper
values -- the 4 torso joints are NOT recorded. Poses are therefore emitted
in the `astribot_torso_link_4` frame (the body both arms mount on), which is
fully determined by the recorded joints. Passing --torso j1 j2 j3 j4 also
emits poses in the chassis base frame for a known torso configuration.
"""
import argparse
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import mujoco

REPO = pathlib.Path(__file__).resolve().parent.parent
MJCF = REPO / "mjcf/astribot_s1_mjcf/astribot_s1_with_gripper.xml"
DATASET = pathlib.Path(
    os.environ.get("ASTRIBOT_DATASET",
                   pathlib.Path.home() / "Documents/Work/data/astri_vlva"))

TOOL = {"left": "astribot_arm_left_tool_link",
        "right": "astribot_arm_right_tool_link"}
ARM_MOUNT = "astribot_torso_link_4"
BASE = "chassis_base"

# column index of each arm's 7 joints inside observation.state / action
STATE_SLICE = {"left": slice(0, 7), "right": slice(8, 15)}
GRIPPER_COL = {"left": 7, "right": 15}


def build():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    qadr = {
        side: np.array([
            model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"astribot_arm_{side}_joint_{j}")]
            for j in range(1, 8)
        ]) for side in ("left", "right")
    }
    torso_adr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"astribot_torso_joint_{j}")]
        for j in range(1, 5)
    ])
    bid = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v)
           for k, v in TOOL.items()}
    bid["mount"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM_MOUNT)
    bid["base"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE)
    return model, data, qadr, torso_adr, bid


def fk(joints, torso, model, data, qadr, torso_adr, bid):
    """joints: (N,16) state/action rows. Returns dict of (N,4,4) arrays."""
    n = len(joints)
    out = {f"{frame}_{side}": np.zeros((n, 4, 4))
           for frame in ("torso", "base") for side in ("left", "right")}
    quat = np.zeros(4)

    for i, row in enumerate(joints):
        data.qpos[:] = 0.0
        data.qpos[torso_adr] = torso
        for side in ("left", "right"):
            data.qpos[qadr[side]] = row[STATE_SLICE[side]]
        mujoco.mj_kinematics(model, data)

        for ref, key in (("mount", "torso"), ("base", "base")):
            Rr = data.xmat[bid[ref]].reshape(3, 3)
            pr = data.xpos[bid[ref]]
            for side in ("left", "right"):
                Rt = data.xmat[bid[side]].reshape(3, 3)
                T = np.eye(4)
                T[:3, :3] = Rr.T @ Rt
                T[:3, 3] = Rr.T @ (data.xpos[bid[side]] - pr)
                out[f"{key}_{side}"][i] = T
    return out


def to_quat(R):
    """(N,3,3) rotation matrices -> (N,4) wxyz quaternions."""
    q = np.zeros((len(R), 4))
    for i, m in enumerate(R):
        mujoco.mju_mat2Quat(q[i], m.reshape(9))
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=pathlib.Path,
                    default=DATASET)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--torso", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0],
                    metavar=("J1", "J2", "J3", "J4"),
                    help="torso joint angles (rad) used for the base-frame output")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    eps = pq.read_table(args.dataset / "meta/episodes/chunk-000/file-000.parquet",
                        columns=["episode_index", "length",
                                 "data/chunk_index", "data/file_index"]).to_pandas()
    row = eps[eps.episode_index == args.episode].iloc[0]
    src = (args.dataset / "data" /
           f"chunk-{row['data/chunk_index']:03d}" /
           f"file-{row['data/file_index']:03d}.parquet")

    tbl = pq.read_table(src, columns=["observation.state", "action", "timestamp",
                                      "frame_index", "episode_index", "index"])
    state = np.stack(tbl["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    action = np.stack(tbl["action"].to_numpy(zero_copy_only=False)).astype(np.float64)
    print(f"episode {args.episode}: {src.name}, {len(state)} frames")

    model, data, qadr, torso_adr, bid = build()
    cols = {
        "index": tbl["index"], "episode_index": tbl["episode_index"],
        "frame_index": tbl["frame_index"], "timestamp": tbl["timestamp"],
    }

    for tag, arr in (("state", state), ("action", action)):
        poses = fk(arr, np.asarray(args.torso), model, data, qadr, torso_adr, bid)
        for key, T in poses.items():
            frame, side = key.split("_")
            base = f"ee.{side}.{frame}_frame.{tag}"
            flat = T.reshape(len(T), 16).astype(np.float32)
            cols[f"{base}.matrix"] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat.ravel(), pa.float32()), 16)
            cols[f"{base}.position"] = pa.FixedSizeListArray.from_arrays(
                pa.array(T[:, :3, 3].astype(np.float32).ravel(), pa.float32()), 3)
            cols[f"{base}.quaternion_wxyz"] = pa.FixedSizeListArray.from_arrays(
                pa.array(to_quat(T[:, :3, :3]).astype(np.float32).ravel(), pa.float32()), 4)
        for side in ("left", "right"):
            cols[f"ee.{side}.gripper.{tag}"] = pa.array(
                arr[:, GRIPPER_COL[side]].astype(np.float32), pa.float32())

    out = args.out or (args.dataset / "derived" /
                       f"ee_se3_episode_{args.episode:03d}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(cols)
    pq.write_table(table, out, compression="zstd")
    print(f"wrote {out}  ({len(table)} rows, {len(table.column_names)} cols, "
          f"{out.stat().st_size/1024:.1f} KiB)")
    print(f"torso assumed at {list(args.torso)} rad for the *_base_frame columns")


if __name__ == "__main__":
    main()

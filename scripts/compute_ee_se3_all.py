#!/usr/bin/env python3
"""Compute EE SE(3) poses for every episode in the dataset.

Reuses compute_ee_se3's kinematics, building the MuJoCo model once instead of
once per episode. Writes one derived/ee_se3_episode_NNN.parquet per episode and
validates each result before moving on.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compute_ee_se3 as C  # noqa: E402
from compute_ee_se3 import fk, to_quat, GRIPPER_COL, STATE_SLICE  # noqa: E402

DATASET = C.DATASET


def joint_limits(model):
    import mujoco
    lim = {}
    for side in ("left", "right"):
        for j in range(1, 8):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"astribot_arm_{side}_joint_{j}")
            lim[(side, j - 1)] = model.jnt_range[jid]
    return lim


def build_table(state, action, torso, kin):
    model, data, qadr, torso_adr, bid = kin
    cols = {}
    for tag, arr in (("state", state), ("action", action)):
        poses = fk(arr, torso, model, data, qadr, torso_adr, bid)
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
    return cols, poses


def validate(cols, n):
    """Return (ok, message). Checks SE(3) validity of every stored matrix."""
    worst_orth = worst_det = 0.0
    for name, col in cols.items():
        if not name.endswith(".matrix"):
            continue
        T = np.stack(col.to_numpy(zero_copy_only=False)).reshape(-1, 4, 4)
        if len(T) != n:
            return False, f"{name}: {len(T)} rows, expected {n}"
        R = T[:, :3, :3]
        worst_orth = max(worst_orth, np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max())
        worst_det = max(worst_det, np.abs(np.linalg.det(R) - 1.0).max())
        if not np.allclose(T[:, 3, :], [0, 0, 0, 1], atol=1e-6):
            return False, f"{name}: bad homogeneous row"
        if not np.isfinite(T).all():
            return False, f"{name}: non-finite values"
    return True, f"orth={worst_orth:.2e} det_err={worst_det:.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torso", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0])
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="specific episode indices (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="recompute episodes whose output already exists")
    args = ap.parse_args()

    eps = pq.read_table(DATASET / "meta/episodes/chunk-000/file-000.parquet",
                        columns=["episode_index", "length", "data/chunk_index",
                                 "data/file_index"]).to_pandas()
    todo = args.only if args.only is not None else eps.episode_index.tolist()

    kin = C.build()
    lim = joint_limits(kin[0])
    torso = np.asarray(args.torso)
    outdir = DATASET / "derived"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_frames = 0
    failures, oor = [], []
    for ep in todo:
        row = eps[eps.episode_index == ep].iloc[0]
        out = outdir / f"ee_se3_episode_{ep:03d}.parquet"
        if out.exists() and not args.force:
            print(f"ep {ep:3d}  skip (exists)")
            continue
        src = (DATASET / "data" / f"chunk-{row['data/chunk_index']:03d}"
               / f"file-{row['data/file_index']:03d}.parquet")
        tbl = pq.read_table(src, columns=["observation.state", "action", "timestamp",
                                          "frame_index", "episode_index", "index"])
        state = np.stack(tbl["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        action = np.stack(tbl["action"].to_numpy(zero_copy_only=False)).astype(np.float64)

        # flag joints outside the model's limits (would mean a convention mismatch)
        bad = []
        for side in ("left", "right"):
            blk = state[:, STATE_SLICE[side]]
            for j in range(7):
                lo, hi = lim[(side, j)]
                if blk[:, j].min() < lo - 0.05 or blk[:, j].max() > hi + 0.05:
                    bad.append(f"{side}_j{j}")
        if bad:
            oor.append((ep, bad))

        cols, _ = build_table(state, action, torso, kin)
        cols = {"index": tbl["index"], "episode_index": tbl["episode_index"],
                "frame_index": tbl["frame_index"], "timestamp": tbl["timestamp"], **cols}
        ok, msg = validate(cols, len(state))
        if not ok:
            failures.append((ep, msg))
            print(f"ep {ep:3d}  FAILED  {msg}")
            continue
        pq.write_table(pa.table(cols), out, compression="zstd")
        total_frames += len(state)
        print(f"ep {ep:3d}  {len(state):5d} frames  {msg}  -> {out.name}"
              + (f"  [LIMIT: {','.join(bad)}]" if bad else ""))

    dt = time.time() - t0
    print(f"\ndone: {len(todo)} episodes, {total_frames} frames in {dt:.1f}s")
    if oor:
        print(f"episodes with joints outside MJCF limits: {oor}")
    else:
        print("all arm joints within MJCF limits")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

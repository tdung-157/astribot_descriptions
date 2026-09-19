#!/usr/bin/env python3
"""Replay an Astribot S1 episode in MuJoCo and draw the EE SE(3) poses.

Joint angles come from the dataset parquet; the end-effector frames drawn as
RGB triads are read back from the derived ee_se3_*.parquet (NOT recomputed),
so the overlay doubles as a visual check that the stored poses are correct.
Stored poses are in the astribot_torso_link_4 frame and are mapped to world
with the live torso transform.

Keys (in addition to the standard MuJoCo viewer bindings):
  SPACE  pause / resume        LEFT/RIGHT  step one frame while paused
  R      restart episode       T           toggle EE trails
"""
import argparse
import os
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import mujoco
import mujoco.viewer

REPO = pathlib.Path(__file__).resolve().parent.parent
MJCF = REPO / "mjcf/astribot_s1_mjcf/astribot_s1_with_gripper.xml"
DATASET = pathlib.Path(
    os.environ.get("ASTRIBOT_DATASET",
                   pathlib.Path.home() / "Documents/Work/data/astri_vlva"))

STATE_SLICE = {"left": slice(0, 7), "right": slice(8, 15)}
GRIPPER_COL = {"left": 7, "right": 15}
GRIPPER_SCALE = 0.0093          # ctrl 0..100 -> master joint 0..0.93 rad
SIDE_RGB = {"left": (0.95, 0.30, 0.30), "right": (0.30, 0.55, 1.00)}

AXIS_ALIGN = {                   # arrow points along local +z; map +z -> axis k
    0: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),   # Ry(+90)
    1: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),   # Rx(-90)
    2: np.eye(3),
}
AXIS_RGB = [(0.90, 0.15, 0.15), (0.15, 0.85, 0.15), (0.20, 0.35, 1.00)]


def add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float),
                        np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return True


def draw_frame(scn, T, length=0.20, radius=0.005):
    """Draw an RGB triad for a 4x4 world pose."""
    R, p = T[:3, :3], T[:3, 3]
    for k in range(3):
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW,
                 [radius, radius, length], p, R @ AXIS_ALIGN[k],
                 (*AXIS_RGB[k], 1.0))


class Player:
    def __init__(self, episode, speed, trail_len, use_action, axis_len=0.20):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.data = mujoco.MjData(self.model)
        self.speed = speed
        self.trail_len = trail_len
        self.axis_len = axis_len
        self.show_trail = True
        self.paused = False
        self.i = 0

        M = self.model
        jid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n)
        self.qadr = {s: np.array([M.jnt_qposadr[jid(f"astribot_arm_{s}_joint_{j}")]
                                  for j in range(1, 8)]) for s in ("left", "right")}
        self.gadr = {s: {k: M.jnt_qposadr[jid(f"astribot_gripper_{s}_joint_{k}")]
                         for k in ("L1", "L11", "L2", "R1", "R11", "R2")}
                     for s in ("left", "right")}
        self.mount = bid("astribot_torso_link_4")

        tag = "action" if use_action else "state"
        src, ee = load_episode(episode)
        self.joints = src
        self.ee = {s: np.stack(
            ee[f"ee.{s}.torso_frame.{tag}.matrix"].to_numpy(zero_copy_only=False)
        ).reshape(-1, 4, 4).astype(np.float64) for s in ("left", "right")}
        self.n = len(self.joints)
        self.trail = {s: [] for s in ("left", "right")}

    def pose(self, i):
        d, M = self.data, self.model
        d.qpos[:] = 0.0
        row = self.joints[i]
        for s in ("left", "right"):
            d.qpos[self.qadr[s]] = row[STATE_SLICE[s]]
            g = float(np.clip(row[GRIPPER_COL[s]] * GRIPPER_SCALE, 0.0, 0.93))
            a = self.gadr[s]
            d.qpos[a["L1"]] = d.qpos[a["L11"]] = g
            d.qpos[a["R1"]] = d.qpos[a["R2"]] = g
            d.qpos[a["L2"]] = d.qpos[a["R11"]] = -g
        mujoco.mj_kinematics(M, d)

    def world_ee(self, i):
        """Stored torso-frame pose -> world, using the live torso transform."""
        Tm = np.eye(4)
        Tm[:3, :3] = self.data.xmat[self.mount].reshape(3, 3)
        Tm[:3, 3] = self.data.xpos[self.mount]
        return {s: Tm @ self.ee[s][i] for s in ("left", "right")}

    def render_overlay(self, scn):
        scn.ngeom = 0
        W = self.world_ee(self.i)
        for s in ("left", "right"):
            draw_frame(scn, W[s], length=self.axis_len)
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                     W[s][:3, 3], np.eye(3), (*SIDE_RGB[s], 0.95))
            self.trail[s].append(W[s][:3, 3].copy())
            if len(self.trail[s]) > self.trail_len:
                self.trail[s].pop(0)
        if self.show_trail:
            for s in ("left", "right"):
                pts = self.trail[s]
                for k, p in enumerate(pts):
                    a = 0.10 + 0.55 * (k / max(len(pts) - 1, 1))
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0],
                             p, np.eye(3), (*SIDE_RGB[s], a))

    def key(self, code):
        if code == 32:                       # space
            self.paused = not self.paused
        elif code == 262 and self.paused:    # right
            self.i = (self.i + 1) % self.n
        elif code == 263 and self.paused:    # left
            self.i = (self.i - 1) % self.n
        elif code in (82, 114):              # R
            self.i = 0
            self.trail = {s: [] for s in ("left", "right")}
        elif code in (84, 116):              # T
            self.show_trail = not self.show_trail

    def run(self):
        dt = 1.0 / (30.0 * self.speed)
        with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.key,
                show_left_ui=False, show_right_ui=False) as v:
            v.cam.lookat[:] = [0, 0, 0.85]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.8, -15, 150
            print(f"playing {self.n} frames | user_scn.maxgeom="
                  f"{v.user_scn.maxgeom} | SPACE pause, R restart, T trails")
            nxt = time.time()
            while v.is_running():
                self.pose(self.i)
                self.render_overlay(v.user_scn)
                v.sync()
                nxt += dt
                time.sleep(max(0.0, nxt - time.time()))
                if not self.paused:
                    self.i += 1
                    if self.i >= self.n:
                        self.i = 0
                        self.trail = {s: [] for s in ("left", "right")}


def load_episode(episode):
    eps = pq.read_table(DATASET / "meta/episodes/chunk-000/file-000.parquet",
                        columns=["episode_index", "data/chunk_index",
                                 "data/file_index"]).to_pandas()
    row = eps[eps.episode_index == episode].iloc[0]
    src = (DATASET / "data" / f"chunk-{row['data/chunk_index']:03d}"
           / f"file-{row['data/file_index']:03d}.parquet")
    joints = np.stack(pq.read_table(src, columns=["observation.state"])
                      ["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    ee_path = DATASET / "derived" / f"ee_se3_episode_{episode:03d}.parquet"
    if not ee_path.exists():
        raise SystemExit(f"missing {ee_path} -- run compute_ee_se3.py --episode {episode}")
    return joints, pq.read_table(ee_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--trail", type=int, default=150, help="trail length in frames")
    ap.add_argument("--axis-len", type=float, default=0.20,
                    help="EE triad axis length in metres (gripper is 0.176 m long)")
    ap.add_argument("--action", action="store_true",
                    help="draw the commanded (action) EE poses instead of state")
    args = ap.parse_args()
    Player(args.episode, args.speed, args.trail, args.action, args.axis_len).run()


if __name__ == "__main__":
    main()

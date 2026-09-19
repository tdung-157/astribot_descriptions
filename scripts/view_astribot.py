#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the Astribot S1 description models."""
import argparse
import pathlib
import sys

import mujoco
import mujoco.viewer

REPO = pathlib.Path(__file__).resolve().parent.parent
MJCF = REPO / "mjcf/astribot_s1_mjcf"

MODELS = {
    "gripper": MJCF / "astribot_s1_with_gripper.xml",
    "hand": MJCF / "astribot_s1_with_hand.xml",
    "gripper-fixed": MJCF / "astribot_s1_chassis_fixed_with_gripper.xml",
    "hand-fixed": MJCF / "astribot_s1_chassis_fixed_with_hand.xml",
    "aloha": MJCF / "astribot_s1_for_aloha_with_gripper.xml",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="gripper", choices=sorted(MODELS),
                    help="which Astribot S1 variant to load (default: gripper)")
    args = ap.parse_args()

    path = MODELS[args.model]
    if not path.exists():
        sys.exit(f"model not found: {path}")

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"loaded {path.name}: {model.nq} dof, {model.nu} actuators, "
          f"{model.nbody} bodies, {model.ngeom} geoms")

    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

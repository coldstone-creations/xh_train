"""Convert xh URDF to USD using Isaac Lab's UrdfConverter."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert xh URDF to USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    # paths
    urdf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model", "xh_urdf", "urdf", "xh.urdf"
    )
    usd_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model", "xh_usd", "xh.usd"
    )

    urdf_path = os.path.abspath(urdf_path)
    usd_path = os.path.abspath(usd_path)

    print(f"[INFO] Converting URDF: {urdf_path}")
    print(f"[INFO] Output USD: {usd_path}")

    # configure converter
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(usd_path),
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        collider_type="convex_hull",
        collision_from_visuals=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )

    # convert
    converter = UrdfConverter(cfg)
    print(f"[INFO] USD conversion complete: {usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

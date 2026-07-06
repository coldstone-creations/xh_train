"""Minimal script: spawn xh, hold default pose, print settled height and contact forces."""
import argparse, sys, torch

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Flat-xh-v0")
parser.add_argument("--base_height", type=float, default=None, help="Override spawn base height (m)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
# ensure local rl_training takes precedence over pip-installed version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
import rl_training.tasks  # noqa: register xh

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.base_height is not None:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.base_height)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    dt = env.unwrapped.step_dt

    n_actions = len(env_cfg.actions.joint_pos.joint_names)
    print(f"[INFO] Holding default pose for 1 second (zero action, {int(1.0/dt)} steps)...")
    for _ in range(int(1.0 / dt)):
        obs = env.step(torch.zeros(1, n_actions))[0]

    base_z = robot.data.root_pos_w[0, 2].item()
    print(f"\n===== RESULTS (thigh=-0.6, calf=0.7) =====")
    print(f"Default joint angles:")
    for name, angle in zip(robot.joint_names, robot.data.default_joint_pos[0]):
        print(f"  {name}: {angle.item():+.4f} rad")
    print(f"\nBase height (root_pos_w.z): {base_z:.4f} m")
    print(f"Foot positions & contact forces:")
    net_forces = contact_sensor.data.net_forces_w[:, :, :].norm(dim=-1)
    for name, pos in zip(robot.body_names, robot.data.body_pos_w[0]):
        if "foot" in name.lower():
            idx = list(contact_sensor.body_names).index(name) if name in contact_sensor.body_names else -1
            force = net_forces[0, idx].item() if idx >= 0 else float('nan')
            print(f"  {name}: z={pos[2].item():.4f}m, rel_base={pos[2].item()-base_z:+.4f}m, contact={force:.2f}N")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

"""Visualize robot with custom joint angles. Supports xh and lite3."""
import argparse, os, sys, time, torch

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="xh", choices=["xh", "lite3"],
                    help="Robot to visualize (default: xh)")
parser.add_argument("--terrain", type=str, default="rough", choices=["flat", "rough"],
                    help="Terrain type (default: rough)")
parser.add_argument("--thigh", type=float, default=None, help="Override thigh joint angle (rad)")
parser.add_argument("--calf", type=float, default=None, help="Override calf/knee joint angle (rad)")
parser.add_argument("--hip", type=float, default=None, help="Override hip joint angle (rad)")
parser.add_argument("--print_contact", action="store_true", help="Print contact forces")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
import rl_training.tasks  # noqa: register all tasks

# ── Robot presets ──────────────────────────────────────────────
ROBOTS = {
    "xh": {
        "task": "Rough-xh-v0",  # terrain prefix replaced dynamically
        "joint_regex": {
            "hip": ".*_hip_joint",
            "thigh": ".*_thigh_joint",
            "calf": ".*_calf_joint",
        },
        "default_angles": {"hip": 0.0, "thigh": 0, "calf": 0},
        "standing_target": {"hip": 0.0, "thigh": -0.6, "calf": 0.7},
        "action_scale": {"hip": 0.125, "other": 0.25},
        "joint_label": {"hip": "HipX", "thigh": "HipY", "calf": "Knee"},
    },
    "lite3": {
        "task": "Rough-Deeprobotics-Lite3-v0",
        "joint_regex": {
            "hip": ".*HipX_joint",
            "thigh": ".*HipY_joint",
            "calf": ".*Knee_joint",
        },
        "default_angles": {"hip": 0.0, "thigh": -0.65, "calf": 1.3},
        "standing_target": {"hip": 0.0, "thigh": -0.65, "calf": 1.3},
        "action_scale": {"hip": 0.125, "other": 0.25},
        "joint_label": {"hip": "HipX", "thigh": "HipY", "calf": "Knee"},
    },
}

# ── Setup ──────────────────────────────────────────────────────
cfg = ROBOTS[args_cli.robot]
terrain_type = "Rough" if args_cli.terrain == "rough" else "Flat"
task = cfg["task"].replace("Rough", terrain_type)

# ── Debug: list all registered gym envs ─────────────────────────
import gymnasium as gym
print(f"\n===== GYM REGISTRY DEBUG =====")
print(f"Resolved task name: '{task}'")
xh_envs = [k for k in sorted(gym.registry) if "xh" in k.lower() or "deeprobotics" in k.lower()]
if xh_envs:
    for e in xh_envs:
        print(f"  REGISTERED: {e}")
else:
    print("  NO xh/Deeprobotics envs found in registry!")
    print(f"  Sample keys: {list(sorted(gym.registry))[:20]}")
print(f"===============================\n")

# Apply CLI overrides to defaults
if args_cli.hip is not None:
    cfg["default_angles"]["hip"] = args_cli.hip
if args_cli.thigh is not None:
    cfg["default_angles"]["thigh"] = args_cli.thigh
if args_cli.calf is not None:
    cfg["default_angles"]["calf"] = args_cli.calf

# ── Main ───────────────────────────────────────────────────────
@hydra_task_config(task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # Override default joint angles
    env_cfg.scene.robot.init_state.joint_pos = {
        cfg["joint_regex"]["hip"]: cfg["default_angles"]["hip"],
        cfg["joint_regex"]["thigh"]: cfg["default_angles"]["thigh"],
        cfg["joint_regex"]["calf"]: cfg["default_angles"]["calf"],
    }
    env_cfg.scene.num_envs = 1
    # env_cfg.sim.use_fabric = False  # CPU PhysX, same as Isaac Sim GUI
    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]

    # ── Diagnostic: verify USD path ─────────────────────────────────
    from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR
    usd_path_cfg = env_cfg.scene.robot.spawn.usd_path
    usd_path_resolved = usd_path_cfg.replace("${ISAACLAB_ASSETS_DATA_DIR}", ISAACLAB_ASSETS_DATA_DIR) if "${ISAACLAB_ASSETS_DATA_DIR}" in usd_path_cfg else usd_path_cfg
    print(f"\n===== USD DIAGNOSTIC =====")
    print(f"ISAACLAB_ASSETS_DATA_DIR = {ISAACLAB_ASSETS_DATA_DIR}")
    print(f"Config usd_path          = {usd_path_cfg}")
    print(f"Resolved usd_path        = {usd_path_resolved}")
    print(f"File exists?             = {os.path.exists(usd_path_resolved)}")
    print(f"Robot prim_path          = {robot.cfg.prim_path}")
    print(f"==========================\n")

    dt = env.unwrapped.step_dt
    n_actions = len(env_cfg.actions.joint_pos.joint_names)

    # Disable velocity commands — we just want to stand still
    zero_cmd = torch.zeros(1, 3, device=env.unwrapped.device)
    def _zero_commands():
        env.unwrapped.command_manager.get_command("base_velocity")[:] = zero_cmd

    print(f"\n===== {args_cli.robot.upper()} Joint Visualizer =====")
    print(f"Terrain: {args_cli.terrain}")
    print(f"Default (spawn) angles:")
    for name, angle in zip(robot.joint_names, robot.data.default_joint_pos[0]):
        print(f"  {name}: {angle.item():+.4f} rad")

    # Phase 1: settle in default pose (no velocity commands)
    print(f"\n[INFO] Phase 1: settle (0.5s, cmd=0)...")
    for _ in range(int(0.5 / dt)):
        env.step(torch.zeros(1, n_actions))[0]
        _zero_commands()

    # Phase 2: ramp to standing
    standing = cfg["standing_target"]
    scale = cfg["action_scale"]
    labels = cfg["joint_label"]

    standing_action = torch.zeros(1, n_actions)
    for i, name in enumerate(robot.joint_names):
        angle_name = None
        for key in ["hip", "thigh", "calf"]:
            if labels[key].lower() in name.lower():
                angle_name = key
                break
        if angle_name is None:
            continue
        s = scale["hip"] if angle_name == "hip" else scale["other"]
        standing_action[0, i] = (standing[angle_name] - cfg["default_angles"][angle_name]) / s

    print(f"[INFO] Phase 2: ramp to standing (1s, cmd=0)...")
    print(f"  Target: hip={standing['hip']}, thigh={standing['thigh']}, calf={standing['calf']}")
    ramp_steps = int(1.0 / dt)
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        env.step(alpha * standing_action)[0]
        _zero_commands()

    # Phase 3: hold
    print(f"[INFO] Phase 3: hold standing (cmd=0)...")
    for _ in range(int(1.0 / dt)):
        env.step(standing_action)[0]
        _zero_commands()

    # ── Report ──
    base_z = robot.data.root_pos_w[0, 2].item()
    print(f"\n===== Standing State =====")
    print(f"Base height: {base_z:.4f} m")
    print(f"Joint angles (target vs actual):")
    for i, name in enumerate(robot.joint_names):
        target = robot.data.default_joint_pos[0, i].item()
        actual = robot.data.joint_pos[0, i].item()
        err = actual - target
        torque = robot.data.applied_torque[0, i].item()
        flag = " ←MAX" if abs(torque) > 0.95 * env_cfg.scene.robot.actuators[list(env_cfg.scene.robot.actuators.keys())[0]].effort_limit else ""
        print(f"  {name}: target={target:+.4f} actual={actual:+.4f} err={err:+.4f} torque={torque:+.3f}Nm{flag}")

    if args_cli.print_contact:
        cs = env.unwrapped.scene["contact_forces"]
        forces = cs.data.net_forces_w[:, :, :].norm(dim=-1)
        print(f"\n===== Contact Forces (>0.1N) =====")
        for i, name in enumerate(cs.body_names):
            f = forces[0, i].item()
            if f > 0.1:
                z = robot.data.body_pos_w[0, i, 2].item()
                tag = "[FOOT]" if "foot" in name.lower() else "       "
                print(f"  {tag} {name}: force={f:.2f}N z={z:.4f}")

    # ── Run ──
    print(f"\n[INFO] Running... Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            start = time.time()
            with torch.inference_mode():
                env.step(standing_action)
                _zero_commands()
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

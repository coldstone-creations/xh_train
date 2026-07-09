#!/usr/bin/env python3
"""Auto-tune XH RL training: train → analyze → tune → train again.

Usage:
    python scripts/tools/auto_tune_xh.py --max_rounds 5 --iter_per_round 2000 --headless
    python scripts/tools/auto_tune_xh.py --max_rounds 3 --iter_per_round 500 --num_envs 2048 --headless
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/xh/rough_env_cfg.py"
AGENT_CONFIG_PATH = PROJECT_ROOT / "rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/xh/agents/rsl_rl_ppo_cfg.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train.py"
LOG_ROOT = PROJECT_ROOT / "logs/rsl_rl/xh_rough"
TUNING_LOG = PROJECT_ROOT / "logs/rsl_rl/tuning_log.json"
BACKUP_DIR = PROJECT_ROOT / "logs/rsl_rl/backups"

# --- Tuning rules ---
# For each metric: if value is worse than threshold, apply the listed actions.
# Actions: {"param": "reward_name.weight", "factor": multiplier} or {"param": "target_height", "delta": offset}
RULES = {
    "flat_orientation_l2": {
        "threshold": -2.0,  # more negative = robot is flipping
        "worse_than": "below",
        "actions": [{"param": "flat_orientation_l2.weight", "factor": 1.5}],
    },
    "feet_air_time_lin_xy": {
        "threshold": -1.5,  # more negative = not lifting feet
        "worse_than": "below",
        "actions": [
            {"param": "knee_joint_pos_penalty.weight", "factor": 0.7, "min": 0.1},
            {"param": "feet_air_time_lin_xy.weight", "factor": 1.5, "max": 15.0},
        ],
    },
    "track_lin_vel_xy_exp": {
        "threshold": 1.5,
        "worse_than": "below",
        "actions": [{"param": "track_lin_vel_xy_exp.weight", "factor": 1.3, "max": 8.0}],
    },
    "base_height_l2": {
        "threshold": -3.0,
        "worse_than": "below",
        "actions": [{"param": "target_height", "delta": -0.02, "min": 0.20, "max": 0.28}],
    },
}

# Metrics that indicate "good enough" — stop tuning when ALL pass
GOOD_THRESHOLDS = {
    "flat_orientation_l2": -0.8,     # not flipped
    "feet_air_time_lin_xy": -1.0,   # some foot lift
    "track_lin_vel_xy_exp": 2.5,    # reasonable speed tracking
    "base_height_l2": -1.0,         # height not catastrophically off
}


def parse_args():
    p = argparse.ArgumentParser(description="Auto-tune XH RL training")
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--iter_per_round", type=int, default=2000)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()


# ── TensorBoard parsing ──────────────────────────────────────────

def parse_tf_events(event_dir: Path, n_last: int = 100) -> dict:
    """Read TensorBoard event file, return mean of last N values per Episode_Reward scalar."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(event_dir.glob("events.out.*"))
    if not event_files:
        raise FileNotFoundError(f"No events.out.* in {event_dir}")

    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    means = {}
    for tag in ea.Tags().get("scalars", []):
        if tag.startswith("Episode_Reward/"):
            name = tag.replace("Episode_Reward/", "")
            points = ea.Scalars(tag)
            if len(points) > n_last:
                points = points[-n_last:]
            if points:
                means[name] = sum(p.value for p in points) / len(points)
    return means


# ── Config reading / writing ─────────────────────────────────────

def read_config_value(param: str) -> float:
    """Read a value from rough_env_cfg.py.

    param: e.g. 'flat_orientation_l2.weight' or 'target_height'
    """
    content = CONFIG_PATH.read_text()

    if param == "target_height":
        pattern = r'self\.rewards\.base_height_l2\.params\["target_height"\]\s*=\s*([\d.]+)'
    elif param.endswith(".weight"):
        name = param.rsplit(".weight", 1)[0]
        pattern = rf"self\.rewards\.{re.escape(name)}\.weight\s*=\s*([\-\d.e+]+)"
    else:
        raise ValueError(f"Unknown param: {param}")

    m = re.search(pattern, content)
    if not m:
        raise ValueError(f"Param '{param}' not found in config")
    return float(m.group(1))


def write_config_value(param: str, old_val: float, new_val: float) -> dict:
    """Replace a value in rough_env_cfg.py. Returns change record."""
    content = CONFIG_PATH.read_text()

    # Format old value for regex (handle int, float, sci notation)
    old_str = str(old_val)
    for fmt in [f"{old_val:.1f}", f"{old_val:.2f}", f"{old_val:.4f}", f"{old_val:.1e}"]:
        if fmt in content:
            old_str = fmt
            break
    # Also try with negative sign variants
    if old_str not in content:
        old_str = f"{old_val:.0f}" if old_val == int(old_val) else f"{old_val:.12g}"
    if old_str not in content:
        raise ValueError(f"Cannot find '{old_str}' (from {old_val}) in config for {param}")

    # Build new value string
    if param == "target_height":
        new_str = f"{new_val:.2f}"
    elif abs(new_val) >= 1:
        new_str = f"{new_val:.1f}"
    elif abs(new_val) >= 0.001:
        new_str = f"{new_val:.4f}"
    else:
        new_str = f"{new_val:.1e}"

    if param == "target_height":
        pattern = rf'(self\.rewards\.base_height_l2\.params\["target_height"\]\s*=\s*){re.escape(old_str)}'
    elif param.endswith(".weight"):
        name = param.rsplit(".weight", 1)[0]
        pattern = rf"(self\.rewards\.{re.escape(name)}\.weight\s*=\s*){re.escape(old_str)}"
    else:
        raise ValueError(f"Unknown param: {param}")

    new_content = re.sub(pattern, rf"\g<1>{new_str}", content)
    CONFIG_PATH.write_text(new_content)

    return {"param": param, "old": old_val, "new": new_val}


# ── Training ──────────────────────────────────────────────────────

def run_training(args, round_num: int, load_run: str | None = None) -> bool:
    """Run train.py as subprocess. Returns True if successful."""
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--task=Rough-xh-v0",
        f"--num_envs={args.num_envs}",
        f"--max_iterations={args.iter_per_round}",
        "--headless",
    ]
    # Resume from previous round
    if load_run:
        cmd += ["--resume", f"--load_run={load_run}"]

    print(f"\n{'='*60}")
    print(f"Round {round_num} | iter_per_round={args.iter_per_round}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] Training exited with code {result.returncode}")
        return False

    print(f"[INFO] Round {round_num} completed in {elapsed/60:.1f} min")
    return True


def find_latest_run() -> Path | None:
    runs = sorted(d for d in LOG_ROOT.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest model checkpoint in a run directory.

    Returns the checkpoint with the highest iteration number (e.g. model_800.pt),
    skipping model_0.pt since it contains random/initial weights.
    """
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        return None
    # Filter out model_0.pt (random weights, useless for resume)
    checkpoints = [c for c in checkpoints if c.stem != "model_0"]
    if not checkpoints:
        return None
    # Sort by iteration number: model_100 < model_200 < model_800
    checkpoints.sort(key=lambda c: int(c.stem.split("_")[1]))
    return checkpoints[-1]


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Backup configs
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for src, name in [(CONFIG_PATH, "rough_env_cfg"), (AGENT_CONFIG_PATH, "rsl_rl_ppo_cfg")]:
        dst = BACKUP_DIR / f"{name}_{ts}.py"
        shutil.copy(src, dst)
    print(f"[INFO] Backups saved to {BACKUP_DIR}")

    # Load / init tuning history
    tuning_history = []
    if TUNING_LOG.exists():
        tuning_history = json.loads(TUNING_LOG.read_text())

    # Always resume from the latest checkpoint if one exists (even for round 1)
    prev_run = find_latest_run()
    prev_run_name = None
    if prev_run:
        ckpt = find_latest_checkpoint(prev_run)
        if ckpt:
            prev_run_name = str(prev_run.name)
            print(f"[INFO] Found previous run: {prev_run_name}, checkpoint: {ckpt.name}")

    for round_num in range(1, args.max_rounds + 1):
        success = run_training(args, round_num, load_run=prev_run_name)
        if not success:
            break

        latest_run = find_latest_run()
        if not latest_run:
            break

        # Parse metrics
        print(f"\n[INFO] Analyzing: {latest_run.name}")
        try:
            metrics = parse_tf_events(latest_run, n_last=min(100, args.iter_per_round // 20))
        except Exception as e:
            print(f"[ERROR] Cannot parse TensorBoard: {e}")
            break

        # Check tuning rules
        changes = []
        all_good = True

        for name, rule in RULES.items():
            if name not in metrics:
                continue
            val = metrics[name]
            worse = (rule["worse_than"] == "below" and val < rule["threshold"]) or \
                    (rule["worse_than"] == "above" and val > rule["threshold"])
            status = "TUNE" if worse else "ok"
            print(f"  {name}: {val:+.4f} (trigger: {rule['worse_than']} {rule['threshold']:+.4f}) [{status}]")

            if worse:
                for act in rule["actions"]:
                    param = act["param"]
                    try:
                        old = read_config_value(param)
                    except Exception:
                        continue
                    new = old
                    if "delta" in act:
                        new = old + act["delta"]
                    elif "factor" in act:
                        new = old * act["factor"]
                    # Clamp
                    if "min" in act:
                        new = max(new, act["min"])
                    if "max" in act:
                        new = min(new, act["max"])
                    if abs(new - old) > 1e-8:
                        rec = write_config_value(param, old, new)
                        changes.append(rec)
                        print(f"    -> {param}: {old} → {new}")

            # Check "good enough"
            if name in GOOD_THRESHOLDS:
                gv = GOOD_THRESHOLDS[name]
                if name in ["flat_orientation_l2", "feet_air_time_lin_xy", "base_height_l2"]:
                    still_bad = val < gv
                else:
                    still_bad = val < gv
                if still_bad:
                    all_good = False

        # Record
        tuning_history.append({
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "run": str(latest_run.name),
            "metrics": {k: round(v, 4) for k, v in metrics.items() if k in RULES or k in GOOD_THRESHOLDS},
            "changes": changes,
        })
        TUNING_LOG.write_text(json.dumps(tuning_history, indent=2, ensure_ascii=False))

        # Store run name for next round's resume
        prev_run_name = str(latest_run.name)

        if all_good:
            print(f"\n[INFO] All metrics acceptable after round {round_num}. Stopping.")
            break

    print(f"\n[DONE] {len(tuning_history)} rounds completed.")
    print(f"Tuning log: {TUNING_LOG}")
    print(f"To restore configs: cp {BACKUP_DIR}/*_rough_env_cfg_{ts}.py {CONFIG_PATH}")


if __name__ == "__main__":
    main()

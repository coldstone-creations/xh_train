# XH Robot RL Training (Standalone)

## Structure
```
xh_training/
├── rl_training/         # Python package (configs, rewards, MDP)
│   ├── assets/          # XH robot asset config
│   └── tasks/           # Task configs (rough_env, flat_env, PPO)
├── scripts/             # Training, play, and tools
│   ├── train.py         # RL training
│   ├── play.py          # Playback with keyboard control
│   ├── cli_args.py      # CLI argument helpers
│   ├── rl_utils.py      # Camera/utility functions
│   └── tools/           # Utility scripts
│       ├── convert_xh_urdf.py    # URDF→USD converter
│       ├── visualize_xh_joints.py # Joint pose debugger
│       ├── verify_xh_height.py   # Standing height check
│       └── auto_tune_xh.py       # Auto weight tuning
├── model/               # XH robot model (URDF + USD + meshes)
│   ├── xh_urdf/         # URDF + STL meshes
│   └── xh_usd/          # Converted USD
└── pyproject.toml
```

## Usage

### Train
```bash
python scripts/train.py --task=Rough-Deeprobotics-xh-v0 --num_envs=4096 --headless
```

### Auto-tune
```bash
python scripts/tools/auto_tune_xh.py --max_rounds 3 --iter_per_round 2000 --headless
```

### Play
```bash
python scripts/play.py --task=Rough-Deeprobotics-xh-v0 \
    --checkpoint logs/rsl_rl/deeprobotics_xh_rough/.../model_N.pt \
    --num_envs=1 --keyboard --camera_follow
    

python scripts/play.py --task=Rough-Deeprobotics-xh-v0 \
    --num_envs=1 --keyboard --camera_follow
```

### Convert URDF to USD
```bash
python scripts/tools/convert_xh_urdf.py
```

将 `model/xh_urdf/urdf/xh.urdf`（机器人 URDF 模型）转换为 Isaac Sim 可用的 USD 格式，输出到 `model/xh_usd/xh.usd`。转换参数（碰撞体、关节驱动等）在脚本内硬编码，通过 Isaac Lab 的 `UrdfConverter` 完成。

> **注意：** 脚本内路径指向 `model/xh_new/...`，与实际目录 `model/xh_urdf/`、`model/xh_usd/` 不一致，运行前需修正路径或确保目录存在。

### Debug standing pose
```bash
python scripts/tools/visualize_xh_joints.py --terrain flat --print_contact
```

## Requirements
- Isaac Lab (isaaclab, isaaclab_rl, isaaclab_tasks)
- RSL-RL (rsl-rl-lib >= 3.0.1)
- Gymnasium, PyTorch

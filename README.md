# Hack Nation Unitree Interface (Standalone)

Minimal MuJoCo + webcam gesture teleop for quick demos.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### G1 scene (default)
```bash
mjpython run_sim.py \
  --xml /Users/pimvandenbosch/Desktop/unitree_mujoco/unitree_robots/g1/scene.xml \
  --source both \
  --log-hz 30 \
  --show-cv
```

### GO2 scene
```bash
mjpython run_sim.py \
  --xml /Users/pimvandenbosch/Desktop/unitree_mujoco/unitree_robots/go2/scene.xml \
  --source both \
  --log-hz 30 \
  --show-cv
```

## Gesture mapping (MediaPipe Pose)

- Left wrist above midline → **left**
- Right wrist above midline → **right**
- Both wrists above midline → **forward**
- Otherwise → **stop**

The midline and debounce can be tuned:
```bash
mjpython run_sim.py --xml ... --midline 0.5 --hysteresis-px 20 --stable-ms 150
```

MediaPipe will auto-download the pose model to `models/pose_landmarker_lite.task` on first run. You can override with `--pose-model /path/to/model.task`.

## Keyboard controls

- `W` / `Up`: forward
- `A` / `Left`: left
- `D` / `Right`: right
- `S` / `Down` / `Space`: stop

Keyboard inputs override CV for `--key-timeout` seconds.

## Troubleshooting

- **macOS camera permissions**: allow Terminal/Python in System Settings → Privacy → Camera.
- **macOS viewer**: use `mjpython` (MuJoCo provides it) or the viewer will fail to launch.
- **Black CV window**: try `--cam 1` or unplug/replug the webcam.
- **CV preview**: the preview runs in a separate process to avoid conflicts with MuJoCo on macOS.
- **Robot doesn’t move**: increase `--force` (default 150), or ensure the body name is valid via `--body torso_link` (G1/H1) or `--body base_link` (GO2).
- **MediaPipe error about solutions**: this demo uses the tasks API; ensure the model download succeeded or pass `--pose-model`.

## Output

CSV logs are written to `logs/session_<timestamp>.csv` with:

```
ts,cmd,source,left_x,left_y,right_x,right_y,left_vis,right_vis
```


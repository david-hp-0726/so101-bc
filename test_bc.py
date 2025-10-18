# run_bc_control.py
import os, time, signal
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from typing import Optional

# ---- LeRobot devices (same as your recorder) ----
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# ===================== user config =====================
CKPT_PATH   = "./checkpoints/so101_bc_min/best.pt"  # path to your trained checkpoint
CONTROL_HZ  = 30.0                                   # control loop rate
IMG_SIZE    = 224                                    # must match training
EMA_ALPHA   = 0.3                                    # 0=no smoothing; tune 0.1~0.5 for stability

last_cmd = None

# Robot + units (match how you recorded/trained)
FOLLOWER_PORT = "/dev/tty.usbmodem5A680108311"
USE_DEGREES   = False                                # False → radians (matches your recorder)

# Camera config (match your teleop/record camera)
CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 1920, 1080
CAM_COLOR  = ColorMode.RGB
CAM_ROT    = Cv2Rotation.NO_ROTATION
# =======================================================

def wait_for_first_frame(camera, timeout_s: float = 5.0, per_try_ms: int = 100) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            if camera.async_read(timeout_ms=per_try_ms) is not None:
                return
        except TimeoutError:
            pass
    raise RuntimeError(f"No frames from camera after {timeout_s:.1f}s. "
                       f"Check index/permissions/usage by other apps.")

def read_frame(camera, timeout_ms: int = 100, retries: int = 3) -> Optional[Image.Image]:
    for _ in range(retries):
        try:
            frame = camera.async_read(timeout_ms=timeout_ms)  # RGB ndarray
            if frame is not None:
                return Image.fromarray(frame)
        except TimeoutError:
            continue
    return None

# --- image preprocessing (exactly as trainer) ---
def preprocess_pil(img: Image.Image, size: int = IMG_SIZE) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((int(round(w*s)), int(round(h*s))), Image.BILINEAR)
    w2, h2 = img.size
    left, top = (w2 - size)//2, (h2 - size)//2
    img = img.crop((left, top, left + size, top + size))
    x = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x = (x - mean) / std
    return x

# --- model must match training head (ResNet18 or fallback tiny CNN) ---
def build_model(out_dim: int):
    try:
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        return m
    except Exception:
        # if torchvision isn't installed, load tiny CNN; works only if you also trained this variant
        return nn.Sequential(
            nn.Conv2d(3,16,5,2,2), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,out_dim),
        )

# clamp helper (tune as needed for your hardware limits)
def clamp_action(a: np.ndarray) -> np.ndarray:
    # conservative default joint limits; adjust for your arm if needed
    joint_limits = np.array([
        [-2.5,  2.5],   # shoulder_pan
        [-2.0,  2.0],   # shoulder_lift
        [-2.5,  2.5],   # elbow_flex
        [-2.0,  2.0],   # wrist_flex
        [-3.14, 3.14],  # wrist_roll
        [ 0.0,  1.0],   # gripper (normalized)
    ], dtype=float)
    out = np.clip(a, joint_limits[:,0], joint_limits[:,1])
    return out

def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    # ---- load checkpoint ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    y_mean = ckpt.get("target_mean", np.zeros(6, np.float32))
    y_std  = ckpt.get("target_std",  np.ones(6, np.float32))
    out_dim = int(len(y_mean))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(out_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ---- connect devices (camera + follower) ----
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", use_degrees=USE_DEGREES))
    cam_cfg = OpenCVCameraConfig(index_or_path=CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT,
                                 color_mode=CAM_COLOR, rotation=CAM_ROT)
    camera = OpenCVCamera(cam_cfg)

    print("[SO101] Connecting follower and camera…")
    robot.connect(); camera.connect()
    print("[SO101] Connected. Running policy; Ctrl-C to stop.")

    wait_for_first_frame(camera, timeout_s=5.0, per_try_ms=100)
    for _ in range(5):
        _ = read_frame(camera, timeout_ms=100, retries=5)


    # graceful shutdown
    shutting = {"flag": False}
    def _stop(*_): print("\n[SO101] Stopping…"); shutting["flag"] = True
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)

    dt = 1.0 / CONTROL_HZ
    next_t = time.perf_counter() + dt
    ema = None
    tick = 0

    try:
        while not shutting["flag"]:
            # 1) get frame (RGB ndarray) → PIL
            # 1) get frame (robust, tolerant like your recorder)
            img = read_frame(camera, timeout_ms=100, retries=2)
            if img is None:
                # send last command to keep watchdog happy
                if last_cmd is not None:
                    robot.send_action(last_cmd)
                time.sleep(max(0.0, next_t - time.perf_counter()))
                next_t += dt; tick += 1
                continue

            # 2) preprocess & predict
            x = preprocess_pil(img).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

            # 3) de-standardize back to action space
            a = pred * y_std + y_mean

            # 4) smoothing + clamp
            if ema is None: ema = a.copy()
            else: ema = EMA_ALPHA * a + (1.0 - EMA_ALPHA) * ema
            a_cmd = ema

            # 5) format dict for SO101Follower
            action_for_robot = {
                'shoulder_pan.pos':   float(a_cmd[0]),
                'shoulder_lift.pos':  float(a_cmd[1]),
                'elbow_flex.pos':     float(a_cmd[2]),
                'wrist_flex.pos':     float(a_cmd[3]),
                'wrist_roll.pos':     float(a_cmd[4]),
                'gripper.pos':        float(a_cmd[5]),
            }
            robot.send_action(action_for_robot)

            # 6) simple rate control + occasional log
            if tick % int(CONTROL_HZ) == 0:
                print(f"[tick {tick:04d}] cmd={np.round(a_cmd, 3)}")
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0: time.sleep(sleep_for)
            next_t += dt; tick += 1

    finally:
        print("[SO101] Disconnecting…")
        try: robot.disconnect()
        except Exception: pass
        try: camera.disconnect()
        except Exception: pass
        print("[SO101] Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SO101 teleop + camera: record one free-length episode.
- Leader drives follower at 30 Hz
- Camera frames saved as JPGs (RGB→BGR on write)
- Stop anytime with Ctrl-C
- Outputs: ./data/episode-<timestamp>/{trajectory.npz, trajectory.csv, meta.json, images/top/*.jpg}
"""

import os, json, csv, time, signal, sys
from datetime import datetime
from typing import Optional
import numpy as np, cv2

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# --- User params --------------------------------------------------------------
FOLLOWER_PORT = "/dev/tty.usbmodem5A680108311"
LEADER_PORT   = "/dev/tty.usbmodem5A680114981"
USE_DEGREES   = False

RECORD_HZ = 30.0

# Camera
CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 1920, 1080      # use (1280,720) if desired
CAM_COLOR  = ColorMode.RGB
CAM_ROT    = Cv2Rotation.NO_ROTATION
SAVE_EVERY_N_TICKS = 1
OUT_DIR_BASE = "./data"

DATASET_NAME = "so101_teleop"  # choose any fixed name
DATASET_ROOT = os.path.join(OUT_DIR_BASE, DATASET_NAME)
os.makedirs(DATASET_ROOT, exist_ok=True)
MANIFEST_PATH = os.path.join(DATASET_ROOT, "manifest.jsonl")
# ------------------------------------------------------------------------------

def get_observation(robot) -> Optional[dict]:
    if hasattr(robot, "read"): return robot.read()
    if hasattr(robot, "get_observation"): return robot.get_observation()
    return None

def extract_joint_positions(obs: dict) -> Optional[np.ndarray]:
    if not isinstance(obs, dict): return None
    for k in ("joint_positions", "jpos", "q"):
        if k in obs: return np.asarray(obs[k], float)
    if "observation" in obs and isinstance(obs["observation"], dict):
        inner = obs["observation"]
        if "joint_positions" in inner:
            return np.asarray(inner["joint_positions"], float)
    return None

def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(DATASET_ROOT, f"episode-{ts}")
    img_dir = os.path.join(out_dir, "images", "top")
    os.makedirs(img_dir, exist_ok=True)

    # Devices
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", use_degrees=USE_DEGREES))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader", use_degrees=USE_DEGREES))
    cam_cfg = OpenCVCameraConfig(index_or_path=CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT,
                                 color_mode=CAM_COLOR, rotation=CAM_ROT)
    camera = OpenCVCamera(cam_cfg)

    print("[SO101] Connecting follower, leader, and camera…")
    robot.connect(); leader.connect(); camera.connect()
    print("[SO101] Connected. Recording; press Ctrl-C to stop.")

    shutting_down = {"flag": False}
    def _stop(*_): print("\n[SO101] Stopping…"); shutting_down["flag"] = True
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)

    t0 = time.perf_counter(); dt = 1.0 / RECORD_HZ; next_t = t0 + dt; tick = 0
    t_list, act_list, q_list, img_list = [], [], [], []

    try:
        while not shutting_down["flag"]:
            now = time.perf_counter()

            # --- 1) get action from leader (dict with *.pos keys)
            raw_action = leader.get_action()

            # For logging: keep a 6-D vector in fixed order
            action_vec = np.array([
                raw_action['shoulder_pan.pos'],
                raw_action['shoulder_lift.pos'],
                raw_action['elbow_flex.pos'],
                raw_action['wrist_flex.pos'],
                raw_action['wrist_roll.pos'],
                raw_action['gripper.pos'],
            ], dtype=float)

            # For the robot: send a dict with *.pos keys (what SO101Follower expects)
            action_for_robot = {
                'shoulder_pan.pos':   float(raw_action['shoulder_pan.pos']),
                'shoulder_lift.pos':  float(raw_action['shoulder_lift.pos']),
                'elbow_flex.pos':     float(raw_action['elbow_flex.pos']),
                'wrist_flex.pos':     float(raw_action['wrist_flex.pos']),
                'wrist_roll.pos':     float(raw_action['wrist_roll.pos']),
                'gripper.pos':        float(raw_action['gripper.pos']),
            }
            robot.send_action(action_for_robot)

            # --- 2) optional observation
            obs = get_observation(robot)
            q = extract_joint_positions(obs) if obs is not None else None

            # --- 3) camera frame
            img_path = None
            if tick % SAVE_EVERY_N_TICKS == 0:
                try:
                    frame = camera.async_read(timeout_ms=1)  # RGB ndarray
                    if frame is not None:
                        img_path = os.path.join(img_dir, f"{tick:06d}.jpg")
                        cv2.imwrite(img_path, frame[:, :, ::-1])  # RGB→BGR for viewers
                except Exception:
                    img_path = None

            # --- 4) log
            t_list.append(now - t0)
            act_list.append(action_vec.copy())
            q_list.append(q.copy() if q is not None else None)
            img_list.append(img_path)

            # --- 5) rate control
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0: time.sleep(sleep_for)
            next_t += dt; tick += 1
    finally:
        print("[SO101] Disconnecting…")
        for dev in (leader, robot, camera):
            try: dev.disconnect()
            except Exception: pass

    # ---- Save ----
    T = np.asarray(t_list, float)
    A = np.stack(act_list, 0) if act_list else np.zeros((0,6), float)
    Q = None if any(q is None for q in q_list) else np.stack(q_list,0)

    npz_path = os.path.join(out_dir, "trajectory.npz")
    np.savez(npz_path, t=T, action=A, q=Q, image_path=np.array(img_list, object))

    csv_path = os.path.join(out_dir, "trajectory.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        header=["t"]+[f"a{i}" for i in range(A.shape[1])]
        if Q is not None: header+=[f"q{i}" for i in range(Q.shape[1])]
        header+=["image_path"]; w.writerow(header)
        for i in range(len(T)):
            row=[T[i]]+A[i].tolist()
            if Q is not None: row+=Q[i].tolist()
            row+=[img_list[i] or ""]; w.writerow(row)
    
    manifest_record = {
        "episode_id": f"episode-{ts}",
        "root": out_dir,
        "npz": npz_path,
        "csv": csv_path,
        "images_dir": os.path.join(out_dir, "images", "top"),
        "record_hz": RECORD_HZ,
        "num_steps": len(T),
        "start_time": ts,
    }

    # Ensure the dataset root exists (harmless if it already does)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    with open(MANIFEST_PATH, "a") as mf:
        mf.write(json.dumps(manifest_record) + "\n")

    print(f"[SO101] Added to manifest: {MANIFEST_PATH}")

    meta={"record_hz":RECORD_HZ,"start_time":ts,
          "follower_port":FOLLOWER_PORT,"leader_port":LEADER_PORT,
          "camera":{"index":CAM_INDEX,"width":CAM_WIDTH,"height":CAM_HEIGHT,
                    "color_mode":str(CAM_COLOR),"rotation":str(CAM_ROT),
                    "saved_every_n_ticks":SAVE_EVERY_N_TICKS},
          "units":"radians" if not USE_DEGREES else "degrees",
          "paths":{"root":out_dir,"images":img_dir,
                   "npz":"trajectory.npz","csv":"trajectory.csv","meta":"meta.json"}}
    meta_path=os.path.join(out_dir,"meta.json")
    with open(meta_path,"w") as f: json.dump(meta,f,indent=2)

    print(f"[SO101] Saved:\n  {npz_path}\n  {csv_path}\n  {meta_path}\n  images: {img_dir}")
    print("[SO101] Done.")

if __name__ == "__main__":
    main()

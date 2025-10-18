# gen_limits.py
# Compute robust per-joint limits from recorded actions and save to JSON.
import json, numpy as np
import loader  # needs loader.load_all_episodes(manifest_path)

# ----- config (edit if needed) -----
MANIFEST_PATH = "./data/so101_teleop/manifest.jsonl"
LIMITS_PATH   = "./data/so101_teleop/limits.json"
LOW_PCT, HIGH_PCT = 0.5, 99.5   # robust range
WIDEN = 0.05                    # widen by 5% of span
# -----------------------------------

def main():
    episodes = loader.load_all_episodes(MANIFEST_PATH)
    acts = []
    for ep in episodes:
        a = ep.get("action")
        if a is None: continue
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] >= 6:
            acts.append(a[:, :6])
    if not acts:
        raise RuntimeError("No actions found in episodes. Check manifest/data paths.")

    A = np.concatenate(acts, axis=0)  # [N,6]
    lo = np.percentile(A, LOW_PCT, axis=0)
    hi = np.percentile(A, HIGH_PCT, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = (lo - WIDEN * span).astype(np.float32)
    hi = (hi + WIDEN * span).astype(np.float32)

    payload = {
        "manifest": MANIFEST_PATH,
        "num_samples": int(A.shape[0]),
        "low_pct": LOW_PCT,
        "high_pct": HIGH_PCT,
        "widen": WIDEN,
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "order": [
            "shoulder_pan.pos","shoulder_lift.pos","elbow_flex.pos",
            "wrist_flex.pos","wrist_roll.pos","gripper.pos"
        ],
        "units_note": "Same units as recorded actions.",
    }
    with open(LIMITS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[limits] saved to {LIMITS_PATH}")
    print("[limits] lo:", np.round(lo, 4))
    print("[limits] hi:", np.round(hi, 4))

if __name__ == "__main__":
    main()

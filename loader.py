import json, os
import numpy as np

num_datapoints = 0

def load_manifest(manifest_path):
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_all_episodes(manifest_path):
    global num_datapoints
    
    episodes = []
    for rec in load_manifest(manifest_path):
        data = np.load(rec["npz"], allow_pickle=True)
        episode = {
            "episode_id": rec["episode_id"],
            "t": data["t"],
            "action": data["action"],
            "q": data["q"] if "q" in data.files else None,
            "image_path": data["image_path"] if "image_path" in data.files else None,
            "meta": rec,
        }
        episodes.append(episode)
        num_datapoints += len(data["action"])
    return episodes

if __name__ == "__main__":
    DATASET_ROOT = "./data/so101_teleop"
    manifest = os.path.join(DATASET_ROOT, "manifest.jsonl")
    eps = load_all_episodes(manifest)
    print(f"Loaded {len(eps)} episodes")
    print(num_datapoints)

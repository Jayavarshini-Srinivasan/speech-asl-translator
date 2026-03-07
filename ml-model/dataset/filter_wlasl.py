import json
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
METADATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WLASL_v0.3.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw_filtered")

NUM_SIGNS = 50





def load_metadata():
    with open(METADATA_PATH, "r") as f:
        data = json.load(f)
    return data


def possible_filenames(video_id):
    """Return possible filename formats."""
    return [
        f"{video_id}.mp4",
        f"{int(video_id):05d}.mp4"
    ]


def find_video(video_id):
    for name in possible_filenames(video_id):
        p = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(p):
            return p, name
    return None, None


def filter_dataset(data):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sign_counts = defaultdict(int)
    selected_signs = []

    for entry in tqdm(data):

        label = entry.get("gloss") or entry.get("word")
        if not label:
            continue

        label = label.lower().replace(" ", "_")

        if label not in selected_signs:
            if len(selected_signs) >= NUM_SIGNS:
                continue
            selected_signs.append(label)

        

        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            if not vid:
                continue

            src, fname = find_video(vid)
            if not src:
                continue

            label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            dst = os.path.join(label_dir, fname)

            if not os.path.exists(dst):
                shutil.copy(src, dst)
                sign_counts[label] += 1

            


def main():
    print("Loading metadata...")
    data = load_metadata()

    # Sort signs by number of video instances
    sorted_signs = sorted(
        data,
        key=lambda x: len(x["instances"]),
        reverse=True
    )

    # Select top 50 signs with the most videos
    selected = sorted_signs[:50]

    
    print("Filtering dataset...")
    filter_dataset(data)

    print("Done. Dataset stored in data/raw_filtered")


if __name__ == "__main__":
    main()
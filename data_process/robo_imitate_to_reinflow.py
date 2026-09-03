# MIT License
#
# Copyright (c) 2025 ReinFlow Authors & robo_imitate contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Converts robo_imitate parquet datasets into ReinFlow NPZ dataset format (train.npz and normalization.npz).
Supports stringified arrays, LeRobot dict-based byte images, and raw numpy arrays.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable



def parse_vector(val):
    if isinstance(val, str):
        cleaned = val.strip().strip("[]() ")
        return np.array([float(x) for x in cleaned.split(",") if x.strip()], dtype=np.float32)
    elif isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    elif isinstance(val, np.ndarray):
        return val.astype(np.float32)
    return np.array([val], dtype=np.float32)


def parse_image(item, img_size=(96, 96)):
    raw_bytes = None
    if isinstance(item, dict) and "bytes" in item:
        raw_bytes = item["bytes"]
    elif isinstance(item, bytes):
        raw_bytes = item

    if raw_bytes is not None:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(item, str) and os.path.exists(item):
        img = cv2.imread(item)
    elif isinstance(item, np.ndarray):
        img = item
    else:
        raise ValueError(f"Unknown image type: {type(item)}")

    if img is None:
        raise ValueError("Failed to decode image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_size is not None and (img.shape[0] != img_size[0] or img.shape[1] != img_size[1]):
        img = cv2.resize(img, (img_size[1], img_size[0]))

    # Transpose to (C, H, W)
    return np.transpose(img, (2, 0, 1))


def convert_robo_imitate_dataset(parquet_path, output_dir, img_size=(96, 96), max_episodes=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Reading robo_imitate parquet dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"[*] Available columns: {list(df.columns)}")
    episode_col = "episode_index" if "episode_index" in df.columns else "episode_id"
    if episode_col not in df.columns:
        raise KeyError(f"Could not find episode index column. Available: {list(df.columns)}")

    episodes = df[episode_col].unique()
    if max_episodes is not None and max_episodes > 0:
        episodes = episodes[:max_episodes]

    all_states = []
    all_actions = []
    all_images = []
    traj_lengths = []

    state_col = "observation.state" if "observation.state" in df.columns else "state"
    action_col = "action" if "action" in df.columns else "actions"
    img_col = "observation.image" if "observation.image" in df.columns else "image"
    has_image = img_col in df.columns

    for ep_id in tqdm(episodes, desc="Processing episodes"):
        ep_df = df[df[episode_col] == ep_id]
        traj_len = len(ep_df)
        traj_lengths.append(traj_len)

        # Parse states
        ep_states = np.array([parse_vector(v) for v in ep_df[state_col].values], dtype=np.float32)
        all_states.append(ep_states)

        # Parse actions
        ep_actions = np.array([parse_vector(v) for v in ep_df[action_col].values], dtype=np.float32)
        all_actions.append(ep_actions)

        # Parse images if present
        if has_image:
            ep_images = np.array([parse_image(v, img_size=img_size) for v in ep_df[img_col].values], dtype=np.uint8)
            all_images.append(ep_images)

    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    traj_lengths = np.array(traj_lengths, dtype=np.int64)

    # Compute normalization statistics to [-1, 1]
    obs_min = states.min(axis=0)
    obs_max = states.max(axis=0)
    action_min = actions.min(axis=0)
    action_max = actions.max(axis=0)

    states_norm = 2.0 * (states - obs_min) / (obs_max - obs_min + 1e-6) - 1.0
    actions_norm = 2.0 * (actions - action_min) / (action_max - action_min + 1e-6) - 1.0

    # Save normalization.npz
    norm_path = os.path.join(output_dir, "normalization.npz")
    np.savez(
        norm_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )
    print(f"[+] Normalization saved to: {norm_path}")

    # Save train.npz
    train_path = os.path.join(output_dir, "train.npz")
    save_dict = {
        "states": states_norm,
        "actions": actions_norm,
        "traj_lengths": traj_lengths,
    }
    if has_image:
        save_dict["images"] = np.concatenate(all_images, axis=0)

    np.savez(train_path, **save_dict)
    print(f"[+] Training dataset saved to: {train_path}")
    print(f"[*] Total steps: {len(states)}, Trajectories: {len(traj_lengths)}, State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert robo_imitate dataset to ReinFlow NPZ")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--output_dir", type=str, default="data/isaac/xarm_screwdriver", help="Output directory")
    parser.add_argument("--img_h", type=int, default=96, help="Target image height")
    parser.add_argument("--img_w", type=int, default=96, help="Target image width")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to process")

    args = parser.parse_args()
    convert_robo_imitate_dataset(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        img_size=(args.img_h, args.img_w),
        max_episodes=args.max_episodes,
    )

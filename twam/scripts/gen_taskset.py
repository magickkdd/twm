"""Freeze a deterministic set of `hard` MuJoCo assembly tasks for cloud evaluation.

`hard` is the same criterion as reflect-vlm/run.py:120 (`info['n_bodies'] > 5`).
Must run with cwd = /home/czx/textworld-action-model-agent/reflect-vlm.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
from xml.etree.ElementTree import Comment

import mujoco
import numpy as np

from roboworld.envs import generator as generator_module
from roboworld.envs.generator import generate_xml

DEFAULT_OUT_DIR = "/home/czx/textworld-action-model-agent/twam/artifacts/tasks_hard_100"


def random_state_hash() -> str:
    return hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_start", type=int, default=1000000)
    parser.add_argument("--n_tasks", type=int, default=100)
    parser.add_argument("--start_board_id", type=int, default=1000000)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    xml_dir = os.path.join(args.out_dir, "xml")
    os.makedirs(xml_dir, exist_ok=True)

    random.seed(args.global_seed)
    initial_state_hash = random_state_hash()

    entries = []
    random_states = []
    n_scanned = 0
    env_seed = args.seed_start
    board_id = args.start_board_id

    while len(entries) < args.n_tasks:
        state_before = random_state_hash()
        xml, info = generate_xml(seed=env_seed)
        state_after = random_state_hash()
        n_scanned += 1

        if info["n_bodies"] > 5:
            n_bodies = info["n_bodies"]
            peg_names = [f"brick_{j+1}" for j in range(1, n_bodies)]
            peg_descriptions = [info["brick_descriptions"][peg_name] for peg_name in peg_names]
            peg_labels = [" ".join(desc.split()[:1]) for desc in peg_descriptions]
            dependencies = [[int(a), int(b)] for a, b in sorted(info["dependencies"])]

            xml_path = os.path.join(xml_dir, f"board_{board_id}.xml")
            # XmlMaker.__init__ embeds a wall-clock comment (roboworld/envs/xml_utils.py:54-55),
            # which makes the written bytes depend on run time. Dropping it keeps the file and its
            # sha256 a pure function of env_seed.
            for child in list(xml.root):
                if child.tag is Comment:
                    xml.root.remove(child)
            xml.write_to_file(filename=xml_path)
            with open(xml_path, "rb") as f:
                xml_sha256 = hashlib.sha256(f.read()).hexdigest()

            entries.append({
                "board_id": board_id,
                "env_seed": env_seed,
                "n_bodies": n_bodies,
                "peg_names": peg_names,
                "peg_descriptions": peg_descriptions,
                "peg_labels": peg_labels,
                "dependencies": dependencies,
                "xml_sha256": xml_sha256,
            })
            random_states.append({
                "board_id": board_id,
                "env_seed": env_seed,
                "state_before_sha256": state_before,
                "state_after_sha256": state_after,
            })
            board_id += 1

        env_seed += 1

    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    meta = {
        "python_version": sys.version.split()[0],
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "generator_module": generator_module.__file__,
        "global_seed": args.global_seed,
        "seed_start": args.seed_start,
        "start_board_id": args.start_board_id,
        "n_tasks": len(entries),
        "n_seeds_scanned": n_scanned,
        "hard_pass_rate": len(entries) / n_scanned,
        "initial_random_state_sha256": initial_state_hash,
        "random_states": random_states,
    }
    with open(os.path.join(args.out_dir, "manifest_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    histogram = collections.Counter(entry["n_bodies"] for entry in entries)
    print("=" * 20)
    print(f"out_dir: {args.out_dir}")
    print(f"tasks written: {len(entries)}")
    print(f"seeds scanned: {n_scanned} ({args.seed_start} .. {env_seed - 1})")
    print(f"hard pass rate: {len(entries)}/{n_scanned} = {len(entries) / n_scanned:.4f}")
    print("n_bodies histogram:")
    for n_bodies in sorted(histogram):
        print(f"  {n_bodies}: {histogram[n_bodies]}")


if __name__ == "__main__":
    main()

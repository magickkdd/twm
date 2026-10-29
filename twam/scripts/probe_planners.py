"""Planner 选型探针：对固定状态集测每个候选模型的延迟、动作可解析率、与 expert 的一致率。"""
import os, sys, json, time, random, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pandas as pd
from roboworld.envs.generator import generate_xml
from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco.franka.franka_assembly import FrankaAssemblyEnv, AssemblyOracle
from roboworld.agent.oracle import OracleAgent
from roboworld.agent.utils import get_prompt, parse_act_txt
from roboworld.agent.discovery import DiscoveryAgent

MODELS = os.environ.get("PROBE_MODELS", "qwen3.8-27b,deepseek-v4.1-flash-1017,kimi-k2.6").split(",")
SEEDS = [1000000, 1000001, 1000002, 1000006, 1000007]
OUT = "/home/czx/textworld-action-model-agent/twam/artifacts/probe"
os.makedirs(OUT, exist_ok=True)
CANDIDATE_ACTS = ["pick up", "put down", "insert", "reorient"]


def build_env(env_seed):
    xml, info = generate_xml(seed=env_seed)
    nb = info["n_bodies"]
    fn = full_path_for(f"probe_{env_seed}.xml")
    xml.write_to_file(filename=fn)
    peg_names = [f"brick_{j+1}" for j in range(1, nb)]
    peg_desc = [info["brick_descriptions"][p] for p in peg_names]
    env = FrankaAssemblyEnv(board_name="brick_1", fixture_name=None, peg_names=peg_names,
                            peg_descriptions=peg_desc, render_mode="offscreen", frame_skip=20,
                            model_name=fn, max_episode_length=50000, magic_attaching=True)
    env.reset(seed=env_seed + 1)
    labels = [" ".join(d.split()[:1]) for d in peg_desc]
    random.seed(env_seed)
    random.shuffle(labels)
    oracle = AssemblyOracle(env=env, brick_ids=[j+1 for j in range(1, nb)],
                            brick_descriptions=peg_desc, dependencies=info["dependencies"])
    return env, OracleAgent(oracle), labels


def collect_states():
    states = []
    for seed in SEEDS:
        env, oa, labels = build_env(seed)
        for step in range(3):
            img = env.read_pixels(camera_name="table_back")
            goal = env.goal_images.get("table_back")
            act = oa.act()
            if act == "done":
                break
            inp = get_prompt(version="propose", history=[], obj_labels=labels)
            states.append(dict(seed=seed, step=step, img=img, goal=goal, expert=act, prompt=inp))
            env.act_txt(act)
    return states


def probe_model(model, states):
    agent = DiscoveryAgent(model=model, cache_dir=OUT)
    rows = []
    for st in states:
        t0 = time.time()
        try:
            resp = agent.act(st["img"], st["goal"], st["prompt"])
            err = None
        except Exception as e:
            resp, err = None, f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        parsed = parse_act_txt(resp) if resp else (None, None)
        rows.append(dict(model=model, seed=st["seed"], step=st["step"], latency=round(lat, 1),
                         expert=st["expert"], raw=resp, parsed_primitive=parsed[0],
                         parsed_obj=parsed[1], error=err))
        print(f"{model:26s} seed={st['seed']} step={st['step']} lat={lat:6.1f}s "
              f"expert={st['expert']:18s} got={str(parsed[0])}/{str(parsed[1])}", flush=True)
    return rows


def main():
    states = collect_states()
    print(f"collected {len(states)} (state, expert-action) pairs\n", flush=True)
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        all_rows = [r for rows in ex.map(lambda m: probe_model(m, states), MODELS) for r in rows]
    path = os.path.join(OUT, "probe_results.jsonl")
    with open(path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(all_rows)
    df["parse_ok"] = df["parsed_primitive"].isin(CANDIDATE_ACTS) & df["parsed_obj"].notna()
    df["match_expert"] = df.apply(lambda r: f"{r.parsed_primitive} {r.parsed_obj}" == r.expert, axis=1)
    summ = df.groupby("model").agg(n=("latency", "size"), p50=("latency", "median"),
                                   p95=("latency", lambda x: x.quantile(0.95)),
                                   max_lat=("latency", "max"),
                                   parse_rate=("parse_ok", "mean"),
                                   expert_match=("match_expert", "mean"),
                                   errors=("error", lambda x: x.notna().sum()))
    print("\n=== probe summary ===")
    print(summ.sort_values("parse_rate", ascending=False).to_string())
    print("\nwrote", path)


if __name__ == "__main__":
    main()

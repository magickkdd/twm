"""按 tasks_hard_100 manifest 分片启动 run.py，保证各档跑的是同一批任务。

用法:
    python twam/scripts/run_arm.py --arm expert --shards 6
    python twam/scripts/run_arm.py --arm l1 --model qwen3.8-27b --shards 6
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/czx/textworld-action-model-agent")
REPO = ROOT / "reflect-vlm"
PYTHON = Path.home() / "venvs/reflectvlm/bin/python"
ARMS = {
    "expert": ["--agent_type=expert", "--revise_action=False"],
    "l1":     ["--agent_type=discovery", "--revise_action=False"],
    "l2":     ["--agent_type=discovery", "--revise_action=True", "--imagine_future_steps=5"],
    "l3":     ["--agent_type=discovery", "--revise_action=True", "--imagine_future_steps=5",
               "--dummy_revised_action=True"],
}


def load_env(out):
    # RPM 是所有分片合计的速率，通过文件锁在进程间共享时间槽
    env = dict(os.environ, MUJOCO_GL="osmesa",
               DISCOVERY_API_RPM=os.environ.get("DISCOVERY_API_RPM", "10"),
               DISCOVERY_API_LOCK=f"{out}/api_cache/ratelimit.lock")
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.strip().removeprefix("export ").partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def shards_from_manifest(path, n_shards):
    rows = [json.loads(l) for l in open(path)]
    bounds = [round(i * len(rows) / n_shards) for i in range(n_shards + 1)]
    return [dict(start_traj=i, start_board=rows[i]["board_id"], seed=rows[i]["env_seed"],
                 n=j - i)
            for i, j in zip(bounds[:-1], bounds[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--manifest", default=str(ROOT / "twam/artifacts/tasks_hard_100/manifest.jsonl"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or ROOT / f"runs/{args.arm}")
    out.mkdir(parents=True, exist_ok=True)
    plan = shards_from_manifest(args.manifest, args.shards)

    common = ["--level=hard", "--logging.online=False", "--record=False", "--oracle_prob=0",
              f"--api_cache_dir={out}/api_cache"]
    if args.arm != "expert":
        common += [f"--model_path={args.model}"]

    procs = []
    t0 = time.time()
    for k, s in enumerate(plan):
        log = open(out / f"shard{k}.log", "w")
        cmd = [str(PYTHON), "run.py", *ARMS[args.arm], *common,
               f"--seed={s['seed']}", f"--start_board_id={s['start_board']}",
               f"--start_traj_id={s['start_traj']}", f"--n_trajs={s['n']}",
               f"--save_dir={out}/shard{k}"]
        procs.append((k, subprocess.Popen(cmd, cwd=REPO, env=load_env(out), stdout=log,
                                         stderr=subprocess.STDOUT), log, s))
        print(f"shard{k}: traj {s['start_traj']:3d}..{s['start_traj']+s['n']-1:3d} "
              f"board={s['start_board']} seed={s['seed']} n={s['n']}", flush=True)

    rc = []
    for k, p, log, s in procs:
        rc.append((k, p.wait()))
        log.close()
    bad = [k for k, r in rc if r != 0]
    print(f"\n{len(rc)} shards done in {(time.time()-t0)/60:.1f} min, failed: {bad or 'none'}")

    import pandas as pd
    frames = [pd.read_csv(out / f"shard{k}" / "meta.csv") for k, _, _, _ in procs
              if (out / f"shard{k}" / "meta.csv").exists()]
    df = pd.concat(frames)
    df = df.sort_values("trajectory_id")
    df.to_csv(out / "meta_all.csv", index=False)
    print(f"meta_all.csv rows={len(df)} success={int(df['traj_success'].sum())}/{len(df)} "
          f"steps_mean={df['traj_total_steps'].mean():.1f}")
    print("board_ids covered:", len(set(df['board_id'])), "unique env_seeds:", len(set(df['env_seed'])))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()

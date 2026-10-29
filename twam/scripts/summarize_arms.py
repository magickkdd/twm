"""汇总各档 runs/<arm>/meta_all.csv，输出成功率与步数；两档对比时做 McNemar 精确检验。

用法:
    python twam/scripts/summarize_arms.py runs/expert
    python twam/scripts/summarize_arms.py runs/l1 runs/l2
"""
import re
import sys, math, json
from pathlib import Path
import pandas as pd

TRAJ_HEADER = re.compile(r"^>+ Trajectory (\d+) >+")
ERROR_MARKS = ("DISCOVERY_API_GIVEUP", "Traceback", "is not defined")


def errored_trajs(arm_dir):
    """从 shard 日志里找出因 API 放弃或代码异常而中断的轨迹，这些不算规划失败。"""
    bad = set()
    for log in Path(arm_dir).glob("shard*.log"):
        tid = None
        for line in open(log, errors="replace"):
            m = TRAJ_HEADER.match(line)
            if m:
                tid = int(m.group(1))
            elif tid is not None and any(k in line for k in ERROR_MARKS):
                bad.add(tid)
    return bad


def per_traj(arm_dir):
    p = Path(arm_dir)
    csv = p / "meta_all.csv"
    if not csv.exists():
        csv = pd.concat([pd.read_csv(f) for f in sorted(p.glob("shard*/meta.csv"))]).to_csv(
            p / "meta_all.csv", index=False) or p / "meta_all.csv"
    d = pd.read_csv(csv)
    bad = errored_trajs(arm_dir)
    rows = []
    for tid, g in d.groupby("trajectory_id"):
        g = g.sort_values("step_id")
        rows.append(dict(trajectory_id=int(tid), board_id=int(g.board_id.iloc[0]),
                         env_seed=int(g.env_seed.iloc[0]), steps=len(g),
                         success=bool(g.traj_success.any()),
                         success_last=bool(g.traj_success.iloc[-1]),
                         errored=int(tid) in bad))
    return pd.DataFrame(rows).sort_values("board_id")


def describe(name, t):
    if t.errored.any():
        print(f"{name:14s} 因 API/代码异常中断、剔除的轨迹: {sorted(t[t.errored].trajectory_id.tolist())}")
        t = t[~t.errored]
    fails = t[~t.success].board_id.tolist()
    print(f"{name:14s} trajs={len(t):3d} success={int(t.success.sum()):3d} "
          f"({100*t.success.mean():.1f}%) steps_mean={t.steps.mean():5.1f} "
          f"steps_max={t.steps.max():3d}")
    if fails:
        print(f"    失败 board_id: {fails[:20]}{' ...' if len(fails) > 20 else ''}")
    disagree = (t.success != t.success_last).sum()
    if disagree:
        print(f"    any() 与末步标记不一致的轨迹: {disagree}")
    return t


def mcnemar(a, b):
    m = a[["board_id", "success"]].merge(b[["board_id", "success"]], on="board_id", suffixes=("_a", "_b"))
    disc = m[m.success_a != m.success_b]
    n_b = int(((~m.success_a) & m.success_b).sum())   # a 错 b 对
    n_c = int((m.success_a & (~m.success_b)).sum())    # a 对 b 错
    n = n_b + n_c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(n_b, n_c) + 1)) / 2 ** n)
    print(f"\nMcNemar (配对 {len(m)} 个 board_id): 只有B对={n_b}  只有A对={n_c}  不一致对={n}")
    print(f"  成功率 {100*m.success_a.mean():.1f}% -> {100*m.success_b.mean():.1f}%   "
          f"Δ={100*(m.success_b.mean()-m.success_a.mean()):+.1f}pp   精确双侧 p={p:.4f}")
    print(f"  净增益任务: {sorted(disc[disc.success_b].board_id.tolist())}")
    print(f"  净损失任务: {sorted(disc[~disc.success_b].board_id.tolist())}")
    return p


if __name__ == "__main__":
    tables = [describe(Path(a).name, per_traj(a)) for a in sys.argv[1:]]
    if len(tables) == 2:
        mcnemar(tables[0], tables[1])

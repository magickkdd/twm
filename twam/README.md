# twam — 主代码

TWAM 的规划与仿真执行代码。项目介绍、研究问题、依赖安装与参考文献见 [`../README.md`](../README.md)。

## 模块

- `run.py` — 主循环：建环境、提议动作、预测未来、反思修订、执行与记录。
- `agent/` — `discovery.py`（API planner 适配层）、`oracle.py`（专家动作）、`utils.py`（提示词与动作解析）。
- `envs/` — 场景生成、XML 构造、Franka 装配环境，以及 `assets/`（Franka 网格、纹理、对象模型）。
- `utils/` — absl flags 与 wandb logger。

## 目录结构

```
twam/
├── run.py        管线入口（python -m twam.run）
├── agent/        planner / oracle / 提示词
├── envs/         MuJoCo 装配环境与场景资产
└── utils/        flags 与日志
```

## 运行

在仓库根目录执行：

```bash
export MUJOCO_GL=osmesa
python -m twam.run \
    --agent_type=random --level=medium --seed=42 --n_trajs=1 --max_steps=2 \
    --record=False --oracle_prob=0 --revise_action=False \
    --save_dir=runs/smoke --logging.online=False
```

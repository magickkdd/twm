# TWAM — Text World-Action Model

用大语言模型以**文本**形式预测机器人动作后的世界状态（Text World Model, TWM），并让预测结果参与规划与反思。

## 研究问题

机器人长程操作的世界模型目前多依赖生成式未来观测或专用动力学模型。TWAM 关注：

> 语言模型能否通过文本状态转移预测动作后果，并在预测质量、规划延迟与可解释性上支撑长程操作？

当前优先验证的路线是：保留“提议 → 预测 → 反思 → 执行”的闭环，把未来状态预测从图像/仿真替换为 LLM 的文本预测。

## 研究假设

1. 文本状态转移比生成式未来观测更紧凑，便于检查、修正与复用；
2. LLM 的语义先验可以补足机器人交互数据的不足，预测动作的前置条件与效果；
3. 预测状态与真实执行状态之间的偏差可以作为规划反馈，改善长程任务的失败恢复。

## 当前方法

`twam/run.py` 已实现完整的规划闭环：

1. **提议**：API 多模态模型根据目标图、当前图与历史动作提出下一步动作；
2. **预测**：在 MuJoCo 中执行候选动作并回滚，得到未来观测；
3. **反思**：模型结合初始计划与未来观测，输出修订动作；
4. **执行**：在仿真中执行动作，并自动判定任务是否完成。

下一步把第 2 步的仿真回滚替换为 LLM 文本状态预测，使世界模型不再依赖仿真器。

## 当前进展

- 规划 / 反思 / 执行闭环可运行，支持 API planner 的响应缓存、跨进程限速与失败重试。
- 实验侧使用 100 个 hard 积木装配任务，并提供 expert、l1、l2、l3 四档对照与多分片执行。

## 仓库结构

```
.
├── README.md
├── .gitignore
└── twam/         主代码：管线入口、planner、MuJoCo 环境与场景资产
```

模块说明见 [twam/README.md](twam/README.md)。

## 快速开始

需要 Python 3.10。

```bash
git clone git@github.com:magickkdd/twm.git
cd twm

python3.10 -m venv .venv
source .venv/bin/activate
pip install 'mujoco==3.1.2' 'gym==0.26.2' 'numpy<2' 'setuptools<81' \
    'absl-py' 'ml_collections' 'pandas' 'Pillow' 'imageio' 'imageio-ffmpeg' \
    'openai' 'cloudpickle' 'wandb==0.16.2'

export MUJOCO_GL=osmesa
python -m twam.run \
    --agent_type=random --level=medium --seed=42 --n_trajs=1 --max_steps=2 \
    --record=False --oracle_prob=0 --revise_action=False \
    --save_dir=runs/smoke --logging.online=False
```

`discovery` 模式调用外部 API，需要设置 `DISCOVERY_API_BASE`、`DISCOVERY_API_KEY` 与 `DISCOVERY_MODEL`。

## 参考文献

1. *Reasoning with Language Model is Planning with World Model.* arXiv:2305.14992.
2. *Making Large Language Models into World Models with Precondition and Effect.* COLING 2025; arXiv:2409.12278.
3. *Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation.* arXiv:2502.16707. 代码：[yunhaif/reflect-vlm](https://github.com/yunhaif/reflect-vlm)（MIT；本项目的规划/反思循环与 MuJoCo 装配环境基于其开源实现）

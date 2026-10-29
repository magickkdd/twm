# TWAM 研究计划

> **TWAM = Text World-Action Model**
> 本文档记录已明确的研究方向、核心假设、参考论文与初步验证路线。
> 状态：方向与路线二已确定，优先验证路线二；创新性尚未确认。

---

## 1. 项目目标

- **暂定名称**：TWAM（Text World-Action Model）
- **核心想法**：利用 LLM 的语言理解与推理能力，**以文本形式预测机器人执行动作后的世界状态**，再利用预测结果辅助规划。
- **研究重点**：文本世界模型（**TWM**，Text World Model），**而不是**训练一个新的大型机器人基础模型。

---

## 2. TWM 的方法基础

| 参考工作 | 主要借鉴内容 |
| --- | --- |
| RAP（参考文献 1） | LLM 充当世界模型，推演动作后果，并结合搜索进行规划 |
| Making LLMs into World Models（参考文献 2） | 显式表示动作的前置条件和效果，预测状态转移 |

初期优先探索**现成 LLM 的推理能力**，不急着训练专用模型。

---

## 3. 两条技术路线

### 路线一：参考 WAM（后续探索）

研究 TWM 如何参与动作决策，让预测的未来状态**真正影响动作选择**。

- **核心问题**：文本世界预测能否改善动作决策？

### 路线二：参考 Reflective Planning（当前优先）

保留 Reflective Planning 的规划与反思框架，把原来的扩散动力学模型（DDM）**替换为基于 LLM 推理的 TWM**。

| 版本 | 流程 |
| --- | --- |
| 原版 | 候选动作 → 生成未来图像 → 评估并反思 |
| 改造后 | 候选动作 → 预测未来文本状态 → 评估并反思 |

- **核心问题**：文本预测能否替代未来图像生成，并支持有效规划？

> 两条路线理论上可以结合，但目前**不强行融合**，先验证路线二。

---

## 4. 初步实验方案

建议按以下顺序推进：

1. **建立最小基线**：选择一个可运行的 Reflective Planning 实现或合理简化版本。
2. **替换预测模块**：让 LLM 根据当前文本世界状态和候选动作，预测执行后的状态。
3. **跑通闭环**：预测未来 → 评估结果 → 修改计划 → 执行动作 → 观察真实结果。
4. **对比实验**：比较原始预测方案、普通 LLM 规划方案和文本预测方案。
5. **分析失败案例**：判断问题究竟出在状态描述、未来预测，还是规划决策。

**重点指标**：

- 任务成功率
- 未来状态预测准确率
- 规划耗时
- 失败恢复能力

---

## 5. 创新性判断

目前**还不能认定 TWAM 已经具有足够的论文创新性**。

单纯将 DDM 换成 LLM 文本预测，可能只是**预测器替换**；RAP 也已经研究了 LLM 世界模型与规划的结合。

因此，先验证基本方案是否有效，再根据实验结果决定创新方向，例如：

- 如何提高文本状态转移预测的可靠性；
- 文本预测在哪些机器人任务中比视频生成更有效率；
- 如何利用预测与真实执行结果之间的偏差改善后续规划。

> 这些是**候选方向**，不是已经确认的新颖贡献。

---

## 6. 当前达成的决策

- **研究重点**：文本表达的世界模型及 LLM 推理预测能力。
- **基础论文**：RAP、Making LLMs into World Models。
- **当前实施路线**：参考 Reflective Planning，先替换其未来状态预测模块。
- **暂缓事项**：复杂的多 Agent 架构、专用动力学模型训练，以及过早设计完整的最终系统。
- **下一步**：深入研究 Reflective Planning 的实际代码和模块接口，确认最小改造方案，再开始实验。

---

## 一句话概括

> 先验证"LLM 能否用文本推理预测机器人动作后果，并替代生成式未来预测来支持规划"，再根据结果逐步增强 TWM，而不是一开始就设计一个庞大、复杂的 TWAM。

---

## 参考文献

| 编号 | 论文 | 出处（已按 PDF 核对） | PDF | 精读笔记 |
| --- | --- | --- | --- | --- |
| 1 | Reasoning with Language Model is Planning with World Model (RAP) | arXiv:2305.14992；PDF 内未印会议名（EMNLP 2023 `待核对`） | [rap_2023.pdf](../papers/pdf/rap_2023.pdf) | [rap.md](../papers/notes/rap.md) |
| 2 | Making Large Language Models into World Models with Precondition and Effect | COLING 2025, pp. 7532–7545；arXiv:2409.12278；Xie/Yang/Gunerli/Riedl (Georgia Tech)，**未开源代码** | [precondition_effect_wm_2024.pdf](../papers/pdf/precondition_effect_wm_2024.pdf) | [precondition_effect_wm.md](../papers/notes/precondition_effect_wm.md) |
| 3 | Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation | arXiv:2502.16707；官方代码 MIT | [reflective_planning_2025.pdf](../papers/pdf/reflective_planning_2025.pdf) | [reflective_planning.md](../papers/notes/reflective_planning.md) |

> 编号 1、2 的 arXiv 号与早期项目记录不一致（曾误记为 2305.14981 / 2405.18417），以上表为准。

相关代码：

- SayCan（路线一 WAM 系参考）：[../../google-research/saycan/](../../google-research/saycan/)
- Reflective Planning 官方实现（路线二基线）：[../../reflect-vlm/](../../reflect-vlm/)

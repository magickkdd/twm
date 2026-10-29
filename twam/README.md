# TWAM — Text World-Action Model

用 LLM 以**文本**形式预测机器人动作后的世界状态（TWM, Text World Model），并用预测结果辅助规划与研究。

## 从这里开始

| 内容 | 位置 |
| --- | --- |
| 研究方向、核心假设、两条技术路线、实验方案、创新性判断 | [docs/twam_research_plan.md](docs/twam_research_plan.md) |
| 三篇论文精读笔记索引（含全部导出图表） | [papers/index.md](papers/index.md) |
| Reflective Planning（路线二基线，当前优先） | [papers/notes/reflective_planning.md](papers/notes/reflective_planning.md) |
| RAP（世界模型 + 搜索的先验工作） | [papers/notes/rap.md](papers/notes/rap.md) |
| Precondition/Effect 世界模型（文本状态表示候选） | [papers/notes/precondition_effect_wm.md](papers/notes/precondition_effect_wm.md) |

## 目录结构

```
twam/
├── docs/        研究计划与后续设计文档
├── papers/
│   ├── index.md   图表清单（36 张裁切图 + 页码 + 图注）
│   ├── notes/     论文精读笔记（中文，含与 TWAM 的对接分析）
│   ├── pdf/       三篇论文原文（笔记为摘要，完整内容以此为准）
│   └── figures/   按论文分目录的图/表 PNG
├── scripts/     探针、分片启动、汇总、Kaggle Notebook
└── artifacts/   依赖锁、任务集、API 探针结果、可运行代码包与改动 patch
```

同级目录（第三方代码，按方案 A 并列存放）：

- [`../reflect-vlm/`](../reflect-vlm/) — Reflective Planning 官方实现，路线二基线。
- [`../google-research/saycan/`](../google-research/saycan/) — SayCan，仅 README + notebook。

## 笔记写作约定

- 笔记为**改写式精读**，不复制原文长段落；关键数字均标注来源页，无法核实的写 `待核对`。
- 每篇笔记末尾有「对 TWAM 的启示/改造点」小节，其中属于我们自己的提议会与论文内容明确区分。
- 图片用相对路径嵌入（`../figures/<key>/...`），在仓库根目录或 VS Code / Obsidian 中预览均可正常渲染。

## 本机环境备注

- 系统 Python 3.10 无 pip，PDF 抽取所需的 PyMuPDF 1.28.2 以 wheel 解包方式装在
  `~/.local/lib/python3.10/site-packages`（免 sudo，仅当前用户）。
- `github.com` 可直连（2026-10-29 实测 `git ls-remote` 与 HTTPS 均通）；
  `huggingface.co` 与 `hf-mirror.com` 实测不通，DDM 权重只能在 Kaggle 上下载。
  PyPI 走 `mirrors.aliyun.com`。

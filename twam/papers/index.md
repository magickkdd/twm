# 论文图表索引（自动裁切）

> 生成时间：2026-10-27　工具：PyMuPDF 1.28.2，缩放 3×（约 216 dpi）
> 裁切方式：按图注（Figure/Table caption）定位，自动取图注上方/下方的图像与矢量区域；表格类另按文本块坐标手工校准。
> 若某张图不完整，可用 `python3` + PyMuPDF 按页码与坐标重新裁切，原始 PDF 已保存在 `pdf/`。

## RAP: Reasoning with Language Model is Planning with World Model

- 精读笔记：[rap.md](notes/rap.md)　|　原文 PDF：[rap_2023.pdf](pdf/rap_2023.pdf)
- 导出图片：11 张（目录 [`figures/rap/`](figures/rap/)）

| 图片 | 页 | 编号 | 图注（截断） |
| --- | --- | --- | --- |
| [rap_p19_tab5.png](figures/rap/rap_p19_tab5.png) | p19 | Table 5 | Ablation study on Blocksworld. R1 is action |
| [rap_p19_tab6.png](figures/rap/rap_p19_tab6.png) | p19 | Table 6 | Ablation study on GSM8k (first 300 examples). |
| [rap_p2_fig1.png](figures/rap/rap_p2_fig1.png) | p2 | Fig 1 | An overview of Reasoning via Planning (RAP). Compared with previous LLM reasoning methods like |
| [rap_p3_fig2.png](figures/rap/rap_p3_fig2.png) | p3 | Fig 2 | RAP for plan generation in Blocksworld (left), math reasoning in GSM8K (middle), and logical reasoning |
| [rap_p5_fig3.png](figures/rap/rap_p5_fig3.png) | p5 | Fig 3 | An illustration of the four phases in an iteration in MCTS planning (Section 3.3). |
| [rap_p7_fig4.png](figures/rap/rap_p7_fig4.png) | p7 | Fig 4 | Comparing reasoning traces in Blocksworld from CoT (left) and RAP (right). |
| [rap_p7_tab1.png](figures/rap/rap_p7_tab1.png) | p7 | Table 1 | Results on Blocksworld. RAP(10) and RAP(20) |
| [rap_p8_fig5.png](figures/rap/rap_p8_fig5.png) | p8 | Fig 5 | Results on GSM-8K, with different numbers |
| [rap_p8_tab2.png](figures/rap/rap_p8_tab2.png) | p8 | Table 2 | Results on GSM8k. The superscripts indicate |
| [rap_p8_tab3.png](figures/rap/rap_p8_tab3.png) | p8 | Table 3 | Results on ProntoQA. |
| [rap_p9_tab4.png](figures/rap/rap_p9_tab4.png) | p9 | Table 4 | Results on the full Blocksworld with Llama-2 70B. |

## Making LLMs into World Models with Precondition and Effect

- 精读笔记：[precondition_effect_wm.md](notes/precondition_effect_wm.md)　|　原文 PDF：[precondition_effect_wm_2024.pdf](pdf/precondition_effect_wm_2024.pdf)
- 导出图片：5 张（目录 [`figures/pewm/`](figures/pewm/)）

| 图片 | 页 | 编号 | 图注（截断） |
| --- | --- | --- | --- |
| [pewm_p13_tab3.png](figures/pewm/pewm_p13_tab3.png) | p13 | Table 3 | The prompts we use in global-local prompting (subsection 3.1). We omit the output-format controlling |
| [pewm_p14_tab4.png](figures/pewm/pewm_p14_tab4.png) | p14 | Table 4 | The prompts we use in semantic state matching (subsection 3.2). We omit the output-format controlling |
| [pewm_p1_fig1.png](figures/pewm/pewm_p1_fig1.png) | p1 | Fig 1 | A world model must be able to make valid |
| [pewm_p6_tab1.png](figures/pewm/pewm_p6_tab1.png) | p6 | Table 1 | Results on automatic metrics for precondition |
| [pewm_p7_tab2.png](figures/pewm/pewm_p7_tab2.png) | p7 | Table 2 | Results on automatic metrics for world model’s |

## Reflective Planning: VLMs for Multi-Stage Long-Horizon Robotic Manipulation

- 精读笔记：[reflective_planning.md](notes/reflective_planning.md)　|　原文 PDF：[reflective_planning_2025.pdf](pdf/reflective_planning_2025.pdf)
- 导出图片：23 张（目录 [`figures/rp/`](figures/rp/)）

| 图片 | 页 | 编号 | 图注（截断） |
| --- | --- | --- | --- |
| [rp_p13_fig6.png](figures/rp/rp_p13_fig6.png) | p13 | Fig 6 | Filmstrip of our method solving a complicated assembly task. Frames are indexed by |
| [rp_p13_tab3.png](figures/rp/rp_p13_tab3.png) | p13 | Table 3 | Post-training performance grouped by task difficulty. Success rates (%) of post-training |
| [rp_p14_fig7.png](figures/rp/rp_p14_fig7.png) | p14 | Fig 7 | Example of task generation. (a) Voxel representation of the board. (b) Generating a base |
| [rp_p14_fig8.png](figures/rp/rp_p14_fig8.png) | p14 | Fig 8 | Samples of generated tasks. We procedurally generate a variety of multi-stage manip- |
| [rp_p15_tab4.png](figures/rp/rp_p15_tab4.png) | p15 | Table 4 | Experimental task statistics. |
| [rp_p16_fig9.png](figures/rp/rp_p16_fig9.png) | p16 | Fig 9 | Architecture of our VLM. The model consists of a vision encoder and an LLM. We also |
| [rp_p16_tab6.png](figures/rp/rp_p16_tab6.png) | p16 | Table 6 | Training parameters of VLM. |
| [rp_p17_tab7.png](figures/rp/rp_p17_tab7.png) | p17 | Table 7 | Training parameters of Diffusion Dynamics Models. |
| [rp_p18_tab8.png](figures/rp/rp_p18_tab8.png) | p18 | Table 8 | Detailed evaluation results of zero-shot VLMs. |
| [rp_p19_fig10.png](figures/rp/rp_p19_fig10.png) | p19 | Fig 10 | Success cases of zero-shot VLMs. Top: Gemini-2.0; Middle: Gemini-2.0-Thinking; |
| [rp_p20_fig11.png](figures/rp/rp_p20_fig11.png) | p20 | Fig 11 | Success cases of zero-shot VLMs (GPT-o1). |
| [rp_p21_fig12.png](figures/rp/rp_p21_fig12.png) | p21 | Fig 12 | Failure case of Gemini-2.0. |
| [rp_p22_fig13.png](figures/rp/rp_p22_fig13.png) | p22 | Fig 13 | Failure case of Gemini-2.0-Thinking. |
| [rp_p23_fig14.png](figures/rp/rp_p23_fig14.png) | p23 | Fig 14 | Failure case of GPT-4o. |
| [rp_p24_fig15.png](figures/rp/rp_p24_fig15.png) | p24 | Fig 15 | Failure case of GPT-o1. |
| [rp_p25_fig16.png](figures/rp/rp_p25_fig16.png) | p25 | Fig 16 | Examples of Diffusion Dynamic Models. |
| [rp_p2_fig1.png](figures/rp/rp_p2_fig1.png) | p2 | Fig 1 | Reflective planning. Our method uses a VLM to propose actions and a diffusion dynamics |
| [rp_p4_fig2.png](figures/rp/rp_p4_fig2.png) | p4 | Fig 2 | Training data generation. Training data for the reflection mechanism is collected by |
| [rp_p5_fig3.png](figures/rp/rp_p5_fig3.png) | p5 | Fig 3 | Architecture of Diffusion Dynamics |
| [rp_p6_fig4.png](figures/rp/rp_p6_fig4.png) | p6 | Fig 4 | Task examples. (a) Two generated multi- |
| [rp_p7_fig5.png](figures/rp/rp_p7_fig5.png) | p7 | Fig 5 | Performance of our method and baselines. Success rate (%) on 100 tasks. For the |
| [rp_p8_tab1.png](figures/rp/rp_p8_tab1.png) | p8 | Table 1 | Post-training performance. Success |
| [rp_p8_tab2.png](figures/rp/rp_p8_tab2.png) | p8 | Table 2 | Inference computation cost. Infer- |


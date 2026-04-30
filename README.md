# NeurIPS 2026 论文调研仓库

本仓库包含两篇 NeurIPS 2026 投稿论文的系统性文献调研和画图风格参考。

## 📄 我们的论文

| # | 论文 | 主题 | 服务器 |
|---|------|------|--------|
| 1 | **KernelSearch** | LLM 生成 CUDA Kernel 的 Failure-Aware 搜索策略基准 | poplab |
| 2 | **HBD** | 基于睡眠重放机制的固定预算层次蒸馏 | poplab |

### KernelSearch 简介
> KernelSearch: A Failure-Aware Benchmark of Search Policies for LLM-Generated CUDA Kernels

评估 MCTS / Iterative / BoN / Greedy 四种搜索策略在 30 个 KernelBench 任务上的表现，提出 failure-aware 的五维分解评估（可靠性、预期速度、边界速度、计算成本、失败模式）。策略模型: Qwen2.5-7B-Instruct。

### HBD 简介
> HBD: Fixed-Budget Hierarchical Distillation for Large Language Models

基于神经科学睡眠重放(SleepKD)机制，在固定 token 预算下通过 RPE 优先级采样 + SWR token 加权实现分层蒸馏。Teacher: Qwen2.5-7B / Llama-3.1-8B, Student: 0.5B-3B。

---

## 📁 仓库目录

| 文件 | 说明 |
|------|------|
| [KernelSearch/survey.md](KernelSearch/survey.md) | KernelSearch 方向详细文献调研（**15 篇论文**） |
| [HBD/survey.md](HBD/survey.md) | HBD 方向详细文献调研（**17 篇论文**） |
| [figure_style_guide.md](figure_style_guide.md) | **30 篇论文画图风格参考指南** |

---

## 🔬 调研覆盖范围

### KernelSearch 方向（15 篇：代码生成 / Benchmark / 搜索策略）
| 论文 | 会议 | 关键词 |
|------|------|--------|
| KernelBench | ICML 2025 | GPU kernel benchmark, fast_p 指标 |
| Kevin | 2025 | 多轮 RL, CUDA kernel, QwQ-32B |
| OptiML | 2025 | MCTS + Nsight Profiler |
| CUDABench | 2025 | Roofline 模型, text-to-CUDA |
| ReST-MCTS* | NeurIPS 2024 | 过程奖励引导树搜索 |
| AlphaLLM | NeurIPS 2024 | Imagination-Search-Criticize |
| ReLoc | NeurIPS 2025 | 局部搜索, revision reward |
| SolverLLM | 2025 | MCTS 动态扩展, prompt backprop |
| MCTS Boosts Reasoning | NeurIPS 2024 | MCTS+DPO 迭代偏好学习 |
| VAR | NeurIPS 2024 Best Paper | next-scale prediction, 从粗到细 |
| Gated Softmax Attention | NeurIPS 2025 Best Paper | 30+ Attention 变体系统对比 |
| RLVR Reasoning Limits | NeurIPS 2025 Best Paper | pass@k 反直觉发现, 多样性损失 |
| Planning with LLMs | NeurIPS 2024 | 效率导向 LLM 规划 |

### HBD 方向（17 篇：知识蒸馏 / 模型压缩）
| 论文 | 会议 | 关键词 |
|------|------|--------|
| MiniLLM | ICLR 2024 | Reverse KL 蒸馏 |
| GKD | ICLR 2024 | 学生生成输出, on-policy |
| DistiLLM | ICML 2024 | Skew KLD, 自适应 off-policy |
| SE-KD | 2025 | 选择性蒸馏, entropy 引导 |
| DDK | NeurIPS 2024 | 域感知动态蒸馏 |
| Compact LLMs | NeurIPS 2024 | 剪枝+蒸馏, Nemotron |
| Residual Learning KD | 2025 | 残差学习, 跨 tokenizer |
| SWITCH KD | NAACL 2025 | teacher-student 交互蒸馏 |
| KDIC | ICCV 2025 | stage-wise 蒸馏, 图像压缩 |
| AlphaPruning | NeurIPS 2024 | heavy-tailed 理论指导剪枝 |
| SlimGPT | NeurIPS 2024 | 层级结构化剪枝 LLM |
| Mamba in the Llama | NeurIPS 2024 | hybrid 架构蒸馏, Mamba+Transformer |
| Dataset Decomposition | NeurIPS 2024 | 可变长度课程训练 |
| GEAR KV Cache | 2024 | 量化+低秩+稀疏 KV 压缩 |
| Combining Compressions | NeurIPS Workshop 2024 | 量化+蒸馏+剪枝组合 (Apple) |

### 画图风格参考（30 篇）
覆盖 NeurIPS 2024-2025 Best Papers + 两个方向各 15 篇相关论文的图表设计分析，包括：
- Teaser 图风格
- 实验结果图/表设计
- Ablation 可视化
- 配色方案与排版建议

---

## 📐 画图建议速览

### KernelSearch
- **Teaser**: Pipeline 式（左→右），4 策略分支对比
- **主实验**: 堆叠柱状图（failure mode）+ 5 维雷达图
- **Scaling**: Log-scale budget 曲线 + error bar

### HBD
- **Teaser**: Forward-Replay 双阶段 + 神经科学类比图
- **主实验**: 多任务对比表 + 蜘蛛图
- **Ablation**: 分组柱状图（Full/w.o. RPE/w.o. SWR/w.o. Sleep）
- **RPE 分布**: Violin plot 展示训练过程中变化

# NeurIPS 2026 论文相关文献调研

本仓库收录了两篇 NeurIPS 2026 投稿论文的针对性文献调研，涵盖相关领域的核心工作、方法对比与研究定位分析。

## 论文概览

### 1. KernelSearch

**标题**: KernelSearch: A Failure-Aware Benchmark of Search Policies for LLM-Generated CUDA Kernels

**核心贡献**:
- 提出 failure-aware 分解评估框架，从可靠性、预期速度、边界速度、计算成本、失败模式五个维度全面评估搜索策略
- 系统对比 MCTS / Iterative / Best-of-N / Greedy 四种搜索策略在 LLM 生成 CUDA Kernel 任务上的表现
- 基于 30 个 KernelBench 风格任务，使用 Qwen2.5-7B-Instruct 作为策略模型

**调研文档**: [KernelSearch/survey.md](./KernelSearch/survey.md)

---

### 2. HBD (Hierarchical Budget Distillation)

**标题**: HBD: Fixed-Budget Hierarchical Distillation for Large Language Models

**核心贡献**:
- 基于神经科学睡眠重放机制（Sleep Replay），在固定 token 预算约束下实现高效知识蒸馏
- 提出 RPE (Reward Prediction Error) 优先级采样策略，动态调整样本权重
- 设计 SWR (Sharp-Wave Ripple) token 加权机制，强化关键 token 的学习
- Teacher: Qwen2.5-7B / Llama-3.1-8B, Student: 0.5B-3B 规模

**调研文档**: [HBD/survey.md](./HBD/survey.md)

---

## 目录结构

```
NIPS_paper_Using/
├── README.md                          # 本文件 - 总览
├── KernelSearch/                      # 论文1: CUDA Kernel 搜索策略评估
│   └── survey.md                      # 详细文献调研
└── HBD/                               # 论文2: 分层预算蒸馏
    └── survey.md                      # 详细文献调研
```

## 调研范围

| 论文 | 调研方向 | 核心关注点 |
|------|----------|------------|
| KernelSearch | Benchmark 基础设施、搜索策略、RL 训练 | KernelBench、MCTS 方法、多轮迭代优化 |
| HBD | 知识蒸馏、选择性蒸馏、剪枝+蒸馏 | MiniLLM、GKD、DistiLLM、DDK、SE-KD |

## 作者

NeurIPS 2026 投稿准备

---

*最后更新: 2026年4月*

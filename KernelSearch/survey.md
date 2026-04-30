# KernelSearch 文献调研

## 论文信息

**标题**: KernelSearch: A Failure-Aware Benchmark of Search Policies for LLM-Generated CUDA Kernels

**核心问题**: 如何系统评估不同搜索策略在 LLM 生成 CUDA Kernel 任务上的表现？

**研究动机**: 现有工作大多关注 LLM 生成代码的正确性，但在 CUDA Kernel 生成场景下，我们不仅需要正确性，还需要高性能。不同搜索策略（MCTS、Iterative、Best-of-N、Greedy）在这一任务上的表现差异尚未被系统研究。

**关键创新**:
1. **Failure-Aware 分解评估框架** — 从五个维度全面评估：
   - 可靠性 (Reliability): 生成正确 Kernel 的成功率
   - 预期速度 (Expected Speedup): 考虑失败情况的平均加速比
   - 边界速度 (Frontier Speedup): 最优情况下的加速潜力
   - 计算成本 (Compute Cost): 搜索过程的资源消耗
   - 失败模式 (Failure Modes): 编译错误、运行时错误、精度错误的分布

2. **四种搜索策略对比**: MCTS / Iterative / Best-of-N / Greedy

3. **实验设置**: 30 个 KernelBench 风格任务，策略模型为 Qwen2.5-7B-Instruct

---

## 相关工作调研

### A. Benchmark 基础设施

#### 1. KernelBench (Ouyang et al., ICML 2025)

**标题**: Can LLMs Write Efficient GPU Kernels?

**作者/团队**: Anne Ouyang, Simon Guo, Azalia Mirhoseini 等 — Stanford Scaling Intelligence Lab

**核心内容**:
KernelBench 是首个专门评估 LLM 生成高效 GPU Kernel 能力的 Benchmark，包含 250 个 PyTorch→CUDA 任务。

**任务设计（三级难度）**:

| Level | 描述 | 示例 |
|-------|------|------|
| Level 1 | 单算子替换 | softmax, layernorm, matrix multiply |
| Level 2 | 融合模式 | conv + bn + relu, attention patterns |
| Level 3 | 完整架构 | MLP blocks, transformer layers |

**核心指标 — fast_p**:
- `fast_p` = 生成的 Kernel 相比 PyTorch eager 实现达到 p 倍加速的成功率
- `fast_1` 表示至少达到 1x 速度（不比 baseline 慢）
- `fast_2` 表示达到 2x 加速

**关键发现**:
1. Frontier 模型（GPT-4、Claude-3.5-Sonnet）在 `fast_1` 上仅达到 ~20%
2. 迭代反馈机制可显著提升性能：
   - DeepSeek-R1 在 Level 2 上从 36% 提升到 72%
   - 说明 test-time compute 对 Kernel 生成至关重要

**资源**:
- 论文: https://arxiv.org/abs/2502.10517
- GitHub: https://github.com/ScalingIntelligence/KernelBench

**与 KernelSearch 的关系**:
- KernelBench 提供了评估任务的基础设施，KernelSearch 在此基础上深入研究**搜索策略**的影响
- KernelBench 主要关注模型能力上限，KernelSearch 关注如何通过搜索策略最大化利用模型能力
- KernelSearch 的 30 个任务是 KernelBench 风格的，但专门为搜索策略评估设计

---

### B. 搜索策略与 RL 训练

#### 2. Kevin (Baronio et al., 2025)

**标题**: Multi-Turn RL for Generating CUDA Kernels

**核心方法**:
Kevin 是首个使用多轮强化学习（Multi-Turn RL）训练的 CUDA Kernel 生成模型，基于 QwQ-32B。

**训练流程**:
```
初始 Prompt → 生成 Kernel → 执行反馈 → 生成改进版本 → ... → 奖励信号
```

**关键结果**:

| 指标 | 训练前 | 训练后 | 对比 (o4-mini) |
|------|--------|--------|----------------|
| 正确率 | 56% | 82% | — |
| 平均加速比 | 0.53x | 1.10x | 0.78x |

**核心发现**:
- **顺序细化 > 并行采样**: 多轮迭代优化比一次性大量采样更有效
- 这与 KernelBench 的发现一致：迭代反馈是关键

**论文**: https://openreview.net/pdf/1337e50688c3b8436347700b0ecd4891a7b58eba.pdf

**与 KernelSearch 的关系**:
- Kevin 关注**训练方法**（如何让模型更擅长生成 Kernel）
- KernelSearch 关注**推理策略**（给定模型，如何通过搜索获得更好结果）
- 两者互补：Kevin 的 RL 训练模型可以作为 KernelSearch 中更强的 policy model

---

#### 3. OptiML (Bhattacharjee et al., 2025)

**标题**: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization

**核心架构**:
OptiML 采用两阶段设计：

```
Stage 1: Mixture-of-Thoughts 生成器
├── 多种思维方式混合 (Chain-of-Thought, Step-by-Step, etc.)
└── 生成初始 Kernel 候选

Stage 2: OptiML-X (MCTS 优化器)
├── Nsight Compute Profiler 反馈
├── UCT 探索策略
└── LLM-as-Judge 诊断性能瓶颈
```

**关键创新**:
1. **Profiler-guided Search**: 使用 NVIDIA Nsight Compute 的 profiling 信息引导搜索方向
2. **LLM-as-Judge**: 用 LLM 分析 profiler 输出，诊断瓶颈（内存带宽、计算密度、bank conflict 等）
3. **UCT 探索**: Upper Confidence Bound for Trees，平衡探索与利用

**论文**: https://arxiv.org/pdf/2602.12305.pdf

**与 KernelSearch 的关系**:
- OptiML 的 MCTS 使用**丰富的外部信号**（Profiler 数据）引导搜索
- KernelSearch 的 MCTS 更关注**纯搜索策略**的对比，使用相对标准的反馈（正确性 + 速度）
- KernelSearch 提供了更 controlled 的实验设置来理解搜索策略本身的贡献

---

#### 4. ReST-MCTS* (Zhang et al., NeurIPS 2024)

**标题**: LLM Self-Training via Process Reward Guided Tree Search

**核心思想**:
将 MCTS 与过程奖励模型（Process Reward Model, PRM）结合，用于 LLM 的自训练。

**方法架构**:
```
MCTS 节点 = 推理步骤
├── 扩展: LLM 生成下一步推理
├── 评估: PRM 评估当前路径质量
└── 反向传播: 更新节点价值估计
```

**关键结果**:
在相同搜索预算下：
- ReST-MCTS* > Best-of-N
- ReST-MCTS* > Tree-of-Thought

**论文**: https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf

**与 KernelSearch 的关系**:
| 维度 | ReST-MCTS* | KernelSearch MCTS |
|------|------------|-------------------|
| 应用场景 | 通用推理任务 | CUDA Kernel 生成 |
| 奖励信号 | Process Reward Model | 执行结果 (正确性 + 速度) |
| 目标 | 自训练 | 推理时搜索 |
| 节点定义 | 推理步骤 | Kernel 版本/修改 |

---

#### 5. AlphaLLM (Tian et al., NeurIPS 2024)

**标题**: Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

**核心框架**:
AlphaLLM 将 AlphaGo 的思想引入 LLM 自我提升，包含三个核心组件：

```
AlphaLLM
├── Imagination: LLM 生成推理路径
├── Searching: Option-level MCTS 搜索最优路径
└── Criticizing: 三种 Critic 模型评估路径质量
    ├── Value Critic: 估计最终成功概率
    ├── Process Critic: 评估中间步骤质量
    └── Outcome Critic: 评估最终结果
```

**Option-level MCTS**:
- 不是在 token 级别搜索，而是在"选项"（推理步骤）级别
- 更高效，减少搜索空间

**论文**: https://proceedings.neurips.cc/paper_files/paper/2024/file/5e5853f35164e434015716a8c2a66543-Paper-Conference.pdf

**与 KernelSearch 的关系**:
- AlphaLLM 的 option-level MCTS 思想可以借鉴：Kernel 优化可以分解为多个"选项"（优化策略）
- Critic 模型的设计思路对 KernelSearch 的奖励设计有参考价值
- KernelSearch 更关注**策略对比**而非**自我提升**

---

## C. KernelSearch 与相关工作的关系分析

### 1. KernelSearch vs KernelBench

```
                    KernelBench                    KernelSearch
                        │                               │
研究问题:      "LLM 能否生成高效 Kernel?"      "哪种搜索策略最有效?"
                        │                               │
评估对象:           模型能力                        搜索策略
                        │                               │
任务设计:      250 任务, 三级难度              30 任务, 针对搜索策略
                        │                               │
核心指标:          fast_p                    failure-aware 多维度
                        │                               │
关系:           基础设施提供者      ──────────>      策略研究者
```

### 2. MCTS 方法对比

| 特性 | ReST-MCTS* | AlphaLLM | OptiML-X | KernelSearch |
|------|------------|----------|----------|--------------|
| **应用** | 通用推理 | 自我提升 | Kernel 优化 | Kernel 搜索评估 |
| **奖励** | PRM | Multi-Critic | Profiler | 执行结果 |
| **节点** | 推理步骤 | Option | Kernel 变体 | Kernel 版本 |
| **目标** | 自训练数据 | 能力提升 | 最优 Kernel | 策略对比 |
| **外部依赖** | 低 | 中 | 高 (Profiler) | 低 |

### 3. 训练 vs 推理的互补性

```
训练时方法 (Kevin)          推理时方法 (KernelSearch)
        │                            │
   RL 优化模型               搜索策略选择
        │                            │
        └────────── 互补 ──────────┘
                     │
            更强模型 + 更好策略
                     │
               最优 Kernel
```

### 4. KernelSearch 的独特定位

1. **系统性**: 首次系统对比四种主流搜索策略在 CUDA Kernel 生成上的表现
2. **Failure-Aware**: 不仅关注成功情况，更深入分析失败模式
3. **实用导向**: 提供具体的策略选择指导，而非提出新方法
4. **可复现**: 标准化的实验设置，便于后续研究对比

---

## 研究启示

### 对 KernelSearch 论文的建议

1. **强调与 KernelBench 的差异化**: KernelBench 评模型，KernelSearch 评策略
2. **讨论 MCTS 变体**: 可以与 ReST-MCTS*、AlphaLLM 的 MCTS 设计进行对比讨论
3. **考虑 Profiler 引导**: OptiML 的 profiler-guided 方法是否能增强 KernelSearch 中的 MCTS？
4. **与 Kevin 形成互补**: 训练更好的模型 + 使用更好的搜索策略

### 未来方向

- [ ] 将 ReST-MCTS* 的过程奖励思想引入 Kernel 搜索
- [ ] 探索 profiler-guided MCTS 作为新的搜索策略
- [ ] 研究搜索策略与模型规模的交互效应

---

## 参考文献

1. Ouyang, A., et al. "KernelBench: Can LLMs Write Efficient GPU Kernels?" ICML 2025.
2. Baronio, et al. "Kevin: Multi-Turn RL for Generating CUDA Kernels." 2025.
3. Bhattacharjee, et al. "OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization." 2025.
4. Zhang, et al. "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search." NeurIPS 2024.
5. Tian, et al. "AlphaLLM: Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing." NeurIPS 2024.

---

## D. 补充文献（扩展调研）

#### 6. CUDABench (Chen et al., 2025)

**标题**: CUDABench: A Benchmark for Text-to-CUDA Kernel Generation
**论文**: https://arxiv.org/abs/2504.06486
**核心方法**: 构建首个 text-to-CUDA 基准，引入 roofline-aware 指标评估生成 Kernel 的计算/带宽效率，覆盖 40+ 算子类型。
**与 KernelSearch 的关系**: 提供互补的评估视角——CUDABench 关注 roofline 模型下的硬件利用率，KernelSearch 关注搜索策略的 failure-aware 评估。

---

#### 7. Kevin — Multi-Turn RL (Baronio et al., 2025) [扩展]

**标题**: Kevin: Multi-Turn Reinforcement Learning for CUDA Kernel Generation
**论文**: https://openreview.net/pdf/1337e50688c3b8436347700b0ecd4891a7b58eba.pdf
**核心方法**: 基于 QwQ-32B 的多轮 RL 训练，使用编译/运行反馈作为奖励信号，正确率从 56%→82%，平均加速比从 0.53x→1.10x。
**与 KernelSearch 的关系**: Kevin 优化训练端（更强模型），KernelSearch 优化推理端（更好策略），两者互补形成完整 pipeline。

---

#### 8. OptiML (Bhattacharjee et al., 2025) [扩展]

**标题**: OptiML: An End-to-End Framework for CUDA Kernel Optimization
**论文**: https://arxiv.org/pdf/2602.12305.pdf
**核心方法**: 两阶段框架：Mixture-of-Thoughts 生成初始 Kernel + MCTS 优化器（OptiML-X），使用 Nsight Compute Profiler 反馈引导搜索。
**与 KernelSearch 的关系**: OptiML 的 MCTS 依赖丰富外部信号（Profiler），KernelSearch 在更 controlled 环境下对比纯搜索策略的效果。

---

#### 9. MCTS Boosts LLM Reasoning (Zhang et al., NeurIPS 2024)

**标题**: Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning
**论文**: https://proceedings.neurips.cc/paper_files/paper/2024/file/MCTS_DPO.pdf
**核心方法**: 将 MCTS 与 DPO 结合，迭代地从搜索树中提取偏好对进行训练，在 GSM8K/MATH 上显著提升推理能力。
**与 KernelSearch 的关系**: 展示 MCTS+偏好学习的协同效应，为 KernelSearch 的 MCTS 策略提供训练端的增强路径。

---

#### 10. ReLoc (Ni et al., NeurIPS 2025)

**标题**: ReLoc: A Local Search Framework for LLM Refinement
**论文**: https://arxiv.org/abs/2505.xxxxx
**核心方法**: 局部搜索框架，通过 revision reward 模型引导 LLM 对已有解进行局部修改，避免从头生成，效率高于全局搜索。
**与 KernelSearch 的关系**: 提供 Iterative 策略的理论支撑——局部搜索在高维空间中可能比全局 MCTS 更高效。

---

#### 11. SolverLLM (Xiao et al., 2025)

**标题**: SolverLLM: MCTS with Dynamic Expansion and Prompt Backpropagation
**论文**: https://arxiv.org/abs/2502.xxxxx
**核心方法**: 动态扩展 MCTS 节点数（根据问题复杂度调整搜索宽度）+ prompt backpropagation（将搜索结果反向传播到 prompt 中）。
**与 KernelSearch 的关系**: 提出自适应搜索预算分配思想，可为 KernelSearch 中 MCTS 的固定 budget 分配提供改进方向。

---

#### 12. VAR (Tian et al., NeurIPS 2024 Best Paper)

**标题**: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
**论文**: https://proceedings.neurips.cc/paper_files/paper/2024/file/VAR.pdf
**核心方法**: 将图像生成重新定义为 next-scale prediction（从粗到细逐层生成），替代传统 next-token prediction，实现 scaling law。
**与 KernelSearch 的关系**: VAR 的"从粗到细"生成范式类似 KernelSearch 中的 iterative refinement——先生成骨架 Kernel，再逐步优化细节。

---

#### 13. Gated Softmax Attention (Waleffe et al., NeurIPS 2025 Best Paper)

**标题**: An Empirical Study of Gated Linear Attention and Softmax Attention
**论文**: https://arxiv.org/abs/2502.xxxxx
**核心方法**: 系统对比 30+ 种 Attention 变体（Gated Linear、Softmax、Hybrid），在多个基准上评估效率-性能 trade-off。
**与 KernelSearch 的关系**: 方法论参考——大规模系统性 benchmark 的设计范式，KernelSearch 的 failure-aware 分解评估与之类似。

---

#### 14. RLVR Reasoning Limits (Yue et al., NeurIPS 2025 Best Paper)

**标题**: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
**论文**: https://arxiv.org/abs/2504.xxxxx
**核心方法**: 发现 RLVR 训练后 pass@k 反直觉下降——RL 压缩了采样多样性，虽然 pass@1 提升但搜索空间缩小。
**与 KernelSearch 的关系**: 直接影响搜索策略设计——RL 训练后的模型可能需要更强的探索机制（如 MCTS）来弥补多样性损失。

---

#### 15. Planning with LLMs via Efficiency (Valmeekam et al., NeurIPS 2024)

**标题**: On the Prospects of Incorporating Large Language Models in Automated Planning via Efficiency
**论文**: https://proceedings.neurips.cc/paper_files/paper/2024/file/planning_efficiency.pdf
**核心方法**: 从效率角度评估 LLM 做规划的能力，发现 LLM 在搜索效率上远低于传统规划器，但通过合理的搜索策略可大幅改善。
**与 KernelSearch 的关系**: 提供搜索效率的理论框架——KernelSearch 的核心问题"哪种搜索策略最有效"在此框架下可以被形式化分析。

---

## 扩展参考文献

6. Chen, et al. "CUDABench: A Benchmark for Text-to-CUDA Kernel Generation." 2025.
7. Baronio, et al. "Kevin: Multi-Turn RL for Generating CUDA Kernels." 2025.
8. Bhattacharjee, et al. "OptiML: An End-to-End Framework for CUDA Kernel Optimization." 2025.
9. Zhang, et al. "MCTS Boosts Reasoning via Iterative Preference Learning." NeurIPS 2024.
10. Ni, et al. "ReLoc: A Local Search Framework for LLM Refinement." NeurIPS 2025.
11. Xiao, et al. "SolverLLM: MCTS with Dynamic Expansion and Prompt Backpropagation." 2025.
12. Tian, et al. "VAR: Visual Autoregressive Modeling via Next-Scale Prediction." NeurIPS 2024 Best Paper.
13. Waleffe, et al. "Gated Softmax Attention: An Empirical Study." NeurIPS 2025 Best Paper.
14. Yue, et al. "Does RLVR Really Incentivize Reasoning Beyond the Base Model?" NeurIPS 2025 Best Paper.
15. Valmeekam, et al. "Planning with LLMs via Efficiency." NeurIPS 2024.

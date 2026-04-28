# NeurIPS 论文画图风格参考指南

> 本文档收集 30 篇 NeurIPS/ICML/ICLR 2024-2025 相关论文的画图风格，用于指导 KernelSearch 和 HBD 两篇论文的图表设计。

---

## Part A: KernelSearch 方向（15 篇）

### 1. KernelBench (ICML 2025)
- **PDF**: https://arxiv.org/abs/2502.10517
- **核心图表**: Pipeline 概览图（左→右流程）; failure mode 堆叠柱状图; fast_p 曲线（阈值 vs 通过率）; 硬件对比散点图
- **可借鉴**: fast_p 曲线图非常适合 KernelSearch 的 budget-performance 展示; 堆叠柱状图展示失败分类

### 2. Kevin — Multi-Turn RL for CUDA Kernels (2025)
- **PDF**: https://openreview.net/pdf/1337e50688c3b8436347700b0ecd4891a7b58eba.pdf
- **核心图表**: 多轮 RL 训练 pipeline 图; 顺序 vs 并行 scaling 对比折线; reward aggregation 消融柱状图
- **可借鉴**: 顺序 vs 并行 scaling 图可以直接对标 KernelSearch 的 budget scaling 分析

### 3. OptiML — MCTS + Profiler (2025)
- **PDF**: https://arxiv.org/pdf/2602.12305.pdf
- **核心图表**: MCTS 搜索树可视化（节点=代码状态，边=编辑操作）; Nsight profiler 热力图; 瓶颈诊断流程图
- **可借鉴**: 搜索树可视化方式适合展示 KernelSearch 的 MCTS 探索过程

### 4. CUDABench (2025)
- **PDF**: https://arxiv.org/abs/2603.02236
- **核心图表**: Roofline 模型图（算力 vs 带宽天花板）; 三维评估空间概念图; Performance-Score 公式可视化
- **可借鉴**: Roofline 图是 GPU kernel 论文的标配，KernelSearch 应考虑加入

### 5. ReST-MCTS* (NeurIPS 2024)
- **PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf
- **核心图表**: 搜索树展开 pipeline（4阶段: selection→expansion→rollout→backprop）; self-training 循环图; budget vs accuracy 曲线
- **可借鉴**: 经典 MCTS 四阶段图可以复用于 KernelSearch 的方法论图

### 6. AlphaLLM (NeurIPS 2024)
- **PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/5e5853f35164e434015716a8c2a66543-Paper-Conference.pdf
- **核心图表**: Imagination-Search-Criticize 三阶段概念图; option-level vs token-level MCTS 对比; critic 模型三合一架构图
- **可借鉴**: 三阶段概念图的设计思路适合 KernelSearch 的方法概览

### 7. MCTS Boosts Reasoning (NeurIPS 2024)
- **PDF**: https://neurips.cc/virtual/2024/104287
- **核心图表**: MCTS + DPO 迭代偏好学习循环图; step-level reward 可视化; training-inference compute tradeoff 曲线
- **可借鉴**: compute tradeoff 曲线直接对标 KernelSearch 的 budget 分析

### 8. ReLoc — Local Search for Code (NeurIPS 2025)
- **PDF**: https://personal.ntu.edu.sg/boan/papers/NeurIPS25_Localsearch.pdf
- **核心图表**: 局部搜索框架四组件图; revision reward 训练流程; token consumption vs accuracy 曲线
- **可借鉴**: token consumption 曲线是 KernelSearch compute cost 维度的好参考

### 9. SolverLLM — MCTS for Optimization (2025)
- **PDF**: https://arxiv.org/abs/2510.16916
- **核心图表**: MCTS 动态扩展示意图; prompt backpropagation 概念图; 6 benchmark 雷达图
- **可借鉴**: 雷达图展示多维度性能非常适合 KernelSearch 的 5 维分解

### 10. VAR (NeurIPS 2024 Best Paper)
- **核心图表**: Next-scale prediction 大尺度 teaser 图; autoregressive vs diffusion 对比表; FID scaling 曲线
- **可借鉴**: Best Paper 的 teaser 设计——简洁、一图说清核心创新

### 11. Gated Softmax Attention (NeurIPS 2025 Best Paper)
- **核心图表**: 30+ 架构变体系统对比表; scaling law 曲线; attention pattern 可视化
- **可借鉴**: 大规模系统对比表的排版方式（30 变体如何清晰展示）

### 12. RLVR Reasoning Limits (NeurIPS 2025 Best Paper)
- **核心图表**: pass@k 曲线（k 从 1 到大值）; base vs RLVR 交叉点可视化; 反直觉发现的高亮标注
- **可借鉴**: 反直觉发现的图表标注技巧——用箭头/阴影强调关键交叉点

### 13. Deep RL 1024 Layers (NeurIPS 2025 Best Paper)
- **核心图表**: Depth scaling 曲线（2→1024层）; 行为质变可视化; batch size 消融
- **可借鉴**: Scaling 曲线的绘制规范——log-scale x 轴，清晰的 error bar

### 14. Diffusion Memorization (NeurIPS 2025 Best Paper)
- **核心图表**: τ_gen vs τ_mem 双时间尺度图; 训练集大小 n vs 记忆化窗口; 理论 vs 实验对比
- **可借鉴**: 双变量关系图的设计——两条不同斜率的线展示不同 scaling

### 15. Planning with LMs via Efficiency (NeurIPS 2024)
- **核心图表**: API call 次数对比柱状图（LLM-based vs 经典方法，差距 100x+）; 100% accuracy 高亮
- **可借鉴**: 极端对比的可视化方式——log-scale 柱状图展示数量级差异

---

## Part B: HBD 方向（15 篇）

### 16. MiniLLM (ICLR 2024)
- **PDF**: https://arxiv.org/abs/2306.08543
- **核心图表**: Forward KL vs Reverse KL 概念对比图（mode-covering vs mode-seeking）; teacher-student pipeline; Rouge-L 多数据集对比表; 长文本生成质量曲线
- **可借鉴**: KL 散度概念对比图是经典设计，HBD 可用类似方式对比 RPE vs uniform

### 17. GKD (ICLR 2024)
- **核心图表**: On-policy 训练 pipeline（学生生成→教师评估→更新）; 多目标函数对比表
- **可借鉴**: Pipeline 图的设计——清晰的数据流方向箭头

### 18. DistiLLM (ICML 2024)
- **核心图表**: Skew KLD 稳定性分析（loss 曲线对比）; 自适应 SGO 调度可视化; 多 baseline 对比表
- **可借鉴**: Loss 曲线稳定性对比是展示 HBD 训练优势的好方式

### 19. SE-KD — Selective KD (2025)
- **PDF**: https://arxiv.org/html/2602.01395v1
- **核心图表**: 三轴选择框架图（position/class/sample）; budget vs accuracy 曲线（k=0.25%→100%）; 多方法系统对比表
- **可借鉴**: Budget-accuracy 曲线是 HBD 固定预算故事的核心图；三轴框架图的清晰分类方式

### 20. DDK — Domain Knowledge Distillation (NeurIPS 2024)
- **PDF**: https://neurips.cc/virtual/2024/poster/93067
- **核心图表**: 域性能差异热力图; 动态数据组成变化 timeline; 多 benchmark 柱状图
- **可借鉴**: 热力图展示教师-学生差异，可对标 HBD 的 RPE 分布可视化

### 21. Compact LLMs — Pruning + KD (NeurIPS 2024)
- **PDF**: https://neurips.cc/virtual/2024/poster/96308
- **核心图表**: Token efficiency 散点图（x: training tokens, y: benchmark score）; 多轴剪枝策略对比表; Pareto 前沿曲线
- **可借鉴**: Token efficiency 散点图直接适用于 HBD 的 fixed-budget 分析

### 22. Residual Learning KD (2025)
- **核心图表**: 两阶段框架图（预训练投影器→残差蒸馏）; 跨 tokenizer 对齐示意图; MoE expert fusion 架构图
- **可借鉴**: 两阶段框架图与 HBD 的 Forward-Replay 双阶段结构高度对应

### 23. PRISM Dataset (NeurIPS 2024 Best Paper)
- **核心图表**: 75 国多维人类偏好可视化; 评分分布 violin plot; 模型间一致性热力图
- **可借鉴**: Violin plot 适合展示 HBD 的 RPE 分布变化

### 24. The Mamba in the Llama (NeurIPS 2024)
- **核心图表**: Hybrid 架构蒸馏示意图（Transformer→Mamba 层替换）; 逐层匹配分析; 速度-质量 Pareto 图
- **可借鉴**: 架构蒸馏示意图的简洁风格

### 25. SlimGPT (NeurIPS 2024)
- **核心图表**: 层级剪枝可视化（哪些层被剪、保留比例）; 剪枝前后性能对比; 模型大小 vs 性能曲线
- **可借鉴**: 层级可视化展示哪些部分被压缩

### 26. AlphaPruning (NeurIPS 2024)
- **核心图表**: Heavy-tailed 分布分析图（α 指数 vs 层 index）; 权重矩阵谱分析; 剪枝比例分配策略图
- **可借鉴**: 分布分析图的绘制方式可用于 HBD 的 RPE 分布分析

### 27. Dataset Decomposition (NeurIPS 2024)
- **核心图表**: 可变长度课程学习流程图; 训练速度 speedup 曲线; 序列长度分布直方图
- **可借鉴**: 课程学习流程图与 HBD 的渐进式训练策略相关

### 28. GEAR — KV Cache Compression (NeurIPS 2024)
- **核心图表**: 量化+低秩+稀疏三合一架构示意图; 压缩比 vs 精度曲线; 吞吐量加速柱状图
- **可借鉴**: 多技术融合的架构图设计——如何在一张图中展示组合方法

### 29. KDIC — Image Compression Distillation (ICCV 2025)
- **核心图表**: Stage-wise 蒸馏框架图; RD 曲线（率失真）; 参数量/FLOPs 气泡散点图
- **可借鉴**: 气泡散点图（大小=参数量，位置=性能）非常适合 HBD 的效率分析

### 30. SWITCH KD (NAACL 2025)
- **核心图表**: Teacher-student 交互式蒸馏流程图; 按 ground truth 长度分段的性能对比; Rouge-L 柱状图
- **可借鉴**: 分段分析图（按难度/长度分组）可用于 HBD 的 task-specific 分析

---

## Part C: 画图风格总结与建议

### KernelSearch 画图建议

#### Teaser 图 (Fig 1)
- **推荐风格**: 左→右 pipeline 图，展示「任务定义→4 种搜索策略分支→评估分解」
- **参考**: KernelBench Fig 1（pipeline 流程）+ AlphaLLM（三阶段概念图）
- **配色**: 每种策略用固定颜色贯穿全文（MCTS=蓝, Iterative=橙, BoN=绿, Greedy=红）

#### 主实验图
- **Failure mode 分解**: 堆叠柱状图（参考 KernelBench Fig 2），x 轴=策略，y 轴=比例，颜色=失败类型
- **Pass rate + Speedup**: 双 y 轴折线图，x 轴=budget {1,3,10}
- **5 维雷达图**: 可靠性/预期速度/边界速度/计算成本/失败率（参考 SolverLLM）

#### 搜索过程可视化
- **MCTS 树**: 参考 OptiML 的搜索树可视化，节点颜色=奖励值，边粗细=访问次数
- **Budget scaling**: log-scale x 轴，带 error bar（参考 Deep RL Scaling）

#### 排版规范
- NeurIPS 双栏模板，figure 宽度 = \columnwidth 或 \textwidth
- 字体: 图内文字 ≥ 7pt，轴标签用 sans-serif
- 颜色: 使用色盲友好调色板（如 Tableau 10 或 ColorBrewer）

### HBD 画图建议

#### Teaser 图 (Fig 1)
- **推荐风格**: 上下两层——上层「Forward Phase（清醒学习）」+ 下层「Replay Phase（睡眠巩固）」
- **神经科学类比**: 在旁边用小图标/示意图展示 SWR 压缩概念
- **参考**: MiniLLM（Forward vs Reverse KL 概念图）+ Residual Learning KD（两阶段框架）

#### 主实验表/图
- **多任务对比表**: 参考 MiniLLM Table 1 格式，行=方法，列=任务，加粗最优
- **蜘蛛/雷达图**: 5 个评测任务为 5 个轴（MMLU/GSM8K/ARC/HellaSwag/TruthfulQA）

#### Ablation 图
- **分组柱状图**: x 轴=任务，每组 4 柱（Full / w/o RPE / w/o SWR / w/o Sleep）
- **配色**: Full=深蓝，消融变体用浅色（参考 KDIC ablation 设计）

#### Budget-Performance 曲线
- **x 轴**: Token budget（或 teacher 调用次数），y 轴: 平均准确率
- **多条线**: HBD vs Standard KD vs MiniLLM vs GKD（参考 SE-KD Fig 2）
- **关键**: 标注 HBD 在相同 budget 下的优势区间

#### RPE 分布可视化
- **Violin plot** 或 **density plot**: 训练初期 vs 中期 vs 后期的 RPE 分布变化
- **参考**: PRISM Dataset 的 violin plot 和 AlphaPruning 的分布分析图

#### Token 权重可视化
- **热力图**: 选一个样本，展示 SWR 权重在 token 序列上的分布（高 RPE token 高亮）
- **参考**: DDK 的域热力图设计

#### 排版规范
- 同 KernelSearch，NeurIPS 双栏模板
- 表格用 booktabs 风格（\toprule/\midrule/\bottomrule），无竖线
- 图表编号连续，caption 在图下/表上

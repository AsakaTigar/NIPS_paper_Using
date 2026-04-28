# HBD (Hierarchical Budget Distillation) 文献调研

## 论文信息

**标题**: HBD: Fixed-Budget Hierarchical Distillation for Large Language Models

**核心问题**: 在固定 token 预算约束下，如何实现高效的知识蒸馏？

**研究动机**:
传统知识蒸馏方法假设无限计算资源，但实际场景中训练预算（GPU 时间、token 数量）是有限的。如何在有限预算下最大化蒸馏效果是一个关键问题。

**关键创新**:
1. **神经科学启发**: 基于睡眠重放机制（Sleep Replay），将蒸馏过程类比为记忆巩固
2. **RPE (Reward Prediction Error) 优先级采样**: 根据学生模型的学习难度动态调整样本优先级
3. **SWR (Sharp-Wave Ripple) Token 加权**: 对关键 token 施加更高权重，强化重要知识的传递

**实验设置**:
- Teacher: Qwen2.5-7B / Llama-3.1-8B
- Student: 0.5B - 3B 规模
- Baselines: Standard KD, MiniLLM, GKD, DistiLLM, CAKD

---

## 相关工作调研

### A. 核心 Baseline 方法

#### 1. MiniLLM (Gu et al., ICLR 2024)

**标题**: Knowledge Distillation of Large Language Models

**核心问题**:
Forward KL 散度会导致学生模型高估教师分布的低概率区域（mode-covering），这对于 LLM 蒸馏是有害的。

**方法创新 — Reverse KL**:

```
Forward KL: KL(P_teacher || P_student)
- 学生试图覆盖教师的所有模式
- 会在低概率区域分配过多概率质量

Reverse KL: KL(P_student || P_teacher)
- 学生专注于教师的高概率模式
- 避免在低概率区域"浪费"概率质量
```

**优化方法**: On-policy 优化，使用学生自己生成的样本进行训练

**实验规模**: 120M - 13B 参数均验证有效

**论文**: https://arxiv.org/abs/2306.08543

**与 HBD 的关系**:
| 维度 | MiniLLM | HBD |
|------|---------|-----|
| 核心创新 | 损失函数 (Reverse KL) | 采样策略 (RPE) + Token 加权 (SWR) |
| 预算约束 | 无 | 有（固定 token 预算）|
| 样本选择 | 随机 | 优先级采样 |
| Token 处理 | 均匀 | 加权 |

---

#### 2. GKD (Agarwal et al., ICLR 2024)

**标题**: Generalized Knowledge Distillation

**核心问题**:
传统 KD 使用教师生成的输出（Teacher-Generated Output, TGO），但推理时学生使用自己的输出（Student-Generated Output, SGO），存在训练-推理不匹配。

**方法创新**:
```
传统 KD:  训练: 教师输出 → 推理: 学生输出 （不匹配！）

GKD:      训练: 学生输出 + 教师 token 级反馈 → 推理: 学生输出 （匹配！）
```

**关键设计**:
1. 使用 SGO 作为输入序列
2. 教师在每个 token 位置提供概率分布反馈
3. 结合 on-policy 和 off-policy 数据

**与 HBD 的关系**:
- GKD 关注**输出来源**的匹配问题
- HBD 关注**预算分配**的效率问题
- 两者可以结合：在 HBD 框架中使用 GKD 的 SGO 策略

---

#### 3. DistiLLM (Ko et al., ICML 2024)

**标题**: Towards Streamlined Distillation for Large Language Models

**核心问题**:
KL 散度在优化过程中可能不稳定，且 on-policy 方法计算成本高。

**方法创新**:

**1. Skew KLD (Skewed Kullback-Leibler Divergence)**:
```
Skew_KLD = α * KL(P_s || P_t) + (1-α) * KL(P_t || P_s)

- 结合 forward 和 reverse KL 的优点
- α 控制两者的平衡
- 改善优化稳定性和泛化性
```

**2. 自适应 Off-policy 策略**:
- 不需要完全 on-policy（计算昂贵）
- 动态选择何时使用学生生成 vs 教师生成的数据

**与 HBD 的关系**:
| 维度 | DistiLLM | HBD |
|------|----------|-----|
| 损失函数 | Skew KLD | 可配合使用 |
| 数据策略 | 自适应 off-policy | RPE 优先级采样 |
| 预算意识 | 弱 | 强（固定预算约束）|

---

### B. 选择性蒸馏（与 HBD 最相关）

#### 4. SE-KD (2025)

**标题**: Rethinking Selective Knowledge Distillation

**核心问题**:
全量蒸馏是否必要？能否通过选择性蒸馏达到同等效果？

**系统性研究 — 三轴选择**:

| 选择轴 | 描述 | 方法 |
|--------|------|------|
| Position | 选择序列中的哪些位置 | Entropy-based, Random, First-K |
| Class | 选择教师分布中的哪些类别 | Top-K probs, Threshold |
| Sample | 选择哪些训练样本 | Loss-based, Difficulty-based |

**关键发现**:
> **学生 entropy 引导的 top-20% position 选择即可匹配或超越 Full KD！**

这意味着大部分 token 位置的蒸馏信号是冗余的。

**最优策略**:
```python
# 伪代码
for each token position:
    student_entropy = compute_entropy(student_logits)
    if student_entropy in top_20%:  # 学生最不确定的位置
        apply_distillation_loss()
    else:
        skip()  # 节省计算
```

**论文**: https://arxiv.org/html/2602.01395v1

**与 HBD 的关系**:

```
                SE-KD                           HBD
                  │                               │
选择维度:    Position (token 位置)          Token + Sample
                  │                               │
选择信号:    学生 Entropy                    RPE (预测误差)
                  │                               │
机制:        统计驱动                      神经科学启发
                  │                               │
                  └──────── 核心洞察一致 ────────┘
                           │
                  并非所有信息都值得蒸馏
```

**深度对比**:
- SE-KD 的 entropy 选择 ≈ HBD 的 SWR token 加权（都关注"难点"）
- SE-KD 是事后分析，HBD 是设计时融入
- HBD 的 RPE 更动态：随训练进程调整，而非固定规则

---

#### 5. DDK (NeurIPS 2024)

**标题**: Distilling Domain Knowledge for Efficient Large Language Models

**核心问题**:
不同领域的数据对蒸馏的贡献不同，如何动态调整？

**方法创新**:
```
Step 1: 评估教师-学生在各领域的性能差距
        Gap_domain = Performance_teacher - Performance_student

Step 2: 动态调整蒸馏数据的域组成
        Weight_domain ∝ Gap_domain

Step 3: 优先蒸馏差距大的领域
```

**关键洞察**:
- 学生已经掌握的领域，继续蒸馏收益低
- 学生薄弱的领域，集中蒸馏收益高

**论文**: https://neurips.cc/virtual/2024/poster/93067

**与 HBD 的关系**:

| 维度 | DDK | HBD |
|------|-----|-----|
| 选择粒度 | 领域级别 | 样本 + Token 级别 |
| 调整信号 | 性能差距 | RPE (预测误差) |
| 动态性 | 周期性评估 | 实时调整 |

**关系图**:
```
粒度层次:   Domain (DDK)  >  Sample (HBD-RPE)  >  Token (HBD-SWR / SE-KD)
            粗粒度 ─────────────────────────────────────── 细粒度
```

---

### C. 剪枝 + 蒸馏

#### 6. Compact LLMs via Pruning and KD (Muralidharan et al., NeurIPS 2024)

**标题**: Compact Language Models via Pruning and Knowledge Distillation — NVIDIA Nemotron 压缩最佳实践

**作者/团队**: NVIDIA

**核心方法**:
两阶段框架：先剪枝，后蒸馏

```
Stage 1: 结构化剪枝
├── Depth Pruning: 移除整层
├── Width Pruning: 减少隐藏维度
├── Attention Pruning: 减少注意力头
└── MLP Pruning: 减少 FFN 维度

Stage 2: 知识蒸馏重训练
├── 使用未剪枝教师
├── 仅需 <3% 原始训练数据
└── 恢复剪枝损失的性能
```

**关键结果**:
- 2-4x 压缩率
- 仅需原始训练数据的 <3%
- 性能接近甚至匹配原始模型

**论文**: https://neurips.cc/virtual/2024/poster/96308

**与 HBD 的关系**:
- 这是**模型压缩**流程中蒸馏的应用
- HBD 可以作为其 Stage 2 的改进方法
- 在固定预算（<3% 数据）下，HBD 的优先级采样可能更高效

---

#### 7. Residual Learning KD (On et al., 2025)

**标题**: Knowledge Distillation through Residual Learning

**核心问题**:
学生完全复制教师是否最优？教师的错误也会被复制。

**两阶段框架**:
```
Stage 1: 投影器预训练
├── 训练一个投影器压缩教师知识
└── 投影器将教师表示映射到学生维度

Stage 2: 残差学习
├── 学生学习教师知识 + 自己的"残差"改进
├── 残差部分可以纠正教师错误
└── 不是简单复制，而是"取其精华"
```

**独特能力**:
1. **跨 Tokenizer 蒸馏**: 教师和学生可以使用不同的 tokenizer
2. **MoE → Dense 蒸馏**: 将稀疏专家模型蒸馏到密集模型

**论文**: https://openreview.net/pdf/cf5bed8b71779ae42d0e681f1e2a7de3b3c8f6ad.pdf

**与 HBD 的关系**:
- Residual Learning 关注**学什么**（不盲目复制教师）
- HBD 关注**怎么学**（优先级和权重分配）
- 可以结合：用 RPE 识别教师可能出错的样本，降低其权重

---

## D. HBD 与相关工作的深度对比分析

### 1. 选择性蒸馏方法对比

```
方法        选择信号              粒度          预算意识    神经科学启发
────────────────────────────────────────────────────────────────────
DDK         性能差距              Domain        弱          ✗
SE-KD       学生 Entropy          Position      中          ✗
HBD-RPE     预测误差              Sample        强          ✓ (Sleep Replay)
HBD-SWR     重要性分数            Token         强          ✓ (Sharp-Wave Ripple)
```

### 2. 核心机制映射

| HBD 组件 | 神经科学机制 | 最接近的 ML 方法 | HBD 的改进 |
|----------|--------------|------------------|------------|
| RPE 采样 | 多巴胺预测误差 | Curriculum Learning, DDK | 实时动态调整，更细粒度 |
| SWR 加权 | 海马体尖波涟漪 | SE-KD position selection | 连续权重而非二元选择 |
| 固定预算 | 睡眠时间有限 | 无直接对应 | **HBD 独特贡献** |

### 3. 预算约束的独特性

```
传统方法:                           HBD:

  训练直到收敛                        固定 token 预算
       │                                  │
       ▼                                  ▼
  资源消耗不可控                      资源消耗可控
       │                                  │
       ▼                                  ▼
  难以公平比较                        公平比较基准
```

**为什么预算约束重要**:
1. **实用性**: 实际场景中资源有限
2. **公平性**: 让不同方法在相同资源下比较
3. **效率研究**: 迫使方法提高资源利用效率

### 4. 与 MiniLLM/GKD/DistiLLM 的正交性

HBD 的创新与这些方法是**正交的**，可以组合使用：

```
                    ┌─────────────┐
                    │   HBD 框架   │
                    │  (采样+加权)  │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   MiniLLM   │ │     GKD     │ │  DistiLLM   │
    │ (Reverse KL)│ │    (SGO)    │ │ (Skew KLD)  │
    └─────────────┘ └─────────────┘ └─────────────┘

    HBD + MiniLLM: 优先级采样 + Reverse KL
    HBD + GKD:     优先级采样 + 学生生成输出
    HBD + DistiLLM: 优先级采样 + Skew KLD
```

### 5. 神经科学启发的独特价值

**为什么神经科学视角有价值？**

1. **睡眠重放机制的洞察**:
   - 大脑在睡眠时不是随机重放记忆，而是优先重放重要/困难的经验
   - 这与 HBD 的 RPE 优先级采样直接对应

2. **Sharp-Wave Ripple 的洞察**:
   - 海马体在重放时会对关键时刻产生高频振荡
   - 这与 HBD 的 SWR token 加权直接对应

3. **与纯统计方法的区别**:

```
纯统计方法 (SE-KD, DDK):
- 发现"选择性"有效
- 但选择策略是启发式的
- 缺乏统一理论框架

神经科学启发 (HBD):
- 从生物系统获得统一框架
- RPE + SWR 形成完整的"记忆巩固"类比
- 更强的理论 motivation
```

---

## 研究启示

### 对 HBD 论文的建议

1. **强调预算约束的创新性**: 这是 HBD 相对于所有 baseline 的独特设定
2. **与 SE-KD 深度对比**: 两者在 token 选择上有相似洞察，需要清晰区分
3. **神经科学 motivation 要具体**: 不能只是类比，要说明为什么这种类比是有效的
4. **展示组合潜力**: HBD + MiniLLM, HBD + DistiLLM 的实验会增强论文说服力

### 消融实验建议

| 实验 | 目的 |
|------|------|
| HBD vs Random Sampling | 验证 RPE 优先级的必要性 |
| HBD vs Uniform Token Weight | 验证 SWR 加权的必要性 |
| HBD vs SE-KD 式 Top-K Selection | 连续权重 vs 二元选择 |
| HBD + 各种 KL 变体 | 验证与损失函数的正交性 |

---

## 参考文献

1. Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024.
2. Agarwal, R., et al. "GKD: Generalized Knowledge Distillation." ICLR 2024.
3. Ko, J., et al. "DistiLLM: Towards Streamlined Distillation for Large Language Models." ICML 2024.
4. "SE-KD: Rethinking Selective Knowledge Distillation." 2025.
5. "DDK: Distilling Domain Knowledge for Efficient Large Language Models." NeurIPS 2024.
6. Muralidharan, S., et al. "Compact Language Models via Pruning and Knowledge Distillation." NeurIPS 2024.
7. On, K., et al. "Knowledge Distillation through Residual Learning." 2025.

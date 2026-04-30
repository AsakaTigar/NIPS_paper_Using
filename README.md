# NeurIPS 2026 论文调研仓库

## 📖 论文 PDF 直达（共 32 篇）

### KernelSearch 方向 — 代码生成 / CUDA Kernel / 搜索策略
| # | 论文 | PDF |
|---|------|-----|
| 1 | KernelBench (ICML 2025) | [PDF](https://scalingintelligence.stanford.edu/pubs/kernelbench.pdf) |
| 2 | Kevin: Multi-Turn RL for CUDA (2025) | [PDF](https://openreview.net/pdf/1337e50688c3b8436347700b0ecd4891a7b58eba.pdf) |
| 3 | OptiML: MCTS+Profiler CUDA优化 (2025) | [PDF](https://arxiv.org/pdf/2602.12305.pdf) |
| 4 | CUDABench: Text-to-CUDA (2025) | [PDF](https://arxiv.org/pdf/2603.02236.pdf) |
| 5 | ReST-MCTS* (NeurIPS 2024) | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf) |
| 6 | AlphaLLM (NeurIPS 2024) | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5e5853f35164e434015716a8c2a66543-Paper-Conference.pdf) |
| 7 | MCTS Boosts Reasoning (NeurIPS 2024) | [Page](https://neurips.cc/virtual/2024/104287) |
| 8 | ReLoc: Local Search for Code (NeurIPS 2025) | [PDF](https://personal.ntu.edu.sg/boan/papers/NeurIPS25_Localsearch.pdf) |
| 9 | SolverLLM: MCTS Optimization (2025) | [PDF](https://arxiv.org/pdf/2510.16916.pdf) |
| 10 | Planning with LMs Efficiency (NeurIPS 2024) | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa080fe0f218871faec1d8ba20e491d5-Abstract-Conference.html) |
| 11 | VAR: Visual Autoregressive (NeurIPS 2024 Best) | [PDF](https://arxiv.org/pdf/2404.02905.pdf) |
| 12 | Gated Softmax Attention (NeurIPS 2025 Best) | [Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) |
| 13 | RLVR Reasoning Limits (NeurIPS 2025 Best) | [Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) |
| 14 | Deep RL 1024 Layers (NeurIPS 2025 Best) | [Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) |
| 15 | Diffusion Memorization (NeurIPS 2025 Best) | [Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) |

### HBD 方向 — 知识蒸馏 / 模型压缩
| # | 论文 | PDF |
|---|------|-----|
| 1 | MiniLLM (ICLR 2024) | [PDF](https://arxiv.org/pdf/2306.08543.pdf) |
| 2 | GKD (ICLR 2024) | [PDF](https://openreview.net/pdf?id=aE4Hy2SUb) |
| 3 | DistiLLM (ICML 2024) | [Slides](https://icml.cc/media/icml-2024/Slides/33197.pdf) |
| 4 | SE-KD: Selective KD (2025) | [PDF](https://arxiv.org/pdf/2602.01395.pdf) |
| 5 | DDK: Domain KD (NeurIPS 2024) | [Page](https://neurips.cc/virtual/2024/poster/93067) |
| 6 | Compact LLMs Pruning+KD (NeurIPS 2024) | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4822991365c962105b1b95b1107d30e5-Abstract-Conference.html) |
| 7 | Residual Learning KD (2025) | [PDF](https://openreview.net/pdf/cf5bed8b71779ae42d0e681f1e2a7de3b3c8f6ad.pdf) |
| 8 | SWITCH KD (NAACL 2025) | [PDF](https://aclanthology.org/2025.findings-naacl.206.pdf) |
| 9 | KDIC: Image Compression KD (ICCV 2025) | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Knowledge_Distillation_for_Learned_Image_Compression_ICCV_2025_paper.pdf) |
| 10 | AlphaPruning (NeurIPS 2024) | [Page](https://nips.cc/virtual/2024/papers.html) |
| 11 | SlimGPT (NeurIPS 2024) | [Page](https://nips.cc/virtual/2024/papers.html) |
| 12 | Mamba in the Llama (NeurIPS 2024) | [Page](https://nips.cc/virtual/2024/papers.html) |
| 13 | Dataset Decomposition (NeurIPS 2024) | [Page](https://nips.cc/virtual/2024/papers.html) |
| 14 | GEAR KV Cache (2024) | [Workshop](https://neurips2024-enlsp.github.io/accepted_papers.html) |
| 15 | Combining Compressions (Apple) | [Page](https://machinelearning.apple.com/research/combining-compressions) |
| 16 | PRISM Dataset (NeurIPS 2024 Best) | [Blog](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/) |
| 17 | Superposition & Scaling (NeurIPS 2025 Best) | [Blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/) |

---

## 📄 我们的两篇论文

| 论文 | 主题 | 服务器 |
|------|------|--------|
| **KernelSearch** | LLM 生成 CUDA Kernel 的 Failure-Aware 搜索策略基准 (MCTS/BoN/Iterative/Greedy) | poplab |
| **HBD** | 基于睡眠重放的固定预算层次蒸馏 (RPE优先级 + SWR token加权) | poplab |

## 📁 详细文档

| 文件 | 说明 |
|------|------|
| [KernelSearch/survey.md](KernelSearch/survey.md) | KernelSearch 方向 15 篇论文详细调研 |
| [HBD/survey.md](HBD/survey.md) | HBD 方向 17 篇论文详细调研 |
| [figure_style_guide.md](figure_style_guide.md) | 30 篇论文画图风格参考 + 画图建议 |

## 📐 画图建议速览

**KernelSearch**: Pipeline Teaser（左→右4策略分支）/ 堆叠柱状图（failure mode）/ 5维雷达图 / log-scale budget曲线

**HBD**: Forward-Replay双阶段Teaser / 多任务蜘蛛图 / 分组柱状图Ablation / Violin plot展示RPE分布变化

<img src="assets/logo.svg" alt="DEL" width="90" align="left"><div align="center">
<h1>&nbsp;DEL: 基于上下文感知的动态退出层高效自投机解码</h1></div>

<p align="center">
<a href="https://arxiv.org/abs/2504.05598">
  <img src="https://img.shields.io/badge/Arxiv-2504.05598-orange.svg"></a> 
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a> 
<a href="https://github.com/hoenza/DEL/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>

## 海报

<p align="center">
  <img src="assets/DEL-CoLM.png" alt="DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding Poster" width="100%">
</p>

<p align="center">
  <a href="assets/DEL-CoLM.pdf">下载 PDF</a>
</p>

## 简介

**DEL** 是一种*即插即用的自投机解码算法*，在 LLM 推理过程中动态选择**退出层**和**投机长度**以最大化吞吐量。与依赖固定超参数或离线调优的方法不同，DEL 利用实时的 token 接受信号自适应地为每个输入配置 draft 模型。

DEL 基于 **LayerSkip** 构建，这是一种自投机框架，复用目标模型的早期层来生成 draft token。DEL 通过以下方式增强了该方法：

- **Token-per-Layer (TPL)**：一种平衡接受率与计算成本的指标，用于指导退出层选择。
- **Shadow Token 分析**：高效利用缓存的隐藏状态，同时估计所有退出层的接受概率。
- **动态 Draft 退出**：一种基于置信度的机制，在投机轮次中途决定何时停止生成 draft token。

这些组件使 DEL 能够针对每个 prompt 和上下文窗口进行实时的投机解码参数优化。

![DEL](./assets/DEL.png)

---

## 安装

```bash
# 创建 Conda 环境
conda create --name del python=3.10
conda activate del

# 安装依赖
pip install -r requirements.txt
```

### 当前环境说明

当前分支保留了一个小范围的 `_safe_lm_head()` workaround，用于规避在本机当前环境下观察到的低精度 `lm_head` CUDA 路径异常：

- 在当前机器上，最小复现实验表明 `lm_head` 的低精度路径存在稳定性问题；
- `fp16` 主要在单 token `[B, 1, C]` 投影时触发 `SIGFPE`；
- `bf16` 在当前环境下更不稳定，而 `fp32` 目前实验中正常；
- 因此当前先保留 `_safe_lm_head()`，以保证 DEL / FLy benchmark 可以稳定跑通。

当前 workaround 位于 [self_speculation/llama_model_utils.py](self_speculation/llama_model_utils.py)，其做法是在 `CUDA + fp16 + seq_len=1` 时，将单 token 投影改写为数学等价的两 token 投影后再切回第一个 token。

请注意：

- 这个 workaround 适合当前阶段的**功能验证和流程跑通**；
- 它可能会给单步 decode 带来额外计算，因此**不应直接视为最终性能实验实现**；
- 后续应优先做环境对照实验（PyTorch / CUDA / 驱动组合），若在更稳定的环境中问题消失，应重新评估是否移除该 workaround。

---

## 复现主要结果

运行完整 benchmark 套件：

```bash
bash run_benchmarks.sh
```

该脚本在 7 个数据集和多个 LayerSkip LLaMA 变体上评估 DEL 及若干基线方法（`self_speculative`、`FSM_speculative`、`DV_speculative`、`autoregressive`）。

- 日志保存在 `./logs/` 目录下
- 可修改 `run_benchmarks.sh` 中的 `num_samples`、`max_steps` 或目标模型

---

## 项目结构

```
.
├── benchmark.py                # 主 benchmark 入口
├── arguments.py                # benchmark 和生成的参数解析
├── generate.py                 # 非 benchmark 用途的生成脚本
├── eval.py                     # 评估和打分工具
├── correctness.py              # 投机正确性的单元级检查
├── sweep.py                    # 超参数搜索支持
├── utils.py                    # 杂项工具
├── run_benchmarks.sh           # 复现所有 benchmark 的 Shell 脚本
├── requirements.txt
├── COMPARISON.md               # 对比实验策略文档（中文）
├── README.md
└── self_speculation/           # 所有生成策略的实现
    ├── DEL.py                           # 动态退出层 (DEL) 核心逻辑
    ├── DEL_speculation_generator.py     # 基于 DEL 的生成（支持 FLy）
    ├── DV_speculation_generator.py      # Draft&Verify 投机解码基线
    ├── DELE_speculation_generator.py    # DEL 无动态 draft 退出变体
    ├── FSM_speculation_generator.py     # FSM 投机基线
    ├── FLy_speculation_generator.py     # 双模型 FLy 投机解码
    ├── autoregressive_generator.py      # 普通贪心解码
    ├── self_speculation_generator.py    # 标准自投机解码
    ├── generator_base.py                
    ├── llama_model_utils.py             
    └── speculative_streamer.py          
```

---

## 数据集与模型

### 模型

- `facebook/layerskip-llama3.2-1B`
- `facebook/layerskip-llama3-8B`
- `facebook/layerskip-llama2-[7B,13B,70B]`

### 数据集

- `gsm8k`、`aqua_rat`（数学推理）
- `cnn_dm_lm`、`cnn_dm_summarization`、`xsum_summarization`（长文本/摘要）
- `wmt14_de_en`（翻译）
- `human_eval`（代码生成）

---

## 核心特性

- **DEL：动态退出层**  
  LayerSkip 的即插即用模块，根据实时上下文动态选择每个生成轮次的退出层和投机长度。

- **上下文感知自适应**  
  跟踪跨层的 token 级接受率，使用基于置信度的阈值机制动态调整投机策略。

- **Token-per-Layer (TPL) 优化**  
  引入新的效率指标 TPL，以极低的开销指导退出层和投机长度的最优选择。

- **Shadow Token 分析**  
  利用缓存的隐藏状态和 shadow token 计算预期接受率，无需额外的模型前向传播。

- **流式生成与可扩展性**  
  在多种任务（推理、摘要、代码）上高效运行，从 1B 到 70B 的 LLM 均可扩展，最高可达 **2.84 倍加速**。

- **完全兼容 LayerSkip**  
  无需重新训练或修改架构即可无缝集成早退模型。

- **轻量且实用**  
  运行时和显存开销极小（约 1-2%），适合实际部署。

---

## FLy 集成：宽松自投机解码

本分支集成了 [FLy (Training-Free Loosely Speculative Decoding)](https://arxiv.org/abs/2511.22972) 到 DEL 中，将**自投机 draft 生成**与**宽松验证**结合，进一步提升接受率。

### 动机

在标准投机解码中，一个不精确匹配目标模型预测的 draft token 会导致其后所有 token 被丢弃。FLy 放宽了这一约束：如果一个被拒绝的 token 后面紧跟 `win_len-1` 个连续被接受的 token，则该拒绝被推翻——周围的上下文确认它在语义上是正确的。

自投机方法（如 DEL）使用较弱的 draft（更少的层），因此其精确匹配接受率天然低于双模型投机解码。FLy 的宽松匹配弥补了这一点，在不引入额外模型或训练的情况下每轮接受更多 token。

### 2x2 对比框架

| | 精确匹配验证 | 宽松匹配验证 (FLy) |
|---|---|---|
| **自投机（单模型）** | DEL 基线 | **DEL + FLy（本文方法）** |
| **双模型投机** | FLy exact 基线 | FLy loosely 基线 |

详细对比策略见 [COMPARISON.md](COMPARISON.md)。

### 用法

`DEL_speculative` 现在保留为原始 DEL baseline；`DEL_fly_speculative` 是本文的 DEL+FLy 方法。`--fly_win_len` 控制 DEL+FLy 的滑动窗口大小（默认 6）。

```bash
# DEL 基线
python benchmark.py --model facebook/layerskip-llama3.2-1B \
  --dataset gsm8k --generation_strategy DEL_speculative \
  --num_samples 50 --max_steps 512 --sample False \
  --exit_layer 3 --num_speculations 6

# DEL + FLy
python benchmark.py --model facebook/layerskip-llama3.2-1B \
  --dataset gsm8k --generation_strategy DEL_fly_speculative \
  --num_samples 50 --max_steps 512 --sample False \
  --exit_layer 3 --num_speculations 6 --fly_win_len 6
```

双模型 FLy 投机解码：

```bash
# 双模型精确匹配
python benchmark.py --model facebook/layerskip-llama3-8B \
  --dataset gsm8k --generation_strategy FLy_speculative \
  --draft_model facebook/layerskip-llama3.2-1B \
  --num_samples 50 --max_steps 512 --sample False \
  --num_speculations 3

# 双模型 FLy 宽松匹配
python benchmark.py --model facebook/layerskip-llama3-8B \
  --dataset gsm8k --generation_strategy FLy_speculative \
  --draft_model facebook/layerskip-llama3.2-1B \
  --num_samples 50 --max_steps 512 --sample False \
  --num_speculations 3 \
  --enable_fly True --fly_win_len 6
```

运行完整 DEL+FLy 扫参实验（win_len ∈ {4, 6, 8}）以及双模型对比实验：

```bash
bash run_benchmarks.sh
```

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `self_speculation/generator_base.py` | 在 `GenerationConfig` 中添加 `fly_win_len` 和 `draft_model`（`enable_fly` 仅保留给双模型 FLy 使用） |
| `self_speculation/DEL_speculation_generator.py` | 保留原始 DEL baseline 实现 |
| `self_speculation/DEL_fly_speculation_generator.py` | 新增 DEL+FLy 单模型宽松匹配策略 |
| `self_speculation/FLy_speculation_generator.py` | 新增双模型 FLy 投机解码策略（含精确/宽松两种模式） |
| `benchmark.py` | 添加 `DEL_fly_speculative`、`FLy_speculative` 策略分支及 draft 模型加载 |
| `run_benchmarks.sh` | 添加 DEL+FLy 扫参、双模型精确/宽松实验配置 |
| `COMPARISON.md` | 新增对比实验策略文档 |

---

## 引用

如果您在研究中使用了 DEL，请引用：

```bibtex
@inproceedings{entezari2025del,
  title={DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding},
  author={Entezari Zarch, Hossein and Gao, Lei and Jiang, Chaoyi and Annavaram, Murali},
  booktitle={Proceedings of the Conference on Language Modeling (COLM) 2025},
  year={2025}
}
```
---

## 致谢

- LayerSkip 模型由 [Meta AI](https://github.com/facebookresearch/LayerSkip) 提供。

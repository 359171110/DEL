# 对比实验策略文档

## 1. 研究定位

将 **FLy（宽松验证）** 集成到 **DEL（动态退出层自投机解码）** 中，提出 **宽松自投机解码（Loosely Self-Speculative Decoding）**。核心论点：自投机方法因 draft 质量较弱（仅用早期层），精确匹配接受率天然较低，FLy 的宽松匹配可显著弥补这一点。

---

## 2. 实验设置

**选定模型**（3 个，覆盖 1B / 7B / 13B 规模）：

| 模型 | 退出层 E | 投机步数 γ |
|------|---------|-----------|
| LLaMA-3.2-1B | 3 | 6 |
| LLaMA-2-7B | 7 | 6 |
| LLaMA-2-13B | 13 | 4 |

**全部数据集**（7 个）：

| 数据集 | 任务类型 |
|-------|---------|
| AQuA-RAT | 数学推理 |
| GSM8K | 数学推理 |
| CNN/DM LM | 语言建模 |
| CNN/DM Summarization | 摘要生成 |
| XSUM | 摘要生成 |
| HumanEval | 代码生成 |
| WMT14 De-En | 机器翻译 |

---

## 3. 可复用数据：DEL 论文基线（已填入）

以下 speedup 数据取自 DEL 论文（COLM 2025）Table 1 和 Table 3。

### 3.1 DEL Speedup 汇总（所有模型 × 所有数据集）

| 模型 | AQuA-RAT | CNN/DM LM | CNN/DM Sum | GSM8K | HumanEval | XSUM | WMT14 | Overall |
|------|----------|-----------|------------|-------|-----------|------|-------|---------|
| **LLaMA-3.2-1B** | 2.24× | 2.56× | 2.25× | 2.12× | 2.17× | 2.37× | 2.36× | **2.36×** |
| **LLaMA-2-7B** | 2.41× | 2.41× | 2.75× | 1.99× | 2.12× | 2.44× | 2.65× | **2.50×** |
| **LLaMA-2-13B** | 2.25× | 2.11× | 2.29× | 1.89× | 2.01× | 1.98× | 1.98× | **2.16×** |

### 3.2 各基线 Speedup 对比（DEL 论文 Table 1）

**LLaMA-3.2-1B (E=3, γ=6)**

| 方法 | AQuA-RAT | CNN/DM LM | CNN/DM Sum | XSUM | Overall Speed | Overall Speedup |
|------|----------|-----------|------------|------|---------------|-----------------|
| Vanilla | 1.00× | 1.00× | 1.00× | 1.00× | 64.34 tok/s | 1.00× |
| LS (3-6) | 2.01× | 2.11× | 1.93× | 1.89× | 127.87 tok/s | 1.99× |
| FS (3-6) | 2.16× | 2.48× | 2.13× | 2.18× | 144.21 tok/s | 2.24× |
| DV (3) | 1.72× | 1.93× | 1.83× | 1.43× | 111.14 tok/s | 1.73× |
| **DEL** | **2.24×** | **2.56×** | **2.25×** | **2.37×** | **151.73 tok/s** | **2.36×** |

**LLaMA-2-7B (E=7, γ=6)**

| 方法 | AQuA-RAT | CNN/DM LM | CNN/DM Sum | XSUM | Overall Speed | Overall Speedup |
|------|----------|-----------|------------|------|---------------|-----------------|
| Vanilla | 1.00× | 1.00× | 1.00× | 1.00× | 36.08 tok/s | 1.00× |
| LS (7-6) | 1.83× | 1.91× | 2.16× | 2.09× | 71.93 tok/s | 1.99× |
| FS (7-6) | 2.17× | 2.17× | 2.33× | 2.25× | 80.48 tok/s | 2.23× |
| DV (7) | 1.39× | 1.44× | 2.52× | 2.20× | 67.86 tok/s | 1.88× |
| **DEL** | **2.41×** | **2.41×** | **2.75×** | **2.44×** | **90.29 tok/s** | **2.50×** |

**LLaMA-2-13B (E=13, γ=4)**

| 方法 | AQuA-RAT | CNN/DM LM | CNN/DM Sum | XSUM | Overall Speed | Overall Speedup |
|------|----------|-----------|------------|------|---------------|-----------------|
| Vanilla | 1.00× | 1.00× | 1.00× | 1.00× | 27.08 tok/s | 1.00× |
| LS (7-4) | 1.99× | 1.89× | 1.72× | 1.87× | 50.68 tok/s | 1.87× |
| FS (7-4) | 2.24× | 2.06× | 1.77× | 1.89× | 53.99 tok/s | 1.99× |
| DV (7) | 1.48× | 1.45× | 1.36× | 1.41× | 38.68 tok/s | 1.43× |
| **DEL** | **2.25×** | **2.11×** | **2.29×** | **1.98×** | **58.39 tok/s** | **2.16×** |

> 注：GSM8K、HumanEval、WMT14 的基线 speedup 详见 DEL 论文附录 Table 3。

---

## 4. 参考数据：FLy 论文（ICLR 2026）

FLy 使用 **Llama-3.1-8B-Instruct → Llama-3.1-70B-Instruct** 双模型配置，与本项目的 LayerSkip 自投机不直接可比，但可作为 discussion 参考。

### FLy In-Domain 结果 (Temperature=0)

| Target | 方法 | GSM8K | HumanEval | MBPP | Mean |
|--------|------|-------|-----------|------|------|
| L31-70B | SpS (精确匹配) | 2.13× | 1.64× | 1.72× | 1.83× |
| L31-70B | FLy (宽松匹配) | 2.98× | 2.86× | 2.79× | 2.88× |
| L31-70B | **FLy 增益** | +0.85× | +1.22× | +1.07× | **+1.05×** |

**关键发现**：FLy 的宽松匹配在双模型设置下将 speedup 从 1.83× 提升到 2.88×（相对提升 57%）。

### FLy 超参数消融 (L31-70B, HumanEval, T=0)

| 参数 | 值 | Speedup | τ (平均接受长度) | 质量保持 |
|------|---|---------|----------------|---------|
| Win_len W | 0 (全接受) | 最高 | 最高 | 下降 |
| | 4 | 较高 | 较高 | 轻微下降 |
| | **6 (默认)** | **2.86×** | **12.61** | **≥99%** |
| | 8 | 较低 | 较低 | 保持 |

---

## 5. 参考数据：SWIFT 论文（ICLR 2025）

SWIFT 用跳层（skip layers）而非早退（early exit），模型为标准 Llama-2（非 LayerSkip），仅作参考引用。

| 模型 | CNN/DM | GSM8K | TinyStories | Overall Speedup |
|------|--------|-------|-------------|-----------------|
| LLaMA-2-13B | 1.37× | 1.31× | 1.53× | **1.41×** |
| LLaMA-2-70B | 1.43× | 1.39× | 1.62× | **1.48×** |

DEL 在 LayerSkip 模型上达到 2.16×~2.50×，显著高于 SWIFT 的 1.41×~1.48×。

---

## 6. 需新运行的实验

### 6.1 主表：DEL + FLy (win_len=6)

3 模型 × 7 数据集 = **21 组实验**。与 DEL 基线直接对比。

| 模型 | 数据集 | DEL Speedup (已知) | DEL+FLy Speedup | DEL+FLy accept_rate | 质量变化 |
|------|--------|-------------------|-----------------|---------------------|---------|
| LLaMA-3.2-1B | AQuA-RAT | 2.24× | | | |
| LLaMA-3.2-1B | GSM8K | 2.12× | | | |
| LLaMA-3.2-1B | CNN/DM LM | 2.56× | | | |
| LLaMA-3.2-1B | CNN/DM Sum | 2.25× | | | |
| LLaMA-3.2-1B | XSUM | 2.37× | | | |
| LLaMA-3.2-1B | HumanEval | 2.17× | | | |
| LLaMA-3.2-1B | WMT14 | 2.36× | | | |
| LLaMA-2-7B | AQuA-RAT | 2.41× | | | |
| LLaMA-2-7B | GSM8K | 1.99× | | | |
| LLaMA-2-7B | CNN/DM LM | 2.41× | | | |
| LLaMA-2-7B | CNN/DM Sum | 2.75× | | | |
| LLaMA-2-7B | XSUM | 2.44× | | | |
| LLaMA-2-7B | HumanEval | 2.12× | | | |
| LLaMA-2-7B | WMT14 | 2.65× | | | |
| LLaMA-2-13B | AQuA-RAT | 2.25× | | | |
| LLaMA-2-13B | GSM8K | 1.89× | | | |
| LLaMA-2-13B | CNN/DM LM | 2.11× | | | |
| LLaMA-2-13B | CNN/DM Sum | 2.29× | | | |
| LLaMA-2-13B | XSUM | 1.98× | | | |
| LLaMA-2-13B | HumanEval | 2.01× | | | |
| LLaMA-2-13B | WMT14 | 1.98× | | | |

### 6.2 消融实验：win_len 的影响

选 1 个模型（LLaMA-2-7B）× 2 个数据集 × 3 个 win_len = **6 组实验**。

| 数据集 | win_len=4 | win_len=6 | win_len=8 | 质量变化 |
|-------|-----------|-----------|-----------|---------|
| CNN/DM Sum | | | | |
| AQuA-RAT | | | | |

### 6.3 质量保持验证

与 6.1 同组实验，额外报告 ROUGE-L 和 BLEU，对比 DEL（无 FLy）的质量基线。

---

## 7. 实验总量汇总

| 类别 | 来源 | 实验组数 |
|------|------|---------|
| 所有基线 (Vanilla/LS/FS/DV/DEL) | DEL 论文 | 可复用 |
| **DEL + FLy 主表 (win_len=6)** | **新实验** | **21** |
| **win_len 消融** | **新实验** | **6** |
| **总计需新运行** | | **27** |

---

## 8. 论文中的数据引用说明

### 可直接引用

| 来源论文 | 引用内容 | 引用方式 |
|---------|---------|---------|
| DEL (COLM 2025) | Table 1 & 3 所有基线 speedup | 基线数据，注明 "reproduced from DEL" |
| FLy (ICLR 2026) | Table 3 双模型 speedup | Discussion 中对比宽松匹配增益 |
| SWIFT (ICLR 2025) | Table 2 自投机 speedup | Related work 参考 |

### 不可直接对比（仅定性讨论）

| 来源 | 原因 |
|------|------|
| FLy 论文数据 | 模型不同（Llama-3.1-Instruct vs LayerSkip）、框架不同、投机步数不同 |
| SWIFT 论文数据 | 方法不同（跳层 vs 早退）、模型不同（标准 Llama vs LayerSkip） |

---

## 9. 执行命令

```bash
# DEL + FLy 主表示例: LLaMA-2-7B on CNN/DM Sum, win_len=6
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 torchrun --master_port=29500 benchmark.py \
  --model facebook/layerskip-llama2-7B --dataset cnn_dm_summarization \
  --generation_strategy DEL_speculative \
  --num_samples 1000 --max_steps 512 \
  --exit_layer 7 --num_speculations 6 \
  --enable_fly True --fly_win_len 6 \
  --output_dir ./logs/ --sample False

# win_len 消融示例
... --enable_fly True --fly_win_len 4 ...
... --enable_fly True --fly_win_len 8 ...

# 一键运行全部实验（含 5 模型 × 7 数据集完整覆盖）
bash run_benchmarks.sh
```

---

## 10. 预期结论

1. **DEL + FLy > DEL**：宽松匹配提升自投机的接受率和吞吐量，预期 speedup 提升 0.1×~0.3×
2. **FLy 在自投机中增益更大**：FLy 论文中双模型场景相对提升 57%（1.83× → 2.88×），自投机 draft 质量更弱，FLy 有更大提升空间
3. **win_len=6 是合理默认值**：与 FLy 论文一致
4. **生成质量基本不变**：ROUGE/BLEU 预计 ≥99% 保持率
5. **零额外开销**：FLy 仅修改验证逻辑，不增加前向传播

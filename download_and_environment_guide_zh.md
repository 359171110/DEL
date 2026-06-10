# 跨项目下载与本机环境规范

这份文档只保留两个重点：

- 所有大文件统一放到 `/data1`，因为这个盘空间更大
- 记录这台机器已经验证通过的一套 Python、PyTorch、FlashAttention、CUDA 等版本，后续项目尽量直接复用

但要特别强调一件事：

- 下面提到的环境和安装命令，很多是“可复用样例”，不是要求每个项目机械地全部安装一遍
- 是否安装某个库，要以当前项目自己的 `README`、`requirements.txt`、`pyproject.toml`、运行报错和实际功能需求为准
- 项目没用到的东西就不要装，避免环境越来越重、依赖越来越乱

## 下载规则

下载规则不需要复杂，核心规则如下：

- 模型、数据集、索引、缓存、临时文件，统一放到 `/data1`
- 日志放在当前项目目录下，单独建一个日志文件夹统一管理

这样做的原因很简单：

- `/data1` 空间更大
- 避免把根目录 `/` 写满
- 避免仓库目录越来越大
- 后续项目可以复用同一套目录和缓存
- 项目日志跟代码放在一起，更方便查看、清理和随项目单独管理

推荐放法：

```text
/data1/xinpeigao/
├── conda_envs/
├── caches/
├── models/
├── datasets/
├── runs/
└── tmp/
```

实际使用时只要记住：

- 代码放仓库里
- 大文件放 `/data1`
- 能传路径的地方尽量传 `/data1/...`
- 日志放当前项目目录，例如 `<repo>/logs/`
- 对于 LLaMA 系列模型，不要默认把仓库里的所有权重格式全部下载一遍，要根据当前项目的加载方式选择最合适的一份
- 如果项目是基于 Hugging Face `transformers` 做推理，通常优先保留 `model.safetensors`
- `pytorch_model.bin` 一般用于兼容旧流程，`original/*.pth` 通常更适合做原始 checkpoint 转换或复现；项目没有明确需要时可以不下

## 安装原则

后续所有项目都按下面这个原则处理，不要照着样例无脑全装：

- 环境配置和依赖安装，优先以每个项目本身的说明、依赖文件、代码和实际报错为准
- 这份文档主要提供通用目录规范、缓存位置和本机已验证版本，只有在项目自身信息不完整或需要补充参考时再使用
- 先看项目自己的依赖声明，再决定装什么
- 通用基础环境优先复用这台机器已验证过的 Python 和 PyTorch 版本
- 只有项目明确依赖，或者实际运行时报缺失，再安装额外组件
- `faiss`、`flash_attn`、`cuda-toolkit` 这类库通常不是“所有项目默认必装”
- 如果只是为了跑一个普通的 Hugging Face / PyTorch 项目，很多时候只装 Python、PyTorch 和项目本身依赖就够了
- 当前项目如果需要日志目录，优先在仓库根目录下单独创建，例如 `logs/`

一个简单判断方法：

- 项目依赖文件里没有某个库，代码里也没有直接用到，通常就不要主动装
- 只有在下面几种情况下再补装：
  - 项目依赖里明确写了
  - 代码里明确 import 了
  - 安装或运行时明确报缺包
  - 你为了性能优化，主动选择启用某种可选后端

## 本机已验证环境版本

下面这套是这台机器上已经验证可用的版本组合，后续项目优先沿用这一套，不要随意换。

### 基础环境

- Python: `3.10.20`
- 通用环境命名建议：`/data1/xinpeigao/conda_envs/<project_name>-py310`
- 本机已验证示例环境：`/data1/xinpeigao/conda_envs/ngram-trie-py310`
- GPU 示例：`NVIDIA H20`

### 关键库版本

- PyTorch: `2.5.1+cu121`
- FlashAttention: `2.7.3`
- FAISS: `1.7.4`
- transformers: `4.44.2`
- numpy: `1.26.4`
- pandas: `2.3.3`

### CUDA 相关信息

- conda 安装的 CUDA toolkit 包：`12.1`
- 本机验证到的 `nvcc -V`：`12.9`
- 系统默认 `/usr/bin/nvcc`：`11.5`

注意：

- 这台机器真正能用来编译 `flash_attn` 的不是系统 CUDA 11.5
- 要用 conda 环境里的那套 CUDA
- 否则 `flash_attn` 很容易编译失败

## 推荐安装步骤

下面这些步骤是“常用样例”，不是每个项目都要完整执行。

### 1. 创建环境

每个项目单独建自己的环境，不要所有项目共用同一个 `ngram-trie-py310`。

```bash
conda create -p /data1/xinpeigao/conda_envs/<project_name>-py310 python=3.10 -y
conda activate /data1/xinpeigao/conda_envs/<project_name>-py310
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 PyTorch

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装 FAISS

只有项目明确依赖向量检索、索引构建、相似度搜索，或者依赖文件里明确包含 `faiss` 时才装。

```bash
conda install -p /data1/xinpeigao/conda_envs/<project_name>-py310 -c conda-forge faiss-cpu=1.7.4 -y
```

### 4. 安装用户态 CUDA

只有项目需要在当前环境里编译 CUDA 扩展，或者像 `flash_attn` 这种包明确依赖本环境内的 CUDA 工具链时才装。

```bash
conda install -p /data1/xinpeigao/conda_envs/<project_name>-py310 -c nvidia cuda-toolkit=12.1 -y
```

### 5. 设置 CUDA 路径

这一节只在你确实安装了当前环境自己的 `cuda-toolkit` 时才需要。

```bash
export CUDA_HOME=/data1/xinpeigao/conda_envs/<project_name>-py310
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

### 6. 安装 FlashAttention

只有项目明确依赖 `flash_attn`，或者你确认要启用它作为注意力后端时才装。

```bash
pip install packaging ninja
pip install flash_attn==2.7.3 --no-build-isolation -i https://pypi.org/simple
```

### 7. 安装其他依赖

```bash
pip install -r requirements.txt
```

如果项目没有 `requirements.txt`，就按项目实际情况选择下面之一：

```bash
pip install -e .
```

或者：

```bash
pip install -r requirements.txt
```

## 建议写入 ~/.bashrc 的环境变量

为了让下载、缓存、临时文件都统一走 `/data1`，建议把下面这些通用变量加入 `~/.bashrc`：

```bash
export WORK_ROOT=/data1/xinpeigao
export PIP_CACHE_DIR=$WORK_ROOT/caches/pip
export HF_HOME=$WORK_ROOT/caches/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$WORK_ROOT/caches/torch
export TMPDIR=$WORK_ROOT/tmp
```

日志目录不建议在这里全局写死。更合适的做法是：

- 每个项目在自己的仓库根目录下创建一个日志目录，例如 `logs/`
- 运行命令时把日志重定向到当前项目的 `logs/` 下
- 这样日志跟项目本身绑定，不会和其他项目混在一起

`CUDA_HOME`、`PATH`、`LD_LIBRARY_PATH` 不建议长期写死到某一个项目环境里。更合适的做法是：进入某个项目环境后，再单独导出该项目自己的 CUDA 路径。

例如：

```bash
export CUDA_HOME=/data1/xinpeigao/conda_envs/<project_name>-py310
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

改完后执行：

```bash
source ~/.bashrc
```

## 为什么后续项目建议直接复用这套版本

这套版本已经在本机验证过，主要避开了下面这些坑：

- Python 太新时，`faiss` 和 `flash_attn` 容易不兼容
- 系统自带的 `/usr/bin/nvcc` 是 CUDA `11.5`，编译 `flash_attn` 不够用
- 没有 `sudo`，不能指望改系统级 CUDA
- 如果缓存和下载不放到 `/data1`，根目录容易被写满

所以后续项目如果没有特别强的版本要求，优先沿用这套版本组合，但环境目录请按项目单独创建：

- Python `3.10`
- torch `2.5.1+cu121`
- flash_attn `2.7.3`（仅在项目需要时）
- conda 用户态 CUDA toolkit `12.1`（仅在项目需要编译 CUDA 扩展时）

## 快速验证

环境装完后，至少检查这些：

```bash
conda activate /data1/xinpeigao/conda_envs/<project_name>-py310
which python
python --version
python -m pip cache dir
which nvcc
nvcc -V
python -c "import torch, faiss, flash_attn, transformers, numpy, pandas; print('core import ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

预期结果：

- `which python` 指向 `/data1/xinpeigao/conda_envs/<project_name>-py310/bin/python`
- pip cache 指向 `/data1/xinpeigao/caches/pip`
- `which nvcc` 指向 `/data1/xinpeigao/conda_envs/<project_name>-py310/bin/nvcc`
- `nvcc -V` 显示 CUDA `12.9`
- 输出 `core import ok`
- `torch.cuda.is_available()` 为 `True`

## 常见坑

只记下面几个最关键的：

- 下载路径别放仓库里，统一放 `/data1`
- 日志不要和跨项目缓存混放，优先放当前项目目录下的独立日志文件夹
- LLaMA 系列模型不要把同一套权重的 `safetensors`、`bin`、`original/*.pth` 都无脑下载，先看项目实际加载哪个格式
- Python 不要随便升到 3.13 这类太新的版本
- 编译 `flash_attn` 时不要误用 `/usr/bin/nvcc`
- `flash_attn` 要在 `torch` 安装完成后再装
- 网络不稳定时，大文件下载失败先重试，不要先怀疑整套环境
- 不要把样例命令当成所有项目都必须执行的固定清单

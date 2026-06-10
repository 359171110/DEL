# DEL 1B 模型 SIGFPE 调试记录

## 背景

在使用 `facebook/layerskip-llama3.2-1B` 运行 DEL 小样本 smoke test 时，命令在 GPU 7 上触发了 `SIGFPE`，进程被 `torchrun` 直接终止，无法从普通 Python traceback 中看到明确的出错行。

本次调试的目标不是优化模型效果，而是定位并修复一个**运行时崩溃**问题，使 DEL 的 1B 小模型至少能够稳定完成基本 benchmark 流程。

## 现象

使用的核心命令类似如下：

```bash
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29611 benchmark.py \
  --model /data1/xinpeigao/models/facebook/layerskip-llama3.2-1B \
  --dataset gsm8k \
  --generation_strategy DEL_speculative \
  --num_samples 2 \
  --max_steps 64 \
  --exit_layer 3 \
  --num_speculations 6 \
  --sample False \
  --output_dir ./logs
```

初始现象有三个特点：

1. 环境本身是通的  
   Python、PyTorch、Transformers、datasets 等依赖都能正常导入，GPU 也能被 PyTorch 识别。

2. 崩溃发生在运行过程中  
   不是启动即报错，也不是模型文件缺失，而是已经进入推理后才在某个时刻收到 `SIGFPE`。

3. 普通 traceback 信息不足  
   即使加了 `CUDA_LAUNCH_BLOCKING=1` 和 `TORCH_SHOW_CPP_STACKTRACES=1`，终端里仍然主要只看到 `Signal 8 (SIGFPE)`，没有直接指出是哪一行代码。

## 为什么不能直接猜

这类错误如果只靠静态读代码，很容易误判。因为可能原因至少包括：

- Python 层除零，例如 tokens per second、time per token 的统计；
- DEL 里 acceptance rate / TPL 统计产生 NaN 或 Inf；
- draft/verify 的 token 张量为空，导致后续处理访问非法状态；
- CUDA 内核在某个张量形状下直接崩溃。

因此本次采用的是逐段插桩的方式，不先改业务逻辑，而是先确定崩溃实际落点。

## 调试过程

### 第一阶段：先排除是不是统计逻辑出错

一开始先在 benchmark 主循环、generator 入口、DEL 统计入口附近加埋点，主要验证：

- 输入样本是不是正常进入；
- prompt token 数是不是正常；
- DEL 的 acceptance stats、confidence、TPL 是否出现 NaN / Inf；
- 第一轮 `single_step_speculation()` 是否能完整跑完。

这一阶段的结论是：

- 样本输入正常；
- generator 已经开始工作；
- DEL 的 acceptance / confidence / TPL 都是有限值；
- 第一轮 speculation 可以完整返回。

这意味着：

- 不是空输入；
- 不是 benchmark 末尾统计除零；
- 不是 DEL 第一轮统计直接坏掉。

### 第二阶段：确认是不是第二轮 draft forward 崩溃

继续把埋点向内移动到 `DEL_speculation_generator.py`，重点观察：

- generation loop 进入与退出；
- `single_step_speculation()` 进入与退出；
- draft iter 进入；
- `forward_early_DEL()` 返回前后。

这一步发现：

- 第一轮 DEL draft 和 verify 都能正常完成；
- 崩溃发生在第二轮；
- 第二轮时 `current_exit_layer` 已经从初始值切到了更深的层；
- 第二轮进入了 `single_step_speculation()`，也进入了 draft iter；
- 但在某个 forward 边界后进程消失。

这时问题已经被缩小到：

- 第二轮 early-exit draft 路径；
- 且不是一开始就崩，而是走到更深 exit layer 后才崩。

### 第三阶段：拆开 `forward_early_DEL()` 的尾部

随后继续细化 `forward_early_DEL()` 的尾部路径，依次观察：

- layer 0/1/2/3 是否能执行完成；
- `exit_query_cache` 更新是否完成；
- `model.model.norm(hidden_states)` 是否完成；
- `model.lm_head(hidden_states)` 是否完成。

最终得到的关键证据是：

- 第二轮 `forward_early_DEL()` 的 decoder layer 循环都完成了；
- `exit_query_cache` 更新完成；
- final `norm` 完成；
- 日志停在 `before lm_head` 和 `after lm_head` 之间。

这说明真正的出错点不是：

- cache 更新；
- DEL 统计；
- norm；
- Python 层 return；

而是：

- **第二轮 DEL draft 路径里的 `lm_head(hidden_states)`**。

## 根因解释

### 直接根因

本项目加载模型时使用的是：

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    local_model_path,
    use_safetensors=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
```

也就是说，模型是以 `fp16` 方式在 GPU 上运行的。

DEL 的第二轮 draft 有一个特殊条件组合：

- 在 GPU 上；
- 使用 `float16`；
- 进入更深的 early-exit layer；
- draft 阶段此时只处理 **1 个 token**，即 `seq_len == 1`；
- 最终进入 `lm_head(hidden_states)` 做线性投影。

本次崩溃的高概率根因是：

- **某个 CUDA fp16 的单 token GEMM/线性投影路径，在这个张量形状下触发了底层内核异常，最终表现为 `SIGFPE`。**

这里要注意，`SIGFPE` 在这种场景下不一定真的是 Python 意义上的“浮点除零”，也可能是底层算子异常被操作系统以 signal 形式抛出。

### 为什么第一轮没事、第二轮出事

第一轮和第二轮的关键区别不是“是不是 DEL”，而是张量形状不同：

- 第一轮 prefill 更像是多 token 输入；
- 第二轮 draft 已经进入增量生成，常见情况是只投影最后 1 个 token；
- 这个 `seq_len == 1` 的条件恰好踩中了问题路径。

因此会出现：

- 第一轮正常；
- 第二轮更深 exit layer 时崩；
- 并且崩在 `lm_head`。

## 修复方案

### 修复思路

修复目标不是改 DEL 算法，而是规避这个明显不稳定的底层投影路径。

采取的方案是增加一个安全包装函数 `_safe_lm_head()`：

```python
def _safe_lm_head(model, hidden_states):
    if hidden_states.is_cuda and hidden_states.dtype == torch.float16 and hidden_states.shape[1] == 1:
        padded_hidden_states = torch.cat([hidden_states, hidden_states], dim=1)
        return model.lm_head(padded_hidden_states)[:, :1, :]
    return model.lm_head(hidden_states)
```

### 这个修复为什么成立

这个 workaround 的核心思想是：

- 原本 `hidden_states` 形状是 `[B, 1, C]`；
- 临时复制一份后变成 `[B, 2, C]`；
- `lm_head` 对两个 token 分别做同样的线性映射；
- 因为两个 token 输入完全一样，所以投影结果也完全一样；
- 最后取回第一个 token 的 logits，数值上与原始单 token 投影一致。

也就是说，这不是“改答案”，而是：

- **用一个数学上等价、但底层更稳定的形状来执行同一个线性投影。**

### 为什么改动范围控制在这里

这次没有去改：

- DEL 的 acceptance 逻辑；
- TPL 选择逻辑；
- tokenizer / dataset 逻辑；
- benchmark 主循环。

因为这些都不是根因。

真正保留下来的修复非常小，只针对：

- CUDA
- fp16
- 单 token
- `lm_head`

这能最大限度降低对原始行为的影响。

## 验证结果

修复后重新运行同一条 smoke test 命令，2 个样本均已完整跑通，终端输出了：

- prompt/reference/model response；
- acceptance rate；
- TPL；
- 最终 metrics；
- 最大显存占用。

并且没有再出现 `SIGFPE`。

这说明：

1. 修复命中了真实故障点；
2. benchmark 主流程可以继续执行；
3. 至少在当前 1B + DEL + gsm8k 小样本 smoke test 条件下，运行稳定性问题已被消除。

## 最小归因实验补充

在完成 smoke test 修复之后，又补做了一组更小的归因实验，用来回答三个问题：

1. 问题是不是 DEL 逻辑本身导致的；
2. 问题是不是 hidden state 的 layout / stride 异常导致的；
3. 问题是不是当前环境下低精度 `lm_head` 的后端实现不稳定。

### 实验一：抓取 DEL 路径真实张量

通过运行时抓取，得到 DEL 路径里真正送入 `lm_head` 的张量，其关键元信息如下：

- `shape = [1, 1, 2048]`
- `dtype = torch.float16`
- `device = cuda:0`
- `is_contiguous = true`
- `stride = [2048, 2048, 1]`
- `storage_offset = 0`

这说明该张量不是一个明显异常的 view，也不是最直观的 non-contiguous 脏张量。

### 实验二：synthetic / captured / dtype 对比

然后把多组张量放进独立子进程里，直接调用 `model.lm_head(...)` 做最小复现，避免 DEL 主流程本身干扰结论。

#### fp16 结果

- `synthetic-seq1`：崩溃
- `synthetic-seq2`：正常
- `synthetic-view-seq1`：崩溃
- `synthetic-clone-seq1`：崩溃
- `synthetic-contiguous-seq1`：崩溃
- `synthetic-double-then-slice`：正常
- `captured-original`：崩溃
- `captured-contiguous`：崩溃
- `captured-clone`：崩溃

这组结果说明：

- 不需要经过 DEL 逻辑，只要是 synthetic 的 `[B, 1, C]` fp16 张量也能复现；
- `.clone()` 和 `.contiguous()` 都不能修复；
- 但把输入变成两个 token 后，再切回一个 token 的方式可以稳定执行。

因此可以排除：

- “只有 DEL 路径才会触发”的解释；
- “只是 layout/stride 异常导致”的解释。

#### bf16 结果

- 所有 synthetic case 都崩溃；
- 所有 captured case 也都崩溃；
- 包括 `synthetic-seq2` 和 `synthetic-double-then-slice` 这样的非单 token case 也会崩。

这说明在当前环境里，`bf16` 的 `lm_head` CUDA 路径甚至比 `fp16` 更不稳定。

#### fp32 结果

- 所有 case 全部正常。

这说明同样的模型、同样的张量 shape、同样的代码路径，只要切到 `fp32`，问题完全消失。

### 归因结论升级

基于这组最小复现实验，可以把前面的“高概率判断”升级为更强的结论：

- **不是 DEL 算法逻辑错误；**
- **不是 hidden state layout / stride 问题；**
- **而是当前环境中，低精度 `lm_head` 的 CUDA 执行路径不稳定。**

更细一点地说：

- `fp16` 主要在单 token `[B, 1, C]` 投影时崩溃；
- `bf16` 在当前环境下更糟，连多 token case 也会崩；
- `fp32` 当前观察到是稳定的。

因此当前保留的 `_safe_lm_head()` 并不是“掩盖 DEL bug”，而是：

- 用一个**数学等价但会走另一条稳定后端路径**的办法，
- 避开当前环境里不稳定的低精度单 token `lm_head` kernel。

## 这次调试的关键结论

### 不是这些问题

- 不是 Python 层 tokens per second / time per token 除零；
- 不是 DEL acceptance rate / confidence / TPL 统计逻辑出错；
- 不是 cache 拼接直接出错；
- 不是 final norm 出错；
- 不是数据集或模型文件缺失。

### 真正的问题

- 是 **DEL 第二轮 draft 路径** 上，
- **当前环境中低精度 `lm_head` 的 CUDA 路径不稳定**；
- 其中 `fp16` 主要在 `seq_len == 1` 时暴露；
- `bf16` 在当前环境下更不稳定；
- `fp32` 目前实验中是正常的。

## 是否要改环境

这个问题既然已经被证明和当前环境强相关，就自然会引出一个问题：是不是应该直接换环境，而不是保留 workaround。

答案要分成短期和中期看。

### 短期：不建议为了当前实验立刻换环境

对于当前 DEL 项目和你现在的目标，短期内**不建议立刻全面换环境**，原因有三点：

1. 当前项目模型加载就是 `fp16`，而不是 `bf16`；  
   换环境并不保证一定马上消失，而且要重新验证所有依赖与 benchmark 流程。

2. 当前 workaround 已经被最小复现实验证明是有针对性的；  
   它不是在掩盖 DEL 逻辑错误，而是在规避当前环境下低精度 `lm_head` 后端的不稳定路径。

3. 你现在的主要目标是“先让实验稳定可跑”；  
   在这个阶段，保留一个小而明确的 workaround，通常比立刻重建整套环境更务实。

因此，**如果你的目标是尽快继续跑 DEL 实验，当前建议是：保留现有环境 + 保留 workaround。**

### 中期：值得做一组环境回归对比

如果你的目标变成“减少 workaround、确认根因是不是特定软件栈导致”，那就值得做环境对比。

优先考虑对比这些维度：

- PyTorch 版本
- CUDA runtime / wheel 版本
- 驱动版本
- 不同 GPU 机型

建议方式不是直接覆盖当前环境，而是**新建一个对照环境**，然后复用同一套最小归因脚本进行对比。

如果未来在新的环境里出现下面的结果：

- `fp16` 的 `[B, 1, C] -> lm_head` 不再崩；
- `bf16` 的 synthetic 和 captured case 也稳定；

那就说明当前问题确实主要由旧环境组合触发，此时可以考虑移除 workaround。

### 什么时候应当优先换环境

如果出现以下任一情况，就更应该优先换环境而不是继续依赖 workaround：

- 你后续计划大量使用 `bf16`；
- 你需要把 `lm_head` 的额外开销压到最低；
- 你希望把修复提交得更“干净”，尽量不保留针对环境缺陷的绕行代码；
- 你准备把这套实现迁移到别的机器/集群，想确认问题是否具有环境局部性。

### 当前建议

综合当前证据，当前最合理的建议是：

- **短期**：继续使用当前环境，保留 `_safe_lm_head()`，优先保证实验稳定；
- **中期**：额外建立一个对照环境，用已有最小归因脚本验证新环境是否修复了低精度 `lm_head` 路径；
- **结论上**：当前问题更像“环境中的低精度后端问题”，而不是“DEL 逻辑 bug”，但是否换环境取决于你当前更看重“立刻可跑”还是“彻底移除 workaround”。

## 以后怎么避免

### 1. 对自定义推理路径优先做 smoke test

像 early-exit、speculative decoding、draft/verify 这种路径，往往和普通 `model.generate()` 走的是不同张量形状。即使模型能正常加载，也不代表所有路径都稳定。

建议每次改动后先跑：

- `num_samples=1` 或 `2`
- 固定单卡
- 固定小 `max_steps`

确认不崩再放大实验规模。

### 2. 警惕低精度 `lm_head` 特殊路径

很多问题只在：

- 单 token；
- 增量 decoding；
- fp16 / bf16；
- 特定 GPU / CUDA / torch 组合

下才会暴露。

如果之后再做：

- 自定义 lm_head 调用；
- 手写 draft model；
- early-exit hidden state 直接接 lm_head；

要优先检查是否存在低精度 `lm_head` 的特殊路径，尤其是 `seq_len == 1` 场景。

### 3. 先用运行时证据定位，不要先改算法

这次如果一开始就去改 DEL 公式，很可能会改错地方。正确做法是：

1. 先确认是不是输入、统计、cache、norm、lm_head 哪一段；
2. 再做最小修复；
3. 最后用 post-fix 复跑验证。

### 4. 换底层版本后重新回归

因为这类问题和底层 CUDA kernel 关系很大，所以如果后面变更了：

- PyTorch 版本；
- CUDA 运行时；
- 驱动版本；
- 模型 dtype（如改成 bf16）；

建议重新跑一次同样的 smoke test，确认这个 workaround 仍然必要，或者问题是否已经在新版本中消失。

## 当前保留的修复

本次调试结束后，所有临时调试埋点和调试文件都已经清理，仅保留了真正的修复逻辑：

- `self_speculation/llama_model_utils.py` 中的 `_safe_lm_head()`
- 以及各条相关路径把 `model.lm_head(hidden_states)` 改为 `_safe_lm_head(model, hidden_states)`

因此当前代码库状态是：

- **没有调试噪音**
- **保留了最小必要修复**

## 一句话总结

这次问题的本质不是 DEL 算法错误，而是 **当前环境中的低精度 `lm_head` CUDA 路径不稳定**；其中 `fp16` 主要在单 token draft 投影时暴露，而 `bf16` 在当前环境下更糟。当前修复通过数学等价的两 token 投影绕开不稳定路径，从而消除了 `SIGFPE`。 

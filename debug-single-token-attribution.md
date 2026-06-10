# Debug Session: single-token-attribution

- **Status**: OPEN
- **Goal**: 设计最小归因实验，区分 `SIGFPE` 是由单 token `lm_head` 的底层 CUDA/fp16 kernel、张量 layout/stride、还是 DEL 自定义路径本身触发。

## Hypotheses

1. **H1: 底层 kernel 假设**  
   只要满足 `CUDA + fp16 + seq_len=1`，直接执行 `lm_head(hidden_states)` 就可能崩溃，与 DEL 逻辑无关。

2. **H2: layout/stride 假设**  
   不是单 token 本身有问题，而是 DEL 路径产生的 `hidden_states` 在 contiguous / stride / storage offset 上比较特殊，导致 `lm_head` 走到不稳定实现。

3. **H3: DEL 路径特异性假设**  
   只有从 DEL early-exit 路径拿到的中间状态才会触发问题；如果用标准模型 forward 人工构造一个同 shape 的张量，则不会复现。

4. **H4: dtype 假设**  
   问题主要绑定 `fp16`；改成 `bf16` 或 `fp32` 后，相同张量与相同 shape 不再崩溃。

5. **H5: 环境组合假设**  
   问题与特定 `torch + cuda + driver + GPU` 组合相关，在当前机器/环境上可复现，但不一定是普遍现象。

## Plan

1. 先做完全脱离 DEL 的最小 `lm_head` 复现。
2. 再对比 contiguous / non-contiguous / cloned / sliced 张量。
3. 再把 DEL 路径真实导出的 `hidden_states` 拿来复现。
4. 最后做 dtype 维度对比：fp16 / bf16 / fp32。
5. 用结果判断是 kernel、layout 还是 DEL 路径特异性问题。

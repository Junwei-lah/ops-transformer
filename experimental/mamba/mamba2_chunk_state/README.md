# mamba2_chunk_state 算子说明

### 功能和实现说明

mamba2_chunk_state 用于在 MambaV2 Prefill 阶段进行 chunk 内的离散时间状态更新，根据 chunk_cumsum 得到的累积量dacs/dacs_chunk和状态更新因子dtout进行状态递推，输出 chunk 内每一步的状态序列，并生成用于下一 chunk 的最终隐藏状态。本算子实现为 Vector+cube 融合算子，支持 FP16/FP32。

**计算流**

<img src="https://raw.gitcode.com/user-images/assets/7673863/88ab3b4c-4940-4b88-9aca-e6a44fd4fc04/image.png" height="300">

### Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| dtout   | BCLH   | FP32   |
| dacs   | BCLH   | FP32   |
| bt   | BCLGN   | FP16   |
| xt   | BCLHP   | FP16   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| states   | BCHNP   | FP32   |

**参数说明：**  

B: batch size  
C: number of chunks  
L: chunk size  
H: number of head  
G: ngroups   
N: state size  
P: head dim  
其中C*L为padding后的序列长度

**调用方式**

```
import npu_ops_transformer_ext

out = torch.ops.npu_ops_transformer_ext.mamba2_chunk_state(dtout, dacs, bt, xt)
```

**测试方法**

见当前目录 tests/

```
python test_chunk_state.py
```

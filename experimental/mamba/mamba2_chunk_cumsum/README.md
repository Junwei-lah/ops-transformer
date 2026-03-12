# mamba2_chunk_cumsum 算子说明

### 功能和实现说明

mamba2_chunk_cumsum 用于在 MambaV2 Prefill 阶段对 chunk 内部执行按时间步的累积求和操作，实现 SSM 中状态量在 chunk 维度上的递推更新。算子对输入序列在 S 维度按 chunk_size 拆分，并在每个 chunk 内按照因果顺序执行 cumulative sum，用于后续 chunk 状态更新与 selective scan 计算。本算子基于 Vector 实现累积求和计算，支持 FP16/FP32 输入输出。

**计算流**

<img src="https://raw.gitcode.com/user-images/assets/7673863/37cc914a-27a6-45a5-a244-2d3756581e37/image.png" height="500">  


### Kernel输入输出（I/O）

**输入**
| Tensor | shape | dtype |
|-----|-----|-----|
| at   | H   | FP32   |
| dt   | BCLH   | FP16   |
| dt_bias  | H   | FP16   |
| dt_mask   | BCLH   | FP16   |

**输出**
| Tensor | shape | dtype |
|-----|-----|-----|
| dtout   | BCLH   | FP32   |
| dacs   | BCLH   | FP32   |
| dacs_chunk  | BCH   | FP32   |

**参数说明：**  
B: batch size  
C: number of chunks  
L: chunk size  
H: number of head  
其中C*L为padding后的序列长度

**调用方式**
```
import npu_ops_transformer_ext

out = torch.ops.npu_ops_transformer_ext.mamba2_chunk_cumsum(at, dt, dt_bias, dt_mask)
```

**测试方法**

见当前目录 tests/
```
python test_chunk_cumsum.py
```
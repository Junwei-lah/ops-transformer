# How to use the benchmarking/correctness program benchmark.py

# Installation

To install the sparse kernel go in the `ops-transformer` project home folder and run:

```bash
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
(cd experimental/attention/blitz_sparse_attention/torch_interface && bash build.sh custom)
```

## Parameters

Using the CAPITAL_LETTERS variables at the top of the file you can ru all the things you need. Wherever there is a list, it means you can set mutliple values, and the file will run correctness/time evaluations for **all the possible combinations**.

* `B_VALS`: batch sizes to test. For block-sparsity only `B_VALS = [1]` works for now.
* `H_VALS`: number of heads
* `S_VALS`: sequence lengths
* `D_VALS`: head dimensions
* `N_REPEATS`: how many runs to do to estimate time
* `N_WARMUP`: how many warmup runs before running `N_REPEATS` calls
* `ATTENTION_MATRIX`: what kind of attention matrix to use. By chosing this value, you create the matrix you want and you also automatically set all the parameters as they are required accordingly by torch_npu.npu_prompt_flash_attention. The possible values are (only the first few ones are probably interesting for you):
    * `blocks_optimized_batched`: our optimized block-sparse kernel. There is a batch dimension, but please keep it 1 for now.
    * `blocks_optimized`: same, without batch dimension (you can use either this or the other)
    * `dense`: all tokens are attended by all tokens
    * `sparse_block_all_same`: like `sparse_block`, but all heads have the same mask: this works.
    * `lower_triangular`: lower triangular matrix, classic case. It's the default torch_npu behavior
    * `band`: torch_npu band mode: diagonal band. You can set the width of the diagonal with `BAND_PRE_TOKENS` and `BAND_POST_TOKENS`
    * `custom`: you can create your own custom mask in `make_custom_mask`. Will use the default `dense` pattern for the torch_npu version
* `SPARSITY_VALS`: how many blocks to activate: randomly chosen, but each Q-block row will have the same number of blocks
* `BLOCK_SIZE_Q`, `BLOCK_SIZE_KV`: For now, only the defaulst 128x512 are supported for the `blocks_optimized*` version
* `BLOCK_MASK_SEED`: for reproducibility of random sampling. Can set to any value to change random sampling
* `USE_FRAME`: Adds a frame on top of the block-sparse mask. If you want to change the frame shape, please modify the method `generate_sparse_blocks_by_row_with_frame`. If `S` (in `S_VALS`) length is too short the mask might be dense. For now, the frame is always:
    * First 8 KV-blocks are active for all rows
    * First 29 Q-rows are completely active
    * Last Q-row is compeletely active
    * Last KV-column is all active
* `PRINT_OUTPUTS`: For manual visualization of output tensors
* `PRINT_MASK`: To visually check if the maks is as expected
* `PRINT_BLOCK_EQUALITY`: if the correctness test fails, you can check which blocks in the outputs match with the reference and which not. The block granularity to visualize is given by `PRINT_HEIGHT` and `PRINT_WIDTH`
* `RUN_REFERENCE`: Besides measuring time, also test correctness. This only is possible for small enough masks, otherwise you will get OOM in the reference run. For this reason, you can disable this for very long contexts, and enable it for short enough sequences.
* `TORCH_REFERENCE`: If `True`, correctness will be tested against the torch attention implementation. If `False`, for block-sparse attention the reference is the torch_npu dense with the corresponding attention mask (like in `sparse_block` mode). Please only use `H=1`, as torch_npu doesn't support different masks for different heads.

## Examples

### Test correctness and speed with multiple values (and shorter sequence lengths)

#### Input

```python
B_VALS = [1]
H_VALS = [16, 14]
S_VALS = [4096, 10_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2
ATTENTION_MATRIX = "blocks_optimized_batched" 

SPARSITY_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = False   # For short inputs, better to disable the frame or the mask will be dense

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = True
TORCH_REFERENCE = True  # If False, will instead run the torch_npu reference
```

#### Output:

```
(base) root@5e47d24afe58:/workspace/ops-transformer/benchmark# python benchmark.py 
==========================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
==========================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------
 16   1   4096   4096  128      0.10             yes           17638.60            1328.46
 16   1   4096   4096  128      0.20             yes           17645.56            1166.37
 16   1   4096   4096  128      0.30             yes           17642.52            1162.32
 16   1   4096   4096  128      0.40             yes           17622.96            1060.28
 16   1   4096   4096  128      0.50             yes           17628.49            1025.70
 16   1   4096   4096  128      0.60             yes           17626.99             980.80
 16   1   4096   4096  128      0.70             yes           17646.98             936.72
 16   1   4096   4096  128      0.80             yes           17669.57             955.56
 16   1   4096   4096  128      0.90             yes           17663.78             910.45
 16   1  10000  10000  128      0.10             yes          110451.61            7973.21
 16   1  10000  10000  128      0.20             yes          110456.47            7199.73
 16   1  10000  10000  128      0.30             yes          110440.53            6303.86
 16   1  10000  10000  128      0.40             yes          110435.28            5396.22
 16   1  10000  10000  128      0.50             yes          110441.05            4525.04
 16   1  10000  10000  128      0.60             yes          110446.19            3694.06
 16   1  10000  10000  128      0.70             yes          110438.79            2830.48
 16   1  10000  10000  128      0.80             yes          110458.57            1968.18
 16   1  10000  10000  128      0.90             yes          110501.12            1105.10
 24   1   4096   4096  128      0.10             yes           26154.51            1991.36
 24   1   4096   4096  128      0.20             yes           26161.87            1757.80
 24   1   4096   4096  128      0.30             yes           26115.79            1740.01
 24   1   4096   4096  128      0.40             yes           26215.35            1492.14
 24   1   4096   4096  128      0.50             yes           26202.39            1243.03
 24   1   4096   4096  128      0.60             yes           26141.22            1091.14
 24   1   4096   4096  128      0.70             yes           26161.22            1028.43
 24   1   4096   4096  128      0.80             yes           26180.69            1025.53
 24   1   4096   4096  128      0.90             yes           26216.21             971.07
```

Reference goes OOM for 10_000 sequence lenght and 24 heads (not shown).

On the left you see the inputs, then the "yes" line checks correctness, after that you see the reference runtime and our runtime. After that you see the memory bandwidth usage, which is much better in our kernel.

Note that for short sequence lengths, the speedups are negligible because overheads are higher than actual computation.

# Benchmark performance (no correctness) for very long sequences

#### Input

```python
DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [B, num_heads, seq_len, head_dim]
B_VALS = [1]
H_VALS = [3]
S_VALS = [60_000, 80_000, 100_000, 118_806, 130_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 10
N_WARMUP = 2
ATTENTION_MATRIX = "blocks_optimized_batched"   # "dense", "sparse_block_all_same", "lower_triangular", "band", "custom", "blocks_optimized" "blocks_optimized_batched"

SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = True

BAND_PRE_TOKENS = 8
BAND_POST_TOKENS = 2

PRINT_OUTPUTS = False
PRINT_MASK = False
PRINT_BLOCK_EQUALITY = False

RUN_REFERENCE = False  # True <--> enables accuracy compariston
TORCH_REFERENCE = False  # If False, will instead run the torch_npu reference
```

#### Output:

```
==========================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'  ATTENTION_MATRIX='blocks_optimized_batched'
==========================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------
  3   1  60000  60000  128      0.00             yes           39540.04           42287.86
  3   1  60000  60000  128      0.10             N/A                N/A           38261.13
  3   1  60000  60000  128      0.20             N/A                N/A           34205.20
  3   1  60000  60000  128      0.30             N/A                N/A           30725.77
  3   1  60000  60000  128      0.40             N/A                N/A           26888.40
  3   1  60000  60000  128      0.50             N/A                N/A           23041.55
  3   1  60000  60000  128      0.60             N/A                N/A           19396.84
  3   1  60000  60000  128      0.70             N/A                N/A           15665.37
  3   1  60000  60000  128      0.80             N/A                N/A           12171.41
  3   1  60000  60000  128      0.90             N/A                N/A            8439.49
  3   1  80000  80000  128      0.00             yes           69145.94           75492.80
  3   1  80000  80000  128      0.10             N/A                N/A           68411.87
  3   1  80000  80000  128      0.20             N/A                N/A           61732.04
  3   1  80000  80000  128      0.30             N/A                N/A           54425.49
  3   1  80000  80000  128      0.40             N/A                N/A           47248.43
  3   1  80000  80000  128      0.50             N/A                N/A           39982.01
  3   1  80000  80000  128      0.60             N/A                N/A           33430.32
  3   1  80000  80000  128      0.70             N/A                N/A           26595.97
  3   1  80000  80000  128      0.80             N/A                N/A           19602.85
  3   1  80000  80000  128      0.90             N/A                N/A           13282.41
  3   1 100000 100000  128      0.00             yes          111174.52          117815.22
  3   1 100000 100000  128      0.10             N/A                N/A          106289.33
  3   1 100000 100000  128      0.20             N/A                N/A           95549.62
  3   1 100000 100000  128      0.30             N/A                N/A           83765.51
  3   1 100000 100000  128      0.40             N/A                N/A           72874.36
  3   1 100000 100000  128      0.50             N/A                N/A           61149.34
  3   1 100000 100000  128      0.60             N/A                N/A           49732.42
  3   1 100000 100000  128      0.70             N/A                N/A           39099.00
  3   1 100000 100000  128      0.80             N/A                N/A           27397.11
  3   1 100000 100000  128      0.90             N/A                N/A           17454.23
  3   1 118806 118806  128      0.00             yes          159593.05          171517.10
  3   1 118806 118806  128      0.10             N/A                N/A          153066.28
  3   1 118806 118806  128      0.20             N/A                N/A          136765.43
  3   1 118806 118806  128      0.30             N/A                N/A          120221.88
  3   1 118806 118806  128      0.40             N/A                N/A          104197.14
  3   1 118806 118806  128      0.50             N/A                N/A           87268.88
  3   1 118806 118806  128      0.60             N/A                N/A           71160.35
  3   1 118806 118806  128      0.70             N/A                N/A           54724.87
  3   1 118806 118806  128      0.80             N/A                N/A           38535.49
  3   1 118806 118806  128      0.90             N/A                N/A           22082.29
  3   1 130000 130000  128      0.00             yes          191301.43          209100.46
  3   1 130000 130000  128      0.10             N/A                N/A          184073.00
  3   1 130000 130000  128      0.20             N/A                N/A          164665.89
  3   1 130000 130000  128      0.30             N/A                N/A          145730.24
  3   1 130000 130000  128      0.40             N/A                N/A          125810.19
  3   1 130000 130000  128      0.50             N/A                N/A          106532.20
  3   1 130000 130000  128      0.60             N/A                N/A           87878.38
  3   1 130000 130000  128      0.70             N/A                N/A           67602.93
  3   1 130000 130000  128      0.80             N/A                N/A           48213.34
  3   1 130000 130000  128      0.90             N/A                N/A           28490.92
==========================================================================================
```

The baseline dense `"dense"` mode (standard `torch_npu.npu_fusion_attention`) appears at every sparsity=0.0 row.

## Test setup

```
Host CPU: aarch64
Device: Ascend 910B2
Device Driver: 25.3.rc1   
Docker image: `docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops`
OS: ubuntu: 22.04
CANN: 8.5.0-beta.1
Python: 3.11.10
torch: 2.8.0+cpu
torch_npu: 2.8.0
```

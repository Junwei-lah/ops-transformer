# flash_attention_score_antiquant_kernel `LocalTensor::Get<T>()` 优化说明

## 1. 问题背景

在流水分析中，`flash_attention_score_antiquant_kernel.h` 热环里的如下语句开销偏大：

```cpp
LocalTensor<T> inputTensorVec1 = this->bmm1ResBuf[runInfo2.taskIdMod2].template Get<T>();
```

该步骤不是单纯的“声明一个局部变量”，而是包含：

- `TBuf -> LocalTensor` 句柄获取
- `LocalTensor` 临时对象物化/拷贝
- 可能的地址/元信息装载与回写

因此，如果只把 `LocalTensor<T> inputTensorVec1;` 的“声明”挪到循环外，而保留循环内的 `Get<T>()`，通常收益会很有限。

更有效的方向是：

- 把生命周期固定的 `LocalTensor` 句柄在 `InitBuffer` / `InitLocalBuffer` 阶段一次性缓存下来
- 热环里只做数组索引和引用绑定
- 对 `ProcessVec1/ProcessVec2` 内部反复出现的 `softmax*Buf/commonTBuf.Get<T>()` 同样做缓存

---

## 2. 优化结论

建议优先做两类修改：

1. 在 kernel 主类中缓存 `bmm1ResBuf[2]` / `bmm2ResBuf[2]` 对应的 `LocalTensor`
2. 在 `FABlockVecAntiquant` 中缓存 `softmaxSumBuf[3]` / `softmaxMaxBuf[3]` / `softmaxExpBuf[3]` / `commonTBuf` 对应的 `LocalTensor`

这样可以把热点路径上的 `Get<T>()` 从每轮循环调用，降低为初始化阶段只调用一次。

---

## 3. 必改文件与位置

## 3.1 主 kernel：缓存 `bmm1/bmm2` 的 `LocalTensor`

文件：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h`

### 3.1.1 新增成员变量

当前位置附近：

- `bmm1ResBuf[2]`
- `bmm2ResBuf[2]`

建议新增：

```cpp
LocalTensor<T> bmm1ResTensor[2];
LocalTensor<T> bmm2ResTensor[2];
```

建议放置位置：

- 在 `TBuf<TPosition::VECIN> bmm1ResBuf[2];`
- 在 `TBuf<TPosition::VECIN> bmm2ResBuf[2];`

的后面，便于与 buffer 一一对应。

### 3.1.2 在 `InitBuffer()` 中完成句柄缓存

当前位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:457`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:460`

当前逻辑：

```cpp
this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);
```

建议在这些初始化后追加：

```cpp
this->bmm1ResTensor[0] = this->bmm1ResBuf[0].template Get<T>();
this->bmm1ResTensor[1] = this->bmm1ResBuf[1].template Get<T>();
this->bmm2ResTensor[0] = this->bmm2ResBuf[0].template Get<T>();
this->bmm2ResTensor[1] = this->bmm2ResBuf[1].template Get<T>();
```

### 3.1.3 在 `Process()` 热环中替换 4 处 `Get<T>()`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:612`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:620`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:632`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h:641`

当前写法：

```cpp
LocalTensor<T> outputTensorBmm1 = this->bmm1ResBuf[runInfo1.taskIdMod2].template Get<T>();
LocalTensor<T> inputTensorVec1 = this->bmm1ResBuf[runInfo2.taskIdMod2].template Get<T>();
LocalTensor<T> outputBufmm2 = this->bmm2ResBuf[runInfo2.taskIdMod2].template Get<T>();
LocalTensor<T> inputBufVec2 = this->bmm2ResBuf[runInfo3.taskIdMod2].template Get<T>();
```

建议替换为：

```cpp
auto &outputTensorBmm1 = this->bmm1ResTensor[runInfo1.taskIdMod2];
auto &inputTensorVec1 = this->bmm1ResTensor[runInfo2.taskIdMod2];
auto &outputBufmm2 = this->bmm2ResTensor[runInfo2.taskIdMod2];
auto &inputBufVec2 = this->bmm2ResTensor[runInfo3.taskIdMod2];
```

说明：

- 这里建议用 `auto &`，避免再次按值构造 `LocalTensor`
- 当前调用链的参数本身就是按 `LocalTensor<T>&` 传递，改成引用最顺滑

相关调用位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_cube.h:120`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_cube.h:202`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:618`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:968`

---

## 3.2 Vec Block：缓存 `softmax/common` 相关 `LocalTensor`

文件：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

### 3.2.1 新增缓存成员

当前已有成员：

- `TBuf<> softmaxMaxBuf[3];`
- `TBuf<> softmaxSumBuf[3];`
- `TBuf<> softmaxExpBuf[3];`
- `TBuf<> commonTBuf;`

建议新增成员：

```cpp
LocalTensor<float> softmaxMaxTensor[3];
LocalTensor<float> softmaxSumTensor[3];
LocalTensor<T> softmaxExpTensor[3];
LocalTensor<uint8_t> commonTmpTensor;
```

建议放在对应 `TBuf` 附近，便于维护。

注意：

- `softmaxExpBuf` 当前代码里有 `LocalTensor<float> expUb = ...Get<T>()` 这样的写法
- 落代码时建议统一改为 `auto &expUb = ...` 或 `LocalTensor<T>`，保持和 `Get<T>()` 返回类型一致
- 文档里统一按“缓存 `Get<T>()` 结果”的思路描述，不强制你保留当前的局部变量显式类型

### 3.2.2 在初始化阶段完成缓存

相关初始化位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:328`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:367`

当前 `SoftmaxInitBuffer()`：

```cpp
this->tPipe->InitBuffer(this->softmaxSumBuf[0], 256);
this->tPipe->InitBuffer(this->softmaxSumBuf[1], 256);
this->tPipe->InitBuffer(this->softmaxSumBuf[2], 256);
...
this->tPipe->InitBuffer(this->softmaxMaxBuf[0], 256);
this->tPipe->InitBuffer(this->softmaxMaxBuf[1], 256);
this->tPipe->InitBuffer(this->softmaxMaxBuf[2], 256);
this->tPipe->InitBuffer(this->softmaxExpBuf[0], 256);
this->tPipe->InitBuffer(this->softmaxExpBuf[1], 256);
this->tPipe->InitBuffer(this->softmaxExpBuf[2], 256);
```

建议在这些初始化后追加：

```cpp
this->softmaxSumTensor[0] = this->softmaxSumBuf[0].template Get<float>();
this->softmaxSumTensor[1] = this->softmaxSumBuf[1].template Get<float>();
this->softmaxSumTensor[2] = this->softmaxSumBuf[2].template Get<float>();

this->softmaxMaxTensor[0] = this->softmaxMaxBuf[0].template Get<float>();
this->softmaxMaxTensor[1] = this->softmaxMaxBuf[1].template Get<float>();
this->softmaxMaxTensor[2] = this->softmaxMaxBuf[2].template Get<float>();

this->softmaxExpTensor[0] = this->softmaxExpBuf[0].template Get<T>();
this->softmaxExpTensor[1] = this->softmaxExpBuf[1].template Get<T>();
this->softmaxExpTensor[2] = this->softmaxExpBuf[2].template Get<T>();
```

当前 `InitLocalBuffer()`：

```cpp
this->InitAntiquantBuffer();
this->SoftmaxInitBuffer();
this->tPipe->InitBuffer(commonTBuf, 512);
```

建议在 `commonTBuf` 初始化后追加：

```cpp
this->commonTmpTensor = this->commonTBuf.template Get<uint8_t>();
```

### 3.2.3 替换 `ProcessVec1()` 中的热点 `Get<T>()`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:655`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:656`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:657`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:659`

当前写法：

```cpp
LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
LocalTensor<uint8_t> apiTmpBuffer;
apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
```

建议替换为：

```cpp
auto &sumUb = this->softmaxSumTensor[runInfo.multiCoreIdxMod3];
auto &maxUb = this->softmaxMaxTensor[runInfo.multiCoreIdxMod3];
auto &expUb = this->softmaxExpTensor[runInfo.taskIdMod3];
auto &apiTmpBuffer = this->commonTmpTensor;
```

### 3.2.4 替换 `ComputeLogSumExpAndCopyToGm()` 中的 `Get<float>()`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:814`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:823`

当前写法：

```cpp
LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
```

建议替换为：

```cpp
auto &sumTensor = softmaxSumTensor[runInfo.multiCoreIdxMod3];
auto &maxTensor = softmaxMaxTensor[runInfo.multiCoreIdxMod3];
```

### 3.2.5 替换 `ProcessVec2()` 中的热点 `Get<T>()`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:985`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:999`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1004`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1014`

当前写法：

```cpp
LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
```

建议替换为：

```cpp
auto &expUb = this->softmaxExpTensor[runInfo.taskIdMod3];
auto &sumUb = this->softmaxSumTensor[runInfo.multiCoreIdxMod3];
```

### 3.2.6 替换按偏移切片的 `Get<float>()[offset]`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1043`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1148`

当前写法：

```cpp
LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
```

建议替换为：

```cpp
auto maxTensor = softmaxMaxTensor[runInfo.multiCoreIdxMod3][vec2MaxBufOffset];
```

说明：

- 这里是“基于已缓存 base tensor 再取切片”
- 这样仍然保留偏移语义，但避免重新执行 `Get<float>()`

### 3.2.7 替换 FD 路径中的 `Get<float>()`

需要替换的位置：

- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1659`
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h:1668`

当前写法：

```cpp
LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
```

建议替换为：

```cpp
auto &sumTensor = softmaxSumTensor[runInfo.multiCoreIdxMod3];
auto &maxTensor = softmaxMaxTensor[runInfo.multiCoreIdxMod3];
```

---

## 4. 推荐改动顺序

建议按收益优先级分三步实施：

1. 先改 `flash_attention_score_antiquant_kernel.h`
2. 再改 `flash_attention_score_antiquant_block_vec.h` 中 `ProcessVec1/ProcessVec2`
3. 最后补齐 `ComputeLogSumExpAndCopyToGm`、FD 路径、偏移切片路径

原因：

- 第 1 步直接命中 4-stage pipeline 外层热环
- 第 2 步命中 vec 计算主路径
- 第 3 步属于补充收尾，收益通常次于前两步

---

## 5. 不建议采用的方式

## 5.1 只把局部变量“声明”提到循环外

不推荐写成：

```cpp
LocalTensor<T> inputTensorVec1;
...
inputTensorVec1 = this->bmm1ResBuf[runInfo2.taskIdMod2].template Get<T>();
```

原因：

- `Get<T>()` 仍然保留在热环中
- `LocalTensor` 赋值动作仍可能触发额外指令
- 相比“缓存句柄”，收益通常明显更小

## 5.2 在热路径中继续按值构造 `LocalTensor`

不推荐写成：

```cpp
LocalTensor<T> tensor = cachedTensor[idx];
```

更推荐：

```cpp
auto &tensor = cachedTensor[idx];
```

原因：

- 引用绑定通常更容易让编译器避免额外对象拷贝
- 当前调用接口本身就是引用传参，更契合现有接口设计

---

## 6. 风险与注意事项

## 6.1 确认 `LocalTensor` 句柄不会被跨轮污染

当前从调用链看：

- `IterateBmm1`
- `IterateBmm2`
- `ProcessVec1`
- `ProcessVec2`

都主要把传入的 `LocalTensor` 作为 buffer/输入输出句柄使用，没有看到对这些入参本身做 `SetShapeInfo/SetSize/SetAddr` 的直接修改，因此适合缓存。

但落代码时仍建议重点复查：

- 是否存在把缓存句柄再次 reinterpret 或重绑地址的路径
- 是否存在把缓存句柄传给会修改 tensor 元信息的 API

## 6.2 `softmaxExpBuf` 的局部变量类型要统一

当前代码存在：

```cpp
LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
```

这一写法在阅读上容易造成误解。落地时建议统一为：

```cpp
auto &expUb = this->softmaxExpTensor[runInfo.taskIdMod3];
```

或直接统一成真实元素类型，避免“声明类型”和 `Get<T>()` 不一致。

---

## 7. 验证建议

建议按下面方式验证：

1. 对比修改前后的流水，确认以下位置的指令数是否下降
2. 重点看热环中是否减少了 `ld/st` 和临时对象搬运
3. 确认功能结果完全一致
4. 确认 AIV/AIC pipeline 同步事件未受影响

重点观察点：

- `flash_attention_score_antiquant_kernel.h` 外层 4 处 `Get<T>()`
- `ProcessVec1`
- `ProcessVec2`
- FD 路径下的 softmax sum/max copy

---

## 8. 最终建议

如果目标是优先拿到最稳、最直接的收益，建议先做下面这组最小闭环：

1. 在 `flash_attention_score_antiquant_kernel.h` 缓存 `bmm1ResTensor[2]` / `bmm2ResTensor[2]`
2. 在外层热环中把 4 处 `Get<T>()` 改为 `auto &`
3. 在 `flash_attention_score_antiquant_block_vec.h` 缓存 `softmaxSumTensor[3]` / `softmaxMaxTensor[3]` / `softmaxExpTensor[3]` / `commonTmpTensor`
4. 把 `ProcessVec1/ProcessVec2` 里的 `Get<T>()` 全部替换掉

这组改动实现成本不高，但能覆盖当前最明显的句柄获取热点。

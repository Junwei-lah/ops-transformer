# AttentionWorkScheduler

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明


- 算子功能：Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。

  **该算子不建议单独使用，建议与FFNToAttention和AttentionWorkerCombine算子配合使用，形成完整的工作流。**

    1. 接收FFNToAttention算子发送的数据。该数据以ScheduleContext结构体存储。该结构体包含CommonArea，ControlArea，AttentionArea，FfnArea域。本接口涉及CommonArea(用于存储配置信息，如session_num，micro_batch_num，micro_batch_size，selected_expert_num)，ControlArea(用于上层控制进程是否退出)，AttentionArea域(负责管理算子计算过程中所需的核心数据缓冲区与状态信息，其中token_info_buf存储了与输入相关的数据信息)。

    2. 读取ScheduleContext.AttentionArea域中token_info_buf存储的flag信息，查看通信数据是否准备就绪。

    3. 数据全部准备就绪后，后续可供AttentionWorkerCombine算子使用。

- 计算公式：

$$
\text{Initialize:} \quad \text{ready\_count} = 0, \quad \text{flag\_num} = \text{micro\_batch\_size} \times \text{selected\_expert\_num}
$$

$$
\text{Check if run\_flag is 0:}
\quad \text{if run\_flag} = 0, \quad \text{exit and log}
$$

$$
\text{Loop:} \quad \text{while run\_flag} \neq 0:
\quad \text{ready\_count} = \sum\_{i=1}^{\text{flag\_num}} \mathbf{1}\_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready\_count} = \text{flag\_num}, \quad \text{break}
$$

$$
\text{Reset flags:}
\quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag\_num}
$$

$$
\text{Set micro\_batch\_id:} \quad \text{micro\_batch\_id} = (\text{micro\_batch\_id} + 1) \% \text{micro\_batch\_num}
$$

备注：micro_batch_size、selected_expert_num、run_flag、micro_batch_id是入参ScheduleContext结构体的参数，该结构体信息在调用示例中进行展示说明。


## 参数说明

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 146px">
  <col style="width: 135px">
  <col style="width: 326px">
  <col style="width: 246px">
  <col style="width: 275px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scheduleContextRef</td>
      <td>输入/输出</td>
      <td>Attention侧接收的待处理数据，表示输入scheduleContext信息，详细结构见调用示例。</td>
      <td>不支持空Tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td rowspan="1">输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td rowspan="1">输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>参数scheduleContextRef是空指针</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数scheduleContextRef维度不为1。</td>
    </tr>
    <tr>
      <td>161002</td>
      <td>参数scheduleContextRef是空tensor。</td>
    </tr>
  </tbody>
  </table>

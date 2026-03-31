# Chunk_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Chunk_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持seqlen序列长度。
- 支持NK、NV head数，NV需要为NK倍数。
- 支持DK、DV 不超过128。
- 支持data_type为BF16。

### 环境配置

#### 前置要求

1、 确认torch_npu为最新版本
2、 参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- chunk_gated_delta_rule_golden.py          # cpu侧算子golden实现以及cpu golden与npu结果精度对比
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_chunk_gated_delta_rule_single.py     # 测试单用例运行主程序
- chunk_gated_delta_rule_operator_single.py # CPU侧算子逻辑实现获取golden与npu算子直调
- test_chunk_gated_delta_rule_paramset.py   # 单用例入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_chunk_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

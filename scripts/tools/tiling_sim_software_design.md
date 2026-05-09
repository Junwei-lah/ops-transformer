# FA 算子 Tiling 模拟脚本软件设计说明书

## 1. 简介

### 1.1 背景

FA 算子 tiling 模拟脚本 `tiling_sim` 是一组面向 Prompt Flash Attention 和 Incre Flash Attention host 侧 tiling 逻辑的 Python 分析工具，当前实现位于 `scripts/tools` 目录：

- `pfa_tiling_sim.py`：模拟 `attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp` 中 PFA v2 主路径的切块、BMM check 参数计划和 N-B-S 分核行为。
- `ifa_tiling_sim.py`：模拟 `attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp` 中 IFA v2 faRun 主路径的切块、softmax 空间、FlashDecode `splitS2` 和 N-B-S 分核行为。
- `pfa_compare_tiling_dump.py`、`ifa_compare_tiling_dump.py`：将 C++ 侧 dump 日志与 Python 模拟输出进行字段级对比。

该工具用于在不依赖完整 CANN tiling 运行环境的情况下，快速解释 FA 算子 host 侧关键 tiling 决策，辅助问题定位、参数调优、分核负载均衡分析和代码变更回归。

### 1.2 设计目标

- 可读：以 JSON 结构暴露归一化输入、切块结果、分核结果、负载均衡指标和限制说明，便于人工分析。
- 可对照：核心计算逻辑尽量保持与 C++ host tiling 代码同名、同粒度，降低跨语言对照成本。
- 可视化：支持生成每个 candidate cube core 的任务量 SVG 图，直观看出 idle core 和负载倾斜。
- 可校验：支持从 C++ dump 日志中抽取 summary / coreRange，与模拟结果做自动 diff。
- 可扩展：PFA、IFA 和 compare 脚本相互独立，公共概念保持一致，后续可继续补充其他 FA 变体。

### 1.3 非目标

- 不替代真实 CANN `MatmulApiTiling::GetTiling` 或 AscendC 内部 API。
- 不保证 kernel 实际耗时与 task block 权重完全一致。
- 不模拟 device 侧 kernel 计算过程，仅覆盖 host 侧 tiling 决策和可解释输出。

## 2. 第零层设计描述

第零层描述系统在仓库中的边界和外部依赖。

```mermaid
flowchart LR
    A["用户 / 调试脚本"] --> B["tiling_sim CLI"]
    C["JSON 输入 / stdin / CLI 参数"] --> B
    B --> D["PFA tiling 模拟"]
    B --> E["IFA tiling 模拟"]
    D --> F["JSON 结果"]
    E --> F
    F --> G["SVG 负载均衡图"]
    H["C++ tiling dump"] --> I["compare_tiling_dump"]
    F --> I
    I --> J["字段级 diff 报告"]
```

系统输入主要来自四段式 JSON：

- `shape`：B/N/S1/S2/D/DV、Q heads、KV heads 等 shape。
- `attrs`：dtype、layout、sparse mode、actual seq length、prefix 等算子属性。
- `platform`：AIC/AIV 核数、L1/L0C 等平台信息。
- `flags`：PFA/IFA/MLA/PA/mask/PSE/FlashDecode 等路径开关。

系统输出包括：

- 标准 JSON：用于自动化消费和字段对照。
- compact JSON：用于脚本串联或落盘。
- summary 文本：保留 IFA 旧的人类可读输出。
- SVG 图：用于展示 candidate cube core 上的 task block 分布。
- compare diff：用于验证 Python 模拟结果与 C++ dump 的一致性。

## 3. 第一层设计描述

第一层按工具能力拆分为四个子系统。

| 子系统 | 主要文件 | 职责 |
|---|---|---|
| CLI / 调度子系统 | `pfa_tiling_sim.py`、`ifa_tiling_sim.py` | 解析参数、读取 JSON/stdin、生成示例输入、选择输出格式、触发 SVG 输出 |
| PFA 模拟子系统 | `pfa_tiling_sim.py` | PFA v2 切块、DN 调整、BMM1/BMM2 check plan、N-B-S 分核和负载均衡 |
| IFA 模拟子系统 | `ifa_tiling_sim.py` | IFA v2 faRun 切块、softmax 信息、FlashDecode splitS2、N-B-S 分核和负载均衡 |
| 对比与报告子系统 | `*_compare_tiling_dump.py`、`render_load_balance_svg` | C++ dump 对比、JSON 报告、SVG 可视化 |

### 3.1 数据流

1. CLI 层读取输入，若用户使用 `--example`，直接输出示例 JSON 并退出。
2. 输入解析层将四段式 JSON 或 legacy CLI 参数归一化为 dataclass 配置对象。
3. 核心模拟层按源码路径计算切块参数、分核起止信息和负载指标。
4. 报告层输出 JSON；如果指定 `--plot`，额外生成 SVG，并在 JSON 中写入 `visualization.loadBalanceSvg`。
5. 对比脚本读取 C++ dump 和模拟 JSON，按字段映射输出 `OK`、`DIFF` 或 `MISSING`。

### 3.2 核心数据结构

- `PFAConfig`：PFA 输入配置，负责 dtype/layout/GQA 等归一化。
- `IFATilingInput`：IFA 输入配置，负责从四段式 JSON 抽取 shape、attrs、platform、flags。
- `BlockSizes`：IFA `Souter`、`Sinner`、tail、align 等切块结果。
- `SoftmaxInfo`：IFA softmax tmp shape、公式和 regbase UB 布局说明。
- `FlashDecodeInfo`：IFA FlashDecode `splitS2`、workspace 和分裂信息。
- `Diff`：compare 脚本中的单字段对比结果。

### 3.3 4+1 视图

4+1 视图用于从不同关注点描述 `tiling_sim` 的软件架构。其中 4 个基础视图分别覆盖逻辑结构、开发组织、运行过程和部署形态，`+1` 场景视图用于串联典型使用流程。

#### 3.3.1 逻辑视图

逻辑视图关注系统对外提供的能力，以及能力之间的依赖关系。

```mermaid
flowchart TB
    A["CLI / 调度层"] --> B["输入解析与归一化"]
    B --> C["PFA v2 模拟服务"]
    B --> D["IFA v2 模拟服务"]
    C --> E["分核与负载均衡分析"]
    D --> E
    C --> F["BMM / Softmax / FlashDecode 信息"]
    D --> F
    E --> G["JSON / SVG 报告"]
    H["C++ dump 对比服务"] --> I["Diff 报告"]
    G --> H
```

逻辑上，系统分为输入层、核心模拟层、分析层和报告层：

- 输入层负责将 JSON、stdin 或 CLI 参数转换为标准配置对象。
- 核心模拟层分别实现 PFA v2 和 IFA v2 的 host 侧 tiling 决策。
- 分析层统一计算 task blocks、candidate core、负载均衡指标和 core range。
- 报告层输出 JSON、SVG 或 dump diff。

#### 3.3.2 开发视图

开发视图关注代码组织和维护边界。

```text
scripts/tools/
├── pfa_tiling_sim.py              # PFA v2 模拟主脚本
├── ifa_tiling_sim.py              # IFA v2 模拟主脚本
├── pfa_compare_tiling_dump.py     # PFA C++ dump 对比
├── ifa_compare_tiling_dump.py     # IFA C++ dump 对比
├── pfa_tiling_sim_README.md       # PFA 使用说明
├── ifa_tiling_sim_README.md       # IFA 使用说明
├── pfa_tiling_args_explain.md     # PFA 入参说明
└── tiling_sim_software_design.md  # 软件设计说明书
```

开发边界如下：

- `pfa_tiling_sim.py` 与 `ifa_tiling_sim.py` 以源码可对照性为优先，核心函数不强行合并。
- compare 脚本只承担 dump 解析和字段对比，不承载 tiling 计算逻辑。
- README 和参数说明面向使用者，设计说明书面向维护者和评审者。
- 后续若抽取公共模块，建议只抽取数学工具、负载均衡统计和 SVG 渲染等稳定逻辑。

#### 3.3.3 进程视图

进程视图关注脚本一次执行过程中的控制流和数据流。`tiling_sim` 当前是单进程、同步执行模型，不引入后台服务或并发任务。

```mermaid
sequenceDiagram
    participant U as 用户
    participant CLI as CLI入口
    participant Parser as 输入解析
    participant Sim as 模拟器
    participant Report as 报告输出

    U->>CLI: 执行 pfa/ifa_tiling_sim.py
    CLI->>Parser: 读取 JSON/stdin/CLI 参数
    Parser->>Parser: 字段归一化与合法性校验
    Parser->>Sim: 构造配置对象
    Sim->>Sim: 计算切块、分核、负载指标
    Sim->>Report: 返回 result dict
    Report->>Report: 可选生成 SVG
    Report->>U: 输出 JSON / compact JSON / summary
```

该视图下的关键约束：

- 所有计算在本地 Python 进程内完成。
- 输入解析失败、shape 非法或分核数组越界时直接终止当前执行。
- `--plot` 是主结果生成后的附加动作，不影响 JSON 主体计算。

#### 3.3.4 物理视图

物理视图关注工具在实际环境中的部署和依赖。

```mermaid
flowchart LR
    A["开发机 / CI 环境"] --> B["Python 3 标准库"]
    A --> C["ops-transformer 仓库"]
    C --> D["scripts/tools/tiling_sim 脚本"]
    D --> E["本地 JSON 输入"]
    D --> F["本地 JSON/SVG 输出"]
    D --> G["C++ dump 日志"]
```

物理部署特征：

- 脚本随仓库源码交付，无单独安装步骤。
- 运行依赖为 Python 3 标准库，包括 `argparse`、`json`、`dataclasses`、`math`、`re` 等。
- 不依赖 CANN 运行环境，不访问网络，不启动常驻服务。
- 输出文件由用户指定路径落盘，适合在开发机、流水线或问题定位脚本中直接调用。

#### 3.3.5 场景视图

场景视图用典型用例验证上述 4 个视图是否能够闭环。

**场景一：分析一个 PFA case 的切块与分核**

1. 用户通过 `--example` 生成 PFA JSON 模板。
2. 用户修改 shape、attrs、platform 和 flags。
3. `pfa_tiling_sim.py` 读取 JSON 并构造 `PFAConfig`。
4. 脚本计算 `Souter`、`Sinner`、BMM check plan 和 N-B-S 分核。
5. 用户查看 `splitCore.loadBalance`，必要时通过 `--plot` 生成 SVG。

**场景二：分析一个 IFA FlashDecode case**

1. 用户输入 Q/KV heads、actual KV length、head dim 和平台核数。
2. `ifa_tiling_sim.py` 推导 GQA group、`Sinner`、softmax tmp shape。
3. 脚本根据 `force_flash_decode` 或自动判定逻辑计算 `splitS2`。
4. 输出 `flashDecodeSplitS2` 和 `splitCore`，用于判断分裂是否合理。

**场景三：校验 Python 模拟与 C++ dump 是否一致**

1. 用户开启 C++ 侧 tiling dump 并获得日志。
2. 用户用同一 case 运行 `pfa_tiling_sim.py` 或 `ifa_tiling_sim.py` 生成 JSON。
3. compare 脚本解析 dump 中的 summary 和 coreRange。
4. compare 脚本按字段映射生成 diff 报告。
5. 若出现 `DIFF` 或 `MISSING`，开发者回到对应模拟函数或字段映射中修正。

## 4. 第二层设计描述

### 4.1 UI / 调度模块

#### 职责

- 提供命令行入口 `main()`。
- 支持 JSON 文件输入、stdin 输入、示例输入生成和紧凑输出。
- 支持 IFA legacy 单参数输入，便于快速构造小 case。
- 负责在核心模拟完成后调用 SVG 渲染函数。

#### 关键接口

PFA：

- `example_input()`：生成 PFA 示例 JSON。
- `load_input(path)`：从文件或 stdin 读取 JSON。
- `main()`：处理 `--input`、`--example`、`--compact`、`--plot`。

IFA：

- `build_arg_parser()`：定义 IFA CLI 参数。
- `merge_args(args)`：合并 JSON 输入和 CLI 覆盖参数。
- `print_summary(result)`：输出 legacy summary 文本。
- `main()`：处理 JSON / legacy CLI / summary / compact / plot。

#### 异常处理

- 输入为空时直接提示用户使用 `--example`。
- IFA 中 `q_heads`、`kv_heads` 非正或不可整除时抛出 `ValueError`。
- 解析 actual lens 时校验长度必须为 1 或 `batch_size`。

### 4.2 核心功能模块一：PFA v2 Tiling 模拟

#### 职责

模拟 PFA v2 host 侧关键路径：

- 归一化输入 shape、dtype、layout 和 G/S1 merge 行为。
- 计算 `Souter`、`CubeSouter`、`Sinner`、`SoftmaxSouter`、`splitS2`。
- 按源码规则执行 DN 相关二次调整。
- 构造 BMM1 / BMM2 check plan。
- 估算 `SPLIT_NBS_CUBE` 分核结果。

#### 关键函数

- `PFAConfig.from_dict()` / `normalize()`：配置构造与归一化。
- `adjust_cv_tiling(cfg)`：选择基础切块参数并返回命中路径说明。
- `apply_dn_adjustment(cfg, tiling)`：模拟 DN 对 `Sinner` 等参数的调整。
- `bmm1_check_plan()` / `bmm2_check_plan()`：输出 BMM shape、orgShape、fixSplit 计划。
- `compute_split_core(cfg, tiling)`：计算 N-B-S 分核、core range、task blocks。
- `simulate(cfg)`：PFA 模拟总入口。

#### 设计要点

- `actual_seq_lengths`、`actual_seq_lengths_kv` 缺省时按 batch 填充默认长度。
- PFA merge / IFA / IFA MLA 路径在 `normalize_gs1_merge=true` 时模拟源码中的 G/S1 归一化：`head_num_size = head_num_size / g_size`，`seq_size = seq_size * g_size`。
- BMM check 输出表示 Python 侧“尝试设置”的参数计划，不表示真实 CANN tiling 成功或失败。
- 分核权重以有效 inner block 数作为 task block 估计，用于解释 host 侧分核策略。

### 4.3 核心功能模块二：IFA v2 Tiling 模拟

#### 职责

模拟 IFA v2 faRun 主路径：

- 根据 `q_seq`、GQA、PSE、head dim 选择基础 `Souter` / `Sinner`。
- 计算 `sInnerLoopTimes`、tail、align 和 tiling key 使用的 `Sinner`。
- 计算 softmax tmp shape、AscendC API 公式和 arch35 regbase UB 布局。
- 根据 sparse mode、mask、prefix 计算每个 batch/head/outer row 的有效 inner block。
- 计算 FlashDecode 是否启用及 `splitS2` workspace。
- 生成 `multiCoreParamsRegbase`、core range 和 per-row assignment。

#### 关键函数

- `IFATilingInput.from_dict()`：配置解析。
- `IFATilingV2Simulator.set_fa_run_base_size()`：选择基础切块。
- `calc_inner_size()`：计算 block size、loop、tail 和 align。
- `softmax_info()`：输出 softmax 临时空间信息。
- `calc_block_nums_one_head()`：计算单 head task block 数。
- `compute_split_nb_seq_farun()`：按 N-B-S 维度切分到 cube core。
- `is_flash_decode_farun()` / `split_s2_info()`：FlashDecode 判定和 workspace 估算。
- `run()`：IFA 模拟总入口。

#### 设计要点

- `group = q_heads / kv_heads`，缺省 `is_gqa` 由 `group > 1` 推导。
- `kv_seq` 为 0 时从 `actual_kv_lens` 推导最大值。
- `force_flash_decode` 支持 `true`、`false`、`null` 三态；`null` 时由源码风格逻辑自动判定。
- softmax tmp size 只输出公式字符串，因为真实 `GetSoftMaxFlashV2MinTmpSize` 在 AscendC 内部。

### 4.4 核心功能模块三：Dump 对比与一致性校验

#### 职责

对比 C++ dump 和 Python 模拟输出，帮助确认模拟逻辑是否贴近 host 侧实现。

#### 输入输出

- 输入一：C++ dump 日志，包含 `[PFA_TILING_DUMP][summary]` / `[PFA_TILING_DUMP][coreRange]` 或 IFA 对应 tag。
- 输入二：`tiling_sim` 输出的 JSON 文件。
- 输出：字段级 diff，包含字段路径、C++ 值、Python 值、状态和差异说明。

#### 关键函数

- `parse_cpp_dump(path)`：提取 summary 和 coreRange。
- `parse_kv_line(line)`：解析 `key=value` 和 `key=[a,b]`。
- `compare_summary()`：比较 summary 字段映射。
- `compare_core_ranges()`：比较每个 core 的范围信息。
- `compare_task_arrays()`：比较 `coreTaskBlocks` 和 `candidateCoreTaskBlocks`。
- `build_report()`：聚合对比结果。

#### 设计要点

- 浮点字段使用 `math.isclose`，允许通过 `float_tol` 调整容忍度。
- 字段映射显式写在脚本中，新增 dump 字段时需要同步补充映射。
- `candidateCoreTaskBlocks` 会把 C++ 使用 core 后面的 idle candidate core 补 0，以便与 SVG 和负载均衡分析一致。

### 4.5 报告 / 输出模块

#### JSON 输出

核心字段包括：

- `normalizedInput`：归一化后的输入。
- `tiling`：切块结果、路径说明和关键 tiling 参数。
- `softmax`：IFA softmax 空间信息。
- `batchLoopInfo`：IFA 每个 batch 的 loop 和 mask 修正信息。
- `splitCore`：分核结果、任务权重、core range、负载均衡。
- `flashDecodeSplitS2`：IFA FlashDecode 拆分和 workspace 信息。
- `notes` / `limitations`：模拟边界说明。

#### SVG 输出

`render_load_balance_svg(result, path)` 根据 `candidateCoreTaskBlocks` 生成柱状图：

- 横轴是 cube core id。
- 一个 cube core 对应两个 AIV core。
- 柱高表示 task blocks。
- idle candidate core 以 0 任务灰色柱显示。
- 红色虚线表示 mean line，蓝色虚线表示 target line。
- 柱颜色按相对 mean 的比例区分负载程度。

#### 负载均衡指标

`build_load_balance()` 生成如下指标：

- `usedCubeCores`
- `candidateCubeCores`
- `idleCandidateCubeCores`
- `totalTaskBlocks`
- `meanTaskBlocks`
- `targetTaskBlocks`
- `maxTaskBlocks`
- `minTaskBlocks`
- `coefficientOfVariation`
- `maxOverMean`
- `minOverMean`
- `rating`
- `interpretation`

## 5. 非功能需求

### 5.1 正确性

- 核心公式、分支命名和变量语义应优先与 C++ host tiling 代码保持一致。
- 每次修改核心模拟逻辑后，应使用对应 compare 脚本和典型 case 校验 summary、coreRange、task block 数组。
- 对真实 CANN API 无法复现的部分，输出必须明确标注为计划、公式或估算值。

### 5.2 可维护性

- PFA 与 IFA 的模拟逻辑保持独立，避免为了复用而打乱与源码的对照关系。
- 新增路径时优先补充 dataclass 字段、`example_input()` 和 README 字段说明。
- dump 字段变更时同步更新 compare 脚本的字段映射。
- README 中的命令名称应与实际文件名保持一致。

### 5.3 可用性

- 默认输出缩进 JSON，便于直接阅读。
- `--compact` 输出适合脚本管道和 CI 落盘。
- `--example` 应始终生成可直接运行的最小可用 case。
- 错误信息应包含字段名和期望长度，减少排查成本。

### 5.4 性能

- 单个 case 模拟应为轻量级 CPU 计算，不依赖外部服务。
- 分核遍历复杂度主要与 batch、head、outer loop 和 candidate core 数相关，应避免引入大规模中间矩阵。
- SVG 渲染使用字符串拼接生成静态文件，避免引入额外绘图库依赖。

### 5.5 兼容性

- 使用 Python 3 标准库实现，避免新增第三方依赖。
- JSON 输入保持四段式结构，同时 IFA 保留 legacy CLI 参数输入。
- 输出字段新增时尽量向后兼容，不随意重命名既有字段。

### 5.6 安全性

- 输入文件只按 JSON 解析，不执行用户输入。
- SVG 输出路径由用户显式指定，脚本仅创建目标目录并写入静态 SVG。
- compare 脚本读取 dump 时使用文本解析，不执行日志内容。

## 6. 开发记录

### 6.1 当前实现状态

- 已实现 PFA v2 主路径模拟：基础切块、DN 调整、BMM check plan、N-B-S 分核、负载均衡 JSON 和 SVG。
- 已实现 IFA v2 faRun 主路径模拟：基础切块、softmax 信息、mask/prefix 有效 block 计算、FlashDecode `splitS2`、N-B-S 分核、负载均衡 JSON 和 SVG。
- 已实现 PFA / IFA dump compare 脚本，用于 C++ dump 与 Python 结果的字段级对比。
- 已提供 PFA / IFA README、PFA 参数说明、示例 JSON 和示例 SVG。

### 6.2 已知限制

- PFA BMM check plan 不调用 `MatmulApiTiling::GetTiling`，不能代表真实平台 tiling 成败。
- IFA softmax tmp size 输出源码 API 公式，不计算 AscendC 内部真实返回值。
- task blocks 是 host 侧分核估算权重，不等价于实际 kernel 耗时。
- 当前主要覆盖 PFA v2 和 IFA v2 faRun 主路径，非主路径需要按源码继续补齐。

### 6.3 后续演进建议

- 统一 README 中 IFA 脚本名称，当前实现文件为 `ifa_tiling_sim.py`。
- 抽取 PFA / IFA 共用的 `ceil_div`、`align_up`、load balance 和 SVG 渲染逻辑，形成轻量公共模块。
- 为典型 case 增加自动化回归：示例输入、mask case、GQA case、prefix case、FlashDecode case、DN case。
- 在 compare 报告中增加 summary 统计，例如 OK/DIFF/MISSING 数量和失败字段列表。
- 增加 `--schema` 或字段说明输出，便于调用方自动生成配置。

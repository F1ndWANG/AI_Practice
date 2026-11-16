# RSIAgent 机制复现

本地任务包含文本清洗和整数聚合，由 Curriculum 提供不同输入，Actor 选择 Bash/Python 工具序列，Verifier 用任务规范生成的预期输出独立验收。成功与失败配方写入 SQLite，按任务和工具序列统计，优先复用成功次数大于失败次数的配方。每次尝试使用独立临时目录并限制工具预算。测试以只读方式打开经验库。

## 本地对照

```bash
mkdir -p runs/rsi
python3 -m rsi_agent.runtime train --memory runs/rsi/memory.sqlite > runs/rsi/train.json
python3 -m rsi_agent.runtime test --no-memory > runs/rsi/baseline.json
python3 -m rsi_agent.runtime test --memory runs/rsi/memory.sqlite > runs/rsi/reuse.json
```

本地 BRS 策略优先尝试不同的短配方，DRS 策略优先尝试组合配方。每次失败后记录验证结果并尝试剩余配方。报告提供任务成功率、工具调用数、命中记录、失败与后续尝试。训练与测试各有两个任务实例，适合验证机制与运行预算。

`--model MODEL_NAME` 可接入 OpenAI 兼容端点，模型根据目标、已有尝试和记忆选择工具。通过环境变量配置 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。模型输出只映射到预置工具。端点错误和无效配方会回退至本地策略。

## OSWorld 和 ALE

完整的 BRS 并行波次、DRS 针对性练习、环境适配和官方评分使用 RSIAgent 上游运行器。此处固定简历所列 [RSIAgent fork](https://github.com/ZiwenWang-code/RSIAgent) 的提交 `a9e56263f6deaa493496ad6b155fe24bf131bc12`。

```bash
mkdir -p runs
python3 -m rsi_agent.benchmark runs/upstream-rsi setup
```

按[固定版本的指南](https://github.com/ZiwenWang-code/RSIAgent/blob/a9e56263f6deaa493496ad6b155fe24bf131bc12/README.md#installation)配置 Linux 主机、Docker/KVM、VM 资产、Python 3.12 环境和模型 API。激活对应环境后，从本仓库根目录调用入口并转发参数。

```bash
python -m rsi_agent.benchmark runs/upstream-rsi osworld -- --help
python -m rsi_agent.benchmark runs/upstream-rsi ale -- --help
```

当前验证覆盖本地机制对照，OSWorld 与 ALE 的官方结果需在对应 VM 中单独生成。

## BRS 与 DRS 阶段消融

```bash
python3 -m rsi_agent.study runs/rsi-study --seed 42 --per-family 8 --budget 6
```

BRS 将训练任务并行交给 Actor。整个波次使用同一初始记忆快照，波次结束后按输入顺序合并经验。DRS 使用独立的压力探针，记录失败并从该任务族选择练习。此处的探针、练习与最终测试使用各自的随机序列。无记忆、BRS、BRS+DRS 三组使用同一批留出任务和同一测试预算，后两组只读加载对应阶段的数据库。

| 本地设置（种子 42） | 成功任务 | 测试工具调用 | 额外探索工具调用 |
| --- | ---: | ---: | ---: |
| 无记忆 | 16/16 | 56 | 0 |
| BRS | 16/16 | 32 | 10 |
| BRS+DRS | 16/16 | 24 | 16 |

`summary.json` 保存成功率、工具调用和失败类型。各阶段 JSONL 保留任务 ID、执行步骤、耗时、记忆命中和探针触发关系。这里的结果来自两个本地任务族，官方 OSWorld/ALE 结果由其独立运行器产生。

## 自定义任务套件

```bash
python -m rsi_agent.study runs/my-agent-suite --suite examples/agent_suite.jsonl
```

[示例 JSONL](../examples/agent_suite.jsonl) 每行包含唯一 `id`、`split`、`family`、输入文本 `source` 和要逐字匹配的 `expected`。四个阶段分别为 `broad`、`probe`、`practice`、`test`。当前工具注册表支持文本清洗 `words` 和整数求和 `sum`。同一任务族的同一输入不可跨阶段重复，每个探针任务族至少提供一个练习。验证器持有预期文本，执行工具只能读取输入文本。

输出中的 `brs-experience.sqlite` 保留广度探索后快照，`experience.sqlite` 保留深度练习后的状态。报告同时记录自定义套件哈希及留出任务分布。

## 使用模型进行三组对照

配置 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 后，通过 `--model` 为无记忆、BRS 和 BRS+DRS 使用同一个 OpenAI 兼容模型。

```bash
python -m rsi_agent.study runs/rsi-model --model MODEL_NAME --workers 1
python -m ai_practice report runs/rsi-model
```

模型收到任务输入、工具说明、执行历史和当前记忆，不接收验证预期。模型响应错误、未知工具或超预算提议将触发本地配方回退。日志区分模型决策、回退、停止和请求错误，并记录请求耗时。Token 总量仅汇总端点有返回用量的请求，报告同时展示覆盖数。并发数可以按端点限流调整。该入口使用本地任务套件，OSWorld/ALE 仍使用上游运行器。

## 扩展任务集

```bash
python -m rsi_agent.study runs/rsi-extended --extended --per-family 8
```

在文本清洗和整数求和之外，增加 `json` 与 `csv`。JSON 任务要求生成递归键排序的紧凑 JSON。CSV 任务要求按类别聚合整数金额，并生成排序后的制表符分隔文本。BRS 使用输入键已排序、类别字段不带引号的样本，探针、练习和留出样本加入无序键或带逗号的引号字段。自定义 JSONL 也支持这两个任务族。

`summary.json` 中的 `families` 提供各任务族在无记忆、BRS、BRS+DRS 三组中的成功率、调用开销和失败类型。默认套件继续使用基础任务，便于比较既有实验。

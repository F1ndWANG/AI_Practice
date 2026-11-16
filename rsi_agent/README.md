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

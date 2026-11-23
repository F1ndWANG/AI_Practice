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

## 跨种子成本对照

```bash
python -m rsi_agent.matrix runs/rsi-matrix --seeds 42 43 44 --extended
python -m ai_practice report runs/rsi-matrix
```

每个种子独立生成任务和记忆。三个设置使用同一批留出任务，保留原始轨迹和数据库。汇总给出成功率、同种子相对无记忆基线的差值，以及测试调用节省。净节省从测试调用节省中扣除该设置实际执行的探索调用。报告中的标准差描述种子间变化。

## 重放轨迹与审计记忆

```bash
python -m rsi_agent.study runs/rsi-audit --extended
python -m rsi_agent.replay runs/rsi-audit
```

运行目录包含带预期输出的 `tasks.jsonl` 快照，须和轨迹一起保存在评测端。重放仅执行注册表中的工具，在独立临时目录按原轨迹重做每次尝试，核验预算、执行结果、失败分类和汇总。再根据探索轨迹重新统计配方成功与失败次数，对照两份只读 SQLite 快照。重放沿用已选配方，模型请求、延迟和 Token 记录沿用原始轨迹。此前未生成任务快照的运行需要重跑 study。

## 调用预算对照

```bash
python -m rsi_agent.study runs/rsi-cost --extended
python -m rsi_agent.budgets runs/rsi-cost runs/rsi-budgets --budgets 1 2 3 4 6
python -m ai_practice report runs/rsi-budgets
```

重复执行同一批留出任务，三个设置使用相同的单任务预算。BRS 和 BRS+DRS 从原实验只读加载记忆，每档预算都重新执行本地工具。输出保留逐样本轨迹、成功率、测试调用和扣除原探索开销后的净节省。

当有记忆设置取得正成功率、成功率达到无记忆基线且有测试调用节省时，报告按「探索调用数 ÷ 每任务平均节省调用数」向上取整估算回本任务数。结果假设后续任务具有近似的调用节省，适合对照本地任务设计。

## 错误记忆压力测试

```bash
python -m rsi_agent.memory_stress runs/rsi-cost runs/rsi-stress --budget 2 --false-wins 64
python -m ai_practice report runs/rsi-stress
```

在内存副本中给每类任务的 `copy` 配方注入错误的成功记录。使用同一批留出任务比较原快照、注入记录、以及用非测试样本重验后的结果。重验失败的配方失去记忆优先级，仍可由正常策略选择。校验按任务族使用一个非测试锚点，优先选探针，其次练习和广度样本。报告单列重验开销，原 SQLite 快照以只读方式打开，并在结束后校验哈希。这个实验针对指定的误记忆，复核效果取决于锚点覆盖。

## 模型端点故障恢复

模型决策的 HTTP 请求遇到超时、连接中断或临时服务状态时，最多额外尝试一次。短的整数 `Retry-After` 会被遵循，长的或日期格式的等待要求会直接触发现有本地回退。鉴权失败、无效 JSON 和无效配方也直接走回退。工具调用预算仍只统计工具执行。

轨迹逐次记录 HTTP 尝试和等待开销。报告区分模型决策数、实际 HTTP 尝试数、重试数、失败尝试和最终回退。先前生成的轨迹仍可用原有字段重放。

## 持久化 Agent Runtime

已有的 study 入口继续用于 BRS/DRS 机制实验。持久化执行入口直接调用同一工具注册表，适用于固定配方、逐步恢复与独立验收。

```bash
python -m rsi_agent.durable runs/durable submit examples/runtime_task.json
python -m rsi_agent.durable runs/durable resume TASK_ID
python -m rsi_agent.durable runs/durable events TASK_ID
python -m rsi_agent.runtime_eval runs/runtime-eval
python -m rsi_agent.runtime_metrics runs/runtime-eval/records.jsonl
```

submit 返回任务 ID。相同幂等键绑定同一份输入、配方、验证规则和资源策略，重复提交返回同一个任务。不同内容复用键会失败。任务状态为 pending → running → verifying → succeeded/rejected，执行错误进入 failed，中断可进入 interrupted，failed/interrupted 可恢复。进程锁阻止同任务并发执行，SQLite WAL 保存事件、步骤索引及文件哈希清单。工具开始前扣除调用预算，成功后先发布文件检查点，再在同一数据库事务提交索引与事件。若进程在文件发布后、事务提交前退出，会从上一已提交检查点重做该步骤。

检查点覆盖任务工作目录，恢复时检查内容哈希。当前配方均为工作目录内可重放的文件变换。外部写入、支付或发消息等工具需自行提供服务端幂等协议与结果核对，才能注册为可重试操作。任务幂等键覆盖本地执行记录。失败调用仍计入预算，任务恢复不清零预算。

### 执行约束

计划中的 limits 支持 max_calls、timeout_seconds、cpu_seconds、memory_bytes、file_bytes、workspace_bytes、concurrency、isolation。默认见 durable.py。注册表控制允许的命令，清空继承环境，每步使用暂存目录，超时终止进程组。CPU、单文件、文件描述符限制在 POSIX 子进程生效，地址空间限制在 Linux 生效。总工作目录字节数在每步完成后校验，同一 runtime 目录使用固定的跨进程并发上限。

默认 isolation 为 process，提供独立工作目录与受控注册命令。它面向可信本地工具。Linux 可显式设置 isolation 为 bubblewrap，要求主机已安装且允许运行 bwrap，否则任务提交失败。此模式隔离网络及其他命名空间，宿主文件系统只读，工作目录可写，/tmp 独立。宿主只读文件仍可见，部署外部代码时应进一步使用容器仅挂载允许读取的目录。Docker 模式另外提供容器级内存、CPU 和进程数配额，见后续说明。

### 可治理记忆

GovernedMemory 独立存储在 memory.sqlite，旧版经验库可继续用于原有 study。记忆按 subject、内容哈希及 provenance 去重，同一证据重复写入不累加可信度、不延长寿命。每条记录携带 confidence、expires、来源及创建时间。检索过滤过期和低置信条目，冲突候选中选择置信度最高者，同分矛盾返回 conflict 且 selected 为空，保留所有候选来源供复核。写入方提供的分数保留为 declared；独立标注导入后，以 Beta(1,1) 先验和实际成败观测计算后验均值。持久化任务验收完成后自动记录配方结果和检查点来源。

### 独立验证

rules 支持 file、text、data、tests、page、paper。文件检查读取真实文件及 SHA-256，text 对照调用方持有的预期输出，data 检查 JSON 行数、必需非空字段及唯一键。tests 通过独立子进程运行调用方提供的 unittest suite，至少实际执行一个用例且全部通过才接受，套件内容在提交时锁定。测试输出、返回码与套件摘要写入验证证据。

page 检查保存的 HTML 页面状态中是否存在指定元素 ID，适合接入浏览器工具输出的状态快照。动态网页可使用 browser 规则，在独立 Chromium 上下文中执行 JavaScript 后读取 DOM。paper 核对原文文件中的逐字引文并记录原文哈希，语义推论仍需额外审核。验证器规则由调用方提供，Actor 工具不接收 text 的预期输出。验证证据不等同于独立人工标注的真值。

### 系统级评测

runtime_eval 实际运行八个本地任务，包含正确配方、错误配方、故障恢复和宽松文件验证器对照。records.jsonl 使用独立预期产物哈希标记 success，单独记录 verifier_passed。故意宽松的对照用于暴露误接受，报告不用于代表开放任务的总体能力。

summary.json 输出任务成功率、平均工具调用数、Token 均值及覆盖率、P95 延迟、恢复成功率及分母、验证器总体误判率、误接受率、误拒绝率、单位成功任务成本及成本覆盖率。P95 使用最近秩，恢复任务以 recoveries > 0 计数。延迟现按首次提交至终态事件计算，包含恢复停机等待；模型规划入口另外将首次规划时间纳入端到端耗时。records.jsonl 的 active_seconds 单独保留可记录的活跃耗时。本地固定配方未调用模型，因此 Token 为 0。外部模型记录可用 runtime_metrics 导入，缺失用量填 null，报告覆盖率。只有所有任务成本齐备且至少一个成功时才给出单位成功成本，分子包含失败任务成本。--tool-cost 可指定每次工具尝试的计价，属于调用方提供的价格假设。


### 记忆参与规划与模型用量

```bash
python -m rsi_agent.durable runs/planned plan examples/runtime_planned_task.json
python -m rsi_agent.durable runs/planned memory family:words
```

规划入口只复用未过期、至少具备五条独立标注、后验置信度达标且无同分冲突的成功配方。无可用记忆时使用对应任务族的确定性配方，或通过 `plan --model MODEL --token-price PRICE_PER_TOKEN` 调用现有 OpenAI 兼容规划器。OPENAI_BASE_URL 和 OPENAI_API_KEY 由父进程读取，工具子进程继续清空继承环境。Actor 不接收 rules 中的验收预期。配方、来源、HTTP 尝试及用量在执行前持久化，相同键重复请求复用已记录决策。

如果模型请求期间进程中断，下一次 plan 使用本地配方并保留未知 Token 和成本，避免自动重复发起结果不明的模型请求。发生 HTTP 重试时，返回响应的用量不能覆盖此前请求，任务总量标记为未知。token-price 是调用方提供的每 Token 统一价格，该成本仅计模型 Token，工具执行成本需要另行计入。自动写入的 declared 分数为 0.9，仅表示规则验收记录；规划复用须通过五条独立标注门槛，之后使用观测估计的后验均值。

### 操作者权限与实时页面验证

可在新目录预先初始化权限，然后提交计划。权限配置与任务计划分开保存，后续任务只能使用已允许的注册工具和 HTTP origin。

```bash
python -m rsi_agent.durable runs/restricted init --allow-tool copy \
  --allow-origin https://example.org
```

规则示例为 `{"kind":"page_live","url":"https://example.org/task","element_id":"task","attributes":{"data-status":"done"}}`。验证器重新请求实际 HTTP 页面，核对元素 ID 与属性，记录观察时间、响应哈希和字节数。它禁用代理与跳转，并限制响应体大小和等待时间。page_live 检查服务端返回的 DOM，browser 规则用于 JavaScript 渲染后的页面。page 规则继续用于已有文件快照。终态任务重复执行返回当时的验证记录，需要重新观察页面时应使用新的任务键。

新任务锁定注册工具的命令，恢复时发现工具命令变化或权限撤销会停止执行。旧版任务缺少命令快照时沿用注册工具，迁移后可重新提交以锁定实现。默认初始化允许现有注册工具、禁止实时 HTTP 页面访问。Bubblewrap 仍保留宿主只读文件可见的边界，部署时需要根据允许读取的数据进一步限制挂载。

### 从真实执行记录导出系统指标

将独立标注写入 labels.json，格式为任务 ID 到布尔成功标签的映射，例如 `{"TASK_ID": true}`。执行：

```bash
python -m rsi_agent.runtime_export runs/planned labels.json runs/planned-report
```

输出 records.jsonl 与 summary.json。模型 Token 和声明价格从持久化用量表读取，缺失值保留为 null。没有调用模型的固定配方任务 Token 为 0。执行在验证前失败时，verifier_passed 为 null，该任务仍计入成功率与执行成本，验证器误判统计仅覆盖实际给出判定的任务，并报告 verifier_coverage。

CI 包含 Linux Bubblewrap 实机用例和本地 HTTP 服务状态变更用例。受限本机禁止监听端口时实时 HTTP 用例跳过，Linux CI 负责执行该项。系统本地样例故意包含错误配方和弱验证器，用于核对分母与误判统计。


### 外部写操作与不确定结果恢复

新增 http_write 网关工具，使用服务端支持的幂等协议。每个任务步骤派生固定操作键，先持久化请求哈希与 intent，再查询服务端操作回执。不存在时才 PUT，同一个操作键只能绑定同一份请求。回执须包含匹配的 idempotency_key、request_sha256、status=succeeded。连接丢失或进程中断后，恢复先查服务端，再确认本地回执，最后提交文件检查点。

参考服务对计数器更新与回执记录执行同一个 SQLite 事务，提供可运行的非幂等业务操作示例。启动服务与提交任务分别在两个终端执行：

```bash
mkdir -p runs
python -m rsi_agent.effect_service runs/effect-service.sqlite --port 8085
```

```bash
python -m rsi_agent.durable runs/external init --allow-tool http_write \
  --allow-origin http://127.0.0.1:8085
python -m rsi_agent.durable runs/external submit examples/runtime_external_task.json
```

参考服务绑定本机地址，不包含生产认证功能。连接其他业务系统时，需要该系统真正实现上述事务、幂等与查询协议。普通网页点击或任意 HTTP API 的副作用无法凭客户端检查点获得同等保障。网关只访问操作者允许的 origin，拒绝跳转，限制请求、响应大小，并通过可终止的子进程执行 HTTP 请求。data 验证器可用 expected_fields 检查回执中的业务结果。

### 动态网页验证

安装浏览器依赖：

```bash
python -m pip install '.[browser]'
python -m playwright install chromium
```

规则示例为 `{"kind":"browser","url":"https://example.org/task","selector":"#status","attributes":{"data-state":"ready"},"text":"Ready"}`。独立 Chromium 上下文等待目标 DOM 状态，保存最终 URL、观察时间、DOM 哈希与匹配文本。HTTP 请求仅允许已批准 origin 的 GET/HEAD，禁用 Service Worker、WebSocket 和下载。父进程为整个验证子进程设置超时并终止其进程组。该适配器观察页面，不自动执行带副作用的表单操作。页面资源与浏览器本身的安全性仍取决于浏览器部署环境。

### 有限挂载与容器配额

准备容器镜像：

```bash
docker build -t ai-practice-runtime:local -f examples/runtime.Dockerfile examples
```

在 limits 中设置 isolation=docker，可另外指定 memory_bytes、cpu_quota、max_processes。提交时固定本地镜像 ID，恢复沿用该 ID，执行不自动拉取镜像。容器关闭网络与新增权限，根文件系统只读，只挂载工具代码和当前工作目录，独立测试套件按需只读挂载。宿主其他文件和 Runtime 数据库不挂入容器。临时目录使用限额 tmpfs，Docker cgroup 执行内存、CPU 及进程数限制。

容器名称在启动前写入事件日志，超时会强制删除容器，恢复任务时会清理该任务此前登记的容器。容器保护适用于工具与测试子进程，模型调用、受控 HTTP 网关和浏览器验证在控制进程侧运行。生产部署还需对控制进程配置服务账户与主机级资源限制。process 和 bubblewrap 模式继续保留，便于已有任务迁移。

### 用独立标注更新与评估记忆置信度

```bash
python -m rsi_agent.runtime_export runs/planned calibration-labels.json runs/calibration-report \
  --calibrate-memory --tool-cost 0.001
python -m rsi_agent.durable runs/planned memory family:words --require-calibrated
python -m rsi_agent.memory_calibration runs/planned/memory.sqlite heldout-memory-labels.jsonl
```

校准标签沿用 task_id 到布尔真值的 JSON 映射，导出时显式启用 --calibrate-memory。每个任务的独立标注最多记一次，同一观测 ID 的标签或来源变化会失败。重复导出不会增加证据数。后验分数为 (成功观测数+1)/(观测总数+2)，它是经验可信度估计，需要在独立数据上检查表现。

留出 JSONL 每行包含 id、subject、value、success，value 对应记忆候选，例如 `{"recipe":["sum"],"passed":true}`。评估拒绝与校准使用相同的候选和观测 ID，报告覆盖数、Brier Score 及十桶 ECE。未达五条观测、已过期或不存在的候选计入未覆盖。实际数据的校准质量由报告体现，样例结果不能替代业务验证。

成本导出可同时保留模型成本与按工具尝试数计价的工具成本。缺失模型用量继续标记未知；提供 --tool-cost 后，只有组成项齐备才计算总成本。Token 与价格来自接口记录及调用方声明，项目不生成虚构账单。


### 任务队列、状态查询与取消

```bash
python -m rsi_agent.durable runs/queue enqueue examples/runtime_task.json
python -m rsi_agent.durable runs/queue status
python -m rsi_agent.task_worker runs/queue --max-attempts 3 --retry-delay 0.25
python -m rsi_agent.durable runs/queue status TASK_ID
python -m rsi_agent.durable runs/queue cancel TASK_ID
```

enqueue 只提交固定配方任务，submit 仍提交后立即执行。status 提供当前状态、已提交检查点、总步骤数、工具调用数、执行尝试数、截止时间、取消请求和外部操作回执状态，省略任务原始输入。运行中的取消先写 cancel_requested，由执行进程确认后进入 cancelled；排队、失败或中断任务可以直接取消。重复取消和重复读取终态不会重新执行任务。

工具、独立测试、受控 HTTP 网关及浏览器子进程在等待期间检查取消与总预算。取消时终止子进程组，Docker 任务还会删除登记的容器。cancelled 与 timed_out 为终态；需要重新执行时创建新的任务键。取消不能撤销已经在外部服务生效的业务操作，status 中的 external_operations 和事件日志可用于核对 intent、uncertain、confirmed。

新任务的 limits.total_seconds 默认 300 秒，提交时将绝对截止时间保存到 SQLite，包含排队、重试等待和中断停机时间，恢复不会重置。它覆盖任务提交后的运行阶段；模型规划请求仍由规划器自己的请求超时控制。单工具 timeout_seconds 独立生效。旧版任务缺少总预算记录时保留原有行为。

task_worker 处理启动时队列快照，完成后退出。针对 OS 错误、单工具超时、工具容量暂不可用和网关请求失败执行有界重试与指数退避。幂等协议不匹配、工具命令变更、预算耗尽等错误停止自动重试。max-attempts 按数据库中累计执行次数判断，重启 worker 不清零；工具调用预算同样跨重试累计。其他进程已持有任务锁时跳过该任务并在报告中保留状态。worker 不重新调用模型规划。

指标导出新增取消数、总预算超时数、平均尝试数及生命周期覆盖率。取消、超时或验证前执行失败的任务，verifier_passed 保留为 null，避免将未给出判定的任务计入验证器误判。


### 可恢复 DAG 工作流

```bash
python -m rsi_agent.workflows runs/graph submit sum-graph examples/runtime_workflow.json
python -m rsi_agent.workflows runs/graph run WORKFLOW_ID --workers 2 --max-attempts 3
python -m rsi_agent.workflows runs/graph status WORKFLOW_ID
```

示例并行计算两个分支，再将检查点中的输出汇合为下游输入。工作流结构使用 Python graphlib 校验环与缺失依赖，通过 ThreadPoolExecutor 调度就绪节点。每个节点复用现有 Runtime 的权限、预算、检查点、取消及重试。工作流与节点绑定写入 SQLite，同一工作流键绑定同一份图，每个节点派生固定任务键。

工作流顶层可设置 `"limits": {"max_tool_calls": 3}`，对整张图的工具启动次数设共享上限。所有并行分支和重试在 SQLite 写事务中预留额度，任务计数、共享计数与 tool_started 事件一起提交；达到上限后不启动新工具，节点以不可自动重试的预算错误失败。已预留的启动即使随后失败或进程崩溃也计费，恢复不会返还或重置额度，复用成功检查点不消耗新额度。status 的 tool_budget 返回 limit、used、remaining。该预算绑定不可修改的工作流计划；未配置的旧计划保持原有节点级限额。这里统计注册工具启动，不包含独立验证器、LLM 请求或服务实际账单。

节点可以指定 source，或使用 inputs 引用上游节点的检查点文件，默认路径为 output.txt，多份输入按 separator 连接。inputs 自动形成依赖，depends_on 可增加执行屏障。文件引用会重新检查检查点、读取预算及内容哈希，输入无法越过已确认的节点产物。成功节点恢复时只核对产物，工具调用不重复；中断节点续接原检查点，失败依赖阻止下游执行。--workers 控制图调度并发，工具并发仍由 Runtime 策略约束。

本实现使用 FIRST_COMPLETED 完成通知推进图，分支完成后立即释放满足依赖的下游节点，运行命令处理完当前可执行节点后返回 succeeded、failed 或 incomplete。正在被其他执行进程持有的节点、主动中断的节点会使图保持 incomplete，可使用同一工作流 ID 再次运行。它面向共享本地文件系统上的多进程执行。

### OpenTelemetry 与事务事件导出

无需额外依赖即可从已提交事件导出带 W3C Trace Context 的 JSON：

```bash
python -m rsi_agent.runtime_traces runs/graph --output runs/graph-traces.json
```

安装可选 SDK 后，可以通过 OTLP/HTTP 发送给 OpenTelemetry Collector：

```bash
python -m pip install '.[telemetry]'
python -m rsi_agent.runtime_traces runs/graph \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

链路包含任务、执行尝试、工具和验证阶段，时间来自持久化事件；模型规划起点在有用量证据时纳入任务跨度。Span ID 由任务与事件标识确定，恢复后的相同事件保持相同标识。外部网关同时发送 traceparent，请求可与本地工具 span 关联。导出通过公开 TracerProvider、IdGenerator 和 SpanProcessor 接口重放事件，再使用官方 OTLP exporter 发送。

事件日志作为事务 outbox 的来源，收到传输确认后才更新 trace_deliveries。失败或导出进程在确认前退出时可以重发，语义为至少一次投递，接收方应处理相同 Span ID。导出只处理终态任务，任务重试后的新版本会重新发送。进程崩溃前无法观测的工具结束时间使用恢复事件界定，并标记 observed_event_bounds；该跨度包含可能的停机等待。

JSON 导出保留固定快照，同一文件路径已含不同内容时会失败，新增任务后可选择新的输出文件。链路属性提供任务 ID、步骤、调用数与结果。它与现有事件日志共同用于故障定位。

### 一致性修复

任务、控制记录、提交事件及规划用量现在在同一个 SQLite 写事务中提交。相同幂等键并发提交被串行化；已有用量记录绑定不同内容时会失败。历史规划任务缺少用量时保留未知，避免当作零 Token。

队列的尝试上限在获得任务锁后、增加计数前再次检查，防止其他 worker 已用完次数后，持有旧队列快照的 worker 再次执行。统计导出在同一数据库读快照中读取任务、事件、控制和用量。独立标注按批次事务导入，任意冲突都会回滚该批次新增记录。检查点目录缺失会直接报错。

回归用例覆盖并发相同键提交、写入用量时故障回滚、两个 worker 的旧快照交错、导出期间另一 worker 重试，以及冲突标注的整批回滚。OpenTelemetry 集成用例由本地 HTTP 接收器解码真实 OTLP protobuf，验证 Span ID 与持久化事件一致。


### 工作流级取消与持续就绪调度

```bash
python -m rsi_agent.workflows runs/graph cancel WORKFLOW_ID
python -m rsi_agent.workflows runs/graph status WORKFLOW_ID
```

调度器按完成事件持续补充空闲执行槽；快分支的下游无需等待无关慢分支。并发上限仍由 --workers 和 Runtime 工具策略共同约束。输入产物校验失败时会立即记录该节点错误，其他独立分支继续推进。

工作流取消在 SQLite 写事务中设置持久化取消标志，并为已经绑定的节点任务发出取消。后续节点的任务登记、工作流绑定及取消校验也处于同一个事务，防止取消与提交交错留下可执行的孤立任务。未提交节点进入 cancelled，运行中节点复用 Runtime 的子进程和容器清理流程。

status 额外显示 cancel_requested、各节点的实际 task_state 与 completed_steps。存在尚待确认的运行节点时返回 cancelling，确认后返回 cancelled。重复取消保持幂等，已成功工作流的取消请求保持其成功结果。编排器收到中断异常时会发出工作流取消；使用同一 ID 再运行可确认已退出执行器留下的取消请求。外部业务操作仍需通过原回执核对已生效结果。

同时修复了产物篡改后的恢复问题：节点即使持有历史成功任务，也会重新核验检查点。连续恢复不会把损坏产物重新标为成功，恢复原校验内容后可再次通过。工作流控制表会为已有图补齐默认记录，原任务键、检查点和重试次数继续复用。

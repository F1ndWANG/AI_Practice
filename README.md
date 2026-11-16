# AI Practice

此仓库实现简历「开源项目与大模型研究」中的两个项目。公开样例由代码生成。

| 项目 | 本仓库入口 |
| --- | --- |
| RSIAgent 执行反馈驱动的递归自改进 | [运行与评测](rsi_agent/README.md) |
| 多模态大模型 SFT 与 GRPO、视觉可靠性 | [数据、训练与评测](vlm_posttraining/README.md) |

在仓库根目录运行 CPU 回归测试。需要 Python 3.9+ 和 Bash。

```bash
python3 -m unittest discover -s tests -v
```

## 一次跑通本地评测

```bash
python3 -m rsi_agent.study runs/rsi-study
python3 -m vlm_posttraining.fixtures runs/vlm-pairs
python3 -m vlm_posttraining.compare runs/vlm-pairs/prepared/test.jsonl runs/vlm-pair-report \
  --prediction repeated=runs/vlm-pairs/fixture-repeated_answer.jsonl \
  --prediction oracle=runs/vlm-pairs/fixture-oracle.jsonl
```

以上三条命令要求对应输出目录尚未创建。RSIAgent 输出消融结果和执行轨迹。视觉评测使用合成图与预置预测验证评分器，模型结果可用同一评测入口接入。

## 安装与验收

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
ai-practice smoke runs/acceptance
ai-practice verify runs/acceptance
```

验收入口在 macOS/Linux 上运行，无需 GPU、模型权重或 API。每个运行目录保存阶段输出、测试轨迹、回归指标及文件 SHA-256 清单。验收命令会创建新目录，校验命令可用于复查留档结果。

| 使用场景 | 入口 |
| --- | --- |
| 导入自定义 Agent 任务 | [任务协议及验证](rsi_agent/README.md#自定义任务套件) |
| 从清单执行完整 GPU 实验 | [训练编排与恢复](vlm_posttraining/README.md#完整实验入口) |

## 导出实验报告

```bash
ai-practice report runs/acceptance > runs/acceptance-report.md
```

同一命令也支持 Agent 单独运行目录、多模态成对评测目录和 GPU 实验目录。报告保留样本数、对照结果、区间及可用的推理指标。

| 扩展入口 | 用途 |
| --- | --- |
| [RSIAgent 扩展任务集](rsi_agent/README.md#扩展任务集) | 增加 JSON 规范化与带引号 CSV 聚合，输出按任务族消融结果 |
| [DocVQA 数据转换](vlm_posttraining/README.md#导入-docvqa) | 导入本地 Task 1 标注和图像，保留答案变体与文档分组 |
| [单图复核与模型导出](vlm_posttraining/README.md#单图复核与模型导出) | 复查错例，或将 LoRA 合并为可独立加载的模型目录 |

跨种子对照和多来源审计均可导出 Markdown 报告。参见 [Agent 实验矩阵](rsi_agent/README.md#跨种子成本对照)与[多模态数据合并](vlm_posttraining/README.md#合并多数据源)。

需要审计实验结果时，可[重放 Agent 轨迹与记忆](rsi_agent/README.md#重放轨迹与审计记忆)，或[生成多模态盲测清单](vlm_posttraining/README.md#盲测清单与评分端校验)。

实验诊断入口包括 [Agent 调用预算对照](rsi_agent/README.md#调用预算对照)和 [GRPO 训练前奖励审计](vlm_posttraining/README.md#grpo-训练前奖励审计)，均可生成 Markdown 报告。

[GRPO 模型候选采集](vlm_posttraining/README.md#从模型采集候选并审计)可与训练前审计直接串联。Agent 的[错误记忆压力测试](rsi_agent/README.md#错误记忆压力测试)报告成功率与重验开销。

长时间运行见 [Agent 模型端点故障恢复](rsi_agent/README.md#模型端点故障恢复)和[候选采集中断恢复](vlm_posttraining/README.md#候选采集中断恢复)。

## 回归验收

```bash
python -m ai_practice.gates agent runs/acceptance/agent runs/acceptance/agent runs/agent-gate \
  --baseline baseline --candidate brs_drs --max-regressions 0
python -m ai_practice.gates vision runs/acceptance/fixtures/prepared/test.jsonl \
  runs/acceptance/fixtures/fixture-repeated_answer.jsonl \
  runs/acceptance/fixtures/fixture-oracle.jsonl runs/vision-gate \
  --max-grounded-drop 0 --max-regressions 0
ai-practice report runs/vision-gate
```

两个入口按样本 ID 配对。Agent 还要求两份运行使用同一任务快照和工具预算。VLM 使用同一份参考清单重新评分。默认允许的平均退化为 0，`--max-regressions 0` 进一步禁止单样本质量退化，即使总体均值持平也会失败。Agent 的 `--max-call-increase` 使用每任务平均调用数。准确率阈值用 0 到 1 的比例，例如 `0.02` 表示两个百分点。严格证据准确率为可选指标，仅使用有证据标注的样本，若请求该检查但没有合适样本则判定失败。

通过时返回 0，指标不达标时返回 1。两种结果均写入 `summary.json` 和 `changes.jsonl`。后者包含指标发生变化的样本 ID 及前后值。GitHub Actions 对内置 CPU 样例执行这两项检查。

### 分组回归验收

在现有验收命令加 `--slice-by task --min-slice-samples 5`，可按任务类型逐组验收。VLM 还支持分别追加 `--slice-by difficulty` 和 `--slice-by dataset`。每个维度独立分组，使用与总体检查相同的阈值。样本量门槛按指标有效分母判断。低于门槛或无标签的检查会在报告中注明，如果指定的维度完全没有有效检查，验收失败。只有总体和所有指定维度均通过时，命令才返回 0。

`summary.json` 保存各组分母、前后指标和判定。`changes.jsonl` 在启用分组时附带变化样本的分组标签，便于复核。

### 固定验收策略

```bash
python -m ai_practice.gates agent runs/acceptance/agent runs/acceptance/agent runs/agent-policy \
  --policy examples/policies/agent.json
python -m ai_practice.gates vision runs/acceptance/fixtures/prepared/test.jsonl \
  runs/acceptance/fixtures/fixture-repeated_answer.jsonl \
  runs/acceptance/fixtures/fixture-oracle.jsonl runs/vision-policy \
  --policy examples/policies/vision.json
```

JSON 策略固定阈值、分组维度及最小有效样本数。`required_slices` 可为指定分组列出必须通过的指标。分组缺失、有效分母不足或指标退化时，必需检查失败。验收报告记录策略内容和 SHA-256。使用策略时，阈值和分组选项由策略提供，基线与候选仍由命令指定。仓库的两个示例覆盖内置 CPU 样例，可按实际实验数据调整覆盖要求。

### 交付验收包

```bash
python -m ai_practice.release build runs/agent-policy runs/vision-policy runs/release
python -m ai_practice.release verify runs/release
```

验收包汇总两份回归判定，保存 Markdown 报告、两个判定摘要和文件哈希清单。可通过 `--revision` 标记完整提交 SHA。默认省略逐样本变化记录，只有显式指定 `--include-changes` 才将其打包。原始图像、预测、权重以及完整工具轨迹留在各自实验目录。

生成前会校验两份验收输出的收据。即使某项指标失败，也会生成可校验的验收包，并返回退出码 1。旧版验收目录如果缺少 `receipt.json`，需要重新运行验收命令。GitHub Actions 为两个 Python 环境分别上传内置 CPU 样例的验收包，包含样例的逐条变化记录。

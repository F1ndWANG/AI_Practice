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

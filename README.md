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

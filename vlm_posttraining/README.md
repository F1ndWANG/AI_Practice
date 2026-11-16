# 多模态 SFT / GRPO 与视觉可靠性

采用 Qwen2.5-VL 与 LoRA，支持 OCR、DocVQA 和视觉推理。SFT 仅监督回答部分。GRPO 从 SFT adapter 继续训练，奖励为答案精确匹配、JSON 格式和证据 F1 的加权和，权重分别为 1、0.2、0.5。

## 数据协议

原始 JSONL 每行描述一张图片中的一个问题，图片路径相对于原始 JSONL。

```json
{"id":"doc-42:q1","source_group":"doc-42","task":"docvqa","difficulty":"medium","image":"images/doc-42.png","question":"发票总额是多少？","answer":"128.00","evidence":["合计栏显示128.00"],"split":"train"}
```

- `task` 取 `ocr`、`docvqa`、`visual_reasoning`。标注规范应为同一事实使用一致写法。
- `split` 可选，取 `train`、`validation`、`test`。无显式划分的数据按来源组和完整图片 SHA-256 的连通分组进行确定性划分。
- 去重键由图片 SHA-256 与标准化问题构成。检测重复 ID、冲突标注及跨分割共享图像或来源组。重编码图片和近重复样本需要在上游指定相同来源组。
- 清洗报告统计重复数和各任务及难度的样本分布。生成的清单使用绝对图片路径，迁移机器时重新执行清洗。

```bash
python3 -m vlm_posttraining.pipeline prepare data/manifest.jsonl runs/vlm/prepared
```

CPU 冒烟命令生成合成图片、标注、预置预测、指标和复核队列。

```bash
python3 -m vlm_posttraining.demo runs/vlm-smoke
```

## 训练与评测

在 Linux CUDA 主机使用 Python 3.10+、支持 BF16 的 GPU 和可访问的模型权重。先按目标 CUDA 环境安装匹配的 PyTorch/torchvision，再安装训练依赖。默认采用 Transformers 生成 GRPO 样本。可安装 `trl[vllm]==0.29.0` 并在 GRPO 命令增加 `--vllm`，共置模式需要额外显存。

```bash
python -m pip install -r vlm_posttraining/requirements.txt
python -m vlm_posttraining.train sft runs/vlm/prepared/train.jsonl runs/vlm/sft --steps 50
python -m vlm_posttraining.train grpo runs/vlm/prepared/train.jsonl runs/vlm/grpo \
  --adapter runs/vlm/sft --steps 50 --generations 4
```

先用 validation 调试奖励和超参数，再在固定 test 清单上对 base、SFT、GRPO 使用相同问题、提示词、解码设置和评测协议。

```bash
for variant in base sft grpo; do
  args=()
  if [ "$variant" != base ]; then args=(--adapter "runs/vlm/$variant"); fi
  python -m vlm_posttraining.train predict runs/vlm/prepared/test.jsonl \
    "runs/vlm/$variant.pred.jsonl" "${args[@]}"
  python -m vlm_posttraining.pipeline evaluate runs/vlm/prepared/test.jsonl \
    "runs/vlm/$variant.pred.jsonl" "runs/vlm/$variant.review.jsonl" \
    > "runs/vlm/$variant.metrics.json"
done
```

输出包括答案准确率、格式合规率、证据 F1、可解析输出中的未标注证据率，以及逐任务错例统计。人工复核时对照原图与 `review.jsonl`，区分标注遗漏、等价表达和视觉幻觉。简历中的 2000 余条样本与历史实验指标需要原始数据、权重和人工复核记录才能复现。

参考 [TRL SFT](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer) 和 [TRL GRPO](https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer)。

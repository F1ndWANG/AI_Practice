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

## 反事实样例与成对模型对比

```bash
python3 -m vlm_posttraining.fixtures runs/vlm-pairs
python3 -m vlm_posttraining.compare runs/vlm-pairs/prepared/test.jsonl runs/vlm-pair-report \
  --prediction repeated=runs/vlm-pairs/fixture-repeated_answer.jsonl \
  --prediction oracle=runs/vlm-pairs/fixture-oracle.jsonl
```

生成器提供 72 张图片及 36 对标签，覆盖 OCR、DocVQA 和视觉问答。同一对保留问题、改变图像中的答案，使用相同来源组及 `pair_id`，整对进入同一 split。附带的预测用于核验评测逻辑，训练后可将 base、SFT、GRPO 的预测按顺序传给同一入口。

- `summary.json` 记录清单哈希、任务与难度切片、答案和证据均完全匹配的比率，以及反事实双样本都答对的比率。
- `paired` 给出相对首个预测的同样本改进、退化数量和按来源连通组重采样的 95% bootstrap 区间。只有一个组时区间为空。样本组很少时区间仅用于描述。
- `review.jsonl` 保存原图路径、标注、两组输出和错例类型，便于逐条核验。

## 训练恢复与证据奖励消融

训练每 10 步保存检查点，可用 `--save-steps` 调整。恢复时使用原阶段、参数和数据，传入 `--resume runs/vlm/sft/checkpoint-10`。输出的 `run.json` 包含清单哈希、初始 adapter 哈希、依赖版本和训练参数。

对同一个 SFT adapter 和训练清单分别使用 `--evidence-weight 0` 和默认值 `0.5` 运行 GRPO，输出到不同目录。用同一验证集比较准确率、证据表现和格式，再用固定测试集报告最终对比。

## 完整实验入口

在支持 BF16 的 Linux CUDA 环境安装训练依赖后，准备带标注的原始 JSONL。以下命令完成清洗、SFT、GRPO、可选的证据奖励消融、验证集预测与统计对比。

```bash
CUDA_VISIBLE_DEVICES=0 python -m vlm_posttraining.experiment \
  data/manifest.jsonl runs/vlm-experiment --ablate-evidence
```

中断后用相同配置增加 `--resume`。训练阶段自动查找最近一份包含 adapter、优化器、调度器及随机数状态的检查点。预测或统计阶段中断后重跑该阶段。完成阶段会验证产物哈希并复用输出。配置、原始清单或已登记的输入图发生变化时，应创建新运行目录。

确认验证集设置后，使用同一配置生成留出测试报告。

```bash
CUDA_VISIBLE_DEVICES=0 python -m vlm_posttraining.experiment \
  data/manifest.jsonl runs/vlm-experiment --ablate-evidence --resume --evaluate-test
```

`experiment.json` 是运行配置，`status.json` 保存阶段状态和哈希，`logs/` 保存训练及推理子进程输出。比较结果位于 `comparison-validation/` 和 `comparison-test/`。此编排入口面向单 GPU，多 GPU 训练可以按上文使用独立阶段命令配置。

## 训练前检查

```bash
python -m vlm_posttraining.doctor data/manifest.jsonl
python -m vlm_posttraining.doctor data/manifest.jsonl --gpu
```

默认检查清单、划分、去重、反事实配对、独立来源组数和无证据样本数。可单独加 `--decode-images` 用 Pillow 解码所有去重后的图像路径。`--gpu` 同时尝试导入训练依赖，检查 CUDA、BF16、可见设备与显存，并解码图片。使用共置 vLLM 时可加 `--vllm` 检查包是否安装。

推理 JSONL 记录同步后的单样本生成耗时和输出 token 数。对比报告汇总 P50/P95 和生成 token 吞吐。计时覆盖 `generate()`，图片读取、预处理和文本解码在计时范围外。

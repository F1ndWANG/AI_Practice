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

## 导入 DocVQA

使用已获得授权并保存在本地的 [DocVQA Task 1 图像与标注](https://site.docvqa.org/datasets)。图像根目录需能解析标注中的 `image` 路径。

```bash
python -m vlm_posttraining.import_docvqa /data/docvqa runs/docvqa.jsonl \
  --annotations auto=/data/docvqa/train.json --annotations validation=/data/docvqa/val.json
python -m vlm_posttraining.doctor runs/docvqa.jsonl
```

`--annotations` 可重复，前缀可用 `auto`、`train`、`validation` 或 `test`。`auto` 让准备阶段按文档来源划分，可用于从有标签的训练数据保留内部测试子集。若已有独立且带标签的划分，可显式传入三个文件。官方无标签测试文件需要提交至对应评测入口。

答案比较接受标注中的各个答案变体。未标注证据的样本走仅回答指令，其证据奖励为零，且从证据 F1、严格证据准确率和对应统计区间的分母中排除。答案准确率及其配对变化覆盖有答案标注的所有样本。导入后按文档 ID 和图像哈希验证隔离。此处使用仓库的精确匹配指标，可用于内部实验。

## 单图复核与模型导出

```bash
python -m vlm_posttraining.inference document.png "What is the invoice total?" \
  --adapter runs/vlm/sft --answer-only
python -m vlm_posttraining.export runs/vlm/sft runs/vlm/sft-merged
python -m vlm_posttraining.inference document.png "What is the invoice total?" \
  --model runs/vlm/sft-merged --answer-only
```

单图复核与批量评测复用相同生成逻辑。默认使用 CUDA BF16，可指定 `--device cpu` 使用 FP32 推理。导出通过 PEFT 安全合并 LoRA，写入模型分片、处理器和 adapter 哈希。合并需准备容纳完整模型的内存和磁盘空间，也可用 `--dtype fp32` 导出。参考 [PEFT 模型存储说明](https://huggingface.co/docs/peft/developer_guides/checkpoint#model-storage)。

## 合并多数据源

```bash
python -m vlm_posttraining.mix runs/data-mix \
  --source docvqa=runs/docvqa.jsonl --source vision=data/vision.jsonl
python -m vlm_posttraining.mix runs/data-mix --verify
python -m ai_practice report runs/data-mix
```

样本 ID、来源组和反事实配对 ID 加入数据集前缀。相对图像路径按各自清单解析。显式划分会保留，与全局图像哈希或来源分组冲突时整次导入失败。完全重复且标注一致的样本保留按数据集名称排序遇到的第一条，审计记录各来源的保留量与去重数。跨来源泄漏检查覆盖相同文件内容。同一原始文档的不同裁图或重编码版本，可在各清单中标注相同的 `shared_source_group`，将它们锁定到同一划分。同一清单中同一来源组的所有样本须使用一致的共享标记。

使用 `runs/data-mix/manifest.jsonl` 作为 GPU 实验入口的输入，实验继续按完整清单做一致性校验。若混合时指定了 `--seed`，实验应传入同一值。训练报告中的 `by_dataset` 提供分来源的答案与证据指标，避免总体平均值掩盖单个数据集退化。

## 盲测清单与评分端校验

实验入口为验证集和留出测试集分别生成 `inference/<split>.jsonl`，推理子进程只接收样本 ID、图像、问题、证据任务模式和图像哈希。原始答案、别名和证据标注留在评分端的 `prepared/`。恢复运行时，预测和对比阶段会核对输入依赖的哈希。首次恢复旧实验时会为原预测阶段补建盲测清单，并重跑相应预测和对比阶段。

可独立执行推理与评测。推理机需要能按清单中的绝对路径访问同一批图像。

```bash
python -m vlm_posttraining.blind build runs/vlm/prepared/test.jsonl \
  runs/dispatch/test.jsonl runs/vlm/test-binding.json
python -m vlm_posttraining.train predict runs/dispatch/test.jsonl runs/test-predictions.jsonl \
  --adapter runs/vlm/sft
python -m vlm_posttraining.blind verify runs/vlm/prepared/test.jsonl \
  runs/dispatch/test.jsonl runs/vlm/test-binding.json
python -m vlm_posttraining.pipeline evaluate runs/vlm/prepared/test.jsonl \
  runs/test-predictions.jsonl runs/test-review.jsonl
```

只交付推理清单和图像给推理端。绑定文件应和带标注清单一起保存于可信评分端。校验核对清单、样本内容及图像文件。评测按 ID 对齐，并拒绝缺少、重复或额外的预测。

## GRPO 训练前奖励审计

```bash
python -m vlm_posttraining.reward_audit runs/vlm/prepared/validation.jsonl runs/grpo-audit
python -m ai_practice report runs/grpo-audit
```

默认对每个问题构造正确答案、错误答案、格式损坏和证据扰动候选。审计与训练共用相同的奖励函数。使用 `--candidates pilot.jsonl` 可审计模型输出。每行是一个问题的候选组：

```json
{"id":"sample-1","completions":["{\"answer\":\"12\",\"evidence\":[]}","{\"answer\":\"15\",\"evidence\":[]}"]}
```

候选可以覆盖参考清单的子集，每组至少两条输出。报告记录覆盖量，并对照证据奖励权重为零时的结果。组内所有候选同分时没有相对奖励差。若同组最低分的正确答案低于最高分的有效错误答案，则记录排序反转。通过 `--evidence-weight` 可以比较不同权重，并查看 `groups.jsonl` 中的候选级评分。合成扰动用于检验奖励逻辑，模型采样后的候选用于估计真实信号。

## 从模型采集候选并审计

```bash
python -m vlm_posttraining.blind build runs/vlm/prepared/validation.jsonl \
  runs/dispatch/validation.jsonl runs/vlm/validation-binding.json
python -m vlm_posttraining.sample_candidates runs/dispatch/validation.jsonl runs/vlm-pilot \
  --adapter runs/vlm/sft --count 4 --temperature 0.7 --top-p 0.9 --limit 64
python -m vlm_posttraining.reward_audit runs/vlm/prepared/validation.jsonl runs/vlm-pilot-audit \
  --candidates runs/vlm-pilot/candidates.jsonl --sampling-receipt runs/vlm-pilot/sampling.json
```

采集端只接收盲测清单，核对图片哈希后复用单图推理的图像预处理。每个问题按全局种子和样本 ID 派生采样种子，候选组保存生成耗时和 token 数。采集产物绑定盲测清单、采样参数、候选文件和本地 adapter 哈希。评分端可校验候选及清单是否匹配。指定 `--limit` 时采集清单前 N 条。跨设备或依赖版本的生成结果可能有差异。

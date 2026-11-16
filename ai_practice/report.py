"""Render a Markdown experiment report from an existing result directory."""

import json
from pathlib import Path


def table(headers, rows):
    def row(items):
        escape = lambda item: str(item).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
        return "| " + " | ".join(escape(item) for item in items) + " |"

    return [row(headers), row(["---"] * len(headers)), *(row(items) for items in rows)]


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def agent_lines(summary):
    rows = []
    for name, value in summary["variants"].items():
        requests = value.get("model_requests", 0)
        coverage = f"{value.get('model_requests_with_usage', 0)}/{requests}" if requests else "—"
        rows.append((name, value["tasks"], percent(value["success_rate"]), value["tool_calls"],
                     requests, value.get("model_fallbacks", 0), value.get("model_request_errors", 0),
                     value.get("model_tokens_reported") if value.get("model_tokens_reported") is not None else "—",
                     coverage))
    return [
        "## RSIAgent", "",
        *table(("设置", "任务数", "成功率", "测试工具调用", "模型请求", "回退", "请求错误",
                "已报告 Tokens", "用量覆盖"), rows), "",
        f"探索工具调用（BRS / DRS）  {summary['exploration_tool_calls']['brs']} / "
        f"{summary['exploration_tool_calls']['drs']}", "",
    ]


def vision_lines(summary, name):
    rows = []
    performance = []
    for variant, result in summary["variants"].items():
        overall = result["overall"]
        pairs = result.get("counterfactual")
        values = overall.get("generation_performance")
        if values:
            performance.append((variant, values["measured"], f"{values['p50_seconds']:.3f}",
                                f"{values['p95_seconds']:.3f}",
                                f"{values['generated_tokens_per_second']:.1f}"))
        rows.append((variant, overall["n"], percent(overall["accuracy"]), percent(overall["format"]),
                     percent(overall["evidence"]), percent(result["grounded_accuracy"]),
                     percent(pairs["both_answers_correct"]) if pairs else "—"))
    deltas = []
    for variant, result in summary["paired"].items():
        bounds = result["bootstrap_95"]
        interval = "—" if bounds is None else " 至 ".join(map(percent, bounds))
        deltas.append((variant, percent(result["delta_grounded_accuracy"]), interval,
                       result["improved"], result["regressed"], result["groups"]))
    speed = (table(("设置", "计时样本", "P50 秒", "P95 秒", "生成 Tokens/秒"), performance) + [""]
             if performance else [])
    return [
        f"## {name}", "",
        *table(("设置", "样本数", "答案准确率", "格式合规率", "证据 F1", "严格证据准确率",
                "双图全对率"), rows), "",
        *speed,
        *table(("对比设置", "严格证据准确率变化", "95% bootstrap 区间", "改善", "退化", "来源组"), deltas),
        "", f"清单 SHA-256  `{summary['manifest_sha256']}`", "",
    ]


def render(root):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    lines = ["# AI Practice 实验报告", ""]
    linked_agent = root / "agent/summary.json"
    local_summary = root / "summary.json"
    local_data = read(local_summary) if local_summary.is_file() else None
    if linked_agent.is_file():
        lines.extend(agent_lines(read(linked_agent)))
    elif local_data and "exploration_tool_calls" in local_data:
        lines.extend(agent_lines(local_data))

    for path, title in (
        ("vision/summary.json", "合成图片评分器验收"),
        ("comparison-validation/summary.json", "多模态验证集"),
        ("comparison-test/summary.json", "多模态留出测试集"),
    ):
        metadata = root / path
        if metadata.is_file():
            lines.extend(vision_lines(read(metadata), title))
    if local_data and "paired" in local_data:
        lines.extend(vision_lines(local_data, "多模态成对评测"))
    if len(lines) == 2:
        raise ValueError("No supported result summary found")
    return "\n".join(lines)

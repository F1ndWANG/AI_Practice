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
    families = [
        (family, variant, value["tasks"], percent(value["success_rate"]), value["tool_calls"])
        for family, variants in summary.get("families", {}).items()
        for variant, value in variants.items()
    ]
    breakdown = (table(("任务族", "设置", "任务数", "成功率", "工具调用"), families) + [""]
                 if families else [])
    return [
        "## RSIAgent", "",
        *table(("设置", "任务数", "成功率", "测试工具调用", "模型请求", "回退", "请求错误",
                "已报告 Tokens", "用量覆盖"), rows), "",
        f"探索工具调用（BRS / DRS）  {summary['exploration_tool_calls']['brs']} / "
        f"{summary['exploration_tool_calls']['drs']}", "",
        *breakdown,
    ]


def matrix_lines(summary):
    rows = []
    for name, result in summary["variants"].items():
        saved = result["tool_calls_saved"]
        net = result["tool_calls_saved_including_exploration"]
        sd = saved["sample_sd"]
        rows.append((name, percent(result["success_rate"]["mean"]), f"{saved['mean']:.1f}",
                     "—" if sd is None else f"{sd:.1f}", f"{net['mean']:.1f}"))
    return [
        "## RSIAgent 跨种子对照", "",
        "种子  " + ", ".join(map(str, summary["seeds"])), "",
        *table(("设置", "平均成功率", "平均节省测试调用", "测试节省标准差", "扣除探索后平均节省"), rows), "",
    ]


def budget_lines(summary):
    rows = []
    for budget, variants in summary["budgets"].items():
        for name, result in variants.items():
            estimate = result["estimated_break_even_tasks"]
            rows.append((budget, name, percent(result["success_rate"]), result["tool_calls"],
                         result["net_calls_saved"], "—" if estimate is None else estimate))
    return [
        "## RSIAgent 调用预算对照", "",
        *table(("单任务预算", "设置", "成功率", "测试调用", "净节省调用", "估算回本任务数"), rows), "",
    ]


def reward_lines(summary):
    overview = {"全部": summary["overall"], **summary["by_task_evidence"]}
    rows = [(name, stats["groups"], stats["zero_signal_groups"], stats["zero_signal_without_evidence"],
             stats["answer_preference_inversions"], stats["answer_preference_ties"],
             stats["malformed_candidates"]) for name, stats in overview.items()]
    return [
        "## GRPO 奖励审计", "",
        f"候选来源  {summary['candidate_source']}，证据权重  {summary['evidence_weight']}", "",
        *table(("任务", "组数", "同分组", "无证据项同分组", "正确答案排序反转", "正确答案并列",
                "格式错误候选"), rows), "",
    ]


def vision_lines(summary, name):
    rows = []
    performance = []
    datasets = []
    for variant, result in summary["variants"].items():
        overall = result["overall"]
        pairs = result.get("counterfactual")
        datasets.extend((variant, dataset, values["n"], percent(values["accuracy"]),
                         percent(values["evidence"]))
                        for dataset, values in result.get("by_dataset", {}).items())
        values = overall.get("generation_performance")
        if values:
            performance.append((variant, values["measured"], f"{values['p50_seconds']:.3f}",
                                f"{values['p95_seconds']:.3f}",
                                f"{values['generated_tokens_per_second']:.1f}"))
        rows.append((variant, overall["n"], percent(overall["accuracy"]), percent(overall["format"]),
                     percent(overall["evidence"]), percent(result["grounded_accuracy"]),
                     percent(pairs["both_answers_correct"]) if pairs else "—"))
    deltas = []
    answers = []
    for variant, result in summary["paired"].items():
        answer = result.get("answer_accuracy")
        if answer:
            bounds = answer["bootstrap_95"]
            interval = "—" if bounds is None else " 至 ".join(map(percent, bounds))
            answers.append((variant, percent(answer["delta_accuracy"]), interval, answer["samples"]))
        bounds = result["bootstrap_95"]
        interval = "—" if bounds is None else " 至 ".join(map(percent, bounds))
        deltas.append((variant, percent(result["delta_grounded_accuracy"]), interval,
                       result["improved"], result["regressed"], result["groups"]))
    speed = (table(("设置", "计时样本", "P50 秒", "P95 秒", "生成 Tokens/秒"), performance) + [""]
             if performance else [])
    answer_table = (table(("对比设置", "答案准确率变化", "95% bootstrap 区间", "样本数"), answers) + [""]
                    if answers else [])
    return [
        f"## {name}", "",
        *table(("设置", "样本数", "答案准确率", "格式合规率", "证据 F1", "严格证据准确率",
                "双图全对率"), rows), "",
        *speed,
        *(table(("设置", "数据源", "样本数", "答案准确率", "证据 F1"), datasets) + [""] if datasets else []),
        *answer_table,
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
    elif local_data and local_data.get("kind") == "agent_matrix":
        lines.extend(matrix_lines(local_data))
    elif local_data and local_data.get("kind") == "agent_budget_sweep":
        lines.extend(budget_lines(local_data))
    elif local_data and local_data.get("kind") == "vlm_reward_audit":
        lines.extend(reward_lines(local_data))
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
    audit_path = root / "audit.json"
    if audit_path.is_file():
        audit = read(audit_path)
        if audit.get("kind") == "multimodal_mix":
            rows = [(split, dataset, count, stats["unannotated_evidence_by_dataset"].get(dataset, 0))
                    for split, stats in audit["splits"].items()
                    for dataset, count in stats["by_dataset"].items()]
            lines.extend(["## 多数据源审计", "", f"删除精确重复样本 {audit['duplicates_removed']} 条", "",
                          *table(("划分", "数据源", "保留样本", "缺少证据标注"), rows), ""])
    if local_data and "paired" in local_data:
        lines.extend(vision_lines(local_data, "多模态成对评测"))
    if len(lines) == 2:
        raise ValueError("No supported result summary found")
    return "\n".join(lines)

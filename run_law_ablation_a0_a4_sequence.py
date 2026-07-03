from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"F:\Anaconda\envs\Camel\python.exe")
SOURCE_DIR = ROOT / "embedding_output" / "law_ablation_v4"
OUTPUT_DIR = ROOT / "embedding_output" / "law_ablation"
REPORT_PATH = OUTPUT_DIR / "LAW_ABLATION_A0_A4_RESULTS.md"
STATE_PATH = OUTPUT_DIR / "sequence_state.json"
MONITOR_LOG_PATH = OUTPUT_DIR / "sequence_monitor.log"
POLL_SECONDS = 10


DATASETS = [
    {
        "id": "cail2018",
        "name": "CAIL2018",
        "path": "data/testset/test_cail_sampled_single_seed_42.json",
        "samples": 1000,
    },
    {
        "id": "cjo22",
        "name": "CJO22",
        "path": "data/testset/testset.json",
        "samples": 1698,
    },
]


SCORE_ARGS = [
    "--retrieval-mode",
    "hybrid",
    "--dense-top-k",
    "10",
    "--bm25-top-k",
    "10",
    "--keyword-top-k",
    "10",
    "--join-top-k",
    "30",
    "--fusion-mode",
    "score",
    "--dense-score-weight",
    "0.65",
    "--bm25-score-weight",
    "0.25",
    "--keyword-score-weight",
    "0.10",
    "--rerank-score-weight",
    "0.20",
    "--lexical-include-numeric",
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def common_args(dataset: dict[str, Any], output_name: str) -> list[str]:
    return [
        "eval_llm_rag_law.py",
        "--dataset_path",
        dataset["path"],
        "--law_dir",
        "data/meta/laws.txt",
        "--limit",
        str(dataset["samples"]),
        "--offset",
        "0",
        "--topk_law",
        "3",
        "--retrieval-record-top-k",
        "30",
        "--retrieval-query-max-chars",
        "2000",
        "--max-output-articles",
        "1",
        "--output_path",
        f"embedding_output\\law_ablation\\{output_name}",
    ]


def build_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_id = dataset["id"]
        variants = [
            {
                "variant": "a0",
                "name": "A0 Embedding",
                "components": "Embedding baseline",
                "existing_source": f"{dataset_id}_e0_embedding.jsonl",
                "args": ["--retrieval-mode", "embedding", "--no-rerank"],
            },
            {
                "variant": "a1",
                "name": "A1 Score Fusion",
                "components": "Score Fusion",
                "args": SCORE_ARGS
                + [
                    "--no-law-query-rewrite",
                    "--no-dense-anchor",
                    "--no-rerank",
                    "--no-legal-aware-rerank",
                ],
            },
            {
                "variant": "a2",
                "name": "A2 + Query Rewrite",
                "components": "Score Fusion + Query Rewrite",
                "args": SCORE_ARGS
                + [
                    "--law-query-rewrite",
                    "--no-dense-anchor",
                    "--no-rerank",
                    "--no-legal-aware-rerank",
                ],
            },
            {
                "variant": "a3",
                "name": "A3 + Dense Anchor",
                "components": "Score Fusion + Query Rewrite + Dense Anchor",
                "existing_source": f"{dataset_id}_e3_score_anchor.jsonl",
                "args": SCORE_ARGS
                + [
                    "--law-query-rewrite",
                    "--dense-anchor",
                    "--dense-margin-threshold",
                    "0.02",
                    "--dense-override-threshold",
                    "0.08",
                    "--no-rerank",
                    "--no-legal-aware-rerank",
                ],
            },
            {
                "variant": "a4",
                "name": "A4 + Legal Rerank",
                "components": (
                    "Score Fusion + Query Rewrite + Dense Anchor + Legal Rerank"
                ),
                "existing_source": f"{dataset_id}_e4_score_anchor_rerank.jsonl",
                "args": SCORE_ARGS
                + [
                    "--law-query-rewrite",
                    "--dense-anchor",
                    "--dense-margin-threshold",
                    "0.02",
                    "--dense-override-threshold",
                    "0.08",
                    "--use-rerank",
                    "--rerank-top-k",
                    "30",
                    "--legal-aware-rerank",
                ],
            },
        ]
        for variant in variants:
            experiment_id = f"{dataset_id}_{variant['variant']}"
            output_name = f"{experiment_id}.jsonl"
            experiments.append(
                {
                    **variant,
                    "id": experiment_id,
                    "dataset_id": dataset_id,
                    "dataset": dataset["name"],
                    "sample_count": dataset["samples"],
                    "output": output_name,
                    "args": common_args(dataset, output_name) + variant["args"],
                }
            )
    return experiments


EXPERIMENTS = build_experiments()
RUN_ORDER = [
    "cail2018_a1",
    "cail2018_a2",
    "cjo22_a1",
    "cjo22_a2",
]


def experiment_by_id(experiment_id: str) -> dict[str, Any]:
    return next(exp for exp in EXPERIMENTS if exp["id"] == experiment_id)


def log(message: str) -> None:
    with MONITOR_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_text()} | {message}\n")


def inspect_jsonl(path: Path, expected_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "records": 0,
        "summary": None,
        "invalid_lines": 0,
        "complete": False,
    }
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                result["invalid_lines"] += 1
                continue
            if record.get("record_type") == "summary":
                result["summary"] = record
            else:
                result["records"] += 1
    summary = result["summary"]
    result["complete"] = (
        result["records"] == expected_samples
        and isinstance(summary, dict)
        and summary.get("num_samples") == expected_samples
        and result["invalid_lines"] == 0
    )
    return result


def parse_key_values(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            parsed[key.strip()] = float(value)
        except ValueError:
            parsed[key.strip()] = value
    return parsed


def run_metric_command(args: list[str]) -> str:
    completed = subprocess.run(
        [str(PYTHON), "calc_metrics.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Metric command failed ({completed.returncode}): "
            f"{' '.join(args)}\n{output}"
        )
    return output


def calculate_metrics(exp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    relative_output = str((OUTPUT_DIR / exp["output"]).relative_to(ROOT))
    final_metrics = parse_key_values(
        run_metric_command(["--input", relative_output, "--mode", "law"])
    )
    retrieval_metrics: dict[str, Any] = {}
    for cutoff in (1, 3, 5, 10):
        retrieval_metrics[str(cutoff)] = parse_key_values(
            run_metric_command(
                [
                    "--input",
                    relative_output,
                    "--mode",
                    "retrieval",
                    "--retrieval-k",
                    str(cutoff),
                ]
            )
        )
    return final_metrics, retrieval_metrics


def calculate_comparison(
    baseline_exp: dict[str, Any],
    candidate_exp: dict[str, Any],
) -> dict[str, Any]:
    output = run_metric_command(
        [
            "--input",
            str((OUTPUT_DIR / candidate_exp["output"]).relative_to(ROOT)),
            "--baseline-input",
            str((OUTPUT_DIR / baseline_exp["output"]).relative_to(ROOT)),
            "--mode",
            "law-compare",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ]
    )
    result = json.loads(output)
    result.update(
        {
            "dataset": candidate_exp["dataset"],
            "comparison": (
                f"{candidate_exp['variant'].upper()} vs "
                f"{baseline_exp['variant'].upper()}"
            ),
            "component": candidate_exp["name"],
        }
    )
    return result


def initial_state() -> dict[str, Any]:
    return {
        "sequence_started_at": now_text(),
        "heartbeat_at": now_text(),
        "status": "preparing",
        "experiments": {
            exp["id"]: {
                "dataset": exp["dataset"],
                "variant": exp["variant"].upper(),
                "name": exp["name"],
                "components": exp["components"],
                "status": "pending",
                "progress": 0,
                "expected_samples": exp["sample_count"],
                "output": exp["output"],
                "process_log": f"{exp['id']}.process.log",
                "started_at": None,
                "completed_at": None,
                "pid": None,
                "metrics": {},
                "retrieval_metrics": {},
                "error": None,
            }
            for exp in EXPERIMENTS
        },
        "comparisons": {},
    }


def load_or_create_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(state, dict) and "experiments" in state:
                return state
        except (json.JSONDecodeError, OSError):
            pass
    return initial_state()


def save_state(state: dict[str, Any]) -> None:
    state["heartbeat_at"] = now_text()
    temp_path = STATE_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(STATE_PATH)


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "-"


def render_report(state: dict[str, Any]) -> None:
    lines = [
        "# 法条混合检索 A0-A4 消融实验",
        "",
        f"- 开始时间：{state['sequence_started_at']}",
        f"- 最近心跳：{state['heartbeat_at']}",
        f"- 总体状态：{state['status']}",
        "- 模型：qwen3-max（读取 `ljp_config.py`）",
        "- LLM 输入最终 Top-3；JSONL 保留 Top-30 排名。",
        "",
        "## 消融设置",
        "",
        "| 编号 | Score Fusion | Query Rewrite | Dense Anchor | Legal Rerank |",
        "|---|---:|---:|---:|---:|",
        "| A0 | × | × | × | × |",
        "| A1 | √ | × | × | × |",
        "| A2 | √ | √ | × | × |",
        "| A3 | √ | √ | √ | × |",
        "| A4 | √ | √ | √ | √ |",
        "",
        "## 执行状态",
        "",
        "| 数据集 | 编号 | 状态 | 进度 | 输出文件 |",
        "|---|---|---|---:|---|",
    ]
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        lines.append(
            f"| {item['dataset']} | {item['variant']} | {item['status']} | "
            f"{item['progress']}/{item['expected_samples']} | "
            f"`{item['output']}` |"
        )

    lines.extend(
        [
            "",
            "## 最终法条预测结果（%）",
            "",
            "| 数据集 | 编号 | Acc | Ma-P | Ma-R | Ma-F | Primary Hit | 单标签 Acc | 多标签 Recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    metric_rows = 0
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        metrics = item.get("metrics") or {}
        if not metrics:
            continue
        metric_rows += 1
        lines.append(
            f"| {item['dataset']} | {item['variant']} | "
            f"{format_percent(metrics.get('acc'))} | "
            f"{format_percent(metrics.get('Ma-P'))} | "
            f"{format_percent(metrics.get('Ma-R'))} | "
            f"{format_percent(metrics.get('Ma-F'))} | "
            f"{format_percent(metrics.get('primary_hit_acc'))} | "
            f"{format_percent(metrics.get('single_gold_acc'))} | "
            f"{format_percent(metrics.get('multi_gold_primary_recall'))} |"
        )
    if not metric_rows:
        lines.append("| 暂无 | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## 检索结果（%）",
            "",
            "| 数据集 | 编号 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    retrieval_rows = 0
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        retrieval = item.get("retrieval_metrics") or {}
        if not all(str(k) in retrieval for k in (1, 3, 5, 10)):
            continue
        retrieval_rows += 1
        lines.append(
            f"| {item['dataset']} | {item['variant']} | "
            f"{format_percent(retrieval['1'].get('retrieval_recall@1'))} | "
            f"{format_percent(retrieval['3'].get('retrieval_recall@3'))} | "
            f"{format_percent(retrieval['5'].get('retrieval_recall@5'))} | "
            f"{format_percent(retrieval['10'].get('retrieval_recall@10'))} | "
            f"{format_percent(retrieval['10'].get('retrieval_mrr@10'))} |"
        )
    if not retrieval_rows:
        lines.append("| 暂无 | - | - | - | - | - | - |")

    lines.extend(["", "## 相邻组件贡献", ""])
    comparisons = state.get("comparisons") or {}
    if comparisons:
        lines.extend(
            [
                "| 数据集 | 比较 | Acc 差值 | Bootstrap 95% CI | McNemar p |",
                "|---|---|---:|---|---:|",
            ]
        )
        for key in sorted(comparisons):
            comparison = comparisons[key]
            ci = comparison.get("bootstrap_ci_95")
            ci_text = (
                f"[{format_percent(ci[0])}, {format_percent(ci[1])}]"
                if isinstance(ci, list) and len(ci) == 2
                else "-"
            )
            lines.append(
                f"| {comparison['dataset']} | {comparison['comparison']} | "
                f"{format_percent(comparison.get('accuracy_difference'))} | "
                f"{ci_text} | "
                f"{comparison.get('mcnemar', {}).get('exact_p_value', '-')} |"
            )
    else:
        lines.append("A0-A4 完整后生成。")

    errors = [
        item
        for item in state["experiments"].values()
        if item.get("error")
    ]
    if errors:
        lines.extend(["", "## 异常", ""])
        for item in errors:
            lines.append(
                f"- **{item['dataset']} {item['variant']}**：{item['error']}"
            )

    lines.extend(["", "## 结果文件", ""])
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        lines.append(
            f"- `{item['output']}`；日志 `{item['process_log']}`"
        )
    lines.append("")

    temp_path = REPORT_PATH.with_suffix(".md.tmp")
    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(REPORT_PATH)


def copy_existing_result(exp: dict[str, Any]) -> None:
    source_name = exp.get("existing_source")
    if not source_name:
        return
    source_path = SOURCE_DIR / source_name
    source_status = inspect_jsonl(source_path, exp["sample_count"])
    if not source_status["complete"]:
        raise RuntimeError(f"Existing result is incomplete: {source_path}")
    destination_path = OUTPUT_DIR / exp["output"]
    destination_status = inspect_jsonl(destination_path, exp["sample_count"])
    if not destination_status["complete"]:
        shutil.copy2(source_path, destination_path)
    source_log = SOURCE_DIR / source_name.replace(".jsonl", ".process.log")
    destination_log = OUTPUT_DIR / f"{exp['id']}.process.log"
    if source_log.exists() and not destination_log.exists():
        shutil.copy2(source_log, destination_log)


def record_completed_metrics(
    state: dict[str, Any],
    exp: dict[str, Any],
    source_label: str,
) -> None:
    status = inspect_jsonl(OUTPUT_DIR / exp["output"], exp["sample_count"])
    if not status["complete"]:
        raise RuntimeError(f"Output is incomplete: {exp['output']}")
    item = state["experiments"][exp["id"]]
    item["status"] = "calculating"
    item["progress"] = status["records"]
    item["started_at"] = item.get("started_at") or source_label
    save_state(state)
    render_report(state)
    metrics, retrieval = calculate_metrics(exp)
    item["metrics"] = metrics
    item["retrieval_metrics"] = retrieval
    item["status"] = "completed"
    item["completed_at"] = item.get("completed_at") or now_text()
    item["error"] = None
    save_state(state)
    render_report(state)


def prepare_existing_results(state: dict[str, Any]) -> None:
    for exp in EXPERIMENTS:
        if not exp.get("existing_source"):
            continue
        copy_existing_result(exp)
        record_completed_metrics(state, exp, "copied existing output")
        log(f"copied and indexed existing result {exp['id']}")


def launch_experiment(exp: dict[str, Any]) -> subprocess.Popen[Any]:
    output_path = OUTPUT_DIR / exp["output"]
    if output_path.exists() and output_path.stat().st_size > 0:
        backup = output_path.with_suffix(
            output_path.suffix + f".stale-{datetime.now():%Y%m%d-%H%M%S}"
        )
        output_path.replace(backup)
        log(f"archived incomplete output {output_path.name} -> {backup.name}")
    process_log_path = OUTPUT_DIR / f"{exp['id']}.process.log"
    process_log = process_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), *exp["args"]],
        cwd=ROOT,
        stdout=process_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    process_log.close()
    log(f"launched {exp['id']} pid={process.pid}")
    return process


def wait_for_experiment(
    state: dict[str, Any],
    exp: dict[str, Any],
    process: subprocess.Popen[Any],
) -> dict[str, Any]:
    item = state["experiments"][exp["id"]]
    output_path = OUTPUT_DIR / exp["output"]
    last_progress = -1
    last_report_at = 0.0
    while True:
        status = inspect_jsonl(output_path, exp["sample_count"])
        return_code = process.poll()
        report_due = time.time() - last_report_at >= 30
        if status["records"] != last_progress or report_due:
            item["progress"] = status["records"]
            save_state(state)
            render_report(state)
            if (
                status["records"] == exp["sample_count"]
                or status["records"] // 25 != last_progress // 25
                or report_due
            ):
                log(
                    f"{exp['id']} progress={status['records']}/"
                    f"{exp['sample_count']} return_code={return_code}"
                )
            last_progress = status["records"]
            last_report_at = time.time()
        if return_code is not None:
            if return_code != 0:
                raise RuntimeError(
                    f"Process exited with code {return_code}; "
                    f"see {exp['id']}.process.log"
                )
            final_status = inspect_jsonl(output_path, exp["sample_count"])
            if not final_status["complete"]:
                raise RuntimeError(
                    "Process exited but output is incomplete: "
                    f"records={final_status['records']} "
                    f"summary={final_status['summary'] is not None}"
                )
            return final_status
        time.sleep(POLL_SECONDS)


def update_adjacent_comparisons(state: dict[str, Any], dataset_id: str) -> None:
    dataset_experiments = sorted(
        (exp for exp in EXPERIMENTS if exp["dataset_id"] == dataset_id),
        key=lambda exp: exp["variant"],
    )
    if not all(
        state["experiments"][exp["id"]]["status"] == "completed"
        for exp in dataset_experiments
    ):
        return
    for baseline, candidate in zip(
        dataset_experiments,
        dataset_experiments[1:],
    ):
        key = f"{candidate['id']}_vs_{baseline['variant']}"
        state["comparisons"][key] = calculate_comparison(
            baseline,
            candidate,
        )
    save_state(state)
    render_report(state)


def validate_configuration() -> None:
    required = [
        PYTHON,
        ROOT / "eval_llm_rag_law.py",
        ROOT / "calc_metrics.py",
        ROOT / "data" / "meta" / "laws.txt",
        ROOT / "data" / "testset" / "testset.json",
        ROOT
        / "data"
        / "testset"
        / "test_cail_sampled_single_seed_42.json",
        SOURCE_DIR / "sequence_state.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare existing A0/A3/A4 and run missing A1/A2 experiments."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Copy existing results and build the report without running A1/A2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the four missing experiment commands.",
    )
    args = parser.parse_args()

    validate_configuration()
    if args.dry_run:
        for experiment_id in RUN_ORDER:
            exp = experiment_by_id(experiment_id)
            print(
                f"{exp['dataset']} {exp['variant'].upper()}\n"
                + " ".join([str(PYTHON), *exp["args"]])
            )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_or_create_state()
    state["status"] = "preparing"
    save_state(state)
    render_report(state)
    log("A0-A4 ablation preparation started")

    try:
        prepare_existing_results(state)
        state["status"] = "prepared" if args.prepare_only else "running"
        save_state(state)
        render_report(state)
        if args.prepare_only:
            log("Existing A0/A3/A4 results prepared")
            return 0

        for experiment_id in RUN_ORDER:
            exp = experiment_by_id(experiment_id)
            item = state["experiments"][exp["id"]]
            existing_status = inspect_jsonl(
                OUTPUT_DIR / exp["output"],
                exp["sample_count"],
            )
            if existing_status["complete"]:
                record_completed_metrics(state, exp, "existing output")
                log(f"reused complete output {exp['id']}")
            else:
                item["status"] = "running"
                item["started_at"] = now_text()
                item["error"] = None
                process = launch_experiment(exp)
                item["pid"] = process.pid
                save_state(state)
                render_report(state)
                wait_for_experiment(state, exp, process)
                record_completed_metrics(state, exp, item["started_at"])
                log(f"completed {exp['id']}")
            update_adjacent_comparisons(state, exp["dataset_id"])

        for dataset in DATASETS:
            update_adjacent_comparisons(state, dataset["id"])
        state["status"] = "completed"
        save_state(state)
        render_report(state)
        log("All A0-A4 ablation experiments completed")
        return 0
    except Exception as exc:
        state["status"] = "stopped_on_error"
        running_items = [
            item
            for item in state["experiments"].values()
            if item["status"] in {"running", "calculating"}
        ]
        if running_items:
            running_items[0]["status"] = "failed"
            running_items[0]["error"] = str(exc)
            running_items[0]["completed_at"] = now_text()
        save_state(state)
        render_report(state)
        log(f"failed: {exc}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

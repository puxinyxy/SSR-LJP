from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"F:\Anaconda\envs\Camel\python.exe")
OUTPUT_DIR = ROOT / "embedding_output" / "law_ablation_v4"
REPORT_PATH = OUTPUT_DIR / "LAW_RAG_V4_RESULTS.md"
STATE_PATH = OUTPUT_DIR / "sequence_state.json"
MONITOR_LOG_PATH = OUTPUT_DIR / "sequence_monitor.log"
POLL_SECONDS = 10


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def common_args(
    dataset_path: str,
    sample_count: int,
    output_name: str,
) -> list[str]:
    return [
        "eval_llm_rag_law.py",
        "--dataset_path",
        dataset_path,
        "--law_dir",
        "data/meta/laws.txt",
        "--limit",
        str(sample_count),
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
        f"embedding_output\\law_ablation_v4\\{output_name}",
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
    "--dense-anchor",
    "--dense-margin-threshold",
    "0.02",
    "--dense-override-threshold",
    "0.08",
    "--lexical-include-numeric",
]


def build_experiments() -> list[dict[str, Any]]:
    datasets = [
        (
            "cail2018",
            "CAIL2018",
            "data/testset/test_cail_sampled_single_seed_42.json",
            1000,
        ),
        (
            "cjo22",
            "CJO22",
            "data/testset/testset.json",
            1698,
        ),
    ]
    experiments: list[dict[str, Any]] = []
    for dataset_id, dataset_name, dataset_path, sample_count in datasets:
        variants = [
            (
                "e0_embedding",
                "E0 Embedding",
                ["--retrieval-mode", "embedding", "--no-rerank"],
            ),
            (
                "e3_score_anchor",
                "E3 Score Fusion + Query Rewrite + Dense Anchor",
                SCORE_ARGS
                + [
                    "--no-rerank",
                    "--no-legal-aware-rerank",
                ],
            ),
            (
                "e4_score_anchor_rerank",
                "E4 Score Fusion + Query Rewrite + Dense Anchor + Legal Rerank",
                SCORE_ARGS
                + [
                    "--use-rerank",
                    "--rerank-top-k",
                    "30",
                    "--legal-aware-rerank",
                ],
            ),
        ]
        for variant_id, variant_name, variant_args in variants:
            experiment_id = f"{dataset_id}_{variant_id}"
            output_name = f"{experiment_id}.jsonl"
            experiments.append(
                {
                    "id": experiment_id,
                    "dataset_id": dataset_id,
                    "dataset": dataset_name,
                    "name": variant_name,
                    "sample_count": sample_count,
                    "output": output_name,
                    "args": common_args(
                        dataset_path,
                        sample_count,
                        output_name,
                    )
                    + variant_args,
                }
            )
    return experiments


EXPERIMENTS = build_experiments()


def log(message: str) -> None:
    line = f"{now_text()} | {message}"
    with MONITOR_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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
    summary_samples = (
        summary.get("num_samples")
        if isinstance(summary, dict)
        else None
    )
    result["complete"] = (
        result["records"] == expected_samples
        and summary_samples == expected_samples
        and result["invalid_lines"] == 0
    )
    return result


def parse_key_values(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
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
    output_path = str((OUTPUT_DIR / exp["output"]).relative_to(ROOT))
    final_metrics = parse_key_values(
        run_metric_command(["--input", output_path, "--mode", "law"])
    )
    retrieval_metrics: dict[str, Any] = {}
    for cutoff in (1, 3, 5, 10):
        retrieval_metrics[str(cutoff)] = parse_key_values(
            run_metric_command(
                [
                    "--input",
                    output_path,
                    "--mode",
                    "retrieval",
                    "--retrieval-k",
                    str(cutoff),
                ]
            )
        )
    return final_metrics, retrieval_metrics


def calculate_comparison(
    baseline_output: str,
    candidate_output: str,
) -> dict[str, Any]:
    output = run_metric_command(
        [
            "--input",
            str((OUTPUT_DIR / candidate_output).relative_to(ROOT)),
            "--baseline-input",
            str((OUTPUT_DIR / baseline_output).relative_to(ROOT)),
            "--mode",
            "law-compare",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ]
    )
    return json.loads(output)


def initial_state() -> dict[str, Any]:
    return {
        "sequence_started_at": now_text(),
        "heartbeat_at": now_text(),
        "status": "running",
        "experiments": {
            exp["id"]: {
                "dataset": exp["dataset"],
                "name": exp["name"],
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
        "# CAIL2018 与 CJO22 法条 RAG V4 消融实验",
        "",
        f"- 开始时间：{state['sequence_started_at']}",
        f"- 最近心跳：{state['heartbeat_at']}",
        f"- 总体状态：{state['status']}",
        "- 执行顺序：CAIL2018 E0/E3/E4，然后 CJO22 E0/E3/E4",
        "- 每组 LLM 仅接收 Top-3；JSONL 保存 Top-30 排名用于检索指标。",
        "",
        "## 执行状态",
        "",
        "| 序号 | 数据集 | 实验 | 状态 | 进度 | 开始 | 完成 |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for index, exp in enumerate(EXPERIMENTS, 1):
        item = state["experiments"][exp["id"]]
        lines.append(
            f"| {index} | {item['dataset']} | {item['name']} | "
            f"{item['status']} | {item['progress']}/{item['expected_samples']} | "
            f"{item['started_at'] or '-'} | {item['completed_at'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## 最终法条预测指标",
            "",
            "| 数据集 | 实验 | Acc | Ma-P | Ma-R | Ma-F | Primary Hit | 单标签 Acc | 多标签 Recall |",
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
            f"| {item['dataset']} | {item['name']} | "
            f"{format_percent(metrics.get('acc'))} | "
            f"{format_percent(metrics.get('Ma-P'))} | "
            f"{format_percent(metrics.get('Ma-R'))} | "
            f"{format_percent(metrics.get('Ma-F'))} | "
            f"{format_percent(metrics.get('primary_hit_acc'))} | "
            f"{format_percent(metrics.get('single_gold_acc'))} | "
            f"{format_percent(metrics.get('multi_gold_primary_recall'))} |"
        )
    if not metric_rows:
        lines.append("| 暂无 | 暂无 | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## 检索指标",
            "",
            "| 数据集 | 实验 | K | Any Hit | All Hit | Recall | MRR | 排名来源 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    retrieval_rows = 0
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        for cutoff in ("1", "3", "5", "10"):
            values = (item.get("retrieval_metrics") or {}).get(cutoff)
            if not values:
                continue
            retrieval_rows += 1
            lines.append(
                f"| {item['dataset']} | {item['name']} | {cutoff} | "
                f"{format_percent(values.get(f'retrieval_any_hit@{cutoff}'))} | "
                f"{format_percent(values.get(f'retrieval_all_hit@{cutoff}'))} | "
                f"{format_percent(values.get(f'retrieval_recall@{cutoff}'))} | "
                f"{format_percent(values.get(f'retrieval_mrr@{cutoff}'))} | "
                f"{values.get(f'ranking_source@{cutoff}', '-')} |"
            )
    if not retrieval_rows:
        lines.append("| 暂无 | 暂无 | - | - | - | - | - | - |")

    lines.extend(["", "## 相对 Embedding 的配对比较", ""])
    comparisons = state.get("comparisons") or {}
    if comparisons:
        lines.extend(
            [
                "| 数据集 | 候选实验 | Acc 差值 | Bootstrap 95% CI | McNemar p |",
                "|---|---|---:|---|---:|",
            ]
        )
        for comparison in comparisons.values():
            ci = comparison.get("bootstrap_ci_95")
            ci_text = (
                f"[{format_percent(ci[0])}, {format_percent(ci[1])}]"
                if isinstance(ci, list) and len(ci) == 2
                else "-"
            )
            lines.append(
                f"| {comparison['dataset']} | {comparison['candidate_name']} | "
                f"{format_percent(comparison.get('accuracy_difference'))} | "
                f"{ci_text} | "
                f"{comparison.get('mcnemar', {}).get('exact_p_value', '-')} |"
            )
    else:
        lines.append("全部实验完成后生成。")

    errors = [
        item
        for item in state["experiments"].values()
        if item.get("error")
    ]
    if errors:
        lines.extend(["", "## 异常", ""])
        for item in errors:
            lines.append(
                f"- **{item['dataset']} / {item['name']}**：{item['error']}"
            )

    lines.extend(["", "## 输出记录", ""])
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        lines.append(
            f"- `{item['output']}`；日志 `{item['process_log']}`"
        )

    lines.extend(["", "## 执行命令", ""])
    for exp in EXPERIMENTS:
        command = " ".join([str(PYTHON), *exp["args"]])
        lines.extend(
            [
                f"### {exp['dataset']} / {exp['name']}",
                "",
                "```cmd",
                command,
                "```",
                "",
            ]
        )

    temp_path = REPORT_PATH.with_suffix(".md.tmp")
    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(REPORT_PATH)


def update_progress(
    state: dict[str, Any],
    exp: dict[str, Any],
    status: dict[str, Any],
) -> None:
    state["experiments"][exp["id"]]["progress"] = status["records"]
    save_state(state)
    render_report(state)


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
    command = [str(PYTHON), *exp["args"]]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=process_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    process_log.close()
    log(f"launched {exp['id']} pid={process.pid}: {' '.join(command)}")
    return process


def wait_for_experiment(
    state: dict[str, Any],
    exp: dict[str, Any],
    process: subprocess.Popen[Any],
) -> dict[str, Any]:
    output_path = OUTPUT_DIR / exp["output"]
    last_progress = -1
    last_report_at = 0.0
    while True:
        status = inspect_jsonl(output_path, exp["sample_count"])
        return_code = process.poll()
        report_due = time.time() - last_report_at >= 30
        if status["records"] != last_progress or report_due:
            update_progress(state, exp, status)
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
                    f"experiment process exited with code {return_code}; "
                    f"see {exp['id']}.process.log"
                )
            final_status = inspect_jsonl(output_path, exp["sample_count"])
            if not final_status["complete"]:
                raise RuntimeError(
                    "process exited successfully but output is incomplete: "
                    f"records={final_status['records']} "
                    f"summary={final_status['summary'] is not None} "
                    f"invalid_lines={final_status['invalid_lines']}"
                )
            return final_status
        time.sleep(POLL_SECONDS)


def update_comparisons(state: dict[str, Any], dataset_id: str) -> None:
    dataset_experiments = [
        exp for exp in EXPERIMENTS if exp["dataset_id"] == dataset_id
    ]
    if not all(
        state["experiments"][exp["id"]]["status"] == "completed"
        for exp in dataset_experiments
    ):
        return
    baseline = next(
        exp for exp in dataset_experiments if exp["id"].endswith("e0_embedding")
    )
    for candidate in dataset_experiments:
        if candidate is baseline:
            continue
        result = calculate_comparison(
            baseline["output"],
            candidate["output"],
        )
        result["dataset"] = candidate["dataset"]
        result["candidate_name"] = candidate["name"]
        state["comparisons"][candidate["id"]] = result
    save_state(state)
    render_report(state)


def validate_configuration() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Camel Python not found: {PYTHON}")
    required_paths = {
        ROOT / "eval_llm_rag_law.py",
        ROOT / "calc_metrics.py",
        ROOT / "data" / "meta" / "laws.txt",
        ROOT / "data" / "testset" / "testset.json",
        ROOT
        / "data"
        / "testset"
        / "test_cail_sampled_single_seed_42.json",
    }
    missing = sorted(str(path) for path in required_paths if not path.exists())
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run six law-RAG V4 ablation experiments sequentially."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print commands without running experiments.",
    )
    args = parser.parse_args()

    validate_configuration()
    if args.dry_run:
        for index, exp in enumerate(EXPERIMENTS, 1):
            print(
                f"[{index}/6] {exp['dataset']} / {exp['name']}\n"
                + " ".join([str(PYTHON), *exp["args"]])
            )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = initial_state()
    save_state(state)
    render_report(state)
    log("V4 law ablation monitor started")

    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        existing = inspect_jsonl(
            OUTPUT_DIR / exp["output"],
            exp["sample_count"],
        )
        try:
            if existing["complete"]:
                item["status"] = "calculating"
                item["started_at"] = "existing output"
                item["progress"] = existing["records"]
                save_state(state)
                render_report(state)
                log(f"reusing complete output {exp['id']}")
                completed_status = existing
            else:
                item["status"] = "running"
                item["started_at"] = now_text()
                process = launch_experiment(exp)
                item["pid"] = process.pid
                save_state(state)
                render_report(state)
                completed_status = wait_for_experiment(
                    state,
                    exp,
                    process,
                )

            item["status"] = "calculating"
            save_state(state)
            render_report(state)
            final_metrics, retrieval_metrics = calculate_metrics(exp)
            item["progress"] = completed_status["records"]
            item["metrics"] = final_metrics
            item["retrieval_metrics"] = retrieval_metrics
            item["status"] = "completed"
            item["completed_at"] = now_text()
            save_state(state)
            render_report(state)
            log(
                f"completed {exp['id']} metrics="
                f"{json.dumps(final_metrics, ensure_ascii=False)}"
            )
            update_comparisons(state, exp["dataset_id"])
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            item["completed_at"] = now_text()
            state["status"] = "stopped_on_error"
            save_state(state)
            render_report(state)
            log(f"failed {exp['id']}: {exc}")
            log(traceback.format_exc())
            return 1

    state["status"] = "completed"
    save_state(state)
    render_report(state)
    log("all six V4 law ablation experiments completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

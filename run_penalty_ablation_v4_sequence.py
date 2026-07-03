from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"F:\Anaconda\envs\Camel\python.exe")
OUTPUT_DIR = ROOT / "embedding_output" / "penalty_ablation_v4"
REPORT_PATH = OUTPUT_DIR / "penalty_ablation_v4_results.md"
STATE_PATH = OUTPUT_DIR / "sequence_state.json"
LOG_PATH = OUTPUT_DIR / "sequence_monitor.log"
POLL_SECONDS = 10


HYBRID_ARGS = [
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
    "--rrf-k",
    "60",
    "--dense-weight",
    "2.0",
    "--bm25-weight",
    "0.7",
    "--keyword-weight",
    "0.3",
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def penalty_args(
    dataset_path: str,
    limit: int,
    output_name: str,
) -> list[str]:
    return [
        "eval_llm_rag_penalty.py",
        "--dataset_path",
        dataset_path,
        "--precedent_file",
        "data/candidates/precedent_case.json",
        "--law_dir",
        "data/law_articles",
        "--limit",
        str(limit),
        "--offset",
        "0",
        "--topk_law",
        "3",
        "--topk_case",
        "3",
        "--retrieval-query-max-chars",
        "2000",
        "--retrieval-record-top-k",
        "30",
        "--output_path",
        f"embedding_output\\penalty_ablation_v4\\{output_name}",
    ]


EXPERIMENTS = [
    {
        "id": "cail18_penalty_embedding",
        "dataset": "CAIL2018",
        "name": "CAIL2018 penalty - Embedding",
        "task": "penalty",
        "expected_samples": 1000,
        "output": "cail18_penalty_embedding.jsonl",
        "args": penalty_args(
            "data/testset/test_cail_sampled_single_seed_42.json",
            1000,
            "cail18_penalty_embedding.jsonl",
        )
        + ["--retrieval-mode", "embedding", "--no-rerank"],
    },
    {
        "id": "cail18_penalty_hybrid_rrf",
        "dataset": "CAIL2018",
        "name": "CAIL2018 penalty - Hybrid RRF",
        "task": "penalty",
        "expected_samples": 1000,
        "output": "cail18_penalty_hybrid_rrf.jsonl",
        "args": penalty_args(
            "data/testset/test_cail_sampled_single_seed_42.json",
            1000,
            "cail18_penalty_hybrid_rrf.jsonl",
        )
        + HYBRID_ARGS
        + ["--no-rerank"],
    },
    {
        "id": "cail18_penalty_hybrid_rerank",
        "dataset": "CAIL2018",
        "name": "CAIL2018 penalty - Hybrid RRF + Rerank",
        "task": "penalty",
        "expected_samples": 1000,
        "output": "cail18_penalty_hybrid_rerank.jsonl",
        "args": penalty_args(
            "data/testset/test_cail_sampled_single_seed_42.json",
            1000,
            "cail18_penalty_hybrid_rerank.jsonl",
        )
        + HYBRID_ARGS
        + ["--use-rerank", "--rerank-top-k", "30"],
    },
    {
        "id": "cjo22_penalty_embedding",
        "dataset": "CJO22",
        "name": "CJO22 penalty - Embedding",
        "task": "penalty",
        "expected_samples": 1698,
        "output": "cjo22_penalty_embedding.jsonl",
        "args": penalty_args(
            "data/testset/testset.json",
            1698,
            "cjo22_penalty_embedding.jsonl",
        )
        + ["--retrieval-mode", "embedding", "--no-rerank"],
    },
    {
        "id": "cjo22_penalty_hybrid_rrf",
        "dataset": "CJO22",
        "name": "CJO22 penalty - Hybrid RRF",
        "task": "penalty",
        "expected_samples": 1698,
        "output": "cjo22_penalty_hybrid_rrf.jsonl",
        "args": penalty_args(
            "data/testset/testset.json",
            1698,
            "cjo22_penalty_hybrid_rrf.jsonl",
        )
        + HYBRID_ARGS
        + ["--no-rerank"],
    },
    {
        "id": "cjo22_penalty_hybrid_rerank",
        "dataset": "CJO22",
        "name": "CJO22 penalty - Hybrid RRF + Rerank",
        "task": "penalty",
        "expected_samples": 1698,
        "output": "cjo22_penalty_hybrid_rerank.jsonl",
        "args": penalty_args(
            "data/testset/testset.json",
            1698,
            "cjo22_penalty_hybrid_rerank.jsonl",
        )
        + HYBRID_ARGS
        + ["--use-rerank", "--rerank-top-k", "30"],
    },
]


def log(message: str) -> None:
    line = f"{now_text()} | {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def command_text(exp: dict[str, Any]) -> str:
    return " ".join([str(PYTHON), *exp["args"]])


def initial_state() -> dict[str, Any]:
    return {
        "sequence_started_at": now_text(),
        "monitor_pid": os.getpid(),
        "heartbeat_at": now_text(),
        "status": "running",
        "output_dir": str(OUTPUT_DIR),
        "report_path": str(REPORT_PATH),
        "experiments": {
            exp["id"]: {
                "dataset": exp["dataset"],
                "name": exp["name"],
                "status": "pending",
                "progress": 0,
                "expected_samples": exp["expected_samples"],
                "output": exp["output"],
                "command": command_text(exp),
                "started_at": None,
                "completed_at": None,
                "pid": None,
                "metrics": {},
                "retrieval_metrics": {},
                "error": None,
            }
            for exp in EXPERIMENTS
        },
    }


def save_state(state: dict[str, Any]) -> None:
    state["heartbeat_at"] = now_text()
    temp_path = STATE_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(STATE_PATH)


def pid_is_running(pid: int) -> bool:
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def inspect_jsonl(path: Path, expected_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "records": 0,
        "summary": None,
        "invalid_lines": 0,
        "complete": False,
    }
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8-sig") as handle:
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
    summary_samples = summary.get("num_samples") if isinstance(summary, dict) else None
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
        if value == "None":
            parsed[key.strip()] = None
            continue
        try:
            parsed[key.strip()] = float(value)
        except ValueError:
            parsed[key.strip()] = value
    return parsed


def run_metric_command(args: list[str], attempts: int = 3) -> tuple[bool, str]:
    command = [str(PYTHON), "calc_metrics.py", *args]
    last_output = ""
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        last_output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode == 0:
            return True, last_output
        log(
            f"metric command failed attempt={attempt}/{attempts} "
            f"returncode={completed.returncode}: {' '.join(command)}"
        )
        time.sleep(3)
    return False, last_output


def calculate_metrics(exp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    relative_output = str((OUTPUT_DIR / exp["output"]).relative_to(ROOT))
    ok, output = run_metric_command(
        ["--input", relative_output, "--mode", "penalty"]
    )
    if not ok:
        raise RuntimeError(f"Final penalty metrics failed:\n{output}")
    metrics = parse_key_values(output)

    retrieval: dict[str, Any] = {}
    for cutoff in (1, 3, 5, 10):
        cutoff_ok, cutoff_output = run_metric_command(
            [
                "--input",
                relative_output,
                "--mode",
                "penalty-retrieval",
                "--retrieval-k",
                str(cutoff),
            ],
            attempts=1,
        )
        if cutoff_ok:
            retrieval[str(cutoff)] = parse_key_values(cutoff_output)
        else:
            retrieval[str(cutoff)] = {
                "available": False,
                "reason": cutoff_output.splitlines()[-1]
                if cutoff_output
                else "metrics unavailable",
            }
    return metrics, retrieval


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "-"


def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def render_report(state: dict[str, Any]) -> None:
    lines = [
        "# Penalty Ablation V4 Results",
        "",
        f"- Started at: {state['sequence_started_at']}",
        f"- Heartbeat: {state['heartbeat_at']}",
        f"- Status: {state['status']}",
        f"- Output dir: `{OUTPUT_DIR.relative_to(ROOT)}`",
        "",
        "## Execution Status",
        "",
        "| # | Dataset | Experiment | Status | Progress | Started | Completed |",
        "|---:|---|---|---|---:|---|---|",
    ]

    for index, exp in enumerate(EXPERIMENTS, start=1):
        item = state["experiments"][exp["id"]]
        lines.append(
            f"| {index} | {item['dataset']} | {item['name']} | {item['status']} | "
            f"{item['progress']}/{item['expected_samples']} | "
            f"{item['started_at'] or '-'} | {item['completed_at'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Final Penalty Metrics",
            "",
            "| Dataset | Experiment | Accuracy | Macro-P | Macro-R | Macro-F1 | Eval Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    metric_found = False
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        metrics = item.get("metrics") or {}
        if not metrics:
            continue
        metric_found = True
        lines.append(
            f"| {item['dataset']} | {item['name']} | "
            f"{format_percent(metrics.get('acc'))} | "
            f"{format_percent(metrics.get('Ma-P'))} | "
            f"{format_percent(metrics.get('Ma-R'))} | "
            f"{format_percent(metrics.get('Ma-F'))} | "
            f"{format_number(metrics.get('eval_samples'))} |"
        )
    if not metric_found:
        lines.append("| - | No completed experiments | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Precedent Retrieval Metrics",
            "",
        ]
    )
    retrieval_found = False
    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]
        retrieval = item.get("retrieval_metrics") or {}
        available = [
            (cutoff, values)
            for cutoff, values in retrieval.items()
            if values.get("available", True) is not False
        ]
        if not available:
            continue
        retrieval_found = True
        lines.extend(
            [
                f"### {item['dataset']} - {item['name']}",
                "",
                "| K | Same Bucket Hit | Bucket MRR | Best Month MAE | Law Hit | Ranking Source |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for cutoff, values in available:
            lines.append(
                f"| {cutoff} | "
                f"{format_percent(values.get(f'penalty_retrieval_cls_hit@{cutoff}'))} | "
                f"{format_percent(values.get(f'penalty_retrieval_cls_mrr@{cutoff}'))} | "
                f"{format_number(values.get(f'penalty_retrieval_month_mae@{cutoff}'))} | "
                f"{format_percent(values.get(f'penalty_retrieval_article_hit@{cutoff}'))} | "
                f"{values.get(f'penalty_ranking_source@{cutoff}', '-')} |"
            )
        lines.append("")
    if not retrieval_found:
        lines.append("No completed retrieval metrics.")

    errors = [item for item in state["experiments"].values() if item.get("error")]
    if errors:
        lines.extend(["", "## Errors", ""])
        for item in errors:
            lines.append(f"- **{item['name']}**: {item['error']}")

    lines.extend(["", "## Experiment Records", ""])
    lines.append(f"- State: `{STATE_PATH.relative_to(ROOT)}`")
    lines.append(f"- Monitor log: `{LOG_PATH.relative_to(ROOT)}`")
    for exp in EXPERIMENTS:
        lines.append(
            f"- {exp['name']}: `{(OUTPUT_DIR / exp['output']).relative_to(ROOT)}`; "
            f"log `{(OUTPUT_DIR / (exp['id'] + '.process.log')).relative_to(ROOT)}`"
        )
    lines.append("")

    temp_path = REPORT_PATH.with_suffix(".md.tmp")
    temp_path.write_text("\n".join(lines), encoding="utf-8")
    temp_path.replace(REPORT_PATH)


def update_progress(
    state: dict[str, Any],
    exp: dict[str, Any],
    status: dict[str, Any],
) -> None:
    item = state["experiments"][exp["id"]]
    item["progress"] = status["records"]
    save_state(state)
    render_report(state)


def wait_for_completion(
    state: dict[str, Any],
    exp: dict[str, Any],
    pid: int,
) -> dict[str, Any]:
    output_path = OUTPUT_DIR / exp["output"]
    expected_samples = int(exp["expected_samples"])
    last_logged_progress = -1
    last_heartbeat_at = 0.0
    process_exit_seen_at: float | None = None

    while True:
        status = inspect_jsonl(output_path, expected_samples)
        alive = pid_is_running(pid)
        progress = status["records"]
        heartbeat_due = time.time() - last_heartbeat_at >= 30

        if progress != last_logged_progress and (
            progress == expected_samples
            or progress // 25 != last_logged_progress // 25
        ):
            log(
                f"{exp['id']} progress={progress}/{expected_samples} "
                f"pid={pid} alive={alive} summary={status['summary'] is not None}"
            )
            last_logged_progress = progress
            update_progress(state, exp, status)
            last_heartbeat_at = time.time()
        elif heartbeat_due:
            update_progress(state, exp, status)
            log(
                f"{exp['id']} heartbeat progress={progress}/{expected_samples} "
                f"pid={pid} alive={alive} summary={status['summary'] is not None}"
            )
            last_heartbeat_at = time.time()

        if status["complete"] and not alive:
            return status

        if not alive:
            if process_exit_seen_at is None:
                process_exit_seen_at = time.time()
                log(f"{exp['id']} process exited; entering 30-second output grace period")
            elif time.time() - process_exit_seen_at >= 30:
                raise RuntimeError(
                    "Process exited before output was complete: "
                    f"records={status['records']}, "
                    f"expected={expected_samples}, "
                    f"summary={status['summary'] is not None}, "
                    f"invalid_lines={status['invalid_lines']}"
                )
        else:
            process_exit_seen_at = None

        time.sleep(POLL_SECONDS)


def launch_experiment(exp: dict[str, Any]) -> int:
    output_path = OUTPUT_DIR / exp["output"]
    if output_path.exists() and output_path.stat().st_size > 0:
        backup = output_path.with_suffix(
            output_path.suffix + f".stale-{datetime.now():%Y%m%d-%H%M%S}"
        )
        output_path.replace(backup)
        log(f"archived stale output {output_path.name} -> {backup.name}")

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
    return process.pid


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = initial_state()
    save_state(state)
    render_report(state)
    log(f"monitor started pid={os.getpid()}")

    for exp in EXPERIMENTS:
        item = state["experiments"][exp["id"]]

        existing_status = inspect_jsonl(
            OUTPUT_DIR / exp["output"],
            int(exp["expected_samples"]),
        )
        if existing_status["complete"]:
            log(f"reusing completed output {exp['id']}")
            try:
                metrics, retrieval = calculate_metrics(exp)
                item["progress"] = existing_status["records"]
                item["metrics"] = metrics
                item["retrieval_metrics"] = retrieval
                item["status"] = "completed"
                item["started_at"] = "existing output"
                item["completed_at"] = now_text()
                save_state(state)
                render_report(state)
                continue
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
                item["completed_at"] = now_text()
                state["status"] = "stopped_on_error"
                save_state(state)
                render_report(state)
                log(f"failed existing {exp['id']}: {exc}")
                log(traceback.format_exc())
                return 1

        item["status"] = "running"
        item["started_at"] = now_text()
        pid = launch_experiment(exp)
        item["pid"] = pid
        save_state(state)
        render_report(state)

        try:
            completed_status = wait_for_completion(state, exp, pid)
            metrics, retrieval = calculate_metrics(exp)
            item["progress"] = completed_status["records"]
            item["metrics"] = metrics
            item["retrieval_metrics"] = retrieval
            item["status"] = "completed"
            item["completed_at"] = now_text()
            save_state(state)
            render_report(state)
            log(f"completed {exp['id']} metrics={json.dumps(metrics, ensure_ascii=False)}")
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
    log("all experiments completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

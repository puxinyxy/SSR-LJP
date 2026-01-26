from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def setup_run_logger(
    run_name: str,
    out_root: str = "output/logs",
    args: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> Tuple[logging.Logger, Path, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(out_root) / run_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    meta = {
        "run_id": run_id,
        "run_name": run_name,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "args": args or {},
        "extra": extra or {},
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("run_start run_id=%s out_dir=%s", run_id, str(out_dir))
    return logger, out_dir, run_id


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], prefix: str = "metrics") -> None:
    if not metrics:
        return
    logger.info("%s %s", prefix, json.dumps(metrics, ensure_ascii=False))

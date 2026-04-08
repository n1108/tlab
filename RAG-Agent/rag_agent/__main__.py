"""
CLI entry: run RAG-Agent on dataset/input.json (or a single UUID).

Usage (from repository root containing `dataset/`):
  PYTHONPATH=RAG-Agent python -m rag_agent --dataset-root dataset --limit 1

Or from inside RAG-Agent:
  PYTHONPATH=. python -m rag_agent --dataset-root ../dataset --limit 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# RAG-Agent root = parent of package `rag_agent`
_RAG_ROOT = Path(__file__).resolve().parent.parent
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from rag_agent.bundled.agent.judge import JudgeAgent
from rag_agent.bundled.utils.log import setup_logger
from rag_agent.orchestrator import run_rag_case


def _default_dataset_root() -> Path:
    # RAG-Agent/ -> tlab/ -> sibling dataset at tlab/dataset
    tlab = _RAG_ROOT.parent
    return tlab / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Agent: tool-calling RCA (self-contained)")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(_default_dataset_root()),
        help="Path to dataset folder (contains input.json and metric-parquet paths as MetricAgent expects)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="input.json path (default: <dataset-root>/input.json)",
    )
    parser.add_argument("--output", type=str, default="", help="Output JSONL path")
    parser.add_argument("--uuid", type=str, default="", help="Process only this UUID")
    parser.add_argument("--limit", type=int, default=0, help="Max cases (0 = all)")
    parser.add_argument("--max-turns", type=int, default=6, help="Max LLM turns for tool loop")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel case workers")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--llm-provider", type=str, default="yuzo")
    parser.add_argument("--llm-model", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-api-url", type=str, default="")

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    input_path = Path(args.input) if args.input else dataset_root / "input.json"
    if not input_path.is_file():
        raise SystemExit(f"input.json not found: {input_path}")

    out_path = Path(args.output) if args.output else _RAG_ROOT / "output" / "rag_agent_run.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setup_logger(str(_RAG_ROOT / "output" / "rag_agent.log"), args.log_level)
    log = logging.getLogger(__name__)
    log.info("dataset_root=%s input=%s output=%s", dataset_root, input_path, out_path)

    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise SystemExit("input.json must be a JSON array")

    if args.uuid:
        cases = [c for c in cases if str(c.get("uuid", "")) == args.uuid]
        if not cases:
            raise SystemExit(f"No case with uuid={args.uuid}")

    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    judge = JudgeAgent(
        args.llm_api_key or None,
        args.llm_api_url or None,
        provider=args.llm_provider,
        model=args.llm_model or None,
    )
    if not judge.api_key:
        raise SystemExit(
            "No LLM API key. Set YUZO_API_KEY (or pass --llm-api-key). "
            "Without it, the tool loop cannot run."
        )

    root_str = str(dataset_root)
    written = 0
    workers = max(1, int(args.max_workers))
    with open(out_path, "w", encoding="utf-8") as out:
        if workers == 1:
            for item in cases:
                uuid = str(item.get("uuid", ""))
                desc = str(item.get("Anomaly Description", ""))
                log.info("RAG case uuid=%s", uuid)
                res = run_rag_case(
                    uuid,
                    desc,
                    root_str,
                    judge,
                    max_turns=int(args.max_turns),
                )
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
        else:
            log.info("Running %s cases with %s workers", len(cases), workers)
            indexed_cases = list(enumerate(cases))
            pending: dict[int, dict] = {}
            next_to_write = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        run_rag_case,
                        str(item.get("uuid", "")),
                        str(item.get("Anomaly Description", "")),
                        root_str,
                        judge,
                        int(args.max_turns),
                    ): idx
                    for idx, item in indexed_cases
                }
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        pending[idx] = fut.result()
                    except Exception as e:
                        case = indexed_cases[idx][1]
                        pending[idx] = {
                            "uuid": str(case.get("uuid", "")),
                            "component": "unknown",
                            "reason": f"case failed: {e}",
                            "reasoning_trace": [],
                            "rag_meta": {"mode": "case_error"},
                        }
                    while next_to_write in pending:
                        out.write(json.dumps(pending.pop(next_to_write), ensure_ascii=False) + "\n")
                        out.flush()
                        written += 1
                        next_to_write += 1

    log.info("Wrote %s cases to %s", written, out_path)


if __name__ == "__main__":
    main()

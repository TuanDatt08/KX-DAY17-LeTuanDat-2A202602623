from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _coverage(answer: str, expected: list[str]) -> float:
    lowered = answer.lower()
    return sum(e.lower() in lowered for e in expected) / len(expected) if expected else 1.0


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    coverage = _coverage(answer, expected)
    return 1.0 if coverage == 1 else 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality proxy in [0, 1]: mostly correctness, plus the user's stated style
    (concise, bullet structure). ponytail: swap for the judge model in live mode."""

    concise = len(answer) <= 400
    structured = answer.lstrip().startswith("- ")
    return round(0.7 * _coverage(answer, expected) + 0.15 * concise + 0.15 * structured, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed every conversation, then ask its recall questions in a FRESH thread."""

    user_ids = {c["user_id"] for c in conversations}
    size = getattr(agent, "memory_file_size", lambda _user: 0)
    start_bytes = sum(size(u) for u in user_ids)

    threads: list[str] = []
    recalls: list[float] = []
    qualities: list[float] = []
    for conv in conversations:
        thread_id = f"{agent_name}-{conv['id']}"
        threads.append(thread_id)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)

        for i, q in enumerate(conv["recall_questions"]):
            recall_thread = f"{thread_id}-recall-{i}"
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, q["question"])["response"]
            recalls.append(recall_points(answer, q["expected_contains"]))
            qualities.append(heuristic_quality(answer, q["expected_contains"]))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=sum(recalls) / len(recalls),
        response_quality=sum(qualities) / len(qualities),
        memory_growth_bytes=sum(size(u) for u in user_ids) - start_bytes,
        compactions=sum(agent.compaction_count(t) for t in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.0%}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def run_suite(title: str, dataset: Path, config) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config), conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config), conversations, config),
    ]
    base, adv = rows
    saved = 1 - adv.prompt_tokens_processed / base.prompt_tokens_processed
    print(f"\n## {title}  ({dataset.name}, {len(conversations)} conversation(s))\n")
    print(format_rows(rows))
    print(f"\nPrompt tokens: Advanced vs Baseline = {-saved:+.1%}")
    return rows


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    # Benchmarks write into their own state dir, wiped each run, so leftover User.md
    # from earlier runs can neither inflate recall nor hide memory growth.
    bench_state = config.state_dir / "benchmark"
    shutil.rmtree(bench_state, ignore_errors=True)
    config = replace(config, state_dir=bench_state)

    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config)
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config)


if __name__ == "__main__":
    main()

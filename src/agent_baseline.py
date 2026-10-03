from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, has_credentials


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    # Facts heard in THIS thread only; thrown away with the thread.
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Remembers what was said in the same `thread_id` (full history goes into every prompt)
    - No persistent `User.md`
    - A new thread starts from zero, so long-term facts are forgotten
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the reply plus token accounting. Live LLM if available, else offline."""

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _record_turn(self, session: SessionState, message: str, response: str) -> dict[str, Any]:
        """Shared accounting: the prompt is the WHOLE thread history (that is the baseline's cost)."""

        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "compactions": 0,
        }

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic reply that can only use facts heard in this same thread."""

        session = self._session(thread_id)
        session.facts.update(extract_profile_updates(message))
        return self._record_turn(session, message, answer_from_facts(message, session.facts))

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._record_turn(self._session(thread_id), message, result["messages"][-1].text)

    def _maybe_build_langchain_agent(self):
        """`create_agent` + `InMemorySaver`: thread-scoped memory only, no tools, no profile."""

        if not has_credentials(self.config.model):
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            model=build_chat_model(self.config.model),
            system_prompt="Bạn là trợ lý hữu ích. Trả lời ngắn gọn bằng tiếng Việt.",
            checkpointer=InMemorySaver(),
        )

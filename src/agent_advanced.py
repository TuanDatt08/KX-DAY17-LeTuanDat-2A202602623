from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model, has_credentials

try:  # live-mode deps; module level because tool type hints are resolved from module globals
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
    from langchain.tools import ToolRuntime, tool
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    create_agent = None


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent with three memory layers:

    1. within-session memory  -> recent messages kept in `CompactMemoryManager`
    2. persistent `User.md`   -> `UserProfileStore`, survives new threads
    3. compact memory         -> older messages folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._remember(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        return self._record_turn(user_id, thread_id, message, self._offline_response(user_id, thread_id, message), prompt_tokens)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._remember(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id))),
        )
        return self._record_turn(user_id, thread_id, message, result["messages"][-1].text, prompt_tokens)

    def _remember(self, user_id: str, thread_id: str, message: str) -> None:
        """Steps 1-3: extract stable facts -> persist to User.md -> append to short-term memory.

        Runs in live mode too, as a deterministic safety net if the LLM skips the save tool.
        """

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)

    def _record_turn(self, user_id: str, thread_id: str, message: str, response: str, prompt_tokens: int) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(message) + estimate_tokens(response)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "compactions": self.compaction_count(thread_id),
            "memory_file_size": self.memory_file_size(user_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Prompt = User.md + compact summary + recent kept messages (not the whole thread)."""

        ctx = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(ctx["summary"])
            + sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Same answer logic as baseline, but facts come from persistent User.md."""

        return answer_from_facts(message, self.profile_store.facts(user_id))

    def _maybe_build_langchain_agent(self):
        """Live agent: provider model + InMemorySaver + User.md tools + profile-injecting
        dynamic prompt + summarization middleware for long threads."""

        if create_agent is None or not has_credentials(self.config.model):
            return None

        store = self.profile_store

        @tool
        def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the persistent User.md profile of the current user."""
            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Save or correct ONE stable fact about the user in User.md (e.g. key='location').
            Only call this for facts the user states about themselves, never for questions,
            jokes or temporary context."""
            changed = store.upsert_fact(runtime.context.user_id, key, value)
            return "saved" if changed else "unchanged"

        @dynamic_prompt
        def profile_prompt(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id)
            return (
                "Bạn là trợ lý có trí nhớ dài hạn. Hồ sơ người dùng (User.md):\n"
                f"{profile}\n"
                "Luôn ưu tiên fact mới nhất trong hồ sơ. Khi người dùng cung cấp hoặc đính chính "
                "fact ổn định, gọi save_user_fact. Trả lời theo style trong hồ sơ."
            )

        model = build_chat_model(self.config.model)
        return create_agent(
            model=model,
            tools=[read_user_profile, save_user_fact],
            middleware=[
                profile_prompt,
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )

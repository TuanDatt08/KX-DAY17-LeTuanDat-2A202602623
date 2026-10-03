from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates

# ~200 tokens per turn, like the stress dataset.
LONG_TURN = "Mình kể thêm về tin Artemis III, X-59, WMO và BC energy plan để tạo áp lực ngữ cảnh. " * 10


def make_config(tmp_path: Path):
    """Isolated state dir + tiny compact threshold so compaction happens quickly."""

    return replace(
        load_config(),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=300,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    assert store.file_size("u1") == 0
    assert store.read_text("u1").startswith("# User Profile: u1")

    path = store.write_text("u1", "# User Profile: u1\n\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert store.facts("u1") == {"location": "Đà Nẵng"}

    assert store.edit_text("u1", "Đà Nẵng", "Huế") is True
    assert store.edit_text("u1", "không tồn tại", "x") is False
    assert store.facts("u1")["location"] == "Huế"

    # Correction replaces the old fact instead of keeping both.
    store.upsert_fact("u1", "location", "Đà Nẵng")
    text = store.read_text("u1")
    assert "Đà Nẵng" in text and "Huế" not in text
    assert store.file_size("u1") == len(text.encode("utf-8"))


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)

    agent.reply("u1", "t1", "Chào bạn, mình tên là An.")
    assert agent.compaction_count("t1") == 0  # short threads do not compact

    for _ in range(6):
        agent.reply("u1", "t1", LONG_TURN)

    ctx = agent.compact_memory.context("t1")
    assert agent.compaction_count("t1") >= 2
    assert ctx["summary"]
    assert len(ctx["messages"]) < 13  # full thread is 13 messages; older ones were folded away


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    session_1 = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer.",
        "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
        "Hà Nội chỉ là nơi mình bay ra họp. Có lúc mình đùa là chuyển sang product manager.",
    ]
    for msg in session_1:
        baseline.reply("dungct", "s1", msg)
        advanced.reply("dungct", "s1", msg)

    question = "Nhắc lại tên, nơi ở hiện tại và nghề nghiệp hiện tại của mình."
    adv_answer = advanced.reply("dungct", "s2", question)["response"]
    base_answer = baseline.reply("dungct", "s2", question)["response"]

    for fact in ("DũngCT", "Huế", "MLOps engineer"):
        assert fact in adv_answer
    for stale_or_noise in ("Đà Nẵng", "backend", "Hà Nội", "product manager"):
        assert stale_or_noise not in adv_answer
    assert "DũngCT" not in base_answer  # baseline must forget across threads

    # A brand-new agent instance still recalls: memory is on disk, not in RAM.
    fresh = AdvancedAgent(config, force_offline=True)
    assert "Huế" in fresh.reply("dungct", "s3", question)["response"]


def test_extractor_ignores_questions_noise_and_negation() -> None:
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Có lúc mình đùa là chuyển sang product manager.") == {}
    assert extract_profile_updates("Giờ mình đang ở Huế chứ không còn ở Đà Nẵng.") == {"location": "Huế"}
    assert extract_profile_updates("Mình chuyển từ Huế sang Đà Nẵng.")["location"] == "Đà Nẵng"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for _ in range(12):
        baseline.reply("u1", "long", LONG_TURN)
        advanced.reply("u1", "long", LONG_TURN)

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < 0.6 * baseline.prompt_token_usage("long")
    # Compaction saves prompt context, not the tokens of the conversation itself.
    assert abs(advanced.token_usage("long") - baseline.token_usage("long")) <= 0.05 * baseline.token_usage("long")

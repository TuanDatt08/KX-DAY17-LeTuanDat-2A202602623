from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token count: ~4 characters per token, 0 for empty text.

    Not tokenizer-exact, but stable and identical for both agents, which is all
    the offline benchmark needs.
    """

    text = (text or "").strip()
    return math.ceil(len(text) / 4) if text else 0


# Keys whose values accumulate (comma-separated traits) instead of being replaced.
MERGE_KEYS = {"response_style", "interests"}
FACT_LINE = re.compile(r"^- (\w+): (.*)$", re.MULTILINE)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` at `<root_dir>/<user_id>/User.md`.

    Facts are stored as `- key: value` lines so the file stays human-readable
    and easy to parse back.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w.-]", "_", user_id.strip()) or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": no CRLF on Windows, so byte sizes are identical across OSes.
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(FACT_LINE.findall(self.read_text(user_id)))

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one fact. Replacing (not appending) is the conflict
        handling: a correction overwrites the stale fact instead of coexisting with it.
        Returns True if the file changed."""

        old = self.facts(user_id).get(key)
        if old is not None and key in MERGE_KEYS:
            traits = [t.strip() for t in f"{old}, {value}".split(",") if t.strip()]
            value = ", ".join(dict.fromkeys(traits))  # ordered de-dup
        if old == value:
            return False
        if old is None:
            content = self.read_text(user_id)
            if not content.endswith("\n"):
                content += "\n"
            self.write_text(user_id, content + f"- {key}: {value}\n")
        else:
            self.edit_text(user_id, f"- {key}: {old}\n", f"- {key}: {value}\n")
        return True


# --- Profile extraction -------------------------------------------------------
# ponytail: closed vocabularies + regex. Fine for this Vietnamese dataset; swap for
# LLM / NER-based extraction when inputs become open-domain.

CITIES = ["Hà Nội", "Huế", "Đà Nẵng", "Hồ Chí Minh", "Sài Gòn", "Hải Phòng", "Cần Thơ", "Nha Trang", "Đà Lạt"]
CITY_RE = "|".join(map(re.escape, CITIES))

NAME_RE = re.compile(r"tên (?:mình |tôi )?là ([^\W\d_]\w*(?: [A-ZĐ]\w*)*)")
LOCATION_RE = re.compile(rf"(?:\bở|sang|là) ({CITY_RE})")
PROFESSION_RE = re.compile(r"(?:làm|là|sang) (\w+ (?:engineer|manager|scientist|developer))")
DRINK_RE = re.compile(r"[Đđ]ồ uống yêu thích là ([^.,!?]+)")
FOOD_RE = re.compile(r"[Mm]ón ăn yêu thích là ([^.,!?]+)")
PET_RE = re.compile(r"nuôi (?:một )?(?:bé |con )?(\w+) tên (\w+)")

STYLE_TRAITS = ["ngắn gọn", "3 bullet", "bullet", "ví dụ thực tế", "ví dụ thực chiến", "có cấu trúc", "trade-off"]
INTEREST_TRAITS = ["Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "memory"]

# A sentence containing one of these is talking *about* a fact, not stating it.
NOISE_MARKERS = ("đùa", "ví dụ cũ")
NEGATION_WINDOW = 12  # chars before a match checked for "không" ("không còn ở Đà Nẵng")


def _split_sentences(message: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", message) if s.strip()]


def _last_affirmed(pattern: re.Pattern[str], sentence: str) -> str | None:
    """Last match in the sentence that is not negated right before it."""

    value = None
    for m in pattern.finditer(sentence):
        if "không" not in sentence[max(0, m.start() - NEGATION_WINDOW): m.start()]:
            value = m.group(1)
    return value


def _traits(sentence: str, vocab: list[str]) -> list[str]:
    return [t for t in vocab if t in sentence]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from one user message.

    Guardrails (the cheap version of a confidence threshold):
    - questions ("...?") never write facts
    - sentences flagged as jokes / old examples are skipped
    - negated mentions ("không còn ở Đà Nẵng") are ignored
    - within a message, the latest affirmed value wins ("từ Huế sang Đà Nẵng")
    """

    facts: dict[str, str] = {}
    style: list[str] = []
    interests: list[str] = []
    for sentence in _split_sentences(message):
        if sentence.endswith("?") or any(marker in sentence for marker in NOISE_MARKERS):
            continue

        for key, pattern in (
            ("name", NAME_RE),
            ("location", LOCATION_RE),
            ("profession", PROFESSION_RE),
            ("favorite_drink", DRINK_RE),
            ("favorite_food", FOOD_RE),
        ):
            value = _last_affirmed(pattern, sentence)
            if value:
                facts[key] = value.strip()

        if pet := PET_RE.search(sentence):
            facts["pet"] = f"{pet.group(1)} tên {pet.group(2)}"

        lowered = sentence.lower()
        if any(w in lowered for w in ("trả lời", "giải thích", "style")):
            style += _traits(sentence, STYLE_TRAITS)
        if any(w in lowered for w in ("thích", "quan tâm", "đang học")):
            interests += _traits(sentence, INTEREST_TRAITS)

    if style:
        facts["response_style"] = ", ".join(dict.fromkeys(style))
    if interests:
        facts["interests"] = ", ".join(dict.fromkeys(interests))
    return facts


# --- Compact memory -----------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: first sentence (<=120 chars) of the last `max_items` user messages.

    Assistant turns are skipped: in offline mode they only echo the user.
    ponytail: lossy truncation summary; replace with an LLM summarizer for live mode.
    Stable facts survive anyway because they live in `User.md`, not in the summary.
    """

    lines = []
    for msg in [m for m in messages if m["role"] == "user"][-max_items:]:
        first = _split_sentences(msg["content"])[:1] or [""]
        lines.append(f"- {msg['role']}: {first[0][:120]}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Keep recent messages in full; fold older ones into a bounded summary
    whenever the kept messages exceed `threshold_tokens`."""

    threshold_tokens: int
    keep_messages: int
    max_summary_lines: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self._message_tokens(thread) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])

    @staticmethod
    def _message_tokens(thread: dict[str, object]) -> int:
        return sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def _compact(self, thread: dict[str, object]) -> None:
        old = thread["messages"][: -self.keep_messages]
        thread["messages"] = thread["messages"][-self.keep_messages:]
        lines = thread["summary"].splitlines() + summarize_messages(old, max_items=len(old)).splitlines()
        thread["summary"] = "\n".join(lines[-self.max_summary_lines:])
        thread["compactions"] += 1


# --- Deterministic offline answering (shared by both agents) -----------------
# Both agents use the same answer logic; they differ only in WHERE facts come from
# (baseline: current thread only; advanced: User.md). That keeps the comparison fair.

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
}
QUESTION_KEYWORDS = {
    "name": ("tên", "là ai", "tóm tắt"),
    "location": ("ở đâu", "nơi ở", "còn ở"),
    "profession": ("nghề", "tóm tắt", "là ai"),
    "response_style": ("style", "kiểu trả lời"),
    "favorite_drink": ("đồ uống",),
    "favorite_food": ("món ăn",),
    "pet": ("nuôi", "con gì"),
    "interests": ("quan tâm", "là ai", "tóm tắt"),
}
RECALL_MARKERS = ("nhắc lại", "là gì", "tóm tắt", "là ai")


def is_recall_request(message: str) -> bool:
    lowered = message.lower()
    return message.rstrip().endswith("?") or any(m in lowered for m in RECALL_MARKERS)


def answer_from_facts(message: str, facts: dict[str, str]) -> str:
    """Short bullet answer for recall questions, short acknowledgement otherwise."""

    if not is_recall_request(message):
        return "Đã ghi nhận."
    lowered = message.lower()
    asked = [k for k, words in QUESTION_KEYWORDS.items() if any(w in lowered for w in words)]
    known = [k for k in asked if k in facts]
    if not known:
        return "Mình chưa có thông tin này trong phiên hiện tại."
    return "\n".join(f"- {FACT_LABELS[k]}: {facts[k]}" for k in known)

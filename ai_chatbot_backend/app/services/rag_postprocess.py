# Standard python libraries
import time
import json
from dataclasses import dataclass, asdict, field
from typing import Any, List, Optional
# Third-party libraries
from openai import OpenAI, AsyncOpenAI
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
# Local libraries
from app.core.models.chat_completion import Message
from app.config import settings
import re


# Environment variables
MEMORY_SYNOPSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "user_goals": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["focus", "user_goals", "constraints", "key_entities",
                 "artifacts", "open_questions", "action_items", "decisions"],
    "additionalProperties": False
}

LTM_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge_profile": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "course_id": {"type": "string"},
                    "weak_topics": {"type": "string"}
                },
                "required": ["course_id", "weak_topics"],
                "additionalProperties": False
            }
        },        
        "current_focus": {
            "type": "object",
            "properties": {
                "long_term_goals": {"type": "array", "items": {"type": "string"}},
                "key_entries_of_interest": {"type": "array", "items": {"type": "string"}},
                "persistent_open_questions": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "learning_preferences": {
            "type": "object",
            "properties": {
                "preferred_topics_and_interests": {"type": "array", "items": {"type": "string"}},
                "preferred_learning_style": {"type": "string"},
                "preferred_file_types": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "user_profile": {"type": "string"},
    },
    "additionalProperties": False
}

_LLM_SYSTEM_LTM = (
    "You are a Long-Term Memory (LTM) Synthesis Agent. Your goal is to evolve a student's cognitive model "
    "by merging EXISTING_LTM, NEW_STM, and CHAT_HISTORY into a refined JSON. Prioritize behavioral "
    "evidence and sentiment over static history.\n\n"

    "CORE SYNTHESIS RULES:\n"
    "1. Hierarchy of Evidence: USER input (Sentiment/Behavior) > NEW_STM (Session Facts) > EXISTING_LTM (History).\n"
    "2. Knowledge Retention: Keep 'weak_topics' until the user demonstrates independent, scaffold-free mastery.\n\n"

    "JSON SCHEMA GUIDELINES:\n"
    "- knowledge_profile: { course_id: str, weak_topics: [str] }. \n"
    "  Logic: Add topics if user expresses confusion, asks unresolved questions, or shows 'worse' signals "
    "(avoidance, fear/anxiety, or inability to generalize). Remove ONLY when user demonstrates scaffold-free mastery.\n"
    
    "- current_focus: { long_term_goals: [str], key_entries_of_interest: [str], persistent_open_questions: [str] }. \n"
    "  Logic: 'long_term_goals' are high-level/abstract (e.g., 'Mastering Backend Dev'); update only via explicit "
    "user input or strong repeated patterns. 'key_entries_of_interest' are the immediate active concepts from "
    "this session. 'persistent_open_questions' tracks open questions or conceptual gaps; remove only upon clear evidence of resolution.\n"
    
    "- learning_preferences: { preferred_topics_and_interests: [str], preferred_learning_style: str, preferred_file_types: [str] }. \n"
    "  Logic: 'preferred_topics_and_interests' captures broad affinities—including hobbies, recurring objects, "
    "or themes mentioned (e.g., 'e-commerce', 'gaming', 'fast-paced'). 'preferred_learning_style' is a behavioral "
    "inference (e.g., 'prefers real world examples, prefers code-first shortcuts').\n"
    
    "- user_profile: A 1-sentence behavioral snapshot. \n"
    "  Logic: Summarize the user’s current mindset, background, and learning stance (e.g., 'A low-effort "
    "beginner seeking practical shortcuts to bypass complex logic like recursion').\n\n"

    "CONSTRAINT: Return ONLY the updated JSON. Be concise. Avoid redundant array items."
)

def _truncate_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens using the provided tokenizer.
    If no tokenizer is available, fallback to character-based truncation.
    """
    if not tokenizer:
        return text[:max_tokens * 4]  # Rough estimate: 1 token ~= 4 chars
    
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])
    except Exception:
        return text[:max_tokens * 4]


@dataclass
class MemorySynopsis:
    """
    Compact, structured memory about a conversation. Keep it small & stable.
    """
    focus: str = ""                                             # What is the main topic / intent of the user?
    user_goals: List[str] = field(default_factory=list)         # User's explicit goals/preferences
    constraints: List[str] = field(default_factory=list)        # Hard constraints (versions, dates, scope, etc.)
    key_entities: List[str] = field(default_factory=list)       # People, products, datasets, repos, courses…
    artifacts: List[str] = field(default_factory=list)          # Files, URLs, IDs, paths mentioned
    open_questions: List[str] = field(default_factory=list)     # Unresolved questions the user asked
    action_items: List[str] = field(default_factory=list)       # TODOs, "next steps"
    decisions: List[str] = field(default_factory=list)          # Agreed choices so far

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "MemorySynopsis":
        data = json.loads(s or "{}")
        return MemorySynopsis(**{k: data.get(k, v) for k, v in asdict(MemorySynopsis()).items()})


from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json

# Helper structure for the 'knowledge_profile' array items
@dataclass
class KnowledgeProfileItem:
    """Represents a specific course and the user's weak topics within it."""
    course_id: str = ""
    weak_topics: str = ""

@dataclass
class LearningPreferences:
    preferred_topics_and_interests: List[str] = field(default_factory=list)
    preferred_learning_style: str = ""
    preferred_file_types: List[str] = field(default_factory=list)

@dataclass
class CurrentFocus:
    long_term_goals: List[str] = field(default_factory=list)
    key_entries_of_interest: List[str] = field(default_factory=list)
    persistent_open_questions: List[str] = field(default_factory=list)

@dataclass
class MemorySynopsisLong:
    """
    Structured memory containing the user's long-term learning profile (LTM_SCHEMA).
    """
    # knowledge_profile is a list of KnowledgeProfileItem objects
    knowledge_profile: List[KnowledgeProfileItem] = field(default_factory=list)
    current_focus: CurrentFocus = field(default_factory=CurrentFocus)
    learning_preferences: LearningPreferences = field(default_factory=LearningPreferences)
    user_profile: str = ""

    def to_json(self) -> str:
        """Converts the dataclass instance to a JSON string."""
        # asdict recursively converts nested dataclasses to dicts
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "MemorySynopsisLong":
        """Creates a MemorySynopsisLong instance from a JSON string."""
        data: Dict[str, Any] = json.loads(s or "{}")
        
        # Deserialize knowledge_profile
        knowledge_profiles = []
        knowledge_profiles_data = data.get("knowledge_profile", [])
        if isinstance(knowledge_profiles_data, dict):
             knowledge_profiles_data = [knowledge_profiles_data]
        if isinstance(knowledge_profiles_data, list):
            for item in knowledge_profiles_data:
                if isinstance(item, dict):
                    knowledge_profiles.append(KnowledgeProfileItem(**item))
        
        # Deserialize CurrentFocus
        cf_data = data.get("current_focus", {})
        if not isinstance(cf_data, dict): cf_data = {}
        # Filter keys to match dataclass fields
        valid_cf_keys = {f.name for f in field(default_factory=CurrentFocus).type.__dataclass_fields__.values()} if hasattr('', 'field') else CurrentFocus.__dataclass_fields__.keys()
        current_focus = CurrentFocus(**{k: v for k, v in cf_data.items() if k in CurrentFocus.__dataclass_fields__})

        # Deserialize LearningPreferences
        lp_data = data.get("learning_preferences", {})
        if not isinstance(lp_data, dict): lp_data = {}
        learning_preferences = LearningPreferences(**{k: v for k, v in lp_data.items() if k in LearningPreferences.__dataclass_fields__})

        return MemorySynopsisLong(
            knowledge_profile=knowledge_profiles,
            current_focus=current_focus,
            learning_preferences=learning_preferences,
            user_profile=data.get("user_profile", "")
        )


async def build_memory_synopsis(
        messages: List[Message],
        engine: Any,
        prev_synopsis: Optional[MemorySynopsis] = None,
        chat_history_sid: Optional[str] = None,
        max_prompt_tokens: int = 3500,
) -> MemorySynopsis:
    """
    Create/refresh the rolling MemorySynopsis.
    - messages: full chat so far (system/assistant/user)
    - prev_synopsis: prior memory to carry forward (we'll merge)
    - chat_history_sid: if provided, retrieves previous memory from MongoDB
    """
    # Graceful MongoDB retrieval for previous memory
    if chat_history_sid and not prev_synopsis:
        try:
            from app.services.memory_synopsis_service import MemorySynopsisService
            memory_service = MemorySynopsisService()
            prev_synopsis = await memory_service.get_by_chat_history_sid(chat_history_sid)
        except Exception as e:
            print(f"[INFO] Failed to retrieve previous memory, generating from scratch: {e}")
            prev_synopsis = None  # Continue without previous memory

    transcript = _render_transcript(messages)
    cur = await _llm_synopsis_from_transcript(engine, transcript, max_prompt_tokens=max_prompt_tokens)
    if prev_synopsis:
        cur = await _llm_merge_synopses(engine, prev_synopsis, cur)

    # tighten fields (keep stable, short)
    cur.focus = _truncate_sentence(cur.focus, 180)
    cur.user_goals = cur.user_goals[:8]
    cur.constraints = cur.constraints[:8]
    cur.key_entities = cur.key_entities[:16]
    cur.artifacts = cur.artifacts[:16]
    cur.open_questions = cur.open_questions[:8]
    cur.action_items = cur.action_items[:8]
    cur.decisions = cur.decisions[:8]
    return cur


async def _llm_synthesize_ltm(
        engine: Any,
        tokenizer: Any,
        new_stm: MemorySynopsis,
        prev_ltm: Optional[MemorySynopsisLong],
        transcript: str,
        course_code: str,
        max_prompt_tokens: int = 3500,
) -> MemorySynopsisLong:
    """
    Using local engine to synthesize Long-Term Memory (LTM) JSON from transcript, STM, and optional existing LTM.
    """
    # Prepare STM, LTM and transcript inputs
    existing_ltm_json = prev_ltm.to_json() if prev_ltm else "{}"
    new_stm_json = new_stm.to_json()

    # LTM_SYSTEM data structure
    LLM_USER_LTM_TEMPLATE = """
COURSE_CODE:
{course_code}

EXISTING_LTM:
{existing_ltm_json}

NEW_STM:
{new_stm_json}

TRANSCRIPT:
{transcript}

Task:
Produce the single best updated LTM JSON object following the system rules. Return ONLY JSON.
"""

    # Prepare system and user messages for LLM
    sys_msg = {"role": "system", "content": _LLM_SYSTEM_LTM}
    usr_content = LLM_USER_LTM_TEMPLATE.format(
        course_code=course_code,
        existing_ltm_json=existing_ltm_json,
        new_stm_json=new_stm_json,
        transcript=_truncate_to_tokens(tokenizer, transcript, max_prompt_tokens)
    )
    usr = {"role": "user", "content": usr_content}
    chat = [sys_msg, usr]
    response = await engine.chat.completions.create(
        model=settings.vllm_chat_model,
        messages=chat,
        temperature=0.0,
        top_p=1.0,
        extra_body={"guided_json": LTM_SCHEMA}
    )
    text = response.choices[0].message.content or "{}"
    print("Generated LTM JSON:", text)

    try:
        # Using MemorySynopsisLong.from_json method to build a MemorySynopsisLong instance
        return MemorySynopsisLong.from_json(text.strip())
    except Exception as e:
        print(f'Failed to parse generated MemorySynopsisLong JSON: {e} | Text: {text}')
        # If parsing fails, return an empty LTM instance
        return MemorySynopsisLong()

# --- Main function：build LTM ---

async def build_memory_synopsis_long(
        messages: List[Message],
        tokenizer: Any,
        engine: Any,
        new_stm: MemorySynopsis, # 新增：必须是已生成的 Short-Term Memory
        course_code: str,
        prev_synopsis_long: Optional[MemorySynopsisLong] = None, # LTM 历史
        chat_history_sid: Optional[str] = None,
        max_prompt_tokens: int = 3500,
) -> MemorySynopsisLong:
    """
    Based on chat history, newly generated STM, and optional existing LTM, create/update Long-Term Memory (LTM).
    - messages: full chat history (for context reasoning)
    - new_stm: recently generated MemorySynopsis (Short-Term Memory)
    - prev_synopsis_long: prior LTM instance (MemorySynopsisLong)
    - chat_history_sid: optional, if provided, attempts to fetch prev_synopsis_long from DB
    """

    # 1. Graceful MongoDB retrieval for previous LTM
    if chat_history_sid and not prev_synopsis_long:
        print("[INFO] Attempting to retrieve previous memory from DB...")
        try:
            from app.services.memory_synopsis_service import MemorySynopsisServiceLong
            memory_service = MemorySynopsisServiceLong()
            prev_synopsis_long = await memory_service.get_by_user_id(chat_history_sid)
        except Exception as e:
            print(f"[INFO] Failed to retrieve previous memory, generating from scratch: {e}")
            prev_synopsis_long = None  # Continue without previous memory

    # 2. render transcript
    transcript = _render_transcript(messages)
    # 3. use LLM to synthesize updated LTM
    updated_ltm = await _llm_synthesize_ltm(
        engine,
        tokenizer,
        new_stm,
        prev_synopsis_long,
        transcript,
        course_code=course_code,
        max_prompt_tokens=max_prompt_tokens,
    )
    return updated_ltm


_LLM_SYSTEM = (
    "You are a memory-synopsis compressor. "
    "Given a chat transcript, produce a STRUCTURED JSON with keys: "
    "focus (string), user_goals (list[str]), constraints (list[str]), key_entities (list[str]), "
    "artifacts (list[str]), open_questions (list[str]), action_items (list[str]), decisions (list[str]). "
    "\nHere's some description of each key: \n"
    "focus - what is the main topic / intent of the user? \n"
    "user_goals - user's explicit goals/preferences. \n"
    "constraints - hard constraints (versions, dates, scope, etc.). \n"
    "key_entities - people, products, datasets, repos, courses… \n"
    "artifacts - files, URLs, IDs, paths mentioned. \n"
    "open_questions - unresolved questions the user asked. \n"
    "action_items - TODOs, “next steps”. \n"
    "decisions - agreed choices so far. \n"
    "\nRules:\n"
    "- Return ONLY a single JSON object that matches the schema keys and types above.\n"
    "- Keep text terse and factual. No markdown, no code fences, no extra commentary.\n"
    "- Arrays must contain strings only; deduplicate items; remove empty strings.\n"
    "- Extract explicit constraints (versions, dates, scope limits) as strings.\n"
)


_LLM_USER_TEMPLATE = """Transcript:
{transcript}

Requirements:
- Summarize tersely.
- Deduplicate entities and URLs/paths.
- Extract explicit constraints (versions, dates, scope limits).
Return ONLY JSON.
"""


async def _llm_synopsis_from_transcript(
        engine: Any,
        transcript: str,
        max_prompt_tokens: int = 3500,
) -> MemorySynopsis:
    """
    Use vLLM server to compress the transcript into MemorySynopsis JSON.
    """
    # Check if engine is OpenAI client
    if not isinstance(engine, (OpenAI, AsyncOpenAI)):
        # Fallback for non-OpenAI engines
        return MemorySynopsis()

    # Truncate transcript if needed (rough estimate: 1 token ~= 4 chars)
    max_chars = max_prompt_tokens * 4
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]

    # Prepare the system and user messages for the LLM
    sys_msg = {"role": "system", "content": _LLM_SYSTEM}
    usr = {
        "role": "user",
        "content": _LLM_USER_TEMPLATE.format(transcript=transcript)
    }
    chat = [sys_msg, usr]

    # Generate the synopsis using the OpenAI API with JSON mode
    response = await engine.chat.completions.create(
        model=settings.vllm_chat_model,
        messages=chat,
        temperature=0.0,
        top_p=1.0,
        extra_body={"guided_json": MEMORY_SYNOPSIS_JSON_SCHEMA}
    )

    # vLLM with --reasoning-parser separates reasoning_content from content
    # Use content directly (final response without thinking)
    text = response.choices[0].message.content or "{}"
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        print('Failed to parse merged MemorySynopsis JSON:', text)
        return MemorySynopsis()
    return MemorySynopsis(**data)


def _render_transcript(messages: List[Message], max_chars: int = 12000) -> str:
    """
    Linearized transcript with role tags. Keep it simple for robustness.
    """
    lines: List[str] = []
    for m in messages:
        role = (getattr(m, "role", None) or "user").lower()
        content = getattr(m, "content", "").strip()
        lines.append(f"{role.capitalize()}: {content}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[-max_chars:]


def _truncate_sentence(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars - 1] + "…"


_LLM_MERGE_SYSTEM = (
    "You merge two conversation memory synopses into ONE, preserving correctness and recency.\n"
    "Output ONLY a JSON object with exactly these keys and types:\n"
    "focus (string), user_goals (list[str]), constraints (list[str]), key_entities (list[str]),\n"
    "artifacts (list[str]), open_questions (list[str]), action_items (list[str]), decisions (list[str]).\n"
    "Rules:\n"
    "- Prefer NEW facts if they add specificity, dates/versions/IDs, or fix errors.\n"
    "- Keep stable facts from OLD if NEW is generic or contradictory.\n"
    "- Deduplicate items; remove empties; keep terse phrasing.\n"
    "- Enforce keeping the most specific and recent at the front of lists.\n"
    "- Do NOT invent facts that are not present in OLD or NEW.\n"
    "- Return ONLY JSON. No markdown, no commentary."
)

_LLM_MERGE_USER_TEMPLATE = """OLD_SYNOPSIS:
{old_json}

NEW_SYNOPSIS:
{new_json}

Task:
Produce the single best merged synopsis following the rules. Return ONLY JSON.
"""


async def _llm_merge_synopses(
    engine: Any,
    old: MemorySynopsis,
    new: MemorySynopsis,
) -> MemorySynopsis:
    # Check if engine is OpenAI client
    if not isinstance(engine, (OpenAI, AsyncOpenAI)):
        # Fallback: just return new synopsis
        return new

    old_json = MemorySynopsis(**asdict(old)).to_json()
    new_json = MemorySynopsis(**asdict(new)).to_json()

    # Prepare the system and user messages for the LLM
    sys_msg = {"role": "system", "content": _LLM_MERGE_SYSTEM}
    usr_msg = {"role": "user", "content": _LLM_MERGE_USER_TEMPLATE.format(old_json=old_json, new_json=new_json)}
    chat = [sys_msg, usr_msg]

    # Generate the merged synopsis using the OpenAI API
    response = await engine.chat.completions.create(
        model=settings.vllm_chat_model,
        messages=chat,
        temperature=0.0,
        top_p=1.0,
        extra_body={"guided_json": MEMORY_SYNOPSIS_JSON_SCHEMA}
    )

    # vLLM with --reasoning-parser separates reasoning_content from content
    # Use content directly (final response without thinking)
    text = response.choices[0].message.content or "{}"
    # try to parse JSON, if fails return empty MemorySynopsis
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        print('Failed to parse merged MemorySynopsis JSON:', text)
        return MemorySynopsis()
    return MemorySynopsis(**data)

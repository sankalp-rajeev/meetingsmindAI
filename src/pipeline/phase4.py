import json
import re
from pathlib import Path
from typing import Any, Dict, List
import requests


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Try hard to extract a JSON object from LLM output.
    """
    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from model output")


def _ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1, timeout: int = 480) -> str:
    """
    Calls local Ollama chat endpoint.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def run_phase4(meeting_root: Path, model: str = "qwen2.5:14b") -> Path:
    """
    Reads labeled_transcript.json (preferred) or transcript.json (fallback),
    generates meeting_notes.json in phase4.
    Returns output path.
    """
    labeled = meeting_root / "phase3" / "labeled_transcript.json"
    fallback = meeting_root / "phase1" / "transcript.json"

    if labeled.exists():
        src = labeled
    elif fallback.exists():
        src = fallback
    else:
        raise FileNotFoundError("No transcript found for Phase 4 (need phase3 or phase1 output).")

    data = _load_json(src)
    segments = data.get("segments", [])

    # Build participants if not present
    participants = data.get("participants", None)
    if not participants:
        speakers = []
        for s in segments:
            spk = s.get("speaker")
            if spk and spk not in speakers:
                speakers.append(spk)
        participants = [{"name": spk} for spk in speakers]


    # Keep prompt short + structured (best for local models)
    system = (
        "You are an assistant that outputs ONLY valid JSON. "
        "No markdown, no commentary."
    )

    user = {
        "input_file": src.name,
        "participants": participants,
        "segments": segments,
        "required_schema": {
            "title": "string",
            "summary": "string",
            "decisions": ["string"],
            "action_items": [{"owner": "string", "item": "string", "due": "string|null"}],
            "risks": ["string"],
            "open_questions": ["string"],
        },
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]

    raw = _ollama_chat(model, messages, temperature=0.1, timeout=480)

    try:
        out = _extract_json(raw)
    except Exception:
        # One repair attempt: ask model to fix to valid JSON
        repair_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Fix this into valid JSON ONLY. Do not add any text.\n\n" + raw},
        ]
        raw2 = _ollama_chat(model, repair_messages, temperature=0.0, timeout=480)
        out = _extract_json(raw2)

    out_path = meeting_root / "phase4" / "meeting_notes.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

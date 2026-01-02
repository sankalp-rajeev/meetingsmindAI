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


def _ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.1, timeout: int = 600) -> str:
    """
    Calls local Ollama chat endpoint.
    """
    import os
    ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
    url = f"http://{ollama_host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 32768},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def run_phase4(meeting_root: Path, model: str = "qwen2.5:14b") -> Path:
    """
    Generates comprehensive meeting summary by fusing transcript with visual insights.
    Creates a summary that allows someone who missed the meeting to understand:
    - What was discussed
    - What was shown on screen
    - Key decisions and action items
    """
    print("\n" + "=" * 60)
    print("PHASE 4: INTELLIGENT SUMMARIZATION")
    print("=" * 60)
    
    # Load transcript
    labeled = meeting_root / "phase3" / "labeled_transcript.json"
    fallback = meeting_root / "phase1" / "transcript.json"

    if labeled.exists():
        src = labeled
        print(f"Using labeled transcript from Phase 3")
    elif fallback.exists():
        src = fallback
        print(f"Using raw transcript from Phase 1")
    else:
        raise FileNotFoundError("No transcript found for Phase 4")

    data = _load_json(src)
    segments = data.get("segments", [])
    print(f"Loaded {len(segments)} transcript segments")

    # Extract participants
    speakers = []
    for s in segments:
        spk = s.get("speaker")
        if spk and spk not in speakers:
            speakers.append(spk)
    print(f"Identified {len(speakers)} speakers: {', '.join(speakers[:5])}...")

    # Load visual insights if available
    visual_insights = []
    visual_path = meeting_root / "phase5" / "visual_insights.json"
    if visual_path.exists():
        try:
            visual_data = _load_json(visual_path)
            for insight in visual_data.get("insights", []):
                # Only include actual content (not camera views or duplicates)
                if insight.get("skipped_vlm"):
                    continue
                content_type = insight.get("content_type", "unknown")
                if content_type in ["camera_only", "duplicate", "unknown"]:
                    continue
                
                visual_insights.append({
                    "time": insight.get("timestamp_formatted"),
                    "type": content_type,
                    "description": insight.get("description", ""),
                    "text": _extract_text_list(insight.get("extracted_text", [])),
                    "chart": insight.get("chart_analysis"),
                })
            print(f"Loaded {len(visual_insights)} visual content items (slides/charts)")
        except Exception as e:
            print(f"Could not load visual insights: {e}")

    # Helper to parse timestamp to seconds
    def parse_time(t):
        if isinstance(t, (int, float)):
            return float(t)
        if not t:
            return 0
        t = str(t).replace(',', '.')
        parts = t.split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except:
            return 0

    # Build visual lookup by timestamp
    visual_by_time = []
    for v in visual_insights:
        ts = parse_time(v.get("time", 0))
        visual_by_time.append({"ts": ts, "data": v})
    visual_by_time.sort(key=lambda x: x["ts"])
    
    # Track which visuals have been inserted
    visual_inserted = set()
    
    # Build the enriched transcript with interleaved visual content
    transcript_text = ""
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        start = seg.get("start", "0:00")
        seg_ts = parse_time(start)
        
        # Check if any visual content was shown around this time (±10 seconds)
        for i, v in enumerate(visual_by_time):
            if i not in visual_inserted and abs(v["ts"] - seg_ts) <= 10:
                # Insert visual context before this speech segment
                vd = v["data"]
                visual_desc = f"\n📊 [VISUAL @ {vd['time']}] {vd['type'].upper()}: {vd['description']}"
                if vd.get('text'):
                    visual_desc += f"\n   Visible text: {', '.join(vd['text'][:3])}"
                if vd.get('chart'):
                    visual_desc += f"\n   Chart: {vd['chart']}"
                transcript_text += visual_desc + "\n"
                visual_inserted.add(i)
        
        transcript_text += f"[{start}] {speaker}: {text}\n"
    
    # Add any remaining visuals that weren't matched to speech
    remaining_visuals = [(i, v) for i, v in enumerate(visual_by_time) if i not in visual_inserted]
    if remaining_visuals:
        transcript_text += "\n--- Additional Visual Content ---\n"
        for i, v in remaining_visuals:
            vd = v["data"]
            transcript_text += f"- At {vd['time']}: [{vd['type']}] {vd['description']}\n"

    # Create the prompt for comprehensive summary
    system_prompt = """You are an expert meeting analyst. Your job is to create a comprehensive meeting summary that allows someone who missed the meeting to fully understand what happened.

You will be given:
1. The full transcript with speaker names and timestamps
2. Visual content that was shown (slides, charts, screen shares)

Create a detailed summary that:
- Explains the PURPOSE and CONTEXT of the meeting
- Describes the KEY TOPICS discussed in chronological order
- Connects what people SAID with what they SHOWED on screen
- Identifies all DECISIONS made
- Lists ACTION ITEMS with owners
- Notes any RISKS or CONCERNS raised
- Captures OPEN QUESTIONS that need follow-up

Write in a clear, professional style. Be thorough but concise.
Output ONLY valid JSON matching the required schema."""

    user_prompt = f"""MEETING TRANSCRIPT (with visual content markers):
{transcript_text}

PARTICIPANTS: {', '.join(speakers)}

Note: Lines marked with "📊 [VISUAL @..." indicate slides/charts that were shown on screen at that moment.
Connect what people said with what was being displayed when they spoke.

Create a comprehensive meeting summary in this exact JSON format:
{{
    "title": "A descriptive title for this meeting (not just the topic, but what was accomplished)",
    
    "summary": {{
        "overview": "2-3 sentences explaining what this meeting was about and its main purpose",
        
        "discussion_topics": [
            "Topic 1: Detailed description of what was discussed, who said what, and any data/visuals shown",
            "Topic 2: ...",
            "Topic 3: ..."
        ],
        
        "key_points": [
            "Important point 1 with context and who made it",
            "Important point 2...",
            "Important point 3..."
        ],
        
        "visual_content_summary": "Description of key slides, charts, or data that was presented and their significance",
        
        "decisions_made": [
            "Decision 1: What was decided, by whom, and why",
            "Decision 2: ..."
        ],
        
        "action_items": [
            {{"owner": "Person Name", "item": "What they need to do", "due_date": "When or null"}},
            ...
        ],
        
        "risks_identified": [
            "Risk 1: Description of the risk and its potential impact",
            ...
        ],
        
        "open_questions": [
            "Question that needs follow-up and who should answer it",
            ...
        ],
        
        "next_steps": "What happens after this meeting"
    }}
}}

Be specific and detailed. Reference actual data, numbers, and names from the transcript.
If something wasn't discussed, use an empty array [] rather than making things up."""

    print("\nGenerating comprehensive summary...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = _ollama_chat(model, messages, temperature=0.2, timeout=600)

    try:
        out = _extract_json(raw)
    except Exception as e:
        print(f"Initial parsing failed: {e}, attempting repair...")
        repair_messages = [
            {"role": "system", "content": "Fix this into valid JSON. Output ONLY the corrected JSON, nothing else."},
            {"role": "user", "content": raw},
        ]
        raw2 = _ollama_chat(model, repair_messages, temperature=0.0, timeout=300)
        out = _extract_json(raw2)

    # Ensure the output structure is correct
    if "summary" not in out:
        # LLM might have flattened the structure - wrap it
        if "overview" in out or "discussion_topics" in out:
            out = {"title": out.get("title", "Meeting Summary"), "summary": out}
    
    # Save output
    out_dir = meeting_root / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "meeting_notes.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"✓ Summary saved to {out_path}")
    _print_summary_preview(out)
    
    return out_path


def _extract_text_list(text_data):
    """Safely extract text as a list."""
    if isinstance(text_data, str):
        return [text_data] if text_data else []
    elif isinstance(text_data, list):
        return text_data[:5]
    return []


def _print_summary_preview(out: dict):
    """Print a preview of the generated summary."""
    print("\n--- Summary Preview ---")
    if "title" in out:
        print(f"Title: {out['title']}")
    
    summary = out.get("summary", {})
    if isinstance(summary, str):
        print(f"Summary: {summary[:200]}...")
    elif isinstance(summary, dict):
        if "overview" in summary:
            print(f"Overview: {summary['overview'][:200]}...")
        if "discussion_topics" in summary:
            print(f"Discussion Topics: {len(summary['discussion_topics'])} topics")
        if "action_items" in summary:
            print(f"Action Items: {len(summary['action_items'])} items")
        if "decisions_made" in summary:
            print(f"Decisions: {len(summary['decisions_made'])} decisions")

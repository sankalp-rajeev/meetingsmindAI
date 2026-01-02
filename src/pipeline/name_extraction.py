"""
Speaker Name Extraction - AI-based extraction of speaker names from introductions.

Uses LLM to identify speaker names from the first few utterances of each speaker.
Handles various introduction patterns like:
- "Hi, this is Eric Johnson"
- "Hey everyone, Eric here"  
- "My name is Eric and I'm the CTO"
- "This is Kristie from engineering"
"""

import json
import requests
from typing import Dict, List, Optional


def extract_speaker_names(
    transcript_segments: List[dict],
    model: str = "qwen2.5:14b",
    max_segments_per_speaker: int = 5,
    timeout: int = 60
) -> Dict[str, str]:
    """
    Extract speaker names from transcript using LLM.
    
    Args:
        transcript_segments: List of transcript segments with speaker, start, text
        model: Ollama model to use
        max_segments_per_speaker: How many segments to analyze per speaker
        timeout: Request timeout
    
    Returns:
        Dict mapping speaker_id to extracted name (e.g., {"SPEAKER_00": "Eric Johnson"})
    """
    print("\n[Name Extraction] Extracting speaker names from introductions...")
    
    # Group first N segments per speaker
    speaker_texts: Dict[str, List[str]] = {}
    speaker_counts: Dict[str, int] = {}
    
    for seg in transcript_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if not text:
            continue
        
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # Only take first few segments (likely to contain introductions)
        if speaker_counts[speaker] <= max_segments_per_speaker:
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)
    
    if not speaker_texts:
        print("  No speaker segments found")
        return {}
    
    # Build prompt
    speakers_content = []
    for speaker_id, texts in speaker_texts.items():
        combined = " ".join(texts)[:500]  # Limit text length
        speakers_content.append(f"{speaker_id}: \"{combined}\"")
    
    prompt = f"""Analyze these meeting transcript excerpts and extract speaker names.

For each SPEAKER_XX, determine if they introduced themselves and extract their name.
Only extract names you are confident about - do not guess.

Transcripts:
{chr(10).join(speakers_content)}

Respond with ONLY a JSON object mapping speaker IDs to names.
If no name found for a speaker, use null.
Example: {{"SPEAKER_00": "Eric Johnson", "SPEAKER_01": null, "SPEAKER_02": "Kristie Thomas"}}

JSON only, no explanation:"""

    try:
        import os
        ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        response = requests.post(
            f"http://{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        # Parse JSON from response
        extracted = _parse_json_response(content)
        
        # Filter out null values and clean names
        names = {}
        for speaker_id, name in extracted.items():
            if name and isinstance(name, str) and name.lower() != "null":
                # Clean up the name
                name = name.strip()
                if len(name) > 1 and len(name) < 50:  # Sanity check
                    names[speaker_id] = name
        
        print(f"  Extracted {len(names)} names: {names}")
        return names
        
    except Exception as e:
        print(f"  Name extraction failed: {e}")
        return {}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response."""
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Find JSON in response
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    
    return {}


if __name__ == "__main__":
    # Quick test
    test_segments = [
        {"speaker": "SPEAKER_00", "text": "Hi, this is Eric Johnson. It's February 18, 2021."},
        {"speaker": "SPEAKER_01", "text": "Thanks Eric. This is Kristie Thomas from the engineering team."},
        {"speaker": "SPEAKER_02", "text": "Let me share some updates on the project."},
    ]
    
    names = extract_speaker_names(test_segments)
    print(f"Extracted: {names}")

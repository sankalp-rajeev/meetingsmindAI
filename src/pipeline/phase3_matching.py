"""
Phase 3: Speaker–Face Association (MVP JSON storage)

Reads:
- transcript.json (Phase 1 output)
- faces.json      (Phase 2 output)

Writes:
- speaker_face_suggestions.json   (auto suggestions per SPEAKER_X)
- prelabeled_transcript.json      (segment-level face_track_id + confidence; names empty)
- labeled_transcript.json         (optional, if you provide a speaker_face_map.json with names)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def seconds_to_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


# ----------------------------
# Core matching
# ----------------------------

@dataclass
class SegmentMatch:
    face_track_id: Optional[str]
    hits: int
    total_hits: int
    coverage: float
    dominance: float
    confidence: float
    ambiguous: bool


def build_face_events(faces_json: dict) -> List[Tuple[float, str]]:
    """
    Returns a list of (timestamp_seconds, track_id), one entry per timeline sample.
    """
    events: List[Tuple[float, str]] = []
    for tr in faces_json.get("tracks", []):
        tid = tr.get("track_id")
        for t in tr.get("timeline", []):
            ts = t.get("timestamp")
            if tid is None or ts is None:
                continue
            events.append((float(ts), str(tid)))
    events.sort(key=lambda x: x[0])
    return events


def match_segment(
    seg_start: float,
    seg_end: float,
    face_events: List[Tuple[float, str]],
    sampling_fps: float,
    min_hits: int = 2,
    min_dominance: float = 0.55,
    ambiguity_ratio: float = 1.2,
) -> SegmentMatch:
    """
    Find the most frequent face track within [seg_start, seg_end] based on sampled timestamps.

    Confidence is a blend of:
    - coverage: how many samples we saw vs expected samples for this segment duration
    - dominance: how much the top track dominates among all samples in the window
    """
    if seg_end <= seg_start:
        return SegmentMatch(None, 0, 0, 0.0, 0.0, 0.0, True)

    # Collect hits in window (linear scan; fast enough for MVP. If needed, add bisect later.)
    counts: Dict[str, int] = {}
    total = 0
    for ts, tid in face_events:
        if ts < seg_start:
            continue
        if ts > seg_end:
            break
        counts[tid] = counts.get(tid, 0) + 1
        total += 1

    if total == 0:
        return SegmentMatch(None, 0, 0, 0.0, 0.0, 0.0, True)

    # Top-2
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_tid, top_hits = top[0]
    second_hits = top[1][1] if len(top) > 1 else 0

    duration = seg_end - seg_start
    expected = max(1.0, duration * sampling_fps)
    coverage = clamp(top_hits / expected, 0.0, 1.0)
    dominance = clamp(top_hits / total, 0.0, 1.0)
    confidence = 0.5 * coverage + 0.5 * dominance

    ambiguous = False
    if top_hits < min_hits:
        ambiguous = True
    if dominance < min_dominance:
        ambiguous = True
    if second_hits > 0 and (top_hits / max(1, second_hits)) < ambiguity_ratio:
        ambiguous = True

    return SegmentMatch(
        face_track_id=top_tid if not ambiguous else None,
        hits=int(top_hits),
        total_hits=int(total),
        coverage=float(coverage),
        dominance=float(dominance),
        confidence=float(confidence),
        ambiguous=bool(ambiguous),
    )


def build_speaker_suggestions(prelabeled_segments: List[dict]) -> List[dict]:
    """
    Aggregate segment-level matches into per-speaker suggestions.
    """
    by_speaker: Dict[str, Dict[str, float]] = {}
    speaker_stats: Dict[str, dict] = {}

    for idx, seg in enumerate(prelabeled_segments):
        spk = seg.get("speaker_id", seg.get("speaker", "UNKNOWN"))
        tid = seg.get("face_track_id")
        conf = float(seg.get("confidence", 0.0) or 0.0)

        speaker_stats.setdefault(spk, {"segments": 0, "matched_segments": 0, "avg_confidence": 0.0})
        speaker_stats[spk]["segments"] += 1

        if tid:
            speaker_stats[spk]["matched_segments"] += 1
            by_speaker.setdefault(spk, {})
            # Weight by confidence to prefer cleaner segments
            by_speaker[spk][tid] = by_speaker[spk].get(tid, 0.0) + conf

        # running avg confidence (all segments)
        prev_n = speaker_stats[spk]["segments"] - 1
        speaker_stats[spk]["avg_confidence"] = (speaker_stats[spk]["avg_confidence"] * prev_n + conf) / max(1, prev_n + 1)

    suggestions = []
    for spk, track_weights in by_speaker.items():
        ranked = sorted(track_weights.items(), key=lambda kv: kv[1], reverse=True)
        best_tid, best_score = ranked[0]
        alt = [{"face_track_id": tid, "score": score} for tid, score in ranked[1:4]]
        match_rate = speaker_stats[spk]["matched_segments"] / max(1, speaker_stats[spk]["segments"])
        suggestions.append({
            "speaker_id": spk,
            "suggested_face_track_id": best_tid,
            "score": best_score,
            "match_rate": round(match_rate, 3),
            "avg_segment_confidence": round(float(speaker_stats[spk]["avg_confidence"]), 3),
            "alternatives": alt,
            "status": "needs_label"
        })

    # Include speakers with zero matches too
    for spk, st in speaker_stats.items():
        if spk not in {s["speaker_id"] for s in suggestions}:
            suggestions.append({
                "speaker_id": spk,
                "suggested_face_track_id": None,
                "score": 0.0,
                "match_rate": 0.0,
                "avg_segment_confidence": 0.0,
                "alternatives": [],
                "status": "no_face_detected"
            })

    suggestions.sort(key=lambda x: (x["speaker_id"]))
    return suggestions


def apply_labels(prelabeled: dict, speaker_face_map: dict, faces_json: dict) -> dict:
    """
    Apply manual labels (speaker_id -> name, face_track_id) to produce labeled_transcript.json.
    """
    speaker_to_name: Dict[str, str] = {}
    speaker_to_track: Dict[str, str] = {}

    for m in speaker_face_map.get("mappings", []):
        spk = m.get("speaker_id")
        name = m.get("name")
        tid = m.get("face_track_id")
        if spk and name:
            speaker_to_name[str(spk)] = str(name)
        if spk and tid:
            speaker_to_track[str(spk)] = str(tid)

    # quick lookup: track_id -> keyframe path
    track_keyframes = {}
    for tr in faces_json.get("tracks", []):
        tid = tr.get("track_id")
        keyframe = tr.get("keyframe", {}) or {}
        if tid:
            track_keyframes[str(tid)] = keyframe.get("path")

    segments_out = []
    speaking_time: Dict[str, float] = {}
    participant_track: Dict[str, str] = {}

    for seg in prelabeled.get("segments", []):
        spk = str(seg.get("speaker_id", seg.get("speaker", "UNKNOWN")))
        name = speaker_to_name.get(spk, spk)  # fall back to speaker id if not labeled
        tid = seg.get("face_track_id") or speaker_to_track.get(spk)

        dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        speaking_time[name] = speaking_time.get(name, 0.0) + max(0.0, dur)
        if tid and name not in participant_track:
            participant_track[name] = str(tid)

        segments_out.append({
            "start": seconds_to_hhmmss(seg["start"]),
            "end": seconds_to_hhmmss(seg["end"]),
            "speaker": name,
            "face_track_id": tid,
            "text": seg.get("text", "")
        })

    participants_out = []
    for name, secs in sorted(speaking_time.items(), key=lambda kv: kv[1], reverse=True):
        tid = participant_track.get(name)
        participants_out.append({
            "name": name,
            "face_track_id": tid,
            "speaking_time": seconds_to_hhmmss(secs),
            "keyframe": track_keyframes.get(tid) if tid else None
        })

    return {"segments": segments_out, "participants": participants_out}


def run_matching(
    transcript_path: Path,
    faces_path: Path,
    out_dir: Path,
    speaker_face_map_path: Optional[Path] = None,
    min_hits: int = 2,
    min_dominance: float = 0.55,
    ambiguity_ratio: float = 1.2,
):
    print(f"Loading transcript: {transcript_path}")
    transcript = load_json(transcript_path)
    print(f"Loading faces: {faces_path}")
    faces = load_json(faces_path)

    sampling_fps = float(faces.get("sampling_fps", 2.0) or 2.0)
    face_events = build_face_events(faces)

    prelabeled_segments: List[dict] = []
    unresolved = []

    print("Matching segments...")
    for i, seg in enumerate(transcript.get("segments", [])):
        start = float(seg["start"])
        end = float(seg["end"])
        spk = str(seg.get("speaker", "UNKNOWN"))
        text = seg.get("text", "")

        m = match_segment(
            start, end, face_events, sampling_fps,
            min_hits=min_hits,
            min_dominance=min_dominance,
            ambiguity_ratio=ambiguity_ratio,
        )

        if m.face_track_id is None:
            unresolved.append(i)

        prelabeled_segments.append({
            "segment_index": i,
            "start": start,
            "end": end,
            "speaker_id": spk,
            "text": text,
            "face_track_id": m.face_track_id,
            "confidence": round(m.confidence, 3),
            "debug": {
                "hits": m.hits,
                "total_hits": m.total_hits,
                "coverage": round(m.coverage, 3),
                "dominance": round(m.dominance, 3),
                "ambiguous": m.ambiguous
            }
        })

    prelabeled = {
        "source": {
            "transcript": str(transcript_path),
            "faces": str(faces_path)
        },
        "segments": prelabeled_segments,
        "unresolved_segments": unresolved
    }

    suggestions = {
        "source": prelabeled["source"],
        "suggestions": build_speaker_suggestions(prelabeled_segments),
        "unresolved_segments": unresolved
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(prelabeled, out_dir / "prelabeled_transcript.json")
    save_json(suggestions, out_dir / "speaker_face_suggestions.json")
    print(f"Saved prelabeled_transcript.json and speaker_face_suggestions.json to {out_dir}")

    # Optional: if speaker_face_map.json exists, also export final labeled transcript
    if speaker_face_map_path and speaker_face_map_path.exists():
        print(f"Applying existing map: {speaker_face_map_path}")
        speaker_face_map = load_json(speaker_face_map_path)
        labeled = apply_labels(prelabeled, speaker_face_map, faces)
        save_json(labeled, out_dir / "labeled_transcript.json")
        print(f"Saved labeled_transcript.json")
    
    return out_dir / "prelabeled_transcript.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--transcript", type=str, required=True)
    p.add_argument("--faces", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--map", type=str, default=None)

    args = p.parse_args()

    run_matching(
        transcript_path=Path(args.transcript),
        faces_path=Path(args.faces),
        out_dir=Path(args.outdir),
        speaker_face_map_path=Path(args.map) if args.map else None,
    )

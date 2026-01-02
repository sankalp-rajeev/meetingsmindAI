"""
Phase 5: Visual Intelligence - Smart Frame Sampling & VLM Analysis

Extracts visual content from meeting videos using:
1. Smart frame sampling (scene change, text detection, transcript triggers)
2. VLM analysis (Qwen2.5-VL via Ollama)

Outputs: phase5/visual_insights.json
"""

from __future__ import annotations

import os
import json
import base64
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests


# ----------------------------
# Visual Cue Keywords (for transcript-triggered sampling)
# ----------------------------
VISUAL_CUE_KEYWORDS = [
    "look at", "as you can see", "on the screen", "on screen",
    "this slide", "this chart", "this graph", "this diagram",
    "let me show", "let me share", "sharing my screen",
    "here's the", "take a look", "you can see",
    "the data shows", "the chart shows", "as shown here",
]


# ----------------------------
# Data Classes
# ----------------------------
@dataclass
class Keyframe:
    timestamp: float
    timestamp_formatted: str
    frame: np.ndarray  # Will not be serialized
    trigger_reason: str  # scene_change, text_detected, transcript_trigger
    text_snippets: List[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_formatted": self.timestamp_formatted,
            "trigger_reason": self.trigger_reason,
            "text_snippets": self.text_snippets or []
        }


@dataclass
class VisualInsight:
    timestamp: float
    timestamp_formatted: str
    frame_path: str
    trigger_reason: str
    content_type: str  # slide, screen_share, whiteboard, camera, unknown
    description: str
    extracted_text: List[str]
    confidence: float


# ----------------------------
# Utilities
# ----------------------------
def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convert OpenCV frame to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def resize_for_vlm(frame: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize frame to fit within max_size while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ----------------------------
# Scene Change Detector
# ----------------------------
class SceneChangeDetector:
    """Detect significant visual changes between frames using SSIM."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.prev_gray = None
    
    def is_new_scene(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if frame represents a new scene.
        Returns: (is_new_scene, similarity_score)
        """
        # Convert to grayscale and resize for faster comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 0.0
        
        # Calculate structural similarity
        try:
            score = ssim(self.prev_gray, gray)
        except Exception:
            score = 1.0
        
        is_new = score < self.threshold
        
        if is_new:
            self.prev_gray = gray
        
        return is_new, score
    
    def reset(self):
        self.prev_gray = None


# ----------------------------
# Text Detection Gate
# ----------------------------
class TextDetectionGate:
    """Quick text detection to filter frames worth analyzing with VLM."""
    
    def __init__(self, min_text_length: int = 10):
        self.min_text_length = min_text_length
        self._reader = None
    
    @property
    def reader(self):
        """Lazy load EasyOCR to avoid slow startup."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        return self._reader
    
    def detect_text(self, frame: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Quick OCR check on frame.
        Returns: (has_significant_text, text_snippets)
        """
        # Resize for faster OCR
        small = cv2.resize(frame, (640, 360))
        
        try:
            results = self.reader.readtext(small, detail=0, paragraph=True)
            text_snippets = [t.strip() for t in results if len(t.strip()) >= 3]
            total_text = " ".join(text_snippets)
            has_text = len(total_text) >= self.min_text_length
            return has_text, text_snippets[:10]  # Limit snippets
        except Exception as e:
            print(f"  OCR error: {e}")
            return False, []


# ----------------------------
# Content Pre-Filter (Camera-Only Detection)
# ----------------------------
class ContentPreFilter:
    """
    Robust pre-filter to detect camera-only frames vs actual content.
    
    SKIPS (camera views):
    - Zoom/Teams gallery grid (multiple webcam tiles)
    - Single person full-screen webcam
    - Virtual backgrounds with faces
    
    PROCESSES (actual content):
    - Slides with text/bullets
    - Charts/graphs/tables
    - Whiteboards
    - Screen shares (code, documents, browser)
    """
    
    def __init__(
        self,
        skin_tone_threshold: float = 0.15,   # If >15% of frame is skin-colored, likely camera
        face_count_threshold: int = 2,        # Multiple faces = gallery view
        text_block_threshold: int = 3,        # Need multiple text blocks for slide
        min_text_area_ratio: float = 0.05,    # Text should cover >5% of frame for slides
    ):
        self.skin_tone_threshold = skin_tone_threshold
        self.face_count_threshold = face_count_threshold
        self.text_block_threshold = text_block_threshold
        self.min_text_area_ratio = min_text_area_ratio
        self._face_cascade = None
    
    @property
    def face_cascade(self):
        """Lazy load face detector."""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade
    
    def _detect_skin_ratio(self, frame: np.ndarray) -> float:
        """
        Detect ratio of skin-colored pixels in frame.
        Camera views have lots of skin tones; slides/charts don't.
        """
        # Convert to YCrCb (better for skin detection)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Skin tone ranges in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1])
        
        return skin_ratio
    
    def _detect_face_grid(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Detect faces and calculate their coverage.
        Returns: (face_count, total_face_area_ratio)
        """
        small = cv2.resize(gray, (640, 360))
        faces = self.face_cascade.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return 0, 0.0
        
        total_face_area = sum(fw * fh for (_, _, fw, fh) in faces)
        frame_area = 640 * 360
        face_ratio = total_face_area / frame_area
        
        return len(faces), face_ratio
    
    def _detect_text_regions(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Detect structured text regions (slides have organized text blocks).
        Uses morphological operations to find text-like regions.
        Returns: (text_block_count, text_area_ratio)
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text characters into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours (text blocks)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for text-like regions (wide, not too tall)
        text_blocks = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            area = w * h
            
            # Text blocks are typically wide (aspect > 2) and have reasonable size
            if aspect_ratio > 2 and area > 500 and area < (gray.shape[0] * gray.shape[1] * 0.3):
                text_blocks.append((x, y, w, h))
        
        total_text_area = sum(w * h for (_, _, w, h) in text_blocks)
        text_area_ratio = total_text_area / (gray.shape[0] * gray.shape[1])
        
        return len(text_blocks), text_area_ratio
    
    def _has_ui_elements(self, frame: np.ndarray) -> bool:
        """
        Detect UI elements that indicate screen share.
        Look for: window borders, menu bars, browser chrome, task bars.
        """
        h, w = frame.shape[:2]
        
        # Check top 40 pixels for menu bar (usually dark or colored strip)
        top_strip = frame[:40, :]
        top_std = np.std(top_strip)
        
        # Check for horizontal lines (window borders, tabs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count strong horizontal edges in top 100 pixels
        top_edges = edges[:100, :]
        horizontal_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        horizontal_edges = cv2.morphologyEx(top_edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_edges > 0) / (100 * w)
        
        # If lots of horizontal lines at top, likely has window chrome
        return horizontal_ratio > 0.1 or top_std > 60
    
    def _has_chart_patterns(self, gray: np.ndarray) -> bool:
        """
        Detect chart/graph patterns: bars, lines, pie sections.
        """
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return False
        
        # Count horizontal and vertical lines (axes, bars)
        h_lines = 0
        v_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Horizontal
                h_lines += 1
            elif 75 < angle < 105:  # Vertical
                v_lines += 1
        
        # Charts typically have both horizontal and vertical lines (axes)
        return h_lines >= 2 and v_lines >= 2
    
    def _has_video_conference_grid(self, frame: np.ndarray) -> bool:
        """
        Detect video conference gallery grid pattern (Zoom, Teams, Meet).
        
        Key characteristics of conference grids:
        - DARK background (black/gray gaps between tiles)
        - Multiple BRIGHT rectangular tiles of SIMILAR SIZE
        - Tiles contain people (skin tones)
        
        NOT a grid:
        - Slides with text blocks (white background, text, no skin in tiles)
        - Screen shares (irregular layout)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Check for DARK frame background (Zoom uses black/dark gray)
        # Sample corners and edges to check for dark borders
        corners = [
            gray[:30, :30].mean(),           # Top-left
            gray[:30, -30:].mean(),          # Top-right  
            gray[-30:, :30].mean(),          # Bottom-left
            gray[-30:, -30:].mean(),         # Bottom-right
        ]
        dark_corners = sum(1 for c in corners if c < 50)
        
        # Also check horizontal strips (gaps between rows)
        strip_samples = []
        for y in range(h // 4, 3 * h // 4, h // 8):
            strip_mean = gray[y:y+5, :].mean()
            strip_samples.append(strip_mean)
        dark_strips = sum(1 for s in strip_samples if s < 40)
        
        # Need dark background indicators
        has_dark_background = dark_corners >= 2 or dark_strips >= 2
        
        if not has_dark_background:
            return False  # Slides/docs have white backgrounds, not grids
        
        # 2. Find bright rectangular regions (participant tiles)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangles that could be face tiles
        tiles = []
        min_tile_area = h * w * 0.015  # At least 1.5% of frame
        max_tile_area = h * w * 0.35   # At most 35% of frame
        
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            
            if min_tile_area < area < max_tile_area:
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                # Video tiles are roughly 4:3 or 16:9 (aspect 1.0 - 2.0)
                if 0.8 < aspect < 2.2:
                    tiles.append({'x': x, 'y': y, 'w': rw, 'h': rh, 'area': area})
        
        # 3. Check if tiles are similar sized (key indicator of grid layout)
        if len(tiles) >= 3:
            areas = [t['area'] for t in tiles]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv_area = std_area / max(mean_area, 1)  # Coefficient of variation
            
            if cv_area < 0.4:  # Tiles are similar size (low variance)
                return True
        
        # 4. Check if tiles are arranged in rows (y-coordinates cluster)
        if len(tiles) >= 4:
            y_coords = sorted([t['y'] for t in tiles])
            # Check for y-coordinate clustering (rows)
            row_gaps = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            small_gaps = sum(1 for g in row_gaps if g < 20)  # Same row
            
            if small_gaps >= 2:  # At least 3 tiles in same row
                return True
        
        return False
    
    def _has_conference_watermark(self, frame: np.ndarray) -> bool:
        """
        Detect video conference watermarks (Zoom, Teams, Meet logos).
        Usually in corners of the frame.
        """
        h, w = frame.shape[:2]
        
        # Check bottom-right corner (common for Zoom)
        corner_size = min(150, w // 6)
        bottom_right = frame[h-80:h-10, w-corner_size:]
        
        # Zoom logo is typically blue/white on dark background
        # Check if corner has the characteristic blue color
        hsv = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2HSV)
        
        # Zoom blue is roughly H=100-120
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / max(blue_mask.size, 1)
        
        # Also check for white text on dark bg
        gray_corner = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray_corner > 200)
        dark_pixels = np.sum(gray_corner < 50)
        
        # Watermark pattern: some white text on mostly dark corner
        has_watermark_pattern = white_pixels > 50 and dark_pixels > (corner_size * 70 * 0.5)
        
        return blue_ratio > 0.05 or has_watermark_pattern
    
    def is_likely_content_frame(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Determine if frame contains visual content worth analyzing.
        
        Decision process:
        1. High skin tone + faces = Camera view → SKIP
        2. Multiple faces in grid = Gallery view → SKIP  
        3. Text blocks detected = Slide/doc → PROCESS
        4. UI elements detected = Screen share → PROCESS
        5. Chart patterns detected = Chart → PROCESS
        6. Low skin + no faces = Possibly content → PROCESS
        7. Default = SKIP (be conservative)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Skin tone detection
        skin_ratio = self._detect_skin_ratio(frame)
        
        # 2. Face detection
        face_count, face_area_ratio = self._detect_face_grid(gray)
        
        # 3. Text region detection
        text_block_count, text_area_ratio = self._detect_text_regions(gray)
        
        # Decision logic:
        
        # SKIP: Video conference grid pattern detected + high skin (safety net for when face detection fails)
        # Note: Only trigger if skin ratio is significant (grids have faces = skin)
        if skin_ratio > 0.12 and self._has_video_conference_grid(frame):
            return False, f"conference_grid (grid pattern + skin={skin_ratio:.1%})"
        
        # SKIP: Video conference watermark detected (Zoom, Teams, etc.)
        if self._has_conference_watermark(frame):
            if skin_ratio > 0.10:  # With watermark + some skin = definitely conference view
                return False, f"conference_watermark (logo detected, skin={skin_ratio:.1%})"
        
        # SKIP: Gallery grid (multiple faces)
        if face_count >= self.face_count_threshold:
            return False, f"gallery_view ({face_count} faces)"
        
        # SKIP: High skin tone + face(s) = webcam view
        if skin_ratio > self.skin_tone_threshold and face_count > 0:
            return False, f"webcam_view (skin={skin_ratio:.1%}, faces={face_count})"
        
        # SKIP: Single large face (full screen person)
        if face_count == 1 and face_area_ratio > 0.15:
            return False, f"fullscreen_person (face_area={face_area_ratio:.1%})"
        
        # SKIP: High skin + even without detected faces (might be low-res or angles)
        if skin_ratio > 0.20:  # Very high skin ratio is almost always camera view
            return False, f"high_skin_ratio (skin={skin_ratio:.1%})"
        
        # PROCESS: Significant text blocks (slide/document)
        if text_block_count >= self.text_block_threshold and text_area_ratio > self.min_text_area_ratio:
            return True, f"text_content ({text_block_count} blocks, {text_area_ratio:.1%} area)"
        
        # PROCESS: UI elements detected (screen share)
        if self._has_ui_elements(frame):
            # But verify low skin (not just webcam with virtual background)
            if skin_ratio < self.skin_tone_threshold:
                return True, f"screen_share (UI detected, skin={skin_ratio:.1%})"
        
        # PROCESS: Chart patterns
        if self._has_chart_patterns(gray):
            if skin_ratio < self.skin_tone_threshold:
                return True, f"chart_detected (skin={skin_ratio:.1%})"
        
        # PROCESS: Very low skin ratio + no faces = likely content
        if skin_ratio < 0.05 and face_count == 0:
            return True, f"no_people (skin={skin_ratio:.1%})"
        
        # Default: SKIP (be conservative to avoid wasting VLM calls)
        return False, f"uncertain (skin={skin_ratio:.1%}, faces={face_count})"


# ----------------------------
# Transcript Trigger Matcher
# ----------------------------
class TranscriptTriggerMatcher:
    """Find timestamps where speakers mention visual cues."""
    
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or VISUAL_CUE_KEYWORDS
    
    def find_visual_cues(self, transcript_segments: List[dict]) -> List[Tuple[float, str]]:
        """
        Find segments that mention visual content.
        Returns: List of (timestamp, matched_keyword)
        """
        triggers = []
        
        for seg in transcript_segments:
            text = seg.get("text", "").lower()
            start = seg.get("start", 0)
            
            # Parse timestamp if string
            if isinstance(start, str):
                parts = start.split(":")
                if len(parts) == 3:
                    start = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    start = int(parts[0]) * 60 + float(parts[1])
            
            for keyword in self.keywords:
                if keyword in text:
                    triggers.append((float(start), keyword))
                    break  # One trigger per segment
        
        return triggers


# ----------------------------
# VLM Analyzer (Gemini Vision via Vertex AI)
# ----------------------------
class VLMAnalyzer:
    """Analyze frames using Gemini Vision through Vertex AI."""
    
    def __init__(self, model: str = "gemini-1.5-flash-002", timeout: int = 120):
        self.model = model
        self.timeout = timeout
        # Import here to avoid circular imports
        from src.app.gemini_client import GeminiClient
        self.client = GeminiClient(model_name=model, temperature=0.1)
    
    def analyze_frame(self, frame: np.ndarray, context: str = "") -> dict:
        """
        Analyze a single frame with VLM for visual context.
        Focus: Content, expressions, gestures - things transcripts miss.
        Returns: {content_type, description, extracted_text, expressions, gestures, engagement, confidence}
        """
        # Resize and encode
        resized = resize_for_vlm(frame, max_size=1024)
        b64_image = frame_to_base64(resized)
        
        # Build prompt focused on CONTENT EXTRACTION (no camera-only frames reach this point)
        prompt = """Analyze this video meeting frame and extract ALL visual content.

This frame contains visual content (slide, chart, whiteboard, or screen share). Extract everything.

EXTRACT THESE:

1. **content_type**: What type of visual content?
   - "slide": Presentation slide (title, bullets, text)
   - "chart": Data visualization (bar, line, pie chart, table)
   - "screen_share": Screen with code, documents, browser, spreadsheet
   - "diagram": Architecture diagram, flowchart, org chart
   - "whiteboard": Physical/digital whiteboard
   - "document": PDF, Word doc, report
   - "unknown": Cannot determine

2. **extracted_text**: ALL readable text (titles, labels, bullet points, numbers). Be thorough.

3. **chart_analysis**: If chart/graph visible, describe in detail:
   {"type": "chart type", "title": "chart title", "data": "what data it shows", "key_insights": ["insight 1", "insight 2"]}

4. **table_data**: If table visible, extract key rows/columns as structured data.

5. **code_or_document**: If code/document visible:
   {"type": "language or doc type", "purpose": "what it shows", "key_elements": ["item1", "item2"]}

6. **slide_content**: If presentation slide:
   {"title": "slide title", "bullets": ["point 1", "point 2"], "key_message": "main takeaway"}

7. **description**: Comprehensive description of visual content (2-3 sentences). Include any numbers, metrics, or data points visible.

8. **confidence**: 0.0-1.0

CRITICAL: Extract ALL text, numbers, and data visible. This visual content will be merged with transcript for summary.

Respond in JSON only, no markdown:"""
        
        if context:
            prompt = f"Context from transcript: '{context}'\n\n{prompt}"
        
        try:
            # Use Gemini Vision API
            import base64
            image_bytes = base64.b64decode(b64_image)
            response_text = self.client.analyze_image(image_bytes, prompt, mime_type="image/jpeg")
            
            # Parse JSON from response
            return self._parse_vlm_response(response_text)
            
        except Exception as e:
            print(f"  VLM error: {e}")
            return {
                "content_type": "unknown",
                "description": "Analysis failed",
                "extracted_text": [],
                "expressions": [],
                "gestures": [],
                "engagement": "unknown",
                "confidence": 0.0
            }
    
    def _parse_vlm_response(self, text: str) -> dict:
        """Extract JSON from VLM response."""
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON in response
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        # Fallback
        return {
            "content_type": "unknown",
            "description": text[:200] if text else "No description",
            "extracted_text": [],
            "confidence": 0.3
        }


# ----------------------------
# Smart Frame Sampler
# ----------------------------
class SmartFrameSampler:
    """Intelligently sample keyframes from video."""
    
    def __init__(
        self,
        scene_threshold: float = 0.85,
        base_interval_sec: float = 5.0,
        min_interval_sec: float = 2.0,
    ):
        self.scene_detector = SceneChangeDetector(threshold=scene_threshold)
        self.text_gate = TextDetectionGate()
        self.trigger_matcher = TranscriptTriggerMatcher()
        self.base_interval = base_interval_sec
        self.min_interval = min_interval_sec
    
    def sample(
        self,
        video_path: Path,
        transcript_segments: List[dict] = None,
        max_frames: int = 50
    ) -> List[Keyframe]:
        """
        Sample keyframes intelligently from video.
        
        Strategy:
        1. Sample at base interval (e.g., every 5 seconds)
        2. Keep frames with scene changes
        3. Prioritize frames with text
        4. Add frames at transcript-triggered timestamps
        5. Deduplicate similar frames
        """
        print(f"\n[Smart Sampler] Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  Duration: {format_timestamp(duration)}, FPS: {fps:.1f}")
        
        # Adaptive sampling: ensure we cover the entire video, not just the first N minutes
        # Calculate interval to spread max_frames across the entire video
        adaptive_interval = max(self.base_interval, duration / max_frames)
        base_interval_sec = min(adaptive_interval, 30.0)  # Cap at 30 seconds max
        
        print(f"  Adaptive sampling interval: {base_interval_sec:.1f}s (to cover {format_timestamp(duration)})")
        
        # Get transcript triggers
        trigger_times = set()
        if transcript_segments:
            triggers = self.trigger_matcher.find_visual_cues(transcript_segments)
            trigger_times = {t[0] for t in triggers}
            print(f"  Found {len(trigger_times)} transcript visual cues")
        
        keyframes: List[Keyframe] = []
        frame_interval = int(fps * base_interval_sec)
        min_gap_frames = int(fps * self.min_interval)
        
        frame_idx = 0
        last_keyframe_idx = -min_gap_frames
        
        self.scene_detector.reset()
        
        print(f"  Sampling every {base_interval_sec:.1f}s (checking scene changes)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Check at regular intervals
            if frame_idx % frame_interval == 0 or frame_idx - last_keyframe_idx >= frame_interval:
                # Skip if too close to last keyframe
                if frame_idx - last_keyframe_idx < min_gap_frames:
                    frame_idx += 1
                    continue
                
                is_new_scene, similarity = self.scene_detector.is_new_scene(frame)
                is_trigger = any(abs(timestamp - t) < 2.0 for t in trigger_times)
                
                should_sample = is_new_scene or is_trigger
                
                if should_sample:
                    reason = "transcript_trigger" if is_trigger else "scene_change"
                    
                    # Quick text check
                    has_text, text_snippets = self.text_gate.detect_text(frame)
                    if has_text:
                        reason = "text_detected"
                    
                    keyframes.append(Keyframe(
                        timestamp=timestamp,
                        timestamp_formatted=format_timestamp(timestamp),
                        frame=frame.copy(),
                        trigger_reason=reason,
                        text_snippets=text_snippets if has_text else None
                    ))
                    
                    last_keyframe_idx = frame_idx
                    
                    if len(keyframes) >= max_frames:
                        print(f"  Reached max frames ({max_frames})")
                        break
            
            frame_idx += 1
        
        cap.release()
        
        print(f"  Selected {len(keyframes)} keyframes")
        
        # Log breakdown
        reasons = {}
        for kf in keyframes:
            reasons[kf.trigger_reason] = reasons.get(kf.trigger_reason, 0) + 1
        print(f"  Breakdown: {reasons}")
        
        return keyframes


# ----------------------------
# Main Phase 5 Runner
# ----------------------------
def run_phase5(
    meeting_root: Path,
    vlm_model: str = "qwen2.5vl:latest",
    max_keyframes: int = 30,
    skip_vlm: bool = False
) -> Path:
    """
    Run Phase 5: Visual Intelligence
    
    1. Smart sample keyframes from video
    2. Analyze with VLM
    3. Save visual_insights.json
    
    Args:
        meeting_root: Meeting directory
        vlm_model: Ollama model name
        max_keyframes: Maximum keyframes to analyze
        skip_vlm: If True, only sample frames (for testing)
    
    Returns:
        Path to visual_insights.json
    """
    print("\n" + "=" * 60)
    print("PHASE 5: VISUAL INTELLIGENCE")
    print("=" * 60)
    
    video_path = meeting_root / "original.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load transcript for trigger matching
    transcript_segments = []
    transcript_path = meeting_root / "phase1" / "transcript.json"
    if transcript_path.exists():
        transcript_data = load_json(transcript_path)
        transcript_segments = transcript_data.get("segments", [])
        print(f"Loaded transcript with {len(transcript_segments)} segments")
    
    # Create output directory
    output_dir = meeting_root / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Sample keyframes
    print("\n[1/3] Smart frame sampling...")
    sampler = SmartFrameSampler()
    keyframes = sampler.sample(video_path, transcript_segments, max_frames=max_keyframes)
    
    if not keyframes:
        print("No keyframes selected!")
        result = {
            "keyframes_analyzed": 0,
            "insights": [],
            "summary": "No visual content detected"
        }
        output_path = output_dir / "visual_insights.json"
        save_json(result, output_path)
        return output_path
    
    # Save keyframe images
    print("\n[2/3] Saving keyframe images...")
    for i, kf in enumerate(keyframes):
        frame_path = frames_dir / f"frame_{i:03d}_{kf.timestamp_formatted.replace(':', '-')}.jpg"
        cv2.imwrite(str(frame_path), kf.frame)
    
    # Analyze with VLM
    insights: List[dict] = []
    
    if not skip_vlm:
        print(f"\n[3/3] Analyzing with VLM ({vlm_model})...")
        print("  [Pre-filtering: Skipping camera-only frames to save VLM calls]")
        analyzer = VLMAnalyzer(model=vlm_model)
        content_filter = ContentPreFilter()
        
        # Create skipped frames directory for debugging
        skipped_dir = output_dir / "skipped"
        skipped_dir.mkdir(exist_ok=True)
        
        vlm_calls = 0
        skipped_calls = 0
        
        for i, kf in enumerate(keyframes):
            print(f"  Frame {i+1}/{len(keyframes)} @ {kf.timestamp_formatted}: ", end="")
            
            # Pre-filter: Check if frame has visual content worth analyzing
            is_content, filter_reason = content_filter.is_likely_content_frame(kf.frame)
            
            frame_filename = f"phase5/frames/frame_{i:03d}_{kf.timestamp_formatted.replace(':', '-')}.jpg"
            
            if not is_content:
                # Skip VLM for camera-only frames
                skipped_calls += 1
                print(f"⏭️  SKIP ({filter_reason})")
                
                # Save skipped frame for debugging
                skip_reason_short = filter_reason.split('(')[0].strip().replace(' ', '_')
                skipped_filename = f"frame_{i:03d}_{kf.timestamp_formatted.replace(':', '-')}_{skip_reason_short}.jpg"
                cv2.imwrite(str(skipped_dir / skipped_filename), kf.frame)
                
                insights.append({
                    "timestamp": kf.timestamp,
                    "timestamp_formatted": kf.timestamp_formatted,
                    "frame_path": frame_filename,
                    "trigger_reason": kf.trigger_reason,
                    "content_type": "camera_only",
                    "description": "Camera view only - no additional visual content",
                    "extracted_text": kf.text_snippets or [],
                    "skipped_vlm": True,
                    "skip_reason": filter_reason,
                    "confidence": 0.3
                })
                continue
            
            # Check for duplicate content (same slide/screen shown for extended time)
            # Compare to last analyzed content frame using SSIM
            gray_current = cv2.cvtColor(kf.frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.resize(gray_current, (320, 180))
            
            if 'last_content_frame' in locals() and last_content_frame is not None:
                try:
                    similarity = ssim(last_content_frame, gray_current)
                    if similarity > 0.90:  # >90% similar = duplicate
                        skipped_calls += 1
                        print(f"⏭️  SKIP (duplicate_content, similarity={similarity:.1%})")
                        
                        # Save to skipped dir
                        skipped_filename = f"frame_{i:03d}_{kf.timestamp_formatted.replace(':', '-')}_duplicate_content.jpg"
                        cv2.imwrite(str(skipped_dir / skipped_filename), kf.frame)
                        
                        insights.append({
                            "timestamp": kf.timestamp,
                            "timestamp_formatted": kf.timestamp_formatted,
                            "frame_path": frame_filename,
                            "trigger_reason": kf.trigger_reason,
                            "content_type": "duplicate",
                            "description": f"Duplicate of previous slide (similarity={similarity:.1%})",
                            "extracted_text": kf.text_snippets or [],
                            "skipped_vlm": True,
                            "skip_reason": f"duplicate_content ({similarity:.1%} similar)",
                            "confidence": 0.3
                        })
                        continue
                except Exception as e:
                    pass  # If comparison fails, proceed with VLM
            
            # Content frame - analyze with VLM
            print(f"🔍 Analyzing ({filter_reason})... ", end="")
            vlm_calls += 1
            last_content_frame = gray_current  # Store for dedup comparison
            
            # Get context from transcript
            context = ""
            if kf.text_snippets:
                context = f"Text visible: {', '.join(kf.text_snippets[:3])}"
            
            result = analyzer.analyze_frame(kf.frame, context)
            
            insight = {
                "timestamp": kf.timestamp,
                "timestamp_formatted": kf.timestamp_formatted,
                "frame_path": frame_filename,
                "trigger_reason": kf.trigger_reason,
                "content_type": result.get("content_type", "unknown"),
                "description": result.get("description", ""),
                "extracted_text": result.get("extracted_text", []) or kf.text_snippets or [],
                "chart_analysis": result.get("chart_analysis"),
                "code_or_document": result.get("code_or_document"),
                "expressions": result.get("expressions", []),
                "gestures": result.get("gestures", []),
                "engagement": result.get("engagement", "unknown"),
                "skipped_vlm": False,
                "confidence": result.get("confidence", 0.5)
            }
            
            insights.append(insight)
            
            # Log with more context
            log_parts = [f"[{insight['content_type']}]"]
            if insight.get("chart_analysis"):
                chart_type = insight["chart_analysis"].get("type", "") if isinstance(insight["chart_analysis"], dict) else ""
                if chart_type:
                    log_parts.append(f"📊 {chart_type}")
            if insight["expressions"]:
                log_parts.append(f"😊 {', '.join(insight['expressions'][:2])}")
            if insight["gestures"]:
                log_parts.append(f"👋 {', '.join(insight['gestures'][:2])}")
            print(" ".join(log_parts))
        
        print(f"\n  Summary: {vlm_calls} VLM calls, {skipped_calls} skipped (saved {skipped_calls} API calls)")
    else:
        print("\n[3/3] Skipping VLM analysis (skip_vlm=True)")
        for i, kf in enumerate(keyframes):
            frame_filename = f"phase5/frames/frame_{i:03d}_{kf.timestamp_formatted.replace(':', '-')}.jpg"
            insights.append({
                "timestamp": kf.timestamp,
                "timestamp_formatted": kf.timestamp_formatted,
                "frame_path": frame_filename,
                "trigger_reason": kf.trigger_reason,
                "content_type": "unknown",
                "description": "",
                "extracted_text": kf.text_snippets or [],
                "confidence": 0.0
            })
    
    # Generate summary
    content_types = [i["content_type"] for i in insights]
    type_counts = {t: content_types.count(t) for t in set(content_types)}
    summary_parts = [f"{count} {ctype}" for ctype, count in type_counts.items() if count > 0]
    summary = f"{len(insights)} keyframes analyzed. Found: {', '.join(summary_parts)}."
    
    # Save results
    output = {
        "keyframes_analyzed": len(insights),
        "insights": insights,
        "summary": summary
    }
    
    output_path = output_dir / "visual_insights.json"
    save_json(output, output_path)
    
    print(f"\n✓ Visual insights saved to: {output_path}")
    print(f"  Summary: {summary}")
    
    return output_path


# ----------------------------
# CLI Entry Point
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 5: Visual Intelligence")
    parser.add_argument("--meeting-dir", type=str, required=True, help="Meeting root directory")
    parser.add_argument("--model", type=str, default="qwen2.5vl:latest", help="VLM model name")
    parser.add_argument("--max-frames", type=int, default=30, help="Max keyframes to analyze")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM analysis (sampling only)")
    
    args = parser.parse_args()
    
    run_phase5(
        meeting_root=Path(args.meeting_dir),
        vlm_model=args.model,
        max_keyframes=args.max_frames,
        skip_vlm=args.skip_vlm
    )

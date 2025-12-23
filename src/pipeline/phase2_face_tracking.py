import os
import cv2
import torch
import numpy as np
from pathlib import Path
import json
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from typing import List, Dict, Tuple
import argparse

class FaceTracker:
    def __init__(self, iou_threshold=0.5, max_frames_missing=5):
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        
    def compute_iou(self, box1, box2):
        '''Compute IoU between two boxes [x, y, w, h]'''
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        xi1 = max(box1_coords[0], box2_coords[0])
        yi1 = max(box1_coords[1], box2_coords[1])
        xi2 = min(box1_coords[2], box2_coords[2])
        yi2 = min(box1_coords[3], box2_coords[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections, frame_idx, timestamp):
        '''Update tracks with new detections'''
        self.frame_count += 1
        
        # Mark all tracks as not seen this frame
        for track_id in self.tracks:
            self.tracks[track_id]['frames_missing'] += 1
        
        matched_tracks = set()
        
        # Match detections to existing tracks
        for det in detections:
            bbox, confidence = det['bbox'], det['confidence']
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                if track['frames_missing'] > self.max_frames_missing:
                    continue
                    
                last_bbox = track['timeline'][-1]['bbox']
                iou = self.compute_iou(bbox, last_bbox)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            # Assign to track or create new
            if best_track_id and best_track_id not in matched_tracks:
                # Update existing track
                self.tracks[best_track_id]['timeline'].append({
                    'timestamp': timestamp,
                    'frame': frame_idx,
                    'confidence': float(confidence),
                    'bbox': bbox
                })
                self.tracks[best_track_id]['frames_missing'] = 0
                self.tracks[best_track_id]['last_seen'] = timestamp
                matched_tracks.add(best_track_id)
            elif best_track_id is None:
                # Create new track
                track_id = f"track_{self.next_track_id}"
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'frames_missing': 0,
                    'timeline': [{
                        'timestamp': timestamp,
                        'frame': frame_idx,
                        'confidence': confidence,
                        'bbox': bbox
                    }]
                }
                matched_tracks.add(track_id)
    
    def get_active_tracks(self):
        '''Return tracks that have been seen recently'''
        return {
            tid: track for tid, track in self.tracks.items()
            if track['frames_missing'] <= self.max_frames_missing
        }
    
    def get_all_tracks(self):
        '''Return all tracks'''
        return self.tracks


def normalize_bbox(bbox, frame_width, frame_height):
    '''Convert bbox to normalized coordinates (0-1)'''
    x, y, w, h = bbox
    return [
        float(x / frame_width),
        float(y / frame_height),
        float(w / frame_width),
        float(h / frame_height)
    ]


def denormalize_bbox(bbox_norm, frame_width, frame_height):
    '''Convert normalized bbox back to pixel coordinates'''
    x_norm, y_norm, w_norm, h_norm = bbox_norm
    return [
        int(x_norm * frame_width),
        int(y_norm * frame_height),
        int(w_norm * frame_width),
        int(h_norm * frame_height)
    ]

def select_keyframe(track, confidence_threshold=0.9):
    '''Select best keyframe for a track'''
    timeline = track['timeline']
    
    # Filter high confidence detections
    high_conf = [entry for entry in timeline if entry['confidence'] >= confidence_threshold]
    
    if not high_conf:
        # Fallback: use all detections
        high_conf = timeline
    
    # Prefer middle 60% of track
    total_entries = len(high_conf)
    start_idx = int(total_entries * 0.2)
    end_idx = int(total_entries * 0.8)
    middle_range = high_conf[start_idx:end_idx] if total_entries > 5 else high_conf
    
    # Select highest confidence in middle range
    if middle_range:
        best = max(middle_range, key=lambda x: x['confidence'])
    else:
        best = max(high_conf, key=lambda x: x['confidence'])
    
    return best

def merge_similar_tracks(tracks, similarity_threshold=0.6):
    """Merge tracks that have similar face embeddings (same person)"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    track_list = list(tracks.values())
    embeddings = []
    valid_tracks = []
    
    # Get embeddings (skip zero embeddings from failed detections)
    for track in track_list:
        emb = np.array(track.get('embedding', []))
        if len(emb) > 0 and not np.allclose(emb, 0):  # Skip zero/missing embeddings
            embeddings.append(emb)
            valid_tracks.append(track)
        else:
            # Keep tracks without valid embeddings separate
            pass
    
    if len(embeddings) < 2:
        # Not enough valid embeddings to merge
        return tracks
    
    # Compute similarity matrix
    embeddings_array = np.array(embeddings)
    similarities = cosine_similarity(embeddings_array)
    
    # Find tracks to merge
    merged_tracks = {}
    merged_ids = set()
    
    for i, track_i in enumerate(valid_tracks):
        if track_i['track_id'] in merged_ids:
            continue
            
        # Start new merged track
        merged_track = track_i.copy()
        merged_track['original_track_ids'] = [track_i['track_id']]
        
        # Find similar tracks to merge
        for j, track_j in enumerate(valid_tracks):
            if i != j and track_j['track_id'] not in merged_ids:
                if similarities[i][j] > similarity_threshold:
                    # Merge timelines
                    merged_track['timeline'].extend(track_j['timeline'])
                    merged_track['original_track_ids'].append(track_j['track_id'])
                    merged_ids.add(track_j['track_id'])
                    print(f"    Merging {track_j['track_id']} into {track_i['track_id']} (similarity: {similarities[i][j]:.2f})")
        
        # Update merged track stats
        merged_track['timeline'].sort(key=lambda x: x['timestamp'])
        merged_track['num_appearances'] = len(merged_track['timeline'])
        merged_track['first_seen'] = merged_track['timeline'][0]['timestamp']
        merged_track['last_seen'] = merged_track['timeline'][-1]['timestamp']
        
        # Re-select best keyframe from merged timeline
        keyframe_entry = select_keyframe(merged_track)
        merged_track['keyframe'] = merged_track['keyframe']  # Keep original keyframe for now
        
        merged_tracks[merged_track['track_id']] = merged_track
    
    return merged_tracks


def process_video(video_path, output_dir, sampling_fps=2):
    '''
    Phase 2: Face Detection and Tracking Pipeline
    '''
    
    print("=" * 60)
    print("PHASE 2: FACE DETECTION & TRACKING")
    print("=" * 60)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    faces_dir = Path(output_dir) / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load models
    print("\n[1/7] Loading models...")
    
    # Face detector (YOLOv8)
    print("  - Loading YOLOv8 face detector...")
    face_detector = YOLO('yolov8n-face.pt')
    
    # Face recognition (InsightFace)
    print("  - Loading InsightFace...")
    # Using 'buffalo_l' for better accuracy, typically downloads automatically if not present
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0 if device == 'cuda' else -1)
    
    # Load video
    print(f"\n[2/7] Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    
    # Calculate sampling
    frame_interval = int(fps / sampling_fps)
    sampled_frames = total_frames // frame_interval
    
    print(f"\n[3/7] Extracting & detecting faces (sampling at {sampling_fps} FPS)...")
    print(f"  Processing ~{sampled_frames} frames...")
    
    tracker = FaceTracker()
    frame_idx = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            
            # Detect faces with YOLO
            results = face_detector(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Convert to [x, y, w, h] normalized
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    bbox_norm = normalize_bbox(bbox, frame_width, frame_height)
                    
                    detections.append({
                        'bbox': bbox_norm,
                        'confidence': conf
                    })
            
            # Update tracker
            tracker.update(detections, frame_idx, timestamp)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{sampled_frames} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"  Completed! Processed {processed_count} frames")
    
    # Get all tracks and filter
    print(f"\n[4/7] Filtering tracks...")
    all_tracks = tracker.get_all_tracks()
    
    # Filter out tracks with too few appearances (likely false positives)
    MIN_APPEARANCES = 10
    filtered_tracks = {
        tid: track for tid, track in all_tracks.items()
        if len(track['timeline']) >= MIN_APPEARANCES
    }
    
    print(f"  Found {len(all_tracks)} raw tracks")
    # print(f"  Filtered to {len(filtered_tracks)} tracks (removed {len(all_tracks) - len(filtered_tracks)} spurious detections)")
    
    # Select keyframes and extract embeddings
    print(f"\n[5/7] Selecting keyframes and extracting embeddings...")
    
    output_tracks = []
    
    for track_id, track in filtered_tracks.items():
        num_appearances = len(track['timeline'])
        # print(f"  Processing {track_id}: {num_appearances} appearances")
        
        # Select keyframe
        keyframe_entry = select_keyframe(track)
        
        # Re-open video to extract keyframe
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_entry['frame'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"    Warning: Could not extract frame for {track_id}")
            continue
        
        # Crop face from frame
        bbox_pixels = denormalize_bbox(keyframe_entry['bbox'], frame_width, frame_height)
        x, y, w, h = bbox_pixels
        
        # Add padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame_width, x + w + padding)
        y2 = min(frame_height, y + h + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Save keyframe
        keyframe_path = faces_dir / f"{track_id}_keyframe.jpg"
        if face_crop.size > 0:
            cv2.imwrite(str(keyframe_path), face_crop)
        else:
            print(f"Warning: Empty face crop for {track_id}")
        
        # Extract embedding with InsightFace
        try:
            faces = face_analyzer.get(face_crop)
            if len(faces) > 0:
                embedding = faces[0].embedding.flatten().tolist()
            else:
                # print(f"    Warning: No face detected in keyframe for {track_id}, using zero embedding")
                embedding = [0.0] * 512
        except Exception as e:
            print(f"InsightFace error for {track_id}: {e}")
            embedding = [0.0] * 512
        
        # Build output track
        output_track = {
            'track_id': track_id,
            'num_appearances': num_appearances,
            'first_seen': track['first_seen'],
            'last_seen': track['last_seen'],
            'keyframe': {
                'path': str(keyframe_path.relative_to(Path(output_dir).parent)), # Relative to meeting root
                'timestamp': keyframe_entry['timestamp'],
                'frame': keyframe_entry['frame'],
                'confidence': float(keyframe_entry['confidence']),
                'bbox': keyframe_entry['bbox']
            },
            'embedding': embedding,
            'timeline': track['timeline']
        }
        
        output_tracks.append(output_track)
    
    # Merge similar tracks
    print(f"\n[6/7] Merging similar tracks (same person)...")
    
    # Convert list to dict for merging
    tracks_dict = {t['track_id']: t for t in output_tracks}
    merged_tracks_dict = merge_similar_tracks(tracks_dict, similarity_threshold=0.6)
    output_tracks = list(merged_tracks_dict.values())
    
    print(f"  Final tracks: {len(output_tracks)} unique people")
    
    # Save results
    print(f"\n[7/7] Saving results...")
    
    output = {
        'video_file': os.path.basename(video_path),
        'duration': float(duration),
        'fps': float(fps),
        'sampling_fps': sampling_fps,
        'total_frames': total_frames,
        'sampled_frames': processed_count,
        'num_tracks': len(output_tracks),
        'tracks': output_tracks
    }
    
    output_file = Path(output_dir) / "faces.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Face Detection & Tracking")
    parser.add_argument("--in", dest="input_file", required=True, help="Input video file path")
    parser.add_argument("--outdir", dest="output_dir", required=True, help="Output directory for faces.json and images")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not input_file.exists():
        print(f"Error: Video file not found: {input_file}")
        exit(1)
        
    try:
        process_video(input_file, output_dir, sampling_fps=2)
        print(f"\n[OK] Phase 2 Complete")
    except Exception as e:
        print(f"\n[ERROR] Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

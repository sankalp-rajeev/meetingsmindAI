import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import whisper
import json
from pathlib import Path
import torchaudio
import numpy as np
import subprocess
import tempfile

# Load environment variables
load_dotenv()

def extract_audio_from_video(video_path, output_audio_path=None):
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path: Path to video file
        output_audio_path: Optional path for output audio. If None, uses temp file
    
    Returns:
        Path to extracted audio file
    """
    if output_audio_path is None:
        # Create temporary audio file
        temp_dir = tempfile.gettempdir()
        output_audio_path = os.path.join(temp_dir, f"{Path(video_path).stem}_audio.wav")
    
    print(f"Extracting audio from video: {video_path}")
    
    # FFmpeg command to extract audio
    # -i: input file
    # -vn: disable video
    # -acodec pcm_s16le: audio codec (16-bit PCM)
    # -ar 16000: sample rate 16kHz (optimal for Whisper)
    # -ac 1: mono audio
    # -y: overwrite output file
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # no video
        '-acodec', 'pcm_s16le',  # audio codec
        '-ar', '16000',  # sample rate
        '-ac', '1',  # mono
        '-y',  # overwrite
        output_audio_path
    ]
    
    try:
        # Run ffmpeg
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"[OK] Audio extracted to: {output_audio_path}")
        return output_audio_path
    
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  - macOS: brew install ffmpeg\n"
            "  - Windows: Download from https://ffmpeg.org/download.html"
        )

def is_video_file(file_path):
    """Check if file is a video based on extension"""
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
        '.webm', '.m4v', '.mpg', '.mpeg', '.3gp'
    }
    return Path(file_path).suffix.lower() in video_extensions

def is_audio_file(file_path):
    """Check if file is an audio file based on extension"""
    audio_extensions = {
        '.wav', '.mp3', '.m4a', '.flac', '.ogg', 
        '.aac', '.wma', '.opus'
    }
    return Path(file_path).suffix.lower() in audio_extensions

def process_audio(audio_path, output_dir="data/output", cleanup_temp=True):
    """
    Process audio file for speaker diarization and transcription
    
    Args:
        audio_path: Path to audio file (or video file - audio will be extracted)
        output_dir: Directory to save output files
        cleanup_temp: Whether to delete temporary extracted audio files
    
    Returns:
        Path to output JSON file
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get HuggingFace token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    print("\n" + "="*60)
    print("PHASE 1: AUDIO PROCESSING")
    print("="*60)
    
    # Check if input is video or audio
    original_input = audio_path
    temp_audio_file = None
    
    if is_video_file(audio_path):
        print("\n[0/3] Video file detected - extracting audio...")
        temp_audio_file = extract_audio_from_video(audio_path)
        audio_path = temp_audio_file
    elif is_audio_file(audio_path):
        print("\n[0/3] Audio file detected")
    else:
        raise ValueError(
            f"Unsupported file format: {Path(audio_path).suffix}\n"
            "Supported formats: Video (mp4, avi, mov, mkv, etc.) or "
            "Audio (wav, mp3, m4a, flac, etc.)"
        )
    
    # Pre-load audio file
    print("\n[0/3] Loading audio file...")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    audio_dict = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }
    duration_seconds = waveform.shape[1] / sample_rate
    print(f"[0/3] Audio loaded: {duration_seconds:.1f} seconds")
    
    # Step 1: Speaker Diarization
    print("\n[1/3] Loading speaker diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    diarization_pipeline.to(torch.device(device))
    
    print("[1/3] Running speaker diarization...")
    diarization_output = diarization_pipeline(audio_dict)
    annotation = diarization_output.speaker_diarization
    
    # Extract speaker segments
    speaker_segments = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        speaker_segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": label
        })
    
    num_speakers = len(set([s['speaker'] for s in speaker_segments]))
    print(f"[1/3] Found {num_speakers} speakers")
    print(f"[1/3] Total segments: {len(speaker_segments)}")
    
    # Step 2: Speech-to-Text
    print("\n[2/3] Loading Whisper model...")
    whisper_model = whisper.load_model("base", device=device)
    
    print("[2/3] Transcribing audio (this may take a few minutes)...")
    
    # Pre-process audio for Whisper (resample to 16kHz, convert to numpy)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        waveform_16k = resampler(waveform.to(device))
    else:
        waveform_16k = waveform
    
    # Convert to numpy array (what Whisper expects)
    audio_np = waveform_16k.squeeze().cpu().numpy()
    
    # Transcribe using pre-loaded audio
    result = whisper_model.transcribe(
        audio_np,
        language="en",
        task="transcribe",
        verbose=False
    )
    
    print(f"[2/3] Transcription complete: {len(result['segments'])} segments")
    
    # Step 3: Merge diarization + transcription
    print("\n[3/3] Merging speaker labels with transcript...")
    
    merged_segments = []
    for segment in result["segments"]:
        # Find which speaker was talking during this segment
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_mid = (seg_start + seg_end) / 2
        
        # Find speaker at midpoint of segment
        speaker = "UNKNOWN"
        for spk_seg in speaker_segments:
            if spk_seg["start"] <= seg_mid <= spk_seg["end"]:
                speaker = spk_seg["speaker"]
                break
        
        merged_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "speaker": speaker,
            "text": segment["text"].strip()
        })
    
    # Create output
    output = {
        "file": os.path.basename(original_input),
        "duration": duration_seconds,
        "num_speakers": num_speakers,
        "segments": merged_segments
    }
    
    # Save results
    output_file = Path(output_dir) / f"{Path(original_input).stem}_transcript.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[3/3] Results saved to: {output_file}")
    
    # Cleanup temporary audio file if it was created
    if temp_audio_file and cleanup_temp:
        try:
            os.remove(temp_audio_file)
            print(f"[3/3] Cleaned up temporary audio file")
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_audio_file}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Speakers: {num_speakers}")
    print(f"Total segments: {len(merged_segments)}")
    
    # Speaker statistics
    print("\nSpeaker breakdown:")
    speaker_times = {}
    for seg in merged_segments:
        speaker = seg['speaker']
        duration = seg['end'] - seg['start']
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
    
    for speaker, total_time in sorted(speaker_times.items()):
        print(f"  {speaker}: {total_time:.1f}s ({total_time/duration_seconds*100:.1f}%)")
    
    print("\nFirst 5 segments:")
    for seg in merged_segments[:5]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text']}")
    
    if len(merged_segments) > 5:
        print(f"\n... and {len(merged_segments) - 5} more segments")
    
    return output_file

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Phase 1: audio/video -> transcript.json")
    parser.add_argument("--in", dest="input_file", required=True, help="Input audio/video file path (.wav/.mp3/.mp4 etc.)")
    parser.add_argument("--out", dest="output_file", required=True, help="Output transcript JSON path")
    parser.add_argument("--outdir", dest="output_dir", required=True, help="Output directory for intermediate/produced files")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        exit(1)

    print(f"Processing: {input_file}")
    if is_video_file(input_file):
        print("Video file detected - will extract audio first")

    try:
        # Your existing function should return a path to a produced JSON.
        produced = process_audio(input_file, output_dir=args.output_dir)

        # Ensure it ends up at args.output_file
        produced_path = Path(produced)
        out_path = Path(output_file)
        
        # Ensure parent exists handled by process_audio inside output_dir mostly, but let's be safe for the rename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Produced by process_audio(): {produced_path}")
        print(f"Final output (--out): {out_path}")

        if produced_path.resolve() != out_path.resolve():
            # copy/rename the produced file to the requested output path
            out_path.write_bytes(produced_path.read_bytes())

        print(f"\n Phase 1 Complete!")
        print(f" Output: {out_path}")

    except Exception as e:
        print(f"\n Error: {e}")
        exit(1)


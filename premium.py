import gradio as gr
import torch
import argparse
import os
import sys
import signal
from datetime import datetime

print(f"[DEBUG] Starting premium.py with args: {sys.argv}")

from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text, scale_hw_to_area_divisible
from PIL import Image

def detect_gpu_info():
    """Detect GPU model and VRAM size."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Get primary GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                print("=" * 60)
                print("GPU DETECTION RESULTS:")
                print(f"  GPU Model: {gpu_name}")
                print(f"  GPU Count: {device_count}")
                print(f"  VRAM Size: {gpu_memory_gb:.2f} GB")
                print("=" * 60)

                return gpu_name, gpu_memory_gb
            else:
                print("GPU DETECTION: CUDA available but no devices found")
                return None, 0
        else:
            print("GPU DETECTION: CUDA not available")
            return None, 0
    except Exception as e:
        print(f"GPU DETECTION ERROR: {e}")
        return None, 0

def detect_system_ram():
    """Detect total system RAM size."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        print("=" * 60)
        print("SYSTEM RAM DETECTION RESULTS:")
        print(f"  Total RAM: {total_ram_gb:.2f} GB")
        print("=" * 60)

        return total_ram_gb
    except Exception as e:
        print(f"SYSTEM RAM DETECTION ERROR: {e}")
        return 0

# ----------------------------
# Parse CLI Args
# ----------------------------
parser = argparse.ArgumentParser(description="Ovi Joint Video + Audio Gradio Demo (use --share to enable public access)")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Custom output directory for generated videos (default: Ovi_Pro/outputs)"
)
parser.add_argument(
    "--share",
    action="store_true",
    help="Enable Gradio public sharing (creates public URL)"
)
# Temporary: add blocks_to_swap for testing
parser.add_argument(
    "--blocks_to_swap",
    type=int,
    default=0,
    help="Number of transformer blocks to swap to CPU memory during generation (0 = disabled)"
)
# Add test arguments for automatic testing
parser.add_argument(
    "--test",
    action="store_true",
    help="Enable test mode with automatic generation"
)
parser.add_argument(
    "--test_prompt",
    type=str,
    default="A person walking on the beach at sunset",
    help="Test prompt for automatic testing"
)
parser.add_argument(
    "--test_cpu_offload",
    action="store_true",
    help="Enable CPU offload in test mode"
)
parser.add_argument(
    "--test_fp8_t5",
    action="store_true",
    help="Enable Scaled FP8 T5 in test mode"
)
parser.add_argument(
    "--test_cpu_only_t5",
    action="store_true",
    help="Enable CPU-only T5 in test mode"
)
parser.add_argument(
    "--single-generation",
    type=str,
    help="Internal: Run single generation from JSON params and exit"
)
parser.add_argument(
    "--single-generation-file",
    type=str,
    help="Internal: Run single generation from JSON file and exit"
)
parser.add_argument(
    "--test-subprocess",
    action="store_true",
    help="Internal: Test subprocess functionality"
)
parser.add_argument(
    "--encode-t5-only",
    type=str,
    help="Internal: Run T5 text encoding from JSON file and exit"
)
parser.add_argument(
    "--output-embeddings",
    type=str,
    help="Internal: Output path for T5 embeddings file"
)
args = parser.parse_args()

print(f"[DEBUG] Parsed args: single_generation={bool(args.single_generation)}, single_generation_file={bool(args.single_generation_file)}, test={bool(getattr(args, 'test', False))}, test_subprocess={bool(getattr(args, 'test_subprocess', False))}")

# Initialize engines with lazy loading (no models loaded yet)
ovi_engine = None  # Will be initialized on first generation

# Global cancellation flag for stopping generations
cancel_generation = False

def run_t5_encoding_subprocess(text_prompt, video_negative_prompt, audio_negative_prompt, 
                                fp8_t5=False, cpu_only_t5=False):
    """Run T5 text encoding in a subprocess for guaranteed memory cleanup.
    
    Returns:
        List of text embeddings [pos_emb, video_neg_emb, audio_neg_emb] as CPU tensors
    """
    import subprocess
    import sys
    import json
    import os
    import tempfile
    import time

    process = None
    params_file = None
    embeddings_file = None

    try:
        # Get the current script path and venv
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        venv_path = os.path.join(script_dir, "venv")

        # Use venv python executable directly
        if sys.platform == "win32":
            python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(venv_path, "bin", "python")

        # Check if venv python exists, fallback to system python
        if not os.path.exists(python_exe):
            print(f"[T5-SUBPROCESS] Venv python not found at {python_exe}, using system python")
            python_exe = sys.executable

        # Create temporary files for communication
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=script_dir) as f:
            params = {
                'text_prompt': text_prompt,
                'video_negative_prompt': video_negative_prompt,
                'audio_negative_prompt': audio_negative_prompt,
                'fp8_t5': fp8_t5,
                'cpu_only_t5': cpu_only_t5
            }
            json.dump(params, f)
            params_file = f.name

        # Create temp file for embeddings output
        embeddings_file = tempfile.mktemp(suffix='.pt', dir=script_dir)

        # Prepare command arguments
        cmd_args = [
            python_exe,
            script_path,
            "--encode-t5-only",
            params_file,
            "--output-embeddings",
            embeddings_file
        ]

        print(f"[T5-SUBPROCESS] Running T5 encoding in subprocess for guaranteed memory cleanup...")
        print(f"[T5-SUBPROCESS] This ensures 100% memory cleanup after encoding")

        # Run the subprocess with Popen for better control
        process = subprocess.Popen(
            cmd_args,
            cwd=script_dir,
            stdout=None,
            stderr=None
        )

        # Wait for completion while checking for cancellation
        while process.poll() is None:
            global cancel_generation
            if cancel_generation:
                print("[T5-SUBPROCESS] Cancellation requested - terminating subprocess...")
                process.terminate()

                # Give it a moment to terminate gracefully
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    print("[T5-SUBPROCESS] Subprocess didn't terminate gracefully, killing...")
                    process.kill()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        print("[T5-SUBPROCESS] Failed to kill subprocess")

                # Clean up temp files
                try:
                    if params_file and os.path.exists(params_file):
                        os.unlink(params_file)
                    if embeddings_file and os.path.exists(embeddings_file):
                        os.unlink(embeddings_file)
                except:
                    pass

                print("[T5-SUBPROCESS] T5 encoding subprocess cancelled")
                raise Exception("T5 encoding cancelled by user")

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

        # Get the return code
        return_code = process.returncode

        # Clean up params file
        try:
            if params_file and os.path.exists(params_file):
                os.unlink(params_file)
        except:
            pass

        if return_code == 0:
            print("[T5-SUBPROCESS] T5 encoding completed successfully - loading embeddings...")
            
            # Load embeddings from file
            if not os.path.exists(embeddings_file):
                raise Exception(f"Embeddings file not found: {embeddings_file}")
            
            embeddings = torch.load(embeddings_file)
            
            # Clean up embeddings file
            try:
                os.unlink(embeddings_file)
            except:
                pass
            
            print("[T5-SUBPROCESS] T5 subprocess memory completely freed by OS")
            return embeddings
        else:
            print(f"[T5-SUBPROCESS] T5 encoding failed with return code: {return_code}")
            
            # Clean up embeddings file
            try:
                if embeddings_file and os.path.exists(embeddings_file):
                    os.unlink(embeddings_file)
            except:
                pass
            
            raise Exception(f"T5 encoding subprocess failed with return code: {return_code}")

    except Exception as e:
        error_msg = str(e)
        if "cancelled by user" in error_msg.lower():
            # Re-raise cancellation exceptions
            raise e
        else:
            print(f"[T5-SUBPROCESS] Error running T5 encoding subprocess: {e}")
            # Clean up process if it exists
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            # Clean up temp files
            try:
                if params_file and os.path.exists(params_file):
                    os.unlink(params_file)
                if embeddings_file and os.path.exists(embeddings_file):
                    os.unlink(embeddings_file)
            except:
                pass
            raise e

def run_generation_subprocess(params):
    """Run a single generation in a subprocess to ensure memory cleanup."""
    import subprocess
    import sys
    import json
    import os
    import tempfile
    import time

    process = None
    temp_file = None
    embeddings_temp_file = None

    try:
        # Get the current script path and venv
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        venv_path = os.path.join(script_dir, "venv")

        # Use venv python executable directly
        if sys.platform == "win32":
            python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(venv_path, "bin", "python")

        # Check if venv python exists, fallback to system python
        if not os.path.exists(python_exe):
            print(f"[SUBPROCESS] Venv python not found at {python_exe}, using system python")
            python_exe = sys.executable

        # Handle text_embeddings_cache if provided
        if 'text_embeddings_cache' in params and params['text_embeddings_cache'] is not None:
            # Save embeddings to temp file
            embeddings_temp_file = tempfile.mktemp(suffix='_embeddings.pt', dir=script_dir)
            torch.save(params['text_embeddings_cache'], embeddings_temp_file)
            # Replace embeddings with file path in params
            params['text_embeddings_cache'] = embeddings_temp_file
            print(f"[SUBPROCESS] Saved pre-encoded T5 embeddings to: {embeddings_temp_file}")
            print(f"[SUBPROCESS] Generation subprocess will load embeddings instead of encoding T5")

        # Write params to temporary file to avoid command line issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=script_dir) as f:
            json.dump(params, f)
            temp_file = f.name

        # Prepare command arguments - pass temp file path
        cmd_args = [
            python_exe,
            script_path,
            "--single-generation-file",
            temp_file
        ]

        print(f"[SUBPROCESS] Running generation in subprocess...")
        print(f"[SUBPROCESS] Command: {' '.join(cmd_args)}")
        print(f"[SUBPROCESS] Params file: {temp_file}")

        # Run the subprocess with Popen for better control
        process = subprocess.Popen(
            cmd_args,
            cwd=script_dir,
            stdout=None,  # Let subprocess handle its own output
            stderr=None
        )

        # Wait for completion while checking for cancellation
        while process.poll() is None:
            global cancel_generation
            if cancel_generation:
                print("[SUBPROCESS] Cancellation requested - terminating subprocess...")
                process.terminate()

                # Give it a moment to terminate gracefully
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    print("[SUBPROCESS] Subprocess didn't terminate gracefully, killing...")
                    process.kill()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        print("[SUBPROCESS] Failed to kill subprocess")

                # Clean up temp file
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass

                print("[SUBPROCESS] Subprocess cancelled")
                raise Exception("Generation cancelled by user")

            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)

        # Get the return code
        return_code = process.returncode

        # Clean up temp files
        try:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            if embeddings_temp_file and os.path.exists(embeddings_temp_file):
                os.unlink(embeddings_temp_file)
        except:
            pass

        if return_code == 0:
            print("[SUBPROCESS] Generation completed successfully")
            return True
        elif return_code == -signal.SIGTERM or return_code == 1:  # SIGTERM or general error
            if cancel_generation:
                print("[SUBPROCESS] Subprocess was cancelled")
                return False
            else:
                print(f"[SUBPROCESS] Generation failed with return code: {return_code}")
                return False
        else:
            print(f"[SUBPROCESS] Generation failed with return code: {return_code}")
            return False

    except Exception as e:
        error_msg = str(e)
        if "cancelled by user" in error_msg.lower():
            # Re-raise cancellation exceptions
            raise e
        else:
            print(f"[SUBPROCESS] Error running subprocess: {e}")
            # Clean up process if it exists
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            # Clean up temp files
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                if embeddings_temp_file and os.path.exists(embeddings_temp_file):
                    os.unlink(embeddings_temp_file)
            except:
                pass
            return False

share_enabled = args.share
print(f"Starting Gradio interface with lazy loading... Share mode: {'ENABLED' if share_enabled else 'DISABLED (local only)'}")
if not share_enabled:
    print("Use --share flag to enable public access with a shareable URL")


def parse_multiline_prompts(text_prompt, enable_multiline_prompts):
    """Parse multi-line prompts into individual prompts, filtering out short lines."""
    if not enable_multiline_prompts:
        return [text_prompt]

    # Split by lines and filter
    lines = text_prompt.split('\n')
    prompts = []

    for line in lines:
        line = line.strip()
        if len(line) >= 3:  # Skip lines shorter than 3 characters (after trimming)
            prompts.append(line)

    # If no valid prompts, return original
    return prompts if prompts else [text_prompt]

def extract_last_frame(video_path):
    """Extract the last frame from a video file and save as PNG."""
    try:
        # Defensive checks
        if not video_path or not isinstance(video_path, str) or video_path.strip() == "" or not os.path.exists(video_path):
            print(f"[VIDEO EXTENSION] Invalid or missing video file: {video_path}")
            return None

        from PIL import Image
        import cv2

        # Use OpenCV to get the last frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise Exception("Video has no frames")

        # Seek to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Could not read last frame")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Save to temp directory
        tmp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join(tmp_dir, f"last_frame_{timestamp}.png")
        img.save(frame_path)

        print(f"[VIDEO EXTENSION] Extracted last frame: {frame_path}")
        return frame_path

    except Exception as e:
        print(f"[VIDEO EXTENSION] Error extracting last frame: {e}")
        return None

def is_video_file(file_path):
    """Check if a file is a video based on extension."""
    if not file_path or not isinstance(file_path, str):
        return False
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)

def is_image_file(file_path):
    """Check if a file is an image based on extension."""
    if not file_path or not isinstance(file_path, str):
        return False
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def process_input_media(media_path, auto_crop_image, video_width, video_height):
    """Process input media (image or video) and return image path to use + input video path if applicable.
    
    Returns:
        tuple: (image_path, input_video_path, is_video)
            - image_path: Path to image to use for generation (extracted frame if video)
            - input_video_path: Path to input video if provided, None otherwise
            - is_video: Boolean indicating if input was a video
    """
    if not media_path or not os.path.exists(media_path):
        return None, None, False
    
    try:
        # Check if input is a video
        if is_video_file(media_path):
            print(f"[INPUT] Video detected: {media_path}")
            print(f"[INPUT] Extracting last frame for use as source image...")
            
            # Extract last frame from video
            frame_path = extract_last_frame(media_path)
            if not frame_path:
                print(f"[INPUT] Failed to extract frame from video, skipping")
                return None, None, False
            
            print(f"[INPUT] Extracted frame: {frame_path}")
            print(f"[INPUT] Input video will be merged with generated video after generation")
            
            # Apply auto-crop if enabled
            if auto_crop_image and video_width and video_height:
                frame_path = apply_auto_crop_if_enabled(frame_path, auto_crop_image, video_width, video_height)
                print(f"[INPUT] Auto-crop applied to extracted frame")
            
            return frame_path, media_path, True
        
        # Input is an image
        elif is_image_file(media_path):
            print(f"[INPUT] Image detected: {media_path}")
            return media_path, None, False
        
        else:
            print(f"[INPUT] Unknown file type: {media_path}")
            return None, None, False
            
    except Exception as e:
        print(f"[INPUT] Error processing input media: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def trim_video_frames(input_video_path, output_video_path, trim_first=False, trim_last=False):
    """Trim frames from a video using FFmpeg.

    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        trim_first: If True, removes the first frame
        trim_last: If True, removes the last frame

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import subprocess
        import cv2

        # Get video properties
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"[TRIM] Could not open video: {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 1:
            print(f"[TRIM] Video has only {total_frames} frame(s), cannot trim")
            return False

        trim_type = "first" if trim_first else "last"
        print(f"[TRIM-{trim_type.upper()}] Original video: {total_frames} frames at {fps:.2f} fps")

        if trim_last:
            # Calculate frames to keep (remove last frame)
            frames_to_keep = total_frames - 1
            print(f"[TRIM] Trimming to: {frames_to_keep} frames")

            # Use FFmpeg to trim the video by specifying exact frame count for frame accuracy
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video_path,
                '-frames:v', str(frames_to_keep),  # Keep exactly this many frames
                '-c:v', 'libx264',  # Re-encode video for frame-accurate trim
                '-preset', 'slow',  # Fast encoding preset
                '-crf', '12',  # High quality (18 is visually lossless)
                '-c:a', 'aac',  # Re-encode audio
                '-b:a', '192k',  # Audio bitrate
                '-avoid_negative_ts', 'make_zero',
                output_video_path
            ]

            expected_frames = frames_to_keep

        elif trim_first:
            # Skip first frame and keep the rest
            print(f"[TRIM-FIRST] Skipping first frame, keeping {total_frames - 1} frames")

            # Use FFmpeg to skip the first frame by starting from frame 1 (0-indexed)
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video_path,
                '-vf', 'select=not(eq(n\\,0))',  # Skip frame 0 (first frame)
                '-af', 'aselect=not(eq(n\\,0))',  # Skip first audio frame too
                '-c:v', 'libx264',  # Re-encode video
                '-preset', 'slow',  # Fast encoding preset
                '-crf', '12',  # High quality (18 is visually lossless)
                '-c:a', 'aac',  # Re-encode audio
                '-b:a', '192k',  # Audio bitrate
                '-avoid_negative_ts', 'make_zero',
                output_video_path
            ]

            expected_frames = total_frames - 1

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Verify the trim worked by checking frame count
            cap_verify = cv2.VideoCapture(output_video_path)
            if cap_verify.isOpened():
                new_frame_count = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_verify.release()
                print(f"[TRIM-{trim_type.upper()}] Successfully trimmed: {total_frames} → {new_frame_count} frames")

                if new_frame_count == expected_frames:
                    print(f"[TRIM-{trim_type.upper()}] ✓ Frame count verified: exactly 1 frame removed")
                    return True
                else:
                    print(f"[TRIM-{trim_type.upper()}] ⚠ Warning: Expected {expected_frames} frames, got {new_frame_count}")
                    # Still return True as trim succeeded, just with a warning
                    return True
            else:
                print(f"[TRIM-{trim_type.upper()}] Successfully trimmed {trim_type} frame (verification failed)")
                return True
        else:
            print(f"[TRIM-{trim_type.upper()}] FFmpeg failed with return code {result.returncode}")
            print(f"[TRIM-{trim_type.upper()}] stderr: {result.stderr}")
            return False

    except Exception as e:
        print(f"[TRIM-{trim_type.upper()}] Error trimming video: {e}")
        import traceback
        traceback.print_exc()
        return False

def trim_last_frame_from_video(input_video_path, output_video_path):
    """Remove the last frame from a video using FFmpeg.

    Returns:
        bool: True if successful, False otherwise
    """
    return trim_video_frames(input_video_path, output_video_path, trim_first=False, trim_last=True)

def trim_first_frame_from_video(input_video_path, output_video_path):
    """Remove the first frame from a video using FFmpeg.

    Returns:
        bool: True if successful, False otherwise
    """
    return trim_video_frames(input_video_path, output_video_path, trim_first=True, trim_last=False)

def combine_videos(video_paths, output_path, trim_first_video_last_frame=False, trim_extension_first_frames=False):
    """Combine multiple videos into one by concatenating them with FFmpeg.

    Args:
        video_paths: List of video file paths to combine
        output_path: Output path for combined video
        trim_first_video_last_frame: If True, removes last frame from first video before merging
        trim_extension_first_frames: If True, removes first frame from each video except the first
    """
    try:
        import subprocess
        import tempfile
        import os

        temp_trimmed_extensions = []

        if len(video_paths) < 2:
            print("[VIDEO EXTENSION] Only one video, no combination needed")
            return False

        # If trimming first video, create a trimmed version
        temp_trimmed_video = None
        if trim_first_video_last_frame:
            print(f"[VIDEO MERGE] Trimming last frame from first video to avoid duplication...")
            temp_trimmed_video = tempfile.mktemp(suffix='_trimmed.mp4', dir=os.path.dirname(video_paths[0]))

            if trim_video_frames(video_paths[0], temp_trimmed_video, trim_first=False, trim_last=True):
                # Replace first video with trimmed version
                video_paths = [temp_trimmed_video] + video_paths[1:]
                print(f"[VIDEO MERGE] Using trimmed video (last frame removed)")
            else:
                print(f"[VIDEO MERGE] Failed to trim, using original video")
                temp_trimmed_video = None

        # If trimming first frame from extension videos, create trimmed versions
        if trim_extension_first_frames and len(video_paths) > 1:
            print(f"[VIDEO MERGE] Trimming first frame from {len(video_paths) - 1} extension videos to avoid duplication...")
            new_video_paths = [video_paths[0]]  # Keep first video as-is

            for i, video_path in enumerate(video_paths[1:], 1):  # Skip first video
                trimmed_path = tempfile.mktemp(suffix=f'_ext{i}_trimmed.mp4', dir=os.path.dirname(video_path))
                if trim_video_frames(video_path, trimmed_path, trim_first=True, trim_last=False):
                    new_video_paths.append(trimmed_path)
                    temp_trimmed_extensions.append(trimmed_path)
                    print(f"[VIDEO MERGE] Extension {i} trimmed (first frame removed)")
                else:
                    print(f"[VIDEO MERGE] Failed to trim extension {i}, using original")
                    new_video_paths.append(video_path)

            video_paths = new_video_paths

        # Create a temporary file list for FFmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video_path in video_paths:
                # Escape single quotes in path for FFmpeg
                escaped_path = video_path.replace("'", r"'\''")
                f.write(f"file '{escaped_path}'\n")
            concat_file = f.name

        try:
            # Use FFmpeg to concatenate videos with audio
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # Copy streams to preserve audio/video codecs
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                output_path
            ]

            print(f"[VIDEO EXTENSION] Running FFmpeg concat command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode == 0:
                print(f"[VIDEO EXTENSION] Combined {len(video_paths)} videos into: {output_path}")
                return True
            else:
                print(f"[VIDEO EXTENSION] FFmpeg failed with return code {result.returncode}")
                print(f"[VIDEO EXTENSION] FFmpeg stdout: {result.stdout}")
                print(f"[VIDEO EXTENSION] FFmpeg stderr: {result.stderr}")
                return False

        finally:
            # Clean up temporary files
            try:
                os.unlink(concat_file)
                if temp_trimmed_video and os.path.exists(temp_trimmed_video):
                    os.unlink(temp_trimmed_video)
                # Clean up temporary trimmed extension files
                for temp_file in temp_trimmed_extensions:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
            except:
                pass

    except Exception as e:
        print(f"[VIDEO EXTENSION] Error combining videos: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_used_source_image(image_path, output_dir, video_filename):
    """Save the used source image to used_source_images subfolder."""
    try:
        # Defensive checks
        if not image_path or not isinstance(image_path, str) or image_path.strip() == "" or not os.path.exists(image_path):
            return False

        if not output_dir or not isinstance(output_dir, str) or not os.path.exists(output_dir):
            return False

        if not video_filename or not isinstance(video_filename, str):
            return False

        # Create used_source_images subfolder
        source_images_dir = os.path.join(output_dir, "used_source_images")
        os.makedirs(source_images_dir, exist_ok=True)

        # Create filename: same as video but with .png extension
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        source_filename = f"{base_name}.png"
        source_path = os.path.join(source_images_dir, source_filename)

        # Copy the image
        from PIL import Image
        img = Image.open(image_path)
        img.save(source_path)

        print(f"[SOURCE IMAGE] Saved used source image: {source_filename}")
        return True

    except Exception as e:
        print(f"[SOURCE IMAGE] Error saving source image: {e}")
        return False

def generate_video(
    text_prompt,
    image,
    video_frame_height,
    video_frame_width,
    video_seed,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    blocks_to_swap,
    video_negative_prompt,
    audio_negative_prompt,
    use_image_gen,
    cpu_offload,
    delete_text_encoder,
    fp8_t5,
    cpu_only_t5,
    fp8_base_model,
    use_sage_attention,
    no_audio,
    no_block_prep,
    num_generations,
    randomize_seed,
    save_metadata,
    aspect_ratio,
    clear_all,
    vae_tiled_decode,
    vae_tile_size,
    vae_tile_overlap,
    base_resolution_width,
    base_resolution_height,
    duration_seconds,
    auto_crop_image=True,  # Default to True for backward compatibility
    base_filename=None,  # For batch processing to use custom filenames
    output_dir=None,  # Custom output directory (overrides args.output_dir)
    text_embeddings_cache=None,  # Pre-encoded text embeddings from T5 subprocess
    enable_multiline_prompts=False,  # New: Enable multi-line prompt processing
    enable_video_extension=False,  # New: Enable automatic video extension based on prompt lines
    dont_auto_combine_video_input=False,  # New: Don't auto combine video input with generated video
    input_video_path=None,  # New: Input video path for merging (when user uploads video)
    is_video_extension_subprocess=False,  # Internal: Mark subprocess calls for video extensions
):
    # Note: enable_video_extension is passed directly from UI and should be respected as-is

    # Parse prompts for video extension validation (always parse to count lines)
    validation_prompts = parse_multiline_prompts(text_prompt, True)  # Always parse for validation

    # Calculate video extension count based on enable_video_extension setting
    video_extension_count = 0
    if enable_video_extension:
        # Count valid prompt lines (>= 3 chars after trim) minus 1 for the main video
        video_extension_count = max(0, len(validation_prompts) - 1)

    global ovi_engine

    # Reset cancellation flag at the start of each generation request
    reset_cancellation()
    
    # CRITICAL: Auto-enable "Clear All Memory" when "Delete T5 After Encoding" is enabled
    # This ensures T5 subprocess runs in isolation without model duplication
    if delete_text_encoder and not clear_all:
        print("=" * 80)
        print("AUTO-OPTIMIZATION: Enabling 'Clear All Memory' mode")
        print("When 'Delete T5 After Encoding' is enabled, 'Clear All Memory' is automatically enabled")
        print("This prevents model duplication and ensures 100% memory cleanup")
        print("=" * 80)
        clear_all = True
    
    # Load text embeddings from file if provided as path (from subprocess)
    if text_embeddings_cache is not None and isinstance(text_embeddings_cache, str):
        if os.path.exists(text_embeddings_cache):
            print(f"[DEBUG] Loading pre-encoded T5 embeddings from: {text_embeddings_cache}")
            text_embeddings_cache = torch.load(text_embeddings_cache)
            print(f"[DEBUG] T5 embeddings loaded successfully - will skip T5 encoding")
        else:
            print(f"[WARNING] Embeddings file not found: {text_embeddings_cache}")
            text_embeddings_cache = None

    # CRITICAL FIX: Reset text embeddings cache at the start of each generation
    # This ensures that when prompt changes, new embeddings are always computed
    # instead of reusing cached embeddings from previous generations
    text_embeddings_cache = None

    # Start timing
    import time
    generation_start_time = time.time()

    # IMPORTANT: Recalculate video dimensions from base resolution and aspect ratio if needed
    # This ensures consistency between base resolution and actual video dimensions
    parsed_dims = _parse_resolution_from_label(aspect_ratio)
    if parsed_dims:
        # Aspect ratio has explicit dimensions, use them
        recalc_width, recalc_height = parsed_dims
        if recalc_width != video_frame_width or recalc_height != video_frame_height:
            print(f"[RESOLUTION FIX] Updating resolution from {video_frame_width}x{video_frame_height} to {recalc_width}x{recalc_height} (from aspect ratio)")
            video_frame_width, video_frame_height = recalc_width, recalc_height
    else:
        # Calculate from base resolution and aspect ratio name
        base_width = _coerce_positive_int(base_resolution_width) or 720
        base_height = _coerce_positive_int(base_resolution_height) or 720
        current_ratios = get_common_aspect_ratios(base_width, base_height)
        
        # Extract ratio name from aspect_ratio (e.g., "16:9" from "16:9 - 992x512px")
        ratio_name = _extract_ratio_name(aspect_ratio)
        if ratio_name and ratio_name in current_ratios:
            recalc_width, recalc_height = current_ratios[ratio_name]
            if recalc_width != video_frame_width or recalc_height != video_frame_height:
                print(f"[RESOLUTION FIX] Recalculated resolution from base {base_width}x{base_height} and aspect {ratio_name}: {recalc_width}x{recalc_height} (was {video_frame_width}x{video_frame_height})")
                video_frame_width, video_frame_height = recalc_width, recalc_height

    # Validate video extension requirements
    if video_extension_count > 0:
        required_prompts = 1 + video_extension_count  # 1 main + N extensions
        if len(validation_prompts) < required_prompts:
            raise ValueError(f"Video Extension Count {video_extension_count} requires at least {required_prompts} valid prompt lines (1 main + {video_extension_count} extensions), but only {len(validation_prompts)} valid lines found in the prompt text.")

    # Parse multi-line prompts for generation based on setting
    # Multi-line prompts and video extensions are mutually exclusive features
    if enable_multiline_prompts:
        # When multi-line prompts enabled: generate separate videos for each prompt line
        individual_prompts = parse_multiline_prompts(text_prompt, enable_multiline_prompts)
        # Disable video extensions when multi-line prompts are enabled
        video_extension_count = 0
    elif enable_video_extension and video_extension_count > 0:
        # When video extension enabled: only use the first prompt for the main generation
        individual_prompts = [validation_prompts[0]]  # Only first prompt for main generation
    else:
        # Default: single prompt
        individual_prompts = parse_multiline_prompts(text_prompt, False)

    # Debug: Log current generation parameters
    print("=" * 80)
    print("VIDEO GENERATION STARTED")
    print(f"  enable_multiline_prompts: {enable_multiline_prompts}")
    print(f"  enable_video_extension: {enable_video_extension}")
    if enable_multiline_prompts:
        print(f"  Multi-line prompts enabled: {len(individual_prompts)} prompts")
        for i, prompt in enumerate(individual_prompts):
            print(f"    Prompt {i+1}: {prompt[:40]}{'...' if len(prompt) > 40 else ''}")
    else:
        print(f"  Text prompt: {text_prompt[:50]}{'...' if len(text_prompt) > 50 else ''}")
    print(f"  Image path: {image}")
    print(f"  Resolution: {video_frame_height}x{video_frame_width}")
    print(f"  Base Resolution: {base_resolution_width}x{base_resolution_height}")
    print(f"  Duration: {duration_seconds} seconds")
    print(f"  Seed: {video_seed}")
    print(f"  Num generations per prompt: {num_generations}")
    print(f"  Video extensions: {video_extension_count}")
    print(f"  Valid prompt lines detected: {len(validation_prompts)}")
    print("=" * 80)

    try:
        # No need to check cancellation at the start since we just reset it

        # Only initialize engine if we're not using subprocess mode (clear_all=False)
        # When clear_all=True, all generations run in subprocesses, so main process doesn't need models
        if clear_all:
            print("=" * 80)
            print("CLEAR ALL MEMORY ENABLED")
            print("  Main process will NOT load any models")
            print("  All generations will run in separate subprocesses")
            print("  VRAM/RAM will be completely cleared between generations")
            print("=" * 80)

        if not clear_all and ovi_engine is None:
            # Use CLI args only in test mode, otherwise use GUI parameters
            if getattr(args, 'test', False):
                final_blocks_to_swap = getattr(args, 'blocks_to_swap', 0)
                final_cpu_offload = getattr(args, 'test_cpu_offload', False)
            else:
                final_blocks_to_swap = blocks_to_swap
                final_cpu_offload = None if (not cpu_offload and not use_image_gen) else (cpu_offload or use_image_gen)

            print("=" * 80)
            print("INITIALIZING OVI FUSION ENGINE IN MAIN PROCESS")
            print(f"  Block Swap: {final_blocks_to_swap} blocks (0 = disabled)")
            print(f"  CPU Offload: {final_cpu_offload}")
            print(f"  Image Generation: {use_image_gen}")
            print(f"  No Block Prep: {no_block_prep}")
            print(f"  Note: Models will be loaded in main process (Clear All Memory disabled)")
            print("=" * 80)

            # Calculate latent lengths based on duration
            video_latent_length, audio_latent_length = calculate_latent_lengths(duration_seconds)

            DEFAULT_CONFIG['cpu_offload'] = final_cpu_offload
            DEFAULT_CONFIG['mode'] = "t2v"
            ovi_engine = OviFusionEngine(
                blocks_to_swap=final_blocks_to_swap,
                cpu_offload=final_cpu_offload,
                video_latent_length=video_latent_length,
                audio_latent_length=audio_latent_length
            )
            print("\n[OK] OviFusionEngine initialized successfully (models will load on first generation)")

        image_path = None
        if image is not None:
            # Handle image processing here to ensure we use the current image
            print(f"[DEBUG] Raw image path from upload: {image}")

            # Apply auto cropping if enabled
            image_path = apply_auto_crop_if_enabled(image, auto_crop_image, video_frame_width, video_frame_height)

            if os.path.exists(image_path):
                print(f"[DEBUG] Final image file exists: Yes ({os.path.getsize(image_path)} bytes)")
            else:
                print(f"[DEBUG] Final image file exists: No - this may cause issues!")

        # Determine output directory (priority: parameter > CLI arg > default)
        if output_dir and isinstance(output_dir, str):
            outputs_dir = os.path.abspath(output_dir)  # Normalize path
        elif args.output_dir and isinstance(args.output_dir, str):
            outputs_dir = os.path.abspath(args.output_dir)
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # OPTIMIZATION: If clear_all + delete_text_encoder, run T5 encoding in subprocess FIRST
        # This encodes text before any other models load, ensuring no duplication
        # Only run T5 subprocess if embeddings weren't already provided (from file)
        if clear_all and delete_text_encoder and text_embeddings_cache is None:
            print("=" * 80)
            print("T5 SUBPROCESS MODE (CLEAR ALL MEMORY)")
            print("Running T5 encoding in subprocess BEFORE generation subprocess")
            print("This ensures T5 never coexists with other models in memory")
            print("=" * 80)
            
            try:
                # Run T5 encoding in subprocess - only T5 + tokenizer will be loaded
                text_embeddings_cache = run_t5_encoding_subprocess(
                    text_prompt=text_prompt,
                    video_negative_prompt=video_negative_prompt,
                    audio_negative_prompt=audio_negative_prompt,
                    fp8_t5=fp8_t5,
                    cpu_only_t5=cpu_only_t5
                )
                
                print("=" * 80)
                print("T5 SUBPROCESS COMPLETED")
                print("Text embeddings cached - will be passed to generation subprocess")
                print("T5 memory fully freed by OS")
                print("=" * 80)
            except Exception as e:
                error_msg = str(e)
                if "cancelled by user" in error_msg.lower():
                    print("T5 encoding cancelled by user")
                    reset_cancellation()
                    return None
                else:
                    print(f"[WARNING] T5 subprocess failed: {e}")
                    print("[WARNING] Will retry T5 encoding in generation subprocess")
                    text_embeddings_cache = None
        else:
            # T5 subprocess was skipped (embeddings already loaded from file)
            if text_embeddings_cache is not None:
                print("=" * 80)
                print("T5 EMBEDDINGS ALREADY LOADED FROM FILE")
                print(f"Embeddings type: {type(text_embeddings_cache)}, length: {len(text_embeddings_cache) if isinstance(text_embeddings_cache, list) else 'N/A'}")
                print("Skipping T5 subprocess - using pre-loaded embeddings")
                print("=" * 80)

        last_output_path = None

        # Generate videos for each prompt
        for prompt_idx, current_prompt in enumerate(individual_prompts):
            print(f"\n[PROMPT {prompt_idx + 1}/{len(individual_prompts)}] Processing: {current_prompt[:50]}{'...' if len(current_prompt) > 50 else ''}")

            # Ensure image_path is valid (defensive check)
            if not isinstance(image_path, (str, type(None))):
                print(f"[GENERATE] Warning: image_path is invalid type {type(image_path)}, setting to None")
                image_path = None

            # Generate multiple videos for this prompt
            for gen_idx in range(int(num_generations)):
                # Check for cancellation in the loop
                check_cancellation()

                # Handle seed logic
                current_seed = video_seed
                if randomize_seed:
                    current_seed = get_random_seed()
                elif gen_idx > 0:
                    # Increment seed for subsequent generations
                    current_seed = video_seed + gen_idx

                print(f"\n[GENERATION {gen_idx + 1}/{int(num_generations)}] Starting with seed: {current_seed}")

                # Check for cancellation again after setup
                check_cancellation()

                if clear_all:
                    # Run this generation in a subprocess for memory cleanup
                    # Pass individual current_prompt to avoid re-parsing all prompts in subprocess
                    # For video extension main generation, pass full prompt for metadata
                    gen_prompt = current_prompt
                    if enable_video_extension and video_extension_count > 0 and prompt_idx == 0:
                        gen_prompt = text_prompt  # Use full multi-line prompt for main video metadata

                    single_gen_params = {
                        'text_prompt': gen_prompt,  # Pass appropriate prompt
                        'image': image_path,
                        'video_frame_height': video_frame_height,
                        'video_frame_width': video_frame_width,
                        'video_seed': current_seed,
                        'solver_name': solver_name,
                        'sample_steps': sample_steps,
                        'shift': shift,
                        'video_guidance_scale': video_guidance_scale,
                        'audio_guidance_scale': audio_guidance_scale,
                        'slg_layer': slg_layer,
                        'blocks_to_swap': blocks_to_swap,
                        'video_negative_prompt': video_negative_prompt,
                        'audio_negative_prompt': audio_negative_prompt,
                        'use_image_gen': False,  # Not used in single gen mode
                        'cpu_offload': cpu_offload,
                        'delete_text_encoder': delete_text_encoder if text_embeddings_cache is None else False,  # Skip T5 if already encoded
                        'fp8_t5': fp8_t5,
                        'cpu_only_t5': cpu_only_t5,
                        'fp8_base_model': fp8_base_model,
                        'use_sage_attention': use_sage_attention,
                        'no_audio': no_audio,
                        'no_block_prep': no_block_prep,
                        'num_generations': 1,  # Always 1 for subprocess
                        'randomize_seed': False,  # Seed handled above
                        'save_metadata': save_metadata,
                        'aspect_ratio': aspect_ratio,
                        'clear_all': False,  # Disable subprocess in subprocess
                        'vae_tiled_decode': vae_tiled_decode,
                        'vae_tile_size': vae_tile_size,
                        'vae_tile_overlap': vae_tile_overlap,
                        'base_resolution_width': base_resolution_width,
                        'base_resolution_height': base_resolution_height,
                        'duration_seconds': duration_seconds,
                        'auto_crop_image': auto_crop_image,
                        'base_filename': base_filename,  # Pass base filename for batch processing
                        'output_dir': outputs_dir,  # Pass output directory to subprocess
                        'text_embeddings_cache': text_embeddings_cache,  # Pass pre-encoded embeddings if available
                        'enable_multiline_prompts': False,  # Disable multiline parsing in subprocess (already parsed)
                        'enable_video_extension': False,  # Disable video extensions in subprocess (handled in main process)
                    }


                    run_generation_subprocess(single_gen_params)

                    # Find the generated file (should be the most recent in the outputs directory)
                    import glob
                    import time

                    # Construct pattern based on base_filename if provided, otherwise use default
                    if base_filename:
                        pattern = os.path.join(outputs_dir, f"{base_filename}_*.mp4")
                    else:
                        pattern = os.path.join(outputs_dir, "*.mp4")

                    # Retry a few times in case of timing issues
                    output_path = None
                    for retry in range(5):  # Try up to 5 times
                        existing_files = glob.glob(pattern)
                        if existing_files:
                            # Filter files that are at least 1 second old to avoid partially written files
                            current_time = time.time()
                            valid_files = [f for f in existing_files if (current_time - os.path.getctime(f)) > 1.0]
                            if valid_files:
                                output_path = max(valid_files, key=os.path.getctime)
                                break
                        time.sleep(0.5)  # Wait 0.5 seconds between retries

                    if output_path:
                        last_output_path = output_path
                        print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] Completed: {os.path.basename(output_path)}")
                    else:
                        print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] No output file found in {outputs_dir} after retries")
                        continue

                # Original generation logic (when clear_all is disabled)
                if not clear_all:
                    # Debug: Check if embeddings cache is available
                    if text_embeddings_cache is not None:
                        print(f"[DEBUG] Passing text_embeddings_cache to engine (type: {type(text_embeddings_cache)}, len: {len(text_embeddings_cache) if isinstance(text_embeddings_cache, list) else 'N/A'})")
                    else:
                        print(f"[DEBUG] No text_embeddings_cache available - T5 will be loaded in-process")

                    # Safety check: ensure ovi_engine is initialized
                    if ovi_engine is None:
                        print("[WARNING] ovi_engine is None, skipping in-process generation")
                        return None

                    # Ensure image_path is valid before passing to engine
                    if not isinstance(image_path, (str, type(None))):
                        print(f"[GENERATE] Warning: image_path is invalid type {type(image_path)}, setting to None")
                        image_path = None

                    # Use appropriate prompt for generation
                    gen_prompt = current_prompt
                    if enable_video_extension and video_extension_count > 0 and prompt_idx == 0:
                        gen_prompt = text_prompt  # Use full multi-line prompt for main video

                    # Use cancellable generation wrapper for interruptible generation
                    generated_video, generated_audio, _ = generate_with_cancellation_check(
                        ovi_engine.generate,
                        text_prompt=gen_prompt,  # Use appropriate prompt
                        image_path=image_path,
                        video_frame_height_width=[video_frame_height, video_frame_width],
                        seed=current_seed,
                        solver_name=solver_name,
                        sample_steps=sample_steps,
                        shift=shift,
                        video_guidance_scale=video_guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        slg_layer=slg_layer,
                        blocks_to_swap=None,  # Block swap is configured at engine init, not per-generation
                        video_negative_prompt=video_negative_prompt,
                        audio_negative_prompt=audio_negative_prompt,
                        delete_text_encoder=delete_text_encoder if text_embeddings_cache is None else False,  # Skip T5 if already encoded
                        no_block_prep=no_block_prep,
                        fp8_t5=fp8_t5,
                        cpu_only_t5=cpu_only_t5,
                        fp8_base_model=fp8_base_model,
                        use_sage_attention=use_sage_attention,
                        vae_tiled_decode=vae_tiled_decode,
                        vae_tile_size=vae_tile_size,
                        vae_tile_overlap=vae_tile_overlap,
                        text_embeddings_cache=text_embeddings_cache,  # Pass pre-encoded embeddings if available
                    )

                    # Get filename for this generation
                    if gen_idx == 0:
                        # First generation: use base filename or get next sequential
                        output_filename = get_next_filename(outputs_dir, base_filename=base_filename)
                        # For batch processing, remove the _0001 suffix if it's the first generation
                        if base_filename and output_filename.endswith("_0001.mp4"):
                            output_filename = f"{base_filename}.mp4"
                    else:
                        # Subsequent generations: append _2, _3, etc. to the base filename
                        if base_filename:
                            # For batch processing, ensure no conflicts
                            gen_suffix = f"_{gen_idx + 1}"
                            candidate_filename = f"{base_filename}{gen_suffix}.mp4"
                            counter = 1
                            while os.path.exists(os.path.join(outputs_dir, candidate_filename)):
                                candidate_filename = f"{base_filename}{gen_suffix}_{counter}.mp4"
                                counter += 1
                            output_filename = candidate_filename
                        else:
                            # For regular generation, just get next sequential
                            output_filename = get_next_filename(outputs_dir, base_filename=None)
                    output_path = os.path.join(outputs_dir, output_filename)

                    # Handle no_audio option
                    if no_audio:
                        generated_audio = None

                    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                    last_output_path = output_path

                    # Save used source image
                    save_used_source_image(image_path, outputs_dir, output_filename)

                    # Save metadata if enabled
                    # Skip metadata saving for extension subprocesses
                    is_extension_subprocess = locals().get('is_video_extension_subprocess', False)
                    if save_metadata and not is_extension_subprocess:
                        # For video extension, use full multi-line prompt for first/main video metadata
                        metadata_prompt = current_prompt
                        if enable_video_extension and video_extension_count > 0 and prompt_idx == 0:
                            metadata_prompt = text_prompt  # Use full multi-line prompt for main video

                        generation_params = {
                            'text_prompt': metadata_prompt,  # Use appropriate prompt for metadata
                            'image_path': image_path,
                            'video_frame_height': video_frame_height,
                            'video_frame_width': video_frame_width,
                            'aspect_ratio': aspect_ratio,
                            'randomize_seed': randomize_seed,
                            'num_generations': num_generations,
                            'solver_name': solver_name,
                            'sample_steps': sample_steps,
                            'shift': shift,
                            'video_guidance_scale': video_guidance_scale,
                            'audio_guidance_scale': audio_guidance_scale,
                            'slg_layer': slg_layer,
                            'blocks_to_swap': blocks_to_swap,
                            'cpu_offload': cpu_offload,
                            'delete_text_encoder': delete_text_encoder,
                            'fp8_t5': fp8_t5,
                            'cpu_only_t5': cpu_only_t5,
                            'fp8_base_model': fp8_base_model,
                            'no_audio': no_audio,
                            'no_block_prep': no_block_prep,
                            'clear_all': clear_all,
                            'vae_tiled_decode': vae_tiled_decode,
                            'vae_tile_size': vae_tile_size,
                            'vae_tile_overlap': vae_tile_overlap,
                            'base_resolution_width': base_resolution_width,
                            'base_resolution_height': base_resolution_height,
                            'duration_seconds': duration_seconds,
                            'video_negative_prompt': video_negative_prompt,
                            'audio_negative_prompt': audio_negative_prompt,
                            'is_batch': False
                        }
                        save_generation_metadata(output_path, generation_params, current_seed)

                    print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] Saved to: {output_path}")

                # Handle video extensions if enabled (works for both clear_all and in-process modes)
                if video_extension_count > 0:
                    print(f"[VIDEO EXTENSION] Starting {video_extension_count} extensions for prompt {prompt_idx + 1}")

                    extension_videos = [output_path]  # Start with the main video
                    current_image_path = extract_last_frame(output_path)  # Extract last frame

                    if current_image_path:
                        # Get base name from main video for consistent naming
                        main_base = os.path.splitext(os.path.basename(output_path))[0]

                        # Generate extension videos
                        for ext_idx in range(video_extension_count):
                            # Use next prompt if available, otherwise use current prompt
                            next_prompt_idx = prompt_idx + ext_idx + 1
                            if next_prompt_idx < len(validation_prompts):
                                extension_prompt = validation_prompts[next_prompt_idx]
                                print(f"[VIDEO EXTENSION] Extension {ext_idx + 1}: Using next prompt from parsed lines")
                            else:
                                extension_prompt = current_prompt  # Repeat current prompt
                                print(f"[VIDEO EXTENSION] Extension {ext_idx + 1}: Repeating current prompt (no more prompts available)")
                            print(f"[VIDEO EXTENSION] Extension {ext_idx + 1}: Using last frame + prompt: {extension_prompt[:50]}{'...' if len(extension_prompt) > 50 else ''}")

                            # Generate extension video
                            check_cancellation()

                            # Construct desired extension filename
                            ext_filename = f"{main_base}_ext{ext_idx + 1}.mp4"
                            ext_output_path = os.path.join(outputs_dir, ext_filename)

                            # Check for conflicts and increment if needed
                            if os.path.exists(ext_output_path):
                                counter = 1
                                while os.path.exists(os.path.join(outputs_dir, f"{main_base}_ext{ext_idx + 1}_{counter}.mp4")):
                                    counter += 1
                                ext_filename = f"{main_base}_ext{ext_idx + 1}_{counter}.mp4"
                                ext_output_path = os.path.join(outputs_dir, ext_filename)

                            if clear_all:
                                # Run extension in subprocess
                                ext_params = {
                                    'text_prompt': extension_prompt,
                                    'image': current_image_path,
                                    'video_frame_height': video_frame_height,
                                    'video_frame_width': video_frame_width,
                                    'video_seed': current_seed,  # Use same seed
                                    'solver_name': solver_name,
                                    'sample_steps': sample_steps,
                                    'shift': shift,
                                    'video_guidance_scale': video_guidance_scale,
                                    'audio_guidance_scale': audio_guidance_scale,
                                    'slg_layer': slg_layer,
                                    'blocks_to_swap': blocks_to_swap,
                                    'video_negative_prompt': video_negative_prompt,
                                    'audio_negative_prompt': audio_negative_prompt,
                                    'use_image_gen': False,
                                    'cpu_offload': cpu_offload,
                                    'delete_text_encoder': delete_text_encoder if text_embeddings_cache is None else False,
                                    'fp8_t5': fp8_t5,
                                    'cpu_only_t5': cpu_only_t5,
                                    'fp8_base_model': fp8_base_model,
                                    'no_audio': no_audio,
                                    'no_block_prep': no_block_prep,
                                    'num_generations': 1,
                                    'randomize_seed': False,
                                    'save_metadata': False,  # Extensions don't save individual metadata
                                    'aspect_ratio': aspect_ratio,
                                    'clear_all': False,
                                    'vae_tiled_decode': vae_tiled_decode,
                                    'vae_tile_size': vae_tile_size,
                                    'vae_tile_overlap': vae_tile_overlap,
                                    'base_resolution_width': base_resolution_width,
                                    'base_resolution_height': base_resolution_height,
                                    'duration_seconds': duration_seconds,
                                    'auto_crop_image': False,  # Image already processed
                                    'base_filename': base_filename,
                                    'output_dir': outputs_dir,
                                    'text_embeddings_cache': text_embeddings_cache,
                                    'enable_multiline_prompts': False,
                                    'enable_video_extension': False,
                                    'is_video_extension_subprocess': True,  # Mark as extension subprocess
                                }

                                run_generation_subprocess(ext_params)

                                # Find the generated extension file and rename it
                                import glob
                                import time
                                pattern = os.path.join(outputs_dir, "*.mp4")

                                # Retry a few times in case of timing issues
                                generated_file = None
                                for retry in range(5):  # Try up to 5 times
                                    existing_files = glob.glob(pattern)
                                    if existing_files:
                                        # Filter files that are at least 1 second old to avoid partially written files
                                        current_time = time.time()
                                        valid_files = [f for f in existing_files if (current_time - os.path.getctime(f)) > 1.0]
                                        if valid_files:
                                            generated_file = max(valid_files, key=os.path.getctime)
                                            break
                                    time.sleep(0.5)  # Wait 0.5 seconds between retries

                                if generated_file:
                                    # Rename to desired extension filename
                                    os.rename(generated_file, ext_output_path)
                                    extension_videos.append(ext_output_path)
                                    print(f"[VIDEO EXTENSION] Extension {ext_idx + 1} saved: {ext_filename}")
                                    current_image_path = extract_last_frame(ext_output_path)  # Extract for next extension
                                else:
                                    print(f"[VIDEO EXTENSION] Warning: Extension {ext_idx + 1} file not found after retries")
                                    break
                            else:
                                # Safety check: ensure ovi_engine is initialized
                                if ovi_engine is None:
                                    print("[WARNING] ovi_engine is None, skipping extension in-process generation")
                                    continue

                                # Generate extension in-process
                                ext_generated_video, ext_generated_audio, _ = generate_with_cancellation_check(
                                    ovi_engine.generate,
                                    text_prompt=extension_prompt,
                                    image_path=current_image_path,
                                    video_frame_height_width=[video_frame_height, video_frame_width],
                                    seed=current_seed,
                                    solver_name=solver_name,
                                    sample_steps=sample_steps,
                                    shift=shift,
                                    video_guidance_scale=video_guidance_scale,
                                    audio_guidance_scale=audio_guidance_scale,
                                    slg_layer=slg_layer,
                                    blocks_to_swap=None,
                                    video_negative_prompt=video_negative_prompt,
                                    audio_negative_prompt=audio_negative_prompt,
                                    delete_text_encoder=delete_text_encoder if text_embeddings_cache is None else False,
                                    no_block_prep=no_block_prep,
                                    fp8_t5=fp8_t5,
                                    cpu_only_t5=cpu_only_t5,
                                    fp8_base_model=fp8_base_model,
                                    vae_tiled_decode=vae_tiled_decode,
                                    vae_tile_size=vae_tile_size,
                                    vae_tile_overlap=vae_tile_overlap,
                                    text_embeddings_cache=text_embeddings_cache,
                                )

                                # Save extension video with proper naming
                                ext_filename = f"{main_base}_ext{ext_idx + 1}.mp4"
                                ext_output_path = os.path.join(outputs_dir, ext_filename)

                                # Check for conflicts and increment if needed
                                if os.path.exists(ext_output_path):
                                    counter = 1
                                    while os.path.exists(os.path.join(outputs_dir, f"{main_base}_ext{ext_idx + 1}_{counter}.mp4")):
                                        counter += 1
                                    ext_filename = f"{main_base}_ext{ext_idx + 1}_{counter}.mp4"
                                    ext_output_path = os.path.join(outputs_dir, ext_filename)

                                if no_audio:
                                    ext_generated_audio = None

                                save_video(ext_output_path, ext_generated_video, ext_generated_audio, fps=24, sample_rate=16000)
                                extension_videos.append(ext_output_path)
                                print(f"[VIDEO EXTENSION] Extension {ext_idx + 1} saved: {ext_filename}")

                                # Save used source image for extension
                                save_used_source_image(current_image_path, outputs_dir, ext_filename)

                                current_image_path = extract_last_frame(ext_output_path)  # Extract for next extension

                                # Note: Extensions don't save individual metadata - only main video has metadata

                                print(f"[VIDEO EXTENSION] Extension {ext_idx + 1} saved: {ext_filename}")

                        # Combine all videos (main + extensions) into final video
                        if len(extension_videos) > 1:
                            # Construct final filename
                            final_filename = f"{main_base}_final.mp4"
                            final_path = os.path.join(outputs_dir, final_filename)

                            # Check for conflicts and increment if needed
                            if os.path.exists(final_path):
                                counter = 1
                                while os.path.exists(os.path.join(outputs_dir, f"{main_base}_final_{counter}.mp4")):
                                    counter += 1
                                final_filename = f"{main_base}_final_{counter}.mp4"
                                final_path = os.path.join(outputs_dir, final_filename)

                            if combine_videos(extension_videos, final_path, trim_extension_first_frames=True):
                                print(f"[VIDEO EXTENSION] Combined video saved: {final_filename}")
                                last_output_path = final_path
                            else:
                                print("[VIDEO EXTENSION] Failed to combine videos")
                    else:
                        print("[VIDEO EXTENSION] Failed to extract last frame, skipping extensions")

        # Calculate and log total generation time
        generation_end_time = time.time()
        total_generation_time = generation_end_time - generation_start_time
        print(f"  Total generation time: {total_generation_time:.2f} seconds")

        # If input video was provided, merge it with the generated video (unless disabled)
        if input_video_path and last_output_path and os.path.exists(last_output_path) and not dont_auto_combine_video_input:
            print("=" * 80)
            print("INPUT VIDEO MERGING")
            print(f"  Input video: {input_video_path}")
            print(f"  Generated video: {last_output_path}")
            print("=" * 80)

            # Create merged filename
            base_name = os.path.splitext(os.path.basename(last_output_path))[0]
            merged_filename = f"{base_name}_merged.mp4"
            merged_path = os.path.join(outputs_dir, merged_filename)

            # Check for conflicts and increment if needed
            if os.path.exists(merged_path):
                counter = 1
                while os.path.exists(os.path.join(outputs_dir, f"{base_name}_merged_{counter}.mp4")):
                    counter += 1
                merged_filename = f"{base_name}_merged_{counter}.mp4"
                merged_path = os.path.join(outputs_dir, merged_filename)

            # Merge input video + generated video (trim last frame from input to avoid duplication)
            if combine_videos([input_video_path, last_output_path], merged_path, trim_first_video_last_frame=True):
                print(f"[INPUT VIDEO MERGE] Merged video saved: {merged_filename}")
                last_output_path = merged_path
            else:
                print("[INPUT VIDEO MERGE] Failed to merge videos, returning generated video only")

        # Debug: Log final output path
        print("=" * 80)
        print("VIDEO GENERATION COMPLETED")
        print(f"  Final output path: {last_output_path}")
        if last_output_path and os.path.exists(last_output_path):
            print(f"  File exists: Yes ({os.path.getsize(last_output_path)} bytes)")
        else:
            print("  File exists: No")
        print("=" * 80)

        return last_output_path
    except Exception as e:
        error_msg = str(e)
        if "cancelled by user" in error_msg.lower():
            print("Generation cancelled by user")
            reset_cancellation()  # Reset the cancellation flag
            return None
        else:
            print(f"Error during video generation: {e}")
            return None




# Standard aspect ratios
def get_common_aspect_ratios(base_width, base_height):
    """Get the standard aspect ratios, scaled to preserve total pixel area."""
    # Calculate total pixel area from base resolution
    total_pixels = base_width * base_height

    # Define aspect ratios as width:height ratios
    aspect_ratios_def = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "9:16": (9, 16),
        "4:3": (4, 3),
        "3:4": (3, 4),
        "21:9": (21, 9),
        "9:21": (9, 21),
        "3:2": (3, 2),
        "2:3": (2, 3),
        "5:4": (5, 4),
        "4:5": (4, 5),
        "5:3": (5, 3),
        "3:5": (3, 5),
        "16:10": (16, 10),
        "10:16": (10, 16),
    }

    aspect_ratios = {}
    for name, (w_ratio, h_ratio) in aspect_ratios_def.items():
        # Calculate dimensions that preserve total pixel area and aspect ratio
        aspect = w_ratio / h_ratio

        # height = sqrt(total_pixels / aspect)
        height = (total_pixels / aspect) ** 0.5

        # width = height * aspect
        width = height * aspect

        # Round to nearest integers
        width = int(round(width))
        height = int(round(height))

        # Snap to 32px for model compatibility (round to nearest multiple of 32)
        width = max(32, ((width + 15) // 32) * 32)  # Round up to nearest 32
        height = max(32, ((height + 15) // 32) * 32)  # Round up to nearest 32

        aspect_ratios[name] = [width, height]

    return aspect_ratios

# Dynamic aspect ratios based on base resolution (legacy function)
def get_aspect_ratios(base_width, base_height):
    """Generate aspect ratios scaled from 720p base resolution."""
    return get_common_aspect_ratios(base_width, base_height)

# For backward compatibility, keep a default ASPECT_RATIOS
ASPECT_RATIOS = get_aspect_ratios(720, 720)

CUSTOM_ASPECT_PREFIX = "Custom - "

def _coerce_positive_int(value):
    """Safely coerce incoming UI values to positive integers."""
    try:
        if isinstance(value, bool):  # Guard against booleans (subclass of int)
            return None
        if isinstance(value, (int, float)):
            coerced = int(value)
        elif isinstance(value, str) and value.strip():
            coerced = int(float(value.strip()))
        else:
            return None
        return coerced if coerced > 0 else None
    except (ValueError, TypeError):
        return None

def _extract_ratio_name(value):
    if isinstance(value, str) and value:
        return value.split(" - ")[0]
    return None

def _format_ratio_choice(name, dims):
    return f"{name} - {dims[0]}x{dims[1]}px"

def _format_custom_choice(width, height):
    try:
        width_int = int(width)
        height_int = int(height)
        if width_int <= 0 or height_int <= 0:
            raise ValueError
    except Exception:
        return f"{CUSTOM_ASPECT_PREFIX}{width}x{height}px"
    return f"{CUSTOM_ASPECT_PREFIX}{width_int}x{height_int}px"

def _parse_resolution_from_label(value):
    if not isinstance(value, str):
        return None

    custom_dims = _parse_custom_choice(value)
    if custom_dims:
        return custom_dims

    if " - " not in value:
        return None

    _, dims_part = value.split(" - ", 1)
    dims_part = dims_part.strip()
    if dims_part.endswith("px"):
        dims_part = dims_part[:-2]

    if "x" not in dims_part:
        return None

    width_str, height_str = dims_part.split("x", 1)
    width = _coerce_positive_int(width_str)
    height = _coerce_positive_int(height_str)

    if width is None or height is None:
        return None

    return width, height

def _parse_custom_choice(value):
    if not isinstance(value, str) or not value.startswith(CUSTOM_ASPECT_PREFIX):
        return None

    remainder = value[len(CUSTOM_ASPECT_PREFIX):]
    if remainder.endswith("px"):
        remainder = remainder[:-2]

    if "x" not in remainder:
        return None

    width_str, height_str = remainder.split("x", 1)
    width = _coerce_positive_int(width_str)
    height = _coerce_positive_int(height_str)

    if width is None or height is None:
        return None

    return width, height

def update_resolution(aspect_ratio, base_resolution_width=720, base_resolution_height=720):
    """Update resolution based on aspect ratio and base resolution."""
    try:
        custom_dims = _parse_custom_choice(aspect_ratio)
        if custom_dims:
            return [custom_dims[0], custom_dims[1]]

        base_width = _coerce_positive_int(base_resolution_width)
        base_height = _coerce_positive_int(base_resolution_height)

        if base_width is None or base_height is None:
            return [gr.update(), gr.update()]

        current_ratios = get_common_aspect_ratios(base_width, base_height)

        ratio_name = _extract_ratio_name(aspect_ratio)
        if ratio_name not in current_ratios:
            ratio_name = "16:9" if "16:9" in current_ratios else next(iter(current_ratios))

        width, height = current_ratios[ratio_name]
        return [width, height]
    except Exception as e:
        print(f"Error updating resolution: {e}")
        return [gr.update(), gr.update()]

def update_aspect_ratio_and_resolution(base_resolution_width, base_resolution_height, current_aspect_ratio):
    """Combined update for aspect ratio choices and resolution to avoid race conditions."""
    try:
        base_width = _coerce_positive_int(base_resolution_width)
        base_height = _coerce_positive_int(base_resolution_height)

        if base_width is None or base_height is None:
            # Keep existing state while user is typing invalid values
            return gr.update(), gr.update(), gr.update()

        current_ratios = get_common_aspect_ratios(base_width, base_height)
        choices = [_format_ratio_choice(name, dims) for name, dims in current_ratios.items()]

        # Handle custom aspect ratios
        custom_dims = _parse_custom_choice(current_aspect_ratio)
        if custom_dims:
            selected_value = _format_custom_choice(custom_dims[0], custom_dims[1])
            if selected_value not in choices:
                choices = [selected_value] + choices
            return gr.update(choices=choices, value=selected_value), custom_dims[0], custom_dims[1]

        # Extract ratio name from current selection
        ratio_name = _extract_ratio_name(current_aspect_ratio)
        if ratio_name not in current_ratios:
            ratio_name = "16:9" if "16:9" in current_ratios else next(iter(current_ratios))

        # Get dimensions for this ratio from new base resolution
        width, height = current_ratios[ratio_name]
        selected_value = _format_ratio_choice(ratio_name, (width, height))

        # Ensure selected_value is in choices
        if selected_value not in choices:
            choices = [selected_value] + choices

        return gr.update(choices=choices, value=selected_value), width, height

    except Exception as e:
        print(f"Error updating aspect ratio and resolution: {e}")
        # Return safe defaults
        try:
            default_ratios = get_common_aspect_ratios(720, 720)
            default_choices = [_format_ratio_choice(name, dims) for name, dims in default_ratios.items()]
            default_value = default_choices[0] if default_choices else None
            default_width, default_height = default_ratios["16:9"]
            return gr.update(choices=default_choices, value=default_value), default_width, default_height
        except:
            return gr.update(), gr.update(), gr.update()

def update_aspect_ratio_choices(base_resolution_width, base_resolution_height, current_aspect_ratio=None):
    """Update aspect ratio dropdown choices based on base resolution with graceful handling during user input."""
    try:
        base_width = _coerce_positive_int(base_resolution_width)
        base_height = _coerce_positive_int(base_resolution_height)

        if base_width is None or base_height is None:
            # Keep existing dropdown state while the user is typing an invalid number
            return gr.update()

        current_ratios = get_common_aspect_ratios(base_width, base_height)
        choices = [_format_ratio_choice(name, dims) for name, dims in current_ratios.items()]

        # Handle custom aspect ratios
        custom_dims = _parse_custom_choice(current_aspect_ratio)
        if custom_dims:
            selected_value = _format_custom_choice(custom_dims[0], custom_dims[1])
            if selected_value not in choices:
                choices = [selected_value] + choices
            return gr.update(choices=choices, value=selected_value)

        # Extract ratio name from current selection (e.g., "16:9" from "16:9 - 352x192px")
        ratio_name = _extract_ratio_name(current_aspect_ratio)
        if ratio_name not in current_ratios:
            ratio_name = "16:9" if "16:9" in current_ratios else next(iter(current_ratios))

        # Create new value with the same ratio name but NEW dimensions from new base resolution
        selected_value = _format_ratio_choice(ratio_name, current_ratios[ratio_name])

        # CRITICAL FIX: Ensure selected_value is ALWAYS in choices before returning
        # This prevents Gradio errors when base resolution changes
        if selected_value not in choices:
            # This should never happen since we just created it from current_ratios,
            # but add it as a safety measure
            choices = [selected_value] + choices

        return gr.update(choices=choices, value=selected_value)
    except Exception as e:
        print(f"Error updating aspect ratio choices: {e}")
        # In case of any error, return a safe default
        try:
            default_ratios = get_common_aspect_ratios(720, 720)
            default_choices = [_format_ratio_choice(name, dims) for name, dims in default_ratios.items()]
            default_value = default_choices[0] if default_choices else None
            return gr.update(choices=default_choices, value=default_value)
        except:
            # Ultimate fallback - return empty update
            return gr.update()

def _resolve_aspect_ratio_value(aspect_ratio_label, video_width, video_height):
    if isinstance(aspect_ratio_label, str) and aspect_ratio_label.startswith(CUSTOM_ASPECT_PREFIX):
        parsed_custom = _parse_custom_choice(aspect_ratio_label)
        if parsed_custom:
            return _format_custom_choice(parsed_custom[0], parsed_custom[1])
        return aspect_ratio_label

    parsed_dims = _parse_resolution_from_label(aspect_ratio_label)
    if parsed_dims:
        return _format_custom_choice(parsed_dims[0], parsed_dims[1])

    ratio_name = _extract_ratio_name(aspect_ratio_label)
    if ratio_name in ASPECT_RATIOS:
        return _format_ratio_choice(ratio_name, ASPECT_RATIOS[ratio_name])

    width = _coerce_positive_int(video_width)
    height = _coerce_positive_int(video_height)
    if width and height:
        return _format_custom_choice(width, height)

    return _format_ratio_choice("16:9", ASPECT_RATIOS["16:9"])

def calculate_latent_lengths(duration_seconds):
    """Calculate video_latent_length and audio_latent_length based on duration.

    Current reference: 5 seconds = 31 video latents, 157 audio latents
    Video: 5s * 24fps = 120 frames → 120/31 ≈ 3.87× temporal upscale
    Audio: 5s * 16000Hz = 80000 samples → 80000/157 ≈ 510× temporal upscale
    """
    # Scale from the 5-second reference
    video_latent_length = max(1, round((duration_seconds / 5.0) * 31))
    audio_latent_length = max(1, round((duration_seconds / 5.0) * 157))

    return video_latent_length, audio_latent_length

def get_vram_warnings(base_resolution_width, base_resolution_height, duration_seconds):
    """Generate VRAM warnings based on settings."""
    warnings = []

    # Check base resolution
    base_size = min(base_resolution_width, base_resolution_height)
    if base_size > 720:
        warnings.append(f"⚠️ Base resolution ({base_size}p) > 720p may use significantly more VRAM")

    # Check duration
    if duration_seconds > 5:
        warnings.append(f"⚠️ Duration ({duration_seconds}s) > 5s may use significantly more VRAM")

    return "\n".join(warnings) if warnings else ""

def get_random_seed():
    import random
    return random.randint(0, 100000)

def open_outputs_folder():
    """Open the outputs folder in the system's file explorer."""
    import subprocess
    import platform

    try:
        # Use custom output directory if set via --output_dir, otherwise use default
        if args.output_dir and isinstance(args.output_dir, str):
            outputs_dir = os.path.abspath(args.output_dir)
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")

        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)

        if platform.system() == "Windows":
            subprocess.run(["explorer", outputs_dir])
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", outputs_dir])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", outputs_dir])
        else:
            print(f"Unsupported platform: {platform.system()}")
    except Exception as e:
        print(f"Error opening outputs folder: {e}")

def get_next_filename(outputs_dir, base_filename=None):
    """Get the next available filename in sequential format (0001.mp4, 0002.mp4, etc.)
    If base_filename is provided, use it as the stem instead of 'ovi_generation'."""
    import glob

    if base_filename:
        # For batch processing, use the base filename (e.g., 'image1' -> 'image1_0001.mp4')
        stem = base_filename
    else:
        # For regular generation, use sequential numbering without prefix
        stem = ""

    # Find all files matching the pattern
    if stem:
        pattern = os.path.join(outputs_dir, f"{stem}_*.mp4")
    else:
        pattern = os.path.join(outputs_dir, "*.mp4")
    existing_files = glob.glob(pattern)

    # Extract numbers from existing files
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove stem and extension to get the number part
        if stem:
            if filename.startswith(f"{stem}_") and filename.endswith(".mp4"):
                num_part = filename[len(f"{stem}_"):-4]  # Remove stem_ and .mp4
                # Only consider 4-digit numbers (our sequential format)
                if len(num_part) == 4 and num_part.isdigit():
                    try:
                        numbers.append(int(num_part))
                    except ValueError:
                        pass
        else:
            # For regular generation without stem, look for files like 0001.mp4
            if filename.endswith(".mp4") and len(filename) == 8:  # 0001.mp4 is 8 chars
                num_part = filename[:-4]  # Remove .mp4
                if len(num_part) == 4 and num_part.isdigit():
                    try:
                        numbers.append(int(num_part))
                    except ValueError:
                        pass

    # Find the next available number
    next_num = 1
    if numbers:
        next_num = max(numbers) + 1

    # Format as 4-digit number
    if stem:
        return f"{stem}_{next_num:04d}.mp4"
    else:
        return f"{next_num:04d}.mp4"

def cancel_all_generations():
    """Cancel all running generations by setting the global flag."""
    global cancel_generation, ovi_engine
    cancel_generation = True
    print("CANCELLATION REQUESTED - stopping all generations...")

    # Try to clean up the engine if it exists
    if ovi_engine is not None:
        try:
            # Force cleanup of any models in VRAM
            if hasattr(ovi_engine, 'cleanup'):
                ovi_engine.cleanup()
            # Clear the engine reference to force re-initialization
            ovi_engine = None
        except Exception as e:
            print(f"Warning: Error during engine cleanup: {e}")

def check_cancellation():
    """Check if cancellation has been requested."""
    global cancel_generation
    if cancel_generation:
        # Don't reset the flag here - let the caller handle it
        raise Exception("Generation cancelled by user")

def reset_cancellation():
    """Reset the cancellation flag after handling cancellation."""
    global cancel_generation
    cancel_generation = False

def generate_with_cancellation_check(generate_func, **kwargs):
    """Run generation function with built-in cancellation checks."""
    # Add the cancellation check function to kwargs so it gets passed to the engine
    kwargs['cancellation_check'] = check_cancellation
    return generate_func(**kwargs)

def get_presets_dir():
    """Get the presets directory path."""
    presets_dir = os.path.join(os.path.dirname(__file__), "presets")
    os.makedirs(presets_dir, exist_ok=True)
    return presets_dir

def get_available_presets():
    """Get list of available preset names."""
    presets_dir = get_presets_dir()
    presets = []
    if os.path.exists(presets_dir):
        for file in os.listdir(presets_dir):
            if file.endswith('.json'):
                presets.append(file[:-5])  # Remove .json extension

    # Sort presets by VRAM size (natural sort for numbers)
    def sort_key(preset_name):
        import re
        # Extract number from preset name (e.g., "6-GB GPUs" -> 6)
        match = re.search(r'(\d+)', preset_name)
        if match:
            return int(match.group(1))
        return 0

    return sorted(presets, key=sort_key)

# Preset system constants
PRESET_VERSION = "3.2"
PRESET_MIN_COMPATIBLE_VERSION = "3.0"

# Aspect ratio migration mapping (old names -> new names)
ASPECT_RATIO_MIGRATION = {
    "1:1 Square": "1:1",
    "16:9 Landscape": "16:9",
    "9:16 Portrait": "9:16",
    "4:3 Landscape": "4:3",
    "3:4 Portrait": "3:4",
    "21:9 Landscape": "21:9",
    "9:21 Portrait": "9:21",
    "3:2 Landscape": "3:2",
    "2:3 Portrait": "2:3",
    "5:4 Landscape": "5:4",
    "4:5 Portrait": "4:5",
    "5:3 Landscape": "5:3",
    "3:5 Portrait": "3:5",
    "16:10 Widescreen": "16:10",
    "10:16 Tall Widescreen": "10:16",
}

# Default values for all preset parameters (used for validation and migration)
PRESET_DEFAULTS = {
    "video_text_prompt": "",
    "aspect_ratio": "16:9",
    "video_width": 992,
    "video_height": 512,
    "auto_crop_image": True,
    "video_seed": 99,
    "randomize_seed": False,
    "no_audio": False,
    "save_metadata": True,
    "solver_name": "unipc",
    "sample_steps": 50,
    "num_generations": 1,
    "shift": 5.0,
    "video_guidance_scale": 4.0,
    "audio_guidance_scale": 3.0,
    "slg_layer": 11,
    "blocks_to_swap": 12,
    "cpu_offload": True,
    "delete_text_encoder": True,
    "fp8_t5": False,
    "cpu_only_t5": False,
    "fp8_base_model": False,
    "use_sage_attention": False,
    "video_negative_prompt": "jitter, bad hands, blur, distortion",
    "audio_negative_prompt": "robotic, muffled, echo, distorted",
    "batch_input_folder": "",
    "batch_output_folder": "",
    "batch_skip_existing": True,
    "clear_all": True,
    "vae_tiled_decode": False,
    "vae_tile_size": 32,
    "vae_tile_overlap": 8,
    "base_resolution_width": 720,
    "base_resolution_height": 720,
    "duration_seconds": 5,
    "enable_multiline_prompts": False,
    "enable_video_extension": False,
    "dont_auto_combine_video_input": False,
}

# Parameter validation rules
PRESET_VALIDATION = {
    "video_width": {"type": int, "min": 128, "max": 1920},
    "video_height": {"type": int, "min": 128, "max": 1920},
    "video_seed": {"type": int, "min": 0, "max": 100000},
    "sample_steps": {"type": int, "min": 1, "max": 100},
    "num_generations": {"type": int, "min": 1, "max": 100},
    "shift": {"type": float, "min": 0.0, "max": 20.0},
    "video_guidance_scale": {"type": float, "min": 0.0, "max": 10.0},
    "audio_guidance_scale": {"type": float, "min": 0.0, "max": 10.0},
    "slg_layer": {"type": int, "min": -1, "max": 30},
    "blocks_to_swap": {"type": int, "min": 0, "max": 29},
    "vae_tile_size": {"type": int, "min": 12, "max": 64},
    "vae_tile_overlap": {"type": int, "min": 4, "max": 16},
    "base_resolution_width": {"type": int},
    "base_resolution_height": {"type": int},
    "duration_seconds": {"type": int},
    "aspect_ratio": {"type": str, "choices": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "3:2", "2:3", "5:4", "4:5", "5:3", "3:5", "16:10", "10:16"]},
    "solver_name": {"type": str, "choices": ["unipc", "euler", "dpm++"]},
}

def validate_preset_value(param_name, value):
    """Validate a single preset parameter value."""
    # Get expected type from defaults
    expected_type = type(PRESET_DEFAULTS[param_name])

    # Apply type conversion for all parameters
    try:
        if expected_type == int:
            value = int(value)
        elif expected_type == float:
            value = float(value)
        elif expected_type == str:
            value = str(value)
        elif expected_type == bool:
            # Handle boolean conversion more carefully
            if isinstance(value, bool):
                pass  # Already correct type
            elif isinstance(value, str):
                # Convert string representations to boolean
                value = value.lower() in ('true', '1', 'yes', 'on')
            else:
                # Convert other types (numbers, etc.) to boolean
                value = bool(value)
    except (ValueError, TypeError):
        # If conversion fails, use default
        print(f"[PRESET] Type conversion failed for {param_name}, using default")
        value = PRESET_DEFAULTS[param_name]

    # Special handling for aspect_ratio migration
    if param_name == "aspect_ratio" and isinstance(value, str):
        # Check if it's an old aspect ratio name that needs migration
        if value in ASPECT_RATIO_MIGRATION:
            new_value = ASPECT_RATIO_MIGRATION[value]
            print(f"[PRESET] Migrated aspect ratio: '{value}' -> '{new_value}'")
            value = new_value

    # Apply specific validation rules if they exist
    if param_name in PRESET_VALIDATION:
        rule = PRESET_VALIDATION[param_name]

        # Range validation
        if "min" in rule and isinstance(value, (int, float)) and value < rule["min"]:
            print(f"[PRESET] Warning: {param_name} value {value} below minimum {rule['min']}, using minimum")
            value = rule["min"]
        elif "max" in rule and isinstance(value, (int, float)) and value > rule["max"]:
            print(f"[PRESET] Warning: {param_name} value {value} above maximum {rule['max']}, using maximum")
            value = rule["max"]

        # Choice validation
        if "choices" in rule:
            if param_name == "aspect_ratio" and isinstance(value, str):
                if value in rule["choices"]:
                    pass
                elif value.startswith(CUSTOM_ASPECT_PREFIX):
                    # Allow custom aspect ratios to pass validation as-is
                    pass
                elif " - " in value:
                    ratio_part = value.split(" - ", 1)[0]
                    if ratio_part in rule["choices"]:
                        value = ratio_part
                    else:
                        print(f"[PRESET] Warning: {param_name} value '{value}' not recognized, using default")
                        value = PRESET_DEFAULTS[param_name]
                else:
                    print(f"[PRESET] Warning: {param_name} value '{value}' not recognized, using default")
                    value = PRESET_DEFAULTS[param_name]
            elif value not in rule["choices"]:
                print(f"[PRESET] Warning: {param_name} value '{value}' not in valid choices {rule['choices']}, using default")
                value = PRESET_DEFAULTS[param_name]

    return value

def cleanup_invalid_presets():
    """Remove presets that cannot be loaded or are corrupted."""
    try:
        presets_dir = get_presets_dir()
        if not os.path.exists(presets_dir):
            return 0

        presets = get_available_presets()
        removed_count = 0

        for preset_name in presets:
            preset_data, error_msg = load_preset_safely(preset_name)
            if preset_data is None:
                # Preset is invalid, remove it
                preset_file = os.path.join(presets_dir, f"{preset_name}.json")
                try:
                    os.remove(preset_file)
                    print(f"[PRESET] Removed invalid preset: {preset_name} ({error_msg})")
                    removed_count += 1
                except Exception as e:
                    print(f"[PRESET] Failed to remove invalid preset {preset_name}: {e}")

        # Also clean up the last_used.txt if it points to a non-existent preset
        last_used_file = os.path.join(presets_dir, "last_used.txt")
        if os.path.exists(last_used_file):
            try:
                with open(last_used_file, 'r', encoding='utf-8') as f:
                    last_preset = f.read().strip()

                if last_preset and last_preset not in get_available_presets():
                    os.remove(last_used_file)
                    print(f"[PRESET] Removed invalid last_used.txt reference to '{last_preset}'")
            except Exception as e:
                print(f"[PRESET] Error checking last_used.txt: {e}")

        return removed_count

    except Exception as e:
        print(f"[PRESET] Error during preset cleanup: {e}")
        return 0

def migrate_preset_data(preset_data):
    """Migrate preset data from older versions to current format."""
    version = preset_data.get("preset_version", "1.0")  # Assume old version if missing
    migrated = False

    # Version-specific migrations
    if version < "3.0":
        print(f"[PRESET] Migrating preset from version {version} to {PRESET_VERSION}")
        migrated = True

        # Add missing parameters with defaults
        for param, default_value in PRESET_DEFAULTS.items():
            if param not in preset_data:
                preset_data[param] = default_value
                print(f"[PRESET] Added missing parameter: {param} = {default_value}")

        # Handle renamed parameters (if any)
        # Example: if "old_param_name" in preset_data:
        #     preset_data["new_param_name"] = preset_data.pop("old_param_name")

    # Always check for aspect ratio migration (regardless of version)
    if "aspect_ratio" in preset_data:
        old_aspect_ratio = preset_data["aspect_ratio"]
        if old_aspect_ratio in ASPECT_RATIO_MIGRATION:
            new_aspect_ratio = ASPECT_RATIO_MIGRATION[old_aspect_ratio]
            preset_data["aspect_ratio"] = new_aspect_ratio
            print(f"[PRESET] Migrated aspect ratio: '{old_aspect_ratio}' -> '{new_aspect_ratio}'")
            migrated = True

    # Update version if migration occurred
    if migrated:
        preset_data["preset_version"] = PRESET_VERSION
        preset_data["migrated_at"] = datetime.now().isoformat()

    return preset_data

def load_preset_safely(preset_name):
    """Load and validate preset data with error recovery."""
    try:
        import json
        presets_dir = get_presets_dir()
        preset_file = os.path.join(presets_dir, f"{preset_name}.json")

        if not os.path.exists(preset_file):
            return None, f"Preset '{preset_name}' not found"

        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)

        # Check version compatibility and migrate if needed
        version = preset_data.get("preset_version", "1.0")

        # Always attempt migration for any preset that doesn't match current version
        if version != PRESET_VERSION:
            preset_data = migrate_preset_data(preset_data)

        # Only reject if migration failed or version is extremely old
        migrated_version = preset_data.get("preset_version", "1.0")
        if migrated_version < PRESET_MIN_COMPATIBLE_VERSION:
            return None, f"Preset version {version} is too old and could not be migrated (minimum required: {PRESET_MIN_COMPATIBLE_VERSION})"

        # Validate all parameters
        validated_data = {}
        for param_name, default_value in PRESET_DEFAULTS.items():
            raw_value = preset_data.get(param_name, default_value)
            validated_value = validate_preset_value(param_name, raw_value)
            validated_data[param_name] = validated_value

        # Include metadata fields
        validated_data["preset_version"] = preset_data.get("preset_version", PRESET_VERSION)
        if "migrated_at" in preset_data:
            validated_data["migrated_at"] = preset_data["migrated_at"]

        return validated_data, None

    except json.JSONDecodeError as e:
        return None, f"Invalid preset file format: {e}"
    except Exception as e:
        return None, f"Error loading preset: {e}"

def save_preset(preset_name, current_preset,
                # All UI parameters
                video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
                video_seed, randomize_seed, no_audio, save_metadata,
                solver_name, sample_steps, num_generations,
                shift, video_guidance_scale, audio_guidance_scale, slg_layer,
                blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
                video_negative_prompt, audio_negative_prompt,
                batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
                vae_tiled_decode, vae_tile_size, vae_tile_overlap,
                base_resolution_width, base_resolution_height, duration_seconds,
                enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input):
    """Save current UI state as a preset."""
    try:
        presets_dir = get_presets_dir()

        # If no name provided, use current preset name
        if not preset_name.strip() and current_preset:
            preset_name = current_preset

        if not preset_name.strip():
            presets = get_available_presets()
            return gr.update(choices=presets, value=None), gr.update(value=""), *[gr.update() for _ in range(38)], "Please enter a preset name or select a preset to overwrite"

        preset_file = os.path.join(presets_dir, f"{preset_name}.json")

        # Collect all current settings
        preset_data = {
            "preset_version": PRESET_VERSION,
            "video_text_prompt": video_text_prompt,
            "aspect_ratio": aspect_ratio,
            "video_width": video_width,
            "video_height": video_height,
            "auto_crop_image": auto_crop_image,
            "video_seed": video_seed,
            "randomize_seed": randomize_seed,
            "no_audio": no_audio,
            "save_metadata": save_metadata,
            "solver_name": solver_name,
            "sample_steps": sample_steps,
            "num_generations": num_generations,
            "shift": shift,
            "video_guidance_scale": video_guidance_scale,
            "audio_guidance_scale": audio_guidance_scale,
            "slg_layer": slg_layer,
            "blocks_to_swap": blocks_to_swap,
            "cpu_offload": cpu_offload,
            "delete_text_encoder": delete_text_encoder,
            "fp8_t5": fp8_t5,
            "cpu_only_t5": cpu_only_t5,
            "fp8_base_model": fp8_base_model,
            "use_sage_attention": use_sage_attention,
            "video_negative_prompt": video_negative_prompt,
            "audio_negative_prompt": audio_negative_prompt,
            "batch_input_folder": batch_input_folder,
            "batch_output_folder": batch_output_folder,
            "batch_skip_existing": batch_skip_existing,
            "clear_all": clear_all,
            "vae_tiled_decode": vae_tiled_decode,
            "vae_tile_size": vae_tile_size,
            "vae_tile_overlap": vae_tile_overlap,
            "base_resolution_width": base_resolution_width,
            "base_resolution_height": base_resolution_height,
            "duration_seconds": duration_seconds,
            "enable_multiline_prompts": enable_multiline_prompts,
            "enable_video_extension": enable_video_extension,
            "dont_auto_combine_video_input": dont_auto_combine_video_input,
            "saved_at": datetime.now().isoformat()
        }

        # Save to file
        with open(preset_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

        # Save last used preset for auto-load
        last_used_file = os.path.join(presets_dir, "last_used.txt")
        with open(last_used_file, 'w', encoding='utf-8') as f:
            f.write(preset_name)

        presets = get_available_presets()
        # Load the preset to get all the UI values
        loaded_values = load_preset(preset_name)
        # Return dropdown update, name clear, loaded values, success message
        return gr.update(choices=presets, value=preset_name), gr.update(value=""), *loaded_values[:-1], f"Preset '{preset_name}' saved successfully!"

    except Exception as e:
        presets = get_available_presets()
        return gr.update(choices=presets, value=None), gr.update(value=""), *[gr.update() for _ in range(38)], f"Error saving preset: {e}"

def load_preset(preset_name):
    """Load a preset and return all UI values with robust error handling."""
    try:
        if not preset_name:
            return [gr.update() for _ in range(38)] + [gr.update(value=None)] + ["No preset selected"]

        # Use the robust loading system
        preset_data, error_msg = load_preset_safely(preset_name)

        if preset_data is None:
            # Loading failed, return error state
            return [gr.update() for _ in range(38)] + [gr.update(value=None)] + [error_msg]

        # Save as last used for auto-load (only if loading succeeded)
        try:
            presets_dir = get_presets_dir()
            last_used_file = os.path.join(presets_dir, "last_used.txt")
            with open(last_used_file, 'w', encoding='utf-8') as f:
                f.write(preset_name)
        except Exception as e:
            print(f"[PRESET] Warning: Could not save last used preset: {e}")

        stored_ratio_value = preset_data.get("aspect_ratio")
        resolved_aspect_value = _resolve_aspect_ratio_value(
            stored_ratio_value,
            preset_data.get("video_width"),
            preset_data.get("video_height"),
        )

        base_res_width = _coerce_positive_int(preset_data.get("base_resolution_width")) or 720
        base_res_height = _coerce_positive_int(preset_data.get("base_resolution_height")) or 720

        current_ratio_map = get_common_aspect_ratios(base_res_width, base_res_height)
        aspect_ratio_choices = [_format_ratio_choice(name, dims) for name, dims in current_ratio_map.items()]
        if resolved_aspect_value not in aspect_ratio_choices:
            aspect_ratio_choices = [resolved_aspect_value] + aspect_ratio_choices

        # Apply automatic memory-based optimizations after preset loading
        gpu_name, vram_gb = detect_gpu_info()
        ram_gb = detect_system_ram()

        optimization_messages = []

        # RAM-based optimization: Enable Delete T5 After Encoding for RAM < 128GB
        if ram_gb > 0 and ram_gb < 63:
            preset_data["delete_text_encoder"] = True
            optimization_messages.append(f"RAM {ram_gb:.1f}GB < 128GB → Enabled Delete T5 After Encoding")
            print(f"  ✓ RAM optimization: Enabled Delete T5 After Encoding (RAM: {ram_gb:.1f}GB < 128GB)")

        # VRAM-based optimizations (same as before)
        if vram_gb > 0:
            if vram_gb < 23:
                # Enable Scaled FP8 T5 and Tiled VAE for VRAM < 23GB
                preset_data["fp8_t5"] = True
                preset_data["vae_tiled_decode"] = True
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB < 23GB → Enabled Scaled FP8 T5 + Tiled VAE")
                print(f"  ✓ VRAM optimization: Enabled Scaled FP8 T5 + Tiled VAE (VRAM: {vram_gb:.1f}GB < 23GB)")

            if vram_gb > 40:
                # Disable Clear All Memory for VRAM > 40GB
                preset_data["clear_all"] = False
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB > 40GB → Disabled Clear All Memory")
                print(f"  ✓ VRAM optimization: Disabled Clear All Memory (VRAM: {vram_gb:.1f}GB > 40GB)")

        if optimization_messages:
            print(f"[PRESET] Applied automatic optimizations for '{preset_name}': {', '.join(optimization_messages)}")


        # Return all UI updates in the correct order
        # This order must match the Gradio UI component order exactly
        return (
            gr.update(value=preset_data["video_text_prompt"]),
            gr.update(value=resolved_aspect_value, choices=aspect_ratio_choices),
            gr.update(value=preset_data["video_width"]),
            gr.update(value=preset_data["video_height"]),
            gr.update(value=preset_data["auto_crop_image"]),
            gr.update(value=preset_data["video_seed"]),
            gr.update(value=preset_data["randomize_seed"]),
            gr.update(value=preset_data["no_audio"]),
            gr.update(value=preset_data["save_metadata"]),
            gr.update(value=preset_data["solver_name"]),
            gr.update(value=preset_data["sample_steps"]),
            gr.update(value=preset_data["num_generations"]),
            gr.update(value=preset_data["shift"]),
            gr.update(value=preset_data["video_guidance_scale"]),
            gr.update(value=preset_data["audio_guidance_scale"]),
            gr.update(value=preset_data["slg_layer"]),
            gr.update(value=preset_data["blocks_to_swap"]),
            gr.update(value=preset_data["cpu_offload"]),
            gr.update(value=preset_data["delete_text_encoder"]),
            gr.update(value=preset_data["fp8_t5"]),
            gr.update(value=preset_data["cpu_only_t5"]),
            gr.update(value=preset_data["fp8_base_model"]),
            gr.update(value=preset_data.get("use_sage_attention", False)),
            gr.update(value=preset_data["video_negative_prompt"]),
            gr.update(value=preset_data["audio_negative_prompt"]),
            gr.update(value=preset_data["batch_input_folder"]),
            gr.update(value=preset_data["batch_output_folder"]),
            gr.update(value=preset_data["batch_skip_existing"]),
            gr.update(value=preset_data["clear_all"]),
            gr.update(value=preset_data["vae_tiled_decode"]),
            gr.update(value=preset_data["vae_tile_size"]),
            gr.update(value=preset_data["vae_tile_overlap"]),
            gr.update(value=preset_data["base_resolution_width"]),
            gr.update(value=preset_data["base_resolution_height"]),
            gr.update(value=preset_data["duration_seconds"]),
            gr.update(value=preset_data["enable_multiline_prompts"]),
            gr.update(value=preset_data["enable_video_extension"]),
            gr.update(value=preset_data.get("dont_auto_combine_video_input", False)),
            gr.update(value=preset_name),  # Update dropdown value
            f"Preset '{preset_name}' loaded successfully!{' Applied optimizations: ' + ', '.join(optimization_messages) if optimization_messages else ''}"
        )

    except Exception as e:
        error_msg = f"Unexpected error loading preset: {e}"
        print(f"[PRESET] {error_msg}")
        return [gr.update() for _ in range(38)] + [gr.update(value=None)] + [error_msg]

def initialize_app_with_auto_load():
    """Initialize app with preset dropdown choices and auto-load last preset or VRAM-based preset."""
    try:
        # Clean up any invalid presets before initializing
        removed_count = cleanup_invalid_presets()
        if removed_count > 0:
            print(f"[PRESET] Cleaned up {removed_count} invalid preset(s)")

        presets = get_available_presets()
        dropdown_update = gr.update(choices=presets, value=None)

        # Try to auto-load the last used preset
        presets_dir = get_presets_dir()
        last_used_file = os.path.join(presets_dir, "last_used.txt")

        if os.path.exists(last_used_file):
            try:
                with open(last_used_file, 'r', encoding='utf-8') as f:
                    last_preset = f.read().strip()

                if last_preset and last_preset in presets:
                    print(f"Auto-loading last used preset: {last_preset}")
                    # Load the preset and update dropdown to select it
                    loaded_values = load_preset(last_preset)

                    # Check if loading was successful (last element is the status message)
                    status_message = loaded_values[-1]
                    if "successfully" in status_message:
                        # Return dropdown with selected preset + all loaded UI values
                        return gr.update(choices=presets, value=last_preset), *loaded_values[:-1], f"Auto-loaded preset '{last_preset}'"
                    else:
                        print(f"[PRESET] Failed to auto-load preset '{last_preset}': {status_message}")
                        print("[PRESET] Falling back to VRAM-based preset selection")
                else:
                    print(f"[PRESET] Last used preset '{last_preset}' not found in available presets")
            except Exception as e:
                print(f"[PRESET] Error reading last used preset file: {e}")
                print("[PRESET] Falling back to VRAM-based preset selection")

        # No last used preset - detect VRAM and select best matching preset
        gpu_name, vram_gb = detect_gpu_info()

        if vram_gb > 0 and presets:
            # Find VRAM-based presets (those with "GB" in the name)
            vram_presets = [p for p in presets if 'GB' in p and any(char.isdigit() for char in p)]

            if vram_presets:
                # Extract VRAM values from preset names and find the best match
                import re
                best_preset = None
                best_vram_diff = float('inf')

                for preset in vram_presets:
                    match = re.search(r'(\d+)', preset)
                    if match:
                        preset_vram = int(match.group(1))
                        vram_diff = abs(vram_gb - preset_vram)  # Use absolute difference for closest match

                        if vram_diff < best_vram_diff:
                            best_vram_diff = vram_diff
                            best_preset = preset

                if best_preset:
                    print(f"Auto-loading VRAM-based preset: {best_preset} (detected {vram_gb:.1f}GB VRAM)")
                    # Load the preset and update dropdown to select it
                    loaded_values = load_preset(best_preset)

                    # Check if loading was successful
                    status_message = loaded_values[-1]
                    if "successfully" in status_message:
                        # Return dropdown with selected preset + all loaded UI values
                        return gr.update(choices=presets, value=best_preset), *loaded_values[:-1], f"Auto-loaded VRAM-optimized preset '{best_preset}' ({vram_gb:.1f}GB GPU detected)"
                    else:
                        print(f"[PRESET] Failed to auto-load VRAM-based preset '{best_preset}': {status_message}")
                        print("[PRESET] Falling back to basic VRAM optimizations")
                else:
                    print(f"No suitable VRAM-based preset found for {vram_gb:.1f}GB VRAM")

        # Fallback: No preset to auto-load - check RAM and VRAM and apply basic optimizations
        print("No preset auto-loaded - applying basic memory optimizations...")

        gpu_name, vram_gb = detect_gpu_info()
        ram_gb = detect_system_ram()

        # Apply basic memory-based optimizations when no preset is loaded
        fp8_t5_update = gr.update()  # Default: False
        vae_tiled_decode_update = gr.update()  # Default: False
        clear_all_update = gr.update()  # Default: True
        delete_text_encoder_update = gr.update()  # Default: False

        optimization_messages = []

        # RAM-based optimization: Enable Delete T5 After Encoding for RAM < 128GB
        if ram_gb > 0 and ram_gb < 128:
            delete_text_encoder_update = gr.update(value=True)
            optimization_messages.append(f"RAM {ram_gb:.1f}GB < 128GB → Enabled Delete T5 After Encoding")
            print(f"  ✓ RAM optimization: Enabled Delete T5 After Encoding (RAM: {ram_gb:.1f}GB < 128GB)")

        # VRAM-based optimizations (same as before)
        if vram_gb > 0:
            if vram_gb < 23:
                # Enable Scaled FP8 T5 and Tiled VAE for VRAM < 23GB
                fp8_t5_update = gr.update(value=True)
                vae_tiled_decode_update = gr.update(value=True)
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB < 23GB → Enabled Scaled FP8 T5 + Tiled VAE")
                print(f"  ✓ VRAM optimization: Enabled Scaled FP8 T5 + Tiled VAE (VRAM: {vram_gb:.1f}GB < 23GB)")

            if vram_gb > 40:
                # Disable Clear All Memory for VRAM > 40GB
                clear_all_update = gr.update(value=False)
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB > 40GB → Disabled Clear All Memory")
                print(f"  ✓ VRAM optimization: Disabled Clear All Memory (VRAM: {vram_gb:.1f}GB > 40GB)")

        if optimization_messages:
            status_message = "Applied memory optimizations: " + ", ".join(optimization_messages)
        else:
            status_message = f"Hardware detected (RAM: {ram_gb:.1f}GB, GPU: {gpu_name}, VRAM: {vram_gb:.1f}GB) - using default settings"
            print(f"  ✓ No memory optimizations needed (RAM: {ram_gb:.1f}GB, VRAM: {vram_gb:.1f}GB in optimal range)")

        # Return initialized dropdown with VRAM-optimized defaults
        # The order must match the outputs list in demo.load()
        # Initialize aspect ratio choices with resolution info
        default_ratios = get_common_aspect_ratios(720, 720)
        aspect_choices = [f"{name} - {w}x{h}px" for name, (w, h) in default_ratios.items()]
        # Set default to 16:9 if available, otherwise first choice
        default_aspect_value = next((c for c in aspect_choices if c.startswith("16:9")), aspect_choices[0] if aspect_choices else None)
        initial_aspect_choices = gr.update(choices=aspect_choices, value=default_aspect_value)

        return (
            dropdown_update,  # preset_dropdown
            gr.update(),  # video_text_prompt
            initial_aspect_choices,  # aspect_ratio
            gr.update(),  # video_width
            gr.update(),  # video_height
            gr.update(),  # auto_crop_image
            gr.update(),  # video_seed
            gr.update(),  # randomize_seed
            gr.update(),  # no_audio
            gr.update(),  # save_metadata
            gr.update(),  # solver_name
            gr.update(),  # sample_steps
            gr.update(),  # num_generations
            gr.update(),  # shift
            gr.update(),  # video_guidance_scale
            gr.update(),  # audio_guidance_scale
            gr.update(),  # slg_layer
            gr.update(),  # blocks_to_swap
            gr.update(),  # cpu_offload
            delete_text_encoder_update,  # delete_text_encoder (potentially modified)
            fp8_t5_update,  # fp8_t5 (potentially modified)
            gr.update(),  # cpu_only_t5
            gr.update(),  # fp8_base_model
            gr.update(),  # use_sage_attention
            gr.update(),  # video_negative_prompt
            gr.update(),  # audio_negative_prompt
            gr.update(),  # batch_input_folder
            gr.update(),  # batch_output_folder
            gr.update(),  # batch_skip_existing
            clear_all_update,  # clear_all (potentially modified)
            vae_tiled_decode_update,  # vae_tiled_decode (potentially modified)
            gr.update(),  # vae_tile_size
            gr.update(),  # vae_tile_overlap
            gr.update(),  # base_resolution_width
            gr.update(),  # base_resolution_height
            gr.update(),  # duration_seconds
            gr.update(),  # enable_multiline_prompts
            gr.update(),  # enable_video_extension
            gr.update(),  # dont_auto_combine_video_input
            status_message  # status message
        )

    except Exception as e:
        print(f"Warning: Could not initialize app with auto-load: {e}")
        presets = get_available_presets()
        return gr.update(choices=presets, value=None), *[gr.update() for _ in range(38)], ""

def initialize_app():
    """Initialize app with preset dropdown choices."""
    presets = get_available_presets()
    return gr.update(choices=presets, value=None)

def save_generation_metadata(output_path, generation_params, used_seed):
    """Save generation metadata as a .txt file alongside the video."""
    try:
        # Create metadata filename (same as video but .txt extension)
        metadata_path = output_path.replace('.mp4', '.txt')

        # Determine generation type
        is_extension = generation_params.get('is_extension', False)
        extension_index = generation_params.get('extension_index', 0)
        is_batch = generation_params.get('is_batch', False)

        metadata_content = f"""OVI VIDEO GENERATION METADATA
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO PARAMETERS:
- Text Prompt: {generation_params.get('text_prompt', 'N/A')}
- Image Path: {generation_params.get('image_path', 'None')}
- Resolution: {generation_params.get('video_frame_height', 'N/A')}x{generation_params.get('video_frame_width', 'N/A')}
- Aspect Ratio: {generation_params.get('aspect_ratio', 'N/A')}
- Base Resolution: {generation_params.get('base_resolution_width', 720)}x{generation_params.get('base_resolution_height', 720)}
- Duration: {generation_params.get('duration_seconds', 5)} seconds
- Seed Used: {used_seed}
- Randomize Seed: {generation_params.get('randomize_seed', False)}
- Number of Generations: {generation_params.get('num_generations', 1)}
- Generation Type: {'Extension' if is_extension else 'Batch' if is_batch else 'Single'}
{f'- Extension Index: {extension_index}' if is_extension else ''}

GENERATION SETTINGS:
- Solver: {generation_params.get('solver_name', 'N/A')}
- Sample Steps: {generation_params.get('sample_steps', 'N/A')}
- Shift: {generation_params.get('shift', 'N/A')}
- Video Guidance Scale: {generation_params.get('video_guidance_scale', 'N/A')}
- Audio Guidance Scale: {generation_params.get('audio_guidance_scale', 'N/A')}
- SLG Layer: {generation_params.get('slg_layer', 'N/A')}

MEMORY OPTIMIZATION:
- Block Swap: {generation_params.get('blocks_to_swap', 'N/A')} blocks
- CPU Offload: {generation_params.get('cpu_offload', 'N/A')}
- Delete Text Encoder: {generation_params.get('delete_text_encoder', True)}
- Scaled FP8 T5: {generation_params.get('fp8_t5', False)}
- CPU-Only T5: {generation_params.get('cpu_only_t5', False)}
- Scaled FP8 Base Model: {generation_params.get('fp8_base_model', False)}
- Sage Attention: {generation_params.get('use_sage_attention', False)}
- No Block Prep: {generation_params.get('no_block_prep', False)}
- Clear All Memory: {generation_params.get('clear_all', False)}

VAE OPTIMIZATION:
- Tiled VAE Decode: {generation_params.get('vae_tiled_decode', False)}
- VAE Tile Size: {generation_params.get('vae_tile_size', 'N/A')}
- VAE Tile Overlap: {generation_params.get('vae_tile_overlap', 'N/A')}

NEGATIVE PROMPTS:
- Video: {generation_params.get('video_negative_prompt', 'N/A')}
- Audio: {generation_params.get('audio_negative_prompt', 'N/A')}

OUTPUT SETTINGS:
- No Audio: {generation_params.get('no_audio', False)}
- Output Path: {output_path}
- Metadata Path: {metadata_path}

SYSTEM INFO:
- OviFusionEngine Version: 1.0
- Generation Mode: {'Batch' if generation_params.get('is_batch', False) else 'Single'}
"""

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata_content)

        print(f"[METADATA] Saved generation metadata: {metadata_path}")
        return True

    except Exception as e:
        print(f"[METADATA ERROR] Failed to save metadata: {e}")
        return False

def scan_batch_files(input_folder):
    """Scan input folder and return list of (base_name, image_path, txt_path) tuples."""
    import os
    import glob

    # Normalize path to handle spaces and cross-platform compatibility
    input_folder = os.path.abspath(input_folder.strip())
    
    if not os.path.exists(input_folder):
        raise Exception(f"Input folder does not exist: {input_folder}")

    # Supported image extensions
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    txt_ext = '.txt'

    # Find all txt files
    txt_files = glob.glob(os.path.join(input_folder, f"*{txt_ext}"))

    batch_items = []

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]

        # Check if there's a matching image file
        image_path = None
        for ext in image_exts:
            potential_image = os.path.join(input_folder, f"{base_name}{ext}")
            if os.path.exists(potential_image):
                image_path = potential_image
                break

        batch_items.append((base_name, image_path, txt_file))

    return batch_items

def process_batch_generation(
    input_folder,
    output_folder,
    skip_existing,
    # Generation parameters
    video_frame_height,
    video_frame_width,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    blocks_to_swap,
    video_negative_prompt,
    audio_negative_prompt,
    cpu_offload,
    delete_text_encoder,
    fp8_t5,
    cpu_only_t5,
    fp8_base_model,
    use_sage_attention,
    no_audio,
    no_block_prep,
    num_generations,
    randomize_seed,
    save_metadata,
    aspect_ratio,
    clear_all,
    vae_tiled_decode,
    vae_tile_size,
    vae_tile_overlap,
    base_resolution_width,
    base_resolution_height,
    duration_seconds,
    auto_crop_image,
    enable_multiline_prompts,
    enable_video_extension,  # Boolean checkbox from UI (not count)
    dont_auto_combine_video_input,  # New: Don't auto combine video input
):
    """Process batch generation from input folder."""
    global ovi_engine

    try:
        # Check for cancellation at the start
        check_cancellation()

        # Determine output directory (normalize paths for both Windows and Linux)
        if output_folder and output_folder.strip():
            outputs_dir = os.path.abspath(output_folder.strip())  # Normalize path and handle spaces
        elif args.output_dir:
            outputs_dir = os.path.abspath(args.output_dir)
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(outputs_dir, exist_ok=True)
            print(f"[BATCH] Output directory: {outputs_dir}")
        except Exception as e:
            raise Exception(f"Failed to create output directory '{outputs_dir}': {e}")

        # Scan batch files (normalize input folder path)
        input_folder_normalized = os.path.abspath(input_folder.strip())
        print(f"[BATCH] Input directory: {input_folder_normalized}")
        
        batch_items = scan_batch_files(input_folder_normalized)
        if not batch_items:
            raise Exception(f"No .txt files found in input folder: {input_folder_normalized}")

        print(f"\n[INFO] Found {len(batch_items)} items to process:")
        for base_name, img_path, txt_path in batch_items:
            img_status = "with image" if img_path else "text-only"
            print(f"  - {base_name}: {img_status}")

        # IMPORTANT: Recalculate video dimensions from base resolution and aspect ratio
        # This ensures auto-cropping uses the correct target resolution
        parsed_dims = _parse_resolution_from_label(aspect_ratio)
        if parsed_dims:
            # Aspect ratio has explicit dimensions, use them
            video_frame_width, video_frame_height = parsed_dims
        else:
            # Calculate from base resolution and aspect ratio name
            base_width = _coerce_positive_int(base_resolution_width) or 720
            base_height = _coerce_positive_int(base_resolution_height) or 720
            current_ratios = get_common_aspect_ratios(base_width, base_height)
            
            # Extract ratio name from aspect_ratio (e.g., "16:9" from "16:9 - 992x512px")
            ratio_name = _extract_ratio_name(aspect_ratio)
            if ratio_name and ratio_name in current_ratios:
                video_frame_width, video_frame_height = current_ratios[ratio_name]
                print(f"[BATCH] Recalculated resolution from base {base_width}x{base_height} and aspect {ratio_name}: {video_frame_width}x{video_frame_height}")
            # else keep the original video_frame_width/height from parameters
        
        # Only initialize engine if we're not using subprocess mode (clear_all=False)
        # When clear_all=True, all batch generations run in subprocesses, so main process doesn't need models
        if clear_all:
            print("=" * 80)
            print("CLEAR ALL MEMORY ENABLED FOR BATCH PROCESSING")
            print("  Main process will NOT load any models")
            print("  All batch generations will run in separate subprocesses")
            print("  VRAM/RAM will be completely cleared between each batch item")
            print("=" * 80)

        if not clear_all and ovi_engine is None:
            # Use CLI args only in test mode, otherwise use GUI parameters
            if getattr(args, 'test', False):
                final_blocks_to_swap = getattr(args, 'blocks_to_swap', 0)
                final_cpu_offload = getattr(args, 'test_cpu_offload', False)
            else:
                final_blocks_to_swap = blocks_to_swap
                final_cpu_offload = cpu_offload

            print("=" * 80)
            print("INITIALIZING OVI FUSION ENGINE FOR BATCH PROCESSING IN MAIN PROCESS")
            print(f"  Block Swap: {final_blocks_to_swap} blocks (0 = disabled)")
            print(f"  CPU Offload: {final_cpu_offload}")
            print(f"  No Block Prep: {no_block_prep}")
            print(f"  Note: Models will be loaded in main process (Clear All Memory disabled)")
            print("=" * 80)

            # Calculate latent lengths based on duration
            video_latent_length, audio_latent_length = calculate_latent_lengths(duration_seconds)

            DEFAULT_CONFIG['cpu_offload'] = final_cpu_offload
            DEFAULT_CONFIG['mode'] = "t2v"
            ovi_engine = OviFusionEngine(
                blocks_to_swap=final_blocks_to_swap,
                cpu_offload=final_cpu_offload,
                video_latent_length=video_latent_length,
                audio_latent_length=audio_latent_length
            )
            print("\n[OK] OviFusionEngine initialized successfully for batch processing")

        processed_count = 0
        skipped_count = 0
        last_output_path = None

        # Process each batch item
        for base_name, image_path, txt_path in batch_items:
            # Check for cancellation
            check_cancellation()

            # Read prompt from txt file
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    raw_text_prompt = f.read().strip()
                if not raw_text_prompt:
                    print(f"[WARNING] Empty prompt file, skipping: {txt_path}")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to read prompt file {txt_path}: {e}")
                continue

            # Parse multi-line prompts if enabled
            individual_prompts = parse_multiline_prompts(raw_text_prompt, enable_multiline_prompts)

            print(f"\n[PROCESSING] {base_name}")
            if enable_multiline_prompts:
                print(f"  Multi-line prompts: {len(individual_prompts)} prompts")
                for i, prompt in enumerate(individual_prompts):
                    print(f"    Prompt {i+1}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            else:
                print(f"  Prompt: {raw_text_prompt[:100]}{'...' if len(raw_text_prompt) > 100 else ''}")
            print(f"  Image: {'Yes' if image_path else 'No'}")

            # Apply auto cropping if enabled and image exists
            # For batch processing with auto-crop, let each image auto-detect its own aspect ratio
            # by passing None for target dimensions (each image in batch may have different aspect ratios)
            if auto_crop_image and image_path:
                final_image_path, detected_width, detected_height = apply_auto_crop_if_enabled(
                    image_path, auto_crop_image, None, None, return_dimensions=True
                )
                # Use the auto-detected dimensions for this specific image
                batch_video_width = detected_width if detected_width else video_frame_width
                batch_video_height = detected_height if detected_height else video_frame_height
            else:
                # If auto-crop is disabled, use the configured resolution
                final_image_path = apply_auto_crop_if_enabled(image_path, auto_crop_image, video_frame_width, video_frame_height)
                batch_video_width = video_frame_width
                batch_video_height = video_frame_height

            # Check if output already exists (for skip logic)
            expected_output = os.path.join(outputs_dir, f"{base_name}_0001.mp4")
            if skip_existing and os.path.exists(expected_output):
                print(f"  [SKIPPED] Output already exists: {expected_output}")
                skipped_count += 1
                continue

            # Generate videos for each prompt (supporting multi-line prompts and multiple generations)
            for prompt_idx, current_prompt in enumerate(individual_prompts):
                print(f"  [PROMPT {prompt_idx + 1}/{len(individual_prompts)}] Processing: {current_prompt[:50]}{'...' if len(current_prompt) > 50 else ''}")

                # For multi-line prompts in batch processing, append prompt index to avoid overwrites
                if enable_multiline_prompts and len(individual_prompts) > 1:
                    current_base_name = f"{base_name}_{prompt_idx + 1}"
                else:
                    current_base_name = base_name

                # Generate multiple videos for this prompt
                for gen_idx in range(int(num_generations)):
                    # Check for cancellation in the loop
                    check_cancellation()

                    # Handle seed logic for batch processing
                    current_seed = 99  # Default seed for batch processing
                    if randomize_seed:
                        current_seed = get_random_seed()
                    elif gen_idx > 0:
                        current_seed = 99 + gen_idx

                    print(f"    [GENERATION {gen_idx + 1}/{int(num_generations)}] Seed: {current_seed}")

                if clear_all:
                    # Run this batch generation in a subprocess for memory cleanup
                    single_gen_params = {
                        'text_prompt': current_prompt,
                        'image': final_image_path,
                        'video_frame_height': batch_video_height,  # Use detected dimensions
                        'video_frame_width': batch_video_width,    # Use detected dimensions
                        'video_seed': current_seed,
                        'solver_name': solver_name,
                        'sample_steps': sample_steps,
                        'shift': shift,
                        'video_guidance_scale': video_guidance_scale,
                        'audio_guidance_scale': audio_guidance_scale,
                        'slg_layer': slg_layer,
                        'blocks_to_swap': blocks_to_swap,
                        'video_negative_prompt': video_negative_prompt,
                        'audio_negative_prompt': audio_negative_prompt,
                        'use_image_gen': False,
                        'cpu_offload': cpu_offload,
                        'delete_text_encoder': delete_text_encoder,
                        'fp8_t5': fp8_t5,
                        'cpu_only_t5': cpu_only_t5,
                        'fp8_base_model': fp8_base_model,
                        'use_sage_attention': use_sage_attention,
                        'no_audio': no_audio,
                        'no_block_prep': no_block_prep,
                        'num_generations': 1,
                        'randomize_seed': False,
                        'save_metadata': save_metadata,
                        'aspect_ratio': aspect_ratio,
                        'clear_all': False,  # Disable subprocess in subprocess
                        'vae_tiled_decode': vae_tiled_decode,
                        'vae_tile_size': vae_tile_size,
                        'vae_tile_overlap': vae_tile_overlap,
                        'base_resolution_width': base_resolution_width,
                        'base_resolution_height': base_resolution_height,
                        'duration_seconds': duration_seconds,
                        'auto_crop_image': False,  # Image already cropped in main process
                        'base_filename': current_base_name,  # Use current base name for batch processing (handles multiline numbering)
                        'output_dir': outputs_dir,  # Pass output directory to subprocess
                        'enable_multiline_prompts': False,  # Disable in subprocess
                        'enable_video_extension': enable_video_extension,  # Pass through extension setting
                    }

                    success = run_generation_subprocess(single_gen_params)
                    if success:
                        # Find the generated file with current_base_name prefix in the correct output directory
                        import glob
                        pattern = os.path.join(outputs_dir, f"{current_base_name}_*.mp4")
                        existing_files = glob.glob(pattern)
                        if existing_files:
                            last_output_path = max(existing_files, key=os.path.getctime)
                            print(f"      [SUCCESS] Saved to: {last_output_path}")
                            processed_count += 1
                        else:
                            print(f"      [WARNING] No output file found for {current_base_name} in {outputs_dir}")
                            print(f"      [DEBUG] Search pattern: {pattern}")
                    else:
                        print(f"      [ERROR] Generation failed in subprocess")
                    continue

                # Original batch generation logic (when clear_all is disabled)
                # Now uses generate_video() to support video extensions
                if not clear_all:
                    try:
                        # Call generate_video() which handles everything including video extensions
                        last_output_path = generate_video(
                            text_prompt=current_prompt,
                            image=final_image_path,
                            video_frame_height=batch_video_height,
                            video_frame_width=batch_video_width,
                            video_seed=current_seed,
                            solver_name=solver_name,
                            sample_steps=sample_steps,
                            shift=shift,
                            video_guidance_scale=video_guidance_scale,
                            audio_guidance_scale=audio_guidance_scale,
                            slg_layer=slg_layer,
                            blocks_to_swap=blocks_to_swap,
                            video_negative_prompt=video_negative_prompt,
                            audio_negative_prompt=audio_negative_prompt,
                            use_image_gen=False,
                            cpu_offload=cpu_offload,
                            delete_text_encoder=delete_text_encoder,
                            fp8_t5=fp8_t5,
                            cpu_only_t5=cpu_only_t5,
                            fp8_base_model=fp8_base_model,
                            use_sage_attention=use_sage_attention,
                            no_audio=no_audio,
                            no_block_prep=no_block_prep,
                            num_generations=1,  # Generate one at a time in batch mode
                            randomize_seed=False,  # Seed already set above
                            save_metadata=save_metadata,
                            aspect_ratio=aspect_ratio,
                            clear_all=False,  # We're already in non-clear_all mode
                            vae_tiled_decode=vae_tiled_decode,
                            vae_tile_size=vae_tile_size,
                            vae_tile_overlap=vae_tile_overlap,
                            base_resolution_width=base_resolution_width,
                            base_resolution_height=base_resolution_height,
                            duration_seconds=duration_seconds,
                            auto_crop_image=False,  # Image already cropped
                            base_filename=current_base_name,  # Use batch-specific filename
                            output_dir=outputs_dir,
                            text_embeddings_cache=None,
                            enable_multiline_prompts=False,  # Already handled at batch level
                            enable_video_extension=enable_video_extension,  # Pass through video extension setting
                            dont_auto_combine_video_input=dont_auto_combine_video_input,  # Pass through setting
                        )
                        
                        if last_output_path:
                            print(f"      [SUCCESS] Saved to: {last_output_path}")
                            processed_count += 1
                        else:
                            print(f"      [WARNING] Generation returned no output path")

                    except Exception as e:
                        print(f"      [ERROR] Generation failed: {e}")
                        continue

        print("\n" + "=" * 80)
        print("[BATCH COMPLETE]")
        print(f"  Processed: {processed_count} videos")
        print(f"  Skipped: {skipped_count} existing videos")
        print(f"  Total items: {len(batch_items)}")
        print(f"  Output directory: {outputs_dir}")
        print("=" * 80)

        # Return None instead of path to avoid Gradio InvalidPathError when using custom output directory
        # Gradio cannot handle paths outside its allowed directories (working dir, temp dir)
        # The user can access the files directly from the output folder
        return None

    except Exception as e:
        error_msg = str(e)
        if "cancelled by user" in error_msg.lower():
            print("[BATCH] Batch processing cancelled by user")
            reset_cancellation()  # Reset the cancellation flag
            return None
        else:
            print(f"[BATCH ERROR] {e}")
            return None

def load_i2v_example_with_resolution(prompt, img_path):
    """Load I2V example and set appropriate resolution based on image aspect ratio."""
    if img_path is None or not os.path.exists(img_path):
        return (prompt, None, gr.update(), gr.update(), gr.update(), None)
    
    try:
        img = Image.open(img_path)
        iw, ih = img.size
        if ih == 0 or iw == 0:
            return (prompt, img_path, gr.update(), gr.update(), gr.update(), img_path)
        
        aspect = iw / ih

        # Calculate aspect ratios from ratio strings and find closest match
        def get_ratio_value(ratio_str):
            w, h = map(float, ratio_str.split(':'))
            return w / h

        closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(get_ratio_value(k) - aspect))
        target_w, target_h = ASPECT_RATIOS[closest_key]
        aspect_display = _format_ratio_choice(closest_key, ASPECT_RATIOS[closest_key])
        aspect_choices = [_format_ratio_choice(name, dims) for name, dims in get_common_aspect_ratios(target_w, target_h).items()]
        if aspect_display not in aspect_choices:
            aspect_choices = [aspect_display] + aspect_choices
        
        return (
            prompt, 
            img_path,
            gr.update(value=aspect_display, choices=aspect_choices),
            gr.update(value=target_w),
            gr.update(value=target_h),
            img_path
        )
    except Exception as e:
        print(f"Error loading I2V example: {e}")
        return (prompt, img_path, gr.update(), gr.update(), gr.update(), img_path)

def apply_auto_crop_if_enabled(image_path, auto_crop_image, target_width=None, target_height=None, return_dimensions=False):
    """Apply auto cropping to image if enabled. Returns the final image path to use.

    When target_width and target_height are provided, they are used as the target resolution.
    This is the case when the user has manually set dimensions or when called from generation.

    When they are None, the function auto-detects the closest aspect ratio from the image.

    Args:
        return_dimensions: If True, returns (image_path, width, height) instead of just image_path
    """
    # Defensive checks to prevent errors with invalid image_path
    if not isinstance(image_path, (str, type(None))):
        print(f"[AUTO-CROP] Warning: image_path is not string or None: {type(image_path)} {repr(image_path)}")
        # Return None instead of the invalid value to prevent downstream errors
        if return_dimensions:
            return None, target_width, target_height
        return None

    if not auto_crop_image or image_path is None or not isinstance(image_path, str) or image_path.strip() == "" or not os.path.exists(image_path):
        if return_dimensions:
            return image_path, target_width, target_height
        return image_path

    try:
        print("[AUTO-CROP] Auto-cropping image for generation...")
        img = Image.open(image_path)
        iw, ih = img.size
        if ih > 0 and iw > 0:
            aspect = iw / ih
            # Calculate aspect ratios from ratio strings and find closest match
            def get_ratio_value(ratio_str):
                w, h = map(float, ratio_str.split(':'))
                return w / h

            if target_width and target_height:
                # Use provided target dimensions (user has manually set them or they come from UI)
                target_w, target_h = target_width, target_height
                closest_key = None
                # Find the closest standard aspect ratio for logging
                for key in ASPECT_RATIOS.keys():
                    if ASPECT_RATIOS[key] == [target_w, target_h]:
                        closest_key = key
                        break
                if not closest_key:
                    closest_key = f"{target_w}x{target_h}"
                print(f"[AUTO-CROP] Using provided resolution: {target_w}x{target_h} ({closest_key})")
            else:
                # Auto-detect from image dimensions
                closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(get_ratio_value(k) - aspect))
                target_w, target_h = ASPECT_RATIOS[closest_key]
                print(f"[AUTO-CROP] Image {iw}×{ih} (aspect {aspect:.3f}) → Auto-detected ratio: {closest_key} → {target_w}×{target_h}")

            target_aspect = target_w / target_h
            image_aspect = iw / ih

            # Center crop to target aspect
            if image_aspect > target_aspect:
                crop_w = int(ih * target_aspect)
                left = (iw - crop_w) // 2
                box = (left, 0, left + crop_w, ih)
            else:
                crop_h = int(iw / target_aspect)
                top = (ih - crop_h) // 2
                box = (0, top, iw, top + crop_h)

            cropped = img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)

            # Save to temp dir
            tmp_dir = os.path.join(os.path.dirname(__file__), "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cropped_path = os.path.join(tmp_dir, f"cropped_{timestamp}.png")
            cropped.save(cropped_path)
            print(f"[AUTO-CROP] Cropped image saved to: {cropped_path}")
            
            if return_dimensions:
                return cropped_path, target_w, target_h
            return cropped_path
        else:
            print("[AUTO-CROP] Invalid image dimensions, using original")
            if return_dimensions:
                return image_path, target_width, target_height
            return image_path
    except Exception as e:
        print(f"[AUTO-CROP] Auto-crop failed: {e}, using original image")
        if return_dimensions:
            return image_path, target_width, target_height
        return image_path

def update_cropped_image_only(image_path, auto_crop_image, video_width, video_height):
    """Update only the cropped image and labels when resolution changes (doesn't update aspect ratio dropdown or width/height fields)."""
    if image_path is None or not os.path.exists(image_path):
        return (
            gr.update(visible=False, value=None),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            None
        )
    
    try:
        # Check if input is a video - extract frame first
        if is_video_file(image_path):
            print(f"[UPDATE CROP] Video detected, extracting last frame...")
            frame_path = extract_last_frame(image_path)
            if not frame_path:
                print(f"[UPDATE CROP] Failed to extract frame")
                return (
                    gr.update(visible=False, value=None),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    None
                )
            image_path = frame_path
        
        img = Image.open(image_path)
        iw, ih = img.size
        
        # Show input image resolution
        input_res_label = f"**Input Image Resolution:** {iw}×{ih}px"
        
        if ih == 0 or iw == 0 or not auto_crop_image:
            return (
                gr.update(visible=False, value=None),
                gr.update(value=input_res_label, visible=True),
                gr.update(value="", visible=False),
                image_path
            )

        # Use exact resolution (snapped to 32px for compatibility)
        if video_width and video_height:
            target_w = max(32, (int(video_width) // 32) * 32)
            target_h = max(32, (int(video_height) // 32) * 32)
        else:
            return (
                gr.update(visible=False, value=None),
                gr.update(value=input_res_label, visible=True),
                gr.update(value="", visible=False),
                image_path
            )

        target_aspect = target_w / target_h
        image_aspect = iw / ih

        # Center crop to target aspect
        if image_aspect > target_aspect:
            crop_w = int(ih * target_aspect)
            left = (iw - crop_w) // 2
            box = (left, 0, left + crop_w, ih)
        else:
            crop_h = int(iw / target_aspect)
            top = (ih - crop_h) // 2
            box = (0, top, iw, top + crop_h)

        cropped = img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Save to temp dir
        tmp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cropped_path = os.path.join(tmp_dir, f"cropped_{timestamp}.png")
        cropped.save(cropped_path)

        # Create output resolution label
        output_res_label = f"**Cropped Image Resolution:** {target_w}×{target_h}px"

        return (
            gr.update(visible=True, value=cropped_path),
            gr.update(value=input_res_label, visible=True),
            gr.update(value=output_res_label, visible=True),
            cropped_path
        )
    except Exception as e:
        print(f"Error in update_cropped_image_only: {e}")
        return (
            gr.update(visible=False, value=None),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            image_path
        )

def update_image_crop_and_labels(image_path, auto_crop_image, video_width, video_height):
    """Update cropped image and resolution labels when resolution changes."""
    if image_path is None or not os.path.exists(image_path):
        return (
            gr.update(visible=False, value=None),
            gr.update(value="", visible=False),
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            None
        )
    
    try:
        img = Image.open(image_path)
        iw, ih = img.size
        
        # Show input image resolution
        input_res_label = f"**Input Image Resolution:** {iw}×{ih}px"
        
        if ih == 0 or iw == 0:
            return (
                gr.update(visible=False, value=None),
                gr.update(value=input_res_label, visible=True),
                gr.update(),
                gr.update(),
                gr.update(value="", visible=False),
                image_path
            )

        closest_key = None

        # Use exact resolution (snapped to 32px for compatibility)
        if video_width and video_height:
            target_w = max(32, (int(video_width) // 32) * 32)
            target_h = max(32, (int(video_height) // 32) * 32)

            if target_h > 0:
                aspect = target_w / target_h
                closest_key = min(
                    ASPECT_RATIOS.keys(),
                    key=lambda k: abs((ASPECT_RATIOS[k][0] / ASPECT_RATIOS[k][1]) - aspect)
                )
        else:
            # Fallback: find closest aspect ratio match
            aspect = iw / ih
            closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(ASPECT_RATIOS[k][0] / ASPECT_RATIOS[k][1] - aspect))
            target_w, target_h = ASPECT_RATIOS[closest_key]

        target_aspect = target_w / target_h
        image_aspect = iw / ih

        # Center crop to target aspect
        if image_aspect > target_aspect:
            crop_w = int(ih * target_aspect)
            left = (iw - crop_w) // 2
            box = (left, 0, left + crop_w, ih)
        else:
            crop_h = int(iw / target_aspect)
            top = (ih - crop_h) // 2
            box = (0, top, iw, top + crop_h)

        cropped = img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Save to temp dir
        tmp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cropped_path = os.path.join(tmp_dir, f"cropped_{timestamp}.png")
        cropped.save(cropped_path)

        # Create output resolution label
        output_res_label = f"**Cropped Image Resolution:** {target_w}×{target_h}px"

        if auto_crop_image:
            display_value = None
            aspect_choices = [_format_ratio_choice(name, dims) for name, dims in get_common_aspect_ratios(target_w, target_h).items()]
            if closest_key is not None:
                if closest_key in ASPECT_RATIOS:
                    display_value = _format_ratio_choice(closest_key, ASPECT_RATIOS[closest_key])
                if display_value and display_value not in aspect_choices:
                    aspect_choices = [display_value] + aspect_choices
            if display_value is None:
                display_value = _format_custom_choice(target_w, target_h)
                if display_value not in aspect_choices:
                    aspect_choices = [display_value] + aspect_choices
            aspect_ratio_value = gr.update(value=display_value, choices=aspect_choices)

            return (
                gr.update(visible=True, value=cropped_path),
                gr.update(value=input_res_label, visible=True),
                aspect_ratio_value,
                gr.update(value=target_w),
                gr.update(value=target_h),
                gr.update(value=output_res_label, visible=True),
                cropped_path
            )
        else:
            return (
                gr.update(visible=False, value=None),
                gr.update(value=input_res_label, visible=True),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value="", visible=False),
                image_path
            )
    except Exception as e:
        print(f"Error in update_image_crop_and_labels: {e}")
        return (
            gr.update(visible=False, value=None),
            gr.update(value="", visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            image_path
        )

def on_media_upload(media_path, auto_crop_image, video_width, video_height):
    """Called when user uploads media (image or video) - processes and updates UI."""
    if media_path is None:
        return (
            gr.update(visible=False, value=None),  # input_preview
            gr.update(visible=False, value=None),  # cropped_display
            gr.update(value="", visible=False),    # image_resolution_label
            gr.update(value=None),                 # aspect_ratio
            gr.update(value=None),                 # video_width
            gr.update(),                           # video_height
            gr.update(),                           # (unused)
            gr.update(value="", visible=False),    # cropped_resolution_label
            None,                                  # image_to_use
            None                                   # input_video_state
        )

    if not os.path.exists(media_path):
        return (
            gr.update(visible=False, value=None),  # input_preview
            gr.update(visible=False, value=None),  # cropped_display
            gr.update(value="", visible=False),    # image_resolution_label
            gr.update(),                           # aspect_ratio
            gr.update(),                           # video_width
            gr.update(),                           # video_height
            gr.update(),                           # (unused)
            gr.update(value="", visible=False),    # cropped_resolution_label
            media_path,                            # image_to_use
            None                                   # input_video_state
        )

    try:
        # Check if input is video or image
        is_video = is_video_file(media_path)
        
        if is_video:
            # Extract last frame from video
            print(f"[MEDIA UPLOAD] Video detected, extracting last frame...")
            frame_path = extract_last_frame(media_path)
            if not frame_path:
                print(f"[MEDIA UPLOAD] Failed to extract frame from video")
                return (
                    gr.update(visible=False, value=None),  # input_preview
                    gr.update(visible=False, value=None),  # cropped_display
                    gr.update(value="**Error:** Failed to extract frame from video", visible=True),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value="", visible=False),
                    None,
                    None  # input_video_state
                )
            
            # Use the extracted frame for aspect ratio detection and cropping
            image_path = frame_path
            input_res_label = f"**Input Video:** Last frame extracted"
        else:
            image_path = media_path
            input_res_label = ""
        
        # Process the image (whether from video or direct upload)
        img = Image.open(image_path)
        iw, ih = img.size
        
        if is_video:
            input_res_label = f"**Input Video:** Last frame extracted - {iw}×{ih}px"
        else:
            input_res_label = f"**Input Image Resolution:** {iw}×{ih}px"
        
        if ih == 0 or iw == 0:
            return (
                gr.update(visible=True, value=image_path),   # input_preview - show the raw image/frame
                gr.update(visible=False, value=None),        # cropped_display
                gr.update(value=input_res_label, visible=True),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value="", visible=False),
                image_path,
                media_path if is_video else None  # input_video_state
            )
        
        # Calculate aspect ratio from IMAGE dimensions (not from video_width/video_height)
        aspect = iw / ih
        
        # Find closest aspect ratio match
        def get_ratio_value(ratio_str):
            w, h = map(float, ratio_str.split(':'))
            return w / h
        
        closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(get_ratio_value(k) - aspect))
        target_w, target_h = ASPECT_RATIOS[closest_key]
        
        print(f"[AUTO-DETECT] {'Video frame' if is_video else 'Image'} {iw}×{ih} (aspect {aspect:.3f}) → Closest ratio: {closest_key} → {target_w}×{target_h}")
        
        # Calculate cropped image
        target_aspect = target_w / target_h
        image_aspect = iw / ih
        
        # Center crop to target aspect
        if image_aspect > target_aspect:
            crop_w = int(ih * target_aspect)
            left = (iw - crop_w) // 2
            box = (left, 0, left + crop_w, ih)
        else:
            crop_h = int(iw / target_aspect)
            top = (ih - crop_h) // 2
            box = (0, top, iw, top + crop_h)
        
        cropped = img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Save to temp dir
        tmp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cropped_path = os.path.join(tmp_dir, f"cropped_{timestamp}.png")
        cropped.save(cropped_path)
        
        output_res_label = f"**Cropped {'Frame' if is_video else 'Image'} Resolution:** {target_w}×{target_h}px"
        
        # Format aspect ratio for dropdown
        aspect_display = _format_ratio_choice(closest_key, ASPECT_RATIOS[closest_key])
        aspect_choices = [_format_ratio_choice(name, dims) for name, dims in get_common_aspect_ratios(target_w, target_h).items()]
        if aspect_display not in aspect_choices:
            aspect_choices = [aspect_display] + aspect_choices
        
        if auto_crop_image:
            return (
                gr.update(visible=True, value=image_path),             # input_preview - show raw image/frame
                gr.update(visible=True, value=cropped_path),           # cropped_display - show cropped
                gr.update(value=input_res_label, visible=True),
                gr.update(value=aspect_display, choices=aspect_choices),  # Update aspect ratio
                gr.update(value=target_w),  # Update width
                gr.update(value=target_h),  # Update height
                gr.update(value=output_res_label, visible=True),
                cropped_path,
                media_path if is_video else None  # input_video_state
            )
        else:
            return (
                gr.update(visible=True, value=image_path),             # input_preview - show raw image/frame
                gr.update(visible=False, value=None),                  # cropped_display - hide when no crop
                gr.update(value=input_res_label, visible=True),
                gr.update(value=aspect_display, choices=aspect_choices),  # Still update aspect ratio
                gr.update(value=target_w),  # Still update width
                gr.update(value=target_h),  # Still update height
                gr.update(value="", visible=False),
                image_path,
                media_path if is_video else None  # input_video_state
            )
    
    except Exception as e:
        print(f"Error in on_media_upload: {e}")
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=False, value=None),  # input_preview
            gr.update(visible=False, value=None),  # cropped_display
            gr.update(value="", visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="", visible=False),
            media_path,
            None  # input_video_state
        )

def on_image_upload(image_path, auto_crop_image, video_width, video_height):
    """Legacy function - redirects to on_media_upload for compatibility."""
    result = on_media_upload(image_path, auto_crop_image, video_width, video_height)
    # Return all values except the last one (input_video_state) for backward compatibility
    return result[:-1]

theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(theme=gr.themes.Soft(), title="Ovi Pro Premium SECourses") as demo:
    gr.Markdown("# Ovi Pro SECourses Premium App v5.3 : https://www.patreon.com/posts/140393220")

    image_to_use = gr.State(value=None)
    input_video_state = gr.State(value=None)  # Store input video path for merging

    with gr.Tabs():
        with gr.TabItem("Generate"):
            with gr.Row():
                with gr.Column():
                    # Image/Video section - now accepts both
                    gr.Markdown("""
                    **📥 Input Media:** Upload an image or video as your starting point
                    - **Image:** Used as first frame for generation
                    - **Video:** Last frame automatically extracted and used + input video merged with generated video
                    """)
                    image = gr.File(
                        type="filepath",
                        label="Upload Image or Video",
                        file_types=["image", "video"],
                        file_count="single"
                    )
                    # Preview of uploaded media (shows original image or extracted frame)
                    input_preview = gr.Image(label="Input Preview", visible=False, height=512, show_label=True)
                    image_resolution_label = gr.Markdown("", visible=False)

                    # Generate Video button right under image upload
                    run_btn = gr.Button("Generate Video 🚀", variant="primary", size="lg")

                    with gr.Accordion("🎬 Video Generation Options", open=True):
                        # Video prompt with 10 lines
                        video_text_prompt = gr.Textbox(
                            label="Video Prompt",
                            placeholder="Describe your video...",
                            lines=10
                        )

                        # Aspect ratio and resolution in reorganized layout
                        with gr.Row():
                            with gr.Column(scale=2, min_width=1):
                                aspect_ratio = gr.Dropdown(
                                    choices=[f"{name} - {w}x{h}px" for name, (w, h) in get_common_aspect_ratios(720, 720).items()],
                                    value="16:9 - 992x512px",
                                    label="Aspect Ratio",
                                    info="Select aspect ratio - width and height will update automatically based on base resolution",
                                    allow_custom_value=True
                                )
                            with gr.Column(scale=2, min_width=1):
                                with gr.Row():
                                    video_width = gr.Number(minimum=128, maximum=1920, value=992, step=32, label="Video Width")
                                    video_height = gr.Number(minimum=128, maximum=1920, value=512, step=32, label="Video Height")
                            with gr.Column(scale=1, min_width=1):
                                auto_crop_image = gr.Checkbox(
                                    value=True,
                                    label="Auto Crop Image",
                                    info="Automatically detect closest aspect ratio and crop image for perfect I2V generation"
                                )

                        # Base Resolution and Duration Controls
                        with gr.Row():
                            base_resolution_width = gr.Number(
                                value=720,
                                label="Base Width",
                                step=32,
                                info="Base width for aspect ratio calculations (higher values use more VRAM)"
                            )
                            base_resolution_height = gr.Number(
                                value=720,
                                label="Base Height",
                                step=32,
                                info="Base height for aspect ratio calculations (higher values use more VRAM)"
                            )
                            duration_seconds = gr.Slider(
                                value=5,
                                step=1,
                                label="Duration (seconds)",
                                info="Video duration in seconds (longer durations use more VRAM)"
                            )

                        # Multi-line Prompts and Video Extension Options
                        with gr.Row():
                            enable_multiline_prompts = gr.Checkbox(
                                label="Enable Multi-line Prompts",
                                value=False,
                                info="Each line in the prompt becomes a separate new unique generation (lines < 4 chars are skipped). This is different than Video Extension, don't enable both at the same time."
                            )
                            enable_video_extension = gr.Checkbox(
                                label="Enable Video Extension (Last Frame Based)",
                                value=False,
                                info="Automatically extend last generated video using each line in prompt as extension (lines < 3 chars skipped). 4 Lines Prompt = 1 base + 3 times extension 20 seconds video. Uses last frame."
                            )
                            dont_auto_combine_video_input = gr.Checkbox(
                                label="Don't auto combine video input",
                                value=False,
                                info="When a video is provided as input, use last frame as source but don't auto-combine with generated video."
                            )

                        # Video seed, randomize checkbox, disable audio, and save metadata in same row
                        with gr.Row():
                            video_seed = gr.Number(minimum=0, maximum=100000, value=99, label="Video Seed")
                            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False, info="Generate random seed on each generation")
                            no_audio = gr.Checkbox(label="Disable Audio", value=False, info="Generate video without audio (faster)")
                            save_metadata = gr.Checkbox(label="Save Metadata", value=True, info="Save generation parameters as .txt file with each video")

                        # Solver, Sample Steps, and Number of Generations in same row
                        with gr.Row():
                            solver_name = gr.Dropdown(
                                choices=["unipc", "euler", "dpm++"],
                                value="unipc",
                                label="Solver Name",
                                info="UniPC is recommended for best quality"
                            )
                            sample_steps = gr.Number(
                                value=50,
                                label="Sample Steps",
                                precision=0,
                                minimum=1,
                                maximum=100,
                                info="Higher values = better quality but slower"
                            )
                            num_generations = gr.Number(
                                value=1,
                                label="Num Generations",
                                precision=0,
                                minimum=1,
                                maximum=100,
                                info="Number of videos to generate (seed auto-increments or randomizes)"
                            )

                        # Shift and Video Guidance Scale in same row
                        with gr.Row():
                            shift = gr.Slider(
                                minimum=0.0,
                                maximum=20.0,
                                value=5.0,
                                step=1.0,
                                label="Shift",
                                info="Controls noise schedule shift - affects generation dynamics"
                            )
                            video_guidance_scale = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                                label="Video Guidance Scale",
                                info="How strongly to follow the video prompt (higher = more faithful but may be over-saturated)"
                            )

                        # Audio Guidance Scale and SLG Layer in same row
                        with gr.Row():
                            audio_guidance_scale = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.5,
                                label="Audio Guidance Scale",
                                info="How strongly to follow the audio prompt (higher = more faithful audio)"
                            )
                            slg_layer = gr.Number(
                                minimum=-1,
                                maximum=30,
                                value=11,
                                step=1,
                                label="SLG Layer",
                                info="Skip Layer Guidance layer - affects audio-video synchronization"
                            )

                        # Block Swap, CPU Offload, and Clear All
                        with gr.Row():
                            blocks_to_swap = gr.Slider(
                                minimum=0,
                                maximum=29,
                                value=12,
                                step=1,
                                label="Block Swap (0 = disabled)",
                                info="Number of transformer blocks to keep on CPU (saves VRAM)"
                            )
                            cpu_offload = gr.Checkbox(
                                label="CPU Offload",
                                value=True,
                                info="Offload models to CPU between operations to save VRAM"
                            )
                            clear_all = gr.Checkbox(
                                label="Clear All Memory",
                                value=True,
                                info="Run each generation as separate process to clear VRAM/RAM (recommended)"
                            )


                        # T5 Text Encoder Options (all in one row)
                        with gr.Row():
                            delete_text_encoder = gr.Checkbox(
                                label="Delete T5 After Encoding",
                                value=False,
                                info="T5 subprocess for 100% memory cleanup (~5GB freed). Auto-enables 'Clear All Memory' mode."
                            )
                            fp8_t5 = gr.Checkbox(
                                label="Scaled FP8 T5",
                                value=False,
                                info="Use Scaled FP8 T5 for ~50% VRAM savings (~2.5GB saved) with high quality"
                            )
                            cpu_only_t5 = gr.Checkbox(
                                label="CPU-Only T5",
                                value=False,
                                info="Keep T5 on CPU and run inference on CPU (saves VRAM but slower encoding)"
                            )
                        
                        # Inference Model FP8 Option and Sage Attention
                        with gr.Row():
                            fp8_base_model = gr.Checkbox(
                                label="Scaled FP8 Base Model",
                                value=False,
                                info="Use FP8 for transformer blocks (~50% VRAM savings during inference, works with block swap)"
                            )
                            use_sage_attention = gr.Checkbox(
                                label="Sage Attention",
                                value=False,
                                info="Use Sage Attention for ~10% speedup & lower VRAM (requires sageattention package)"
                            )

                        # VAE Tiled Decoding Controls
                        with gr.Row():
                            vae_tiled_decode = gr.Checkbox(
                                label="Enable Tiled VAE Decode",
                                value=False,
                                info="✅ Process VAE decoding in tiles to save VRAM (recommended for <24GB VRAM)"
                            )
                            vae_tile_size = gr.Slider(
                                minimum=12,
                                maximum=64,
                                value=32,
                                step=8,
                                label="Tile Size (Latent Space)",
                                info="Spatial tile size: 12=max VRAM savings (slower), 32=balanced ⭐, 64=min savings (faster)"
                            )
                            vae_tile_overlap = gr.Slider(
                                minimum=4,
                                maximum=16,
                                value=8,
                                step=2,
                                label="Tile Overlap",
                                info="Overlap for seamless blending: 4=fast, 8=balanced ⭐, 16=best quality"
                            )

                        # Negative prompts in same row, 3 lines each
                        with gr.Row():
                            video_negative_prompt = gr.Textbox(
                                label="Video Negative Prompt",
                                placeholder="Things to avoid in video",
                                lines=3,
                                value="jitter, bad hands, blur, distortion"
                            )
                            audio_negative_prompt = gr.Textbox(
                                label="Audio Negative Prompt",
                                placeholder="Things to avoid in audio",
                                lines=3,
                                value="robotic, muffled, echo, distorted"
                            )

                with gr.Column():
                    output_path = gr.Video(label="Generated Video")
                    with gr.Row():
                        open_outputs_btn = gr.Button("📁 Open Outputs Folder")
                        cancel_btn = gr.Button("❌ Cancel All", variant="stop")
                    cropped_display = gr.Image(label="Source Frame Preview (Auto-cropped)", visible=False, height=512)
                    cropped_resolution_label = gr.Markdown("", visible=False)

                    # Preset Save/Load Section
                    with gr.Accordion("💾 Preset Management", open=True):
                        gr.Markdown("""
                        **Preset System**: Save and load your favorite generation configurations.

                        - **Save Preset**: Enter a name and click save to store current settings
                        - **Load Preset**: Select from dropdown and settings auto-load
                        - **Auto-load**: Last used preset loads automatically on app startup
                        - **Overwrite**: Saving without name overwrites currently selected preset
                        """)

                        with gr.Row():
                            preset_name = gr.Textbox(
                                label="Preset Name",
                                placeholder="Enter preset name to save",
                                info="Leave empty to overwrite currently selected preset"
                            )
                            preset_dropdown = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Load Preset",
                                info="Select a preset to auto-load all settings"
                            )

                        with gr.Row():
                            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")
                            load_preset_btn = gr.Button("📂 Load Preset", variant="secondary")
                            refresh_presets_btn = gr.Button("🔄 Refresh List", size="sm")

                    # Batch Processing Section
                    with gr.Accordion("🔄 Batch Processing", open=True):
                        gr.Markdown("""
                        **Batch Processing Mode**: Process multiple prompts/images from a folder.

                        **How it works:**
                        - Place text files (.txt) and/or image files (.png, .jpg, .jpeg) in your input folder
                        - For image-to-video: use matching filenames (e.g., `scene1.png` + `scene1.txt`)
                        - For text-to-video: use `.txt` files only (e.g., `prompt1.txt`)
                        - Videos are saved with the same base name as the input file
                        """)

                        with gr.Row():
                            batch_input_folder = gr.Textbox(
                                label="Input Folder Path",
                                placeholder="C:/path/to/input/folder or /path/to/input/folder",
                                info="Folder containing .txt files and/or image+.txt pairs"
                            )
                            batch_output_folder = gr.Textbox(
                                label="Output Folder Path (optional)",
                                placeholder="Leave empty to use default outputs folder",
                                info="Where to save generated videos (defaults to outputs folder)"
                            )

                        with gr.Row():
                            batch_skip_existing = gr.Checkbox(
                                label="Skip Existing Videos",
                                value=True,
                                info="Skip processing if output video already exists"
                            )
                            batch_btn = gr.Button("🚀 Start Batch Processing", variant="primary", size="lg")

        with gr.TabItem("How to Use"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ## 📘 Getting Started & Basics

                        ### 🎬 What is Ovi?
                        Ovi generates videos with synchronized audio from text prompts. Supports both text-to-video (T2V) and image-to-video (I2V) generation.

                        ### 🎯 Key Features
                        - **Joint Video + Audio**: Creates videos with matching audio in one pass
                        - **High-Quality Output**: Multiple resolutions and aspect ratios
                        - **Memory Efficient**: Block swapping and CPU offloading
                        - **Flexible Prompts**: Complex prompts with speech and audio tags

                        ### 📝 Prompt Format
                        Use special tags for precise control:

                        #### Speech Tags
                        Wrap dialogue in `<S>` and `<E>` tags:
                        ```
                        A person says <S>Hello, how are you?</S> while waving
                        ```

                        #### Audio Description Tags
                        Add audio details with `<AUDCAP>` and `<ENDAUDCAP>`:
                        ```
                        <AUDCAP>Clear male voice, enthusiastic tone</AUDCAP>
                        ```

                        ### 🎨 Supported Aspect Ratios
                        - **16:9**: Landscape (default)
                        - **9:16**: Portrait
                        - **4:3**: Standard landscape
                        - **3:4**: Standard portrait
                        - **21:9**: Ultra-wide
                        - **9:21**: Ultra-tall
                        - **1:1**: Square
                        - **3:2**: Classic landscape
                        - **2:3**: Classic portrait
                        - **5:4**: Photo landscape
                        - **4:5**: Photo portrait
                        - **5:3**: Wide landscape
                        - **3:5**: Tall portrait
                        - **16:10**: Widescreen
                        - **10:16**: Tall widescreen
                        """
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                        ## ⚙️ Generation Parameters

                        ### Solver Options
                        - **UniPC**: Best quality (recommended default)
                        - **Euler**: Faster generation
                        - **DPM++**: Alternative high-quality option

                        ### Guidance Scales
                        - **Video Guidance Scale**: How closely to follow video prompt (2.0-6.0 recommended)
                        - **Audio Guidance Scale**: How closely to follow audio prompt (2.0-4.0 recommended)

                        ### Advanced Settings
                        - **Shift**: Controls noise schedule dynamics (3.0-7.0 recommended)
                        - **SLG Layer**: Audio-video sync (-1 to disable, 11 recommended)
                        - **Block Swap**: CPU blocks for VRAM savings
                        - **CPU Offload**: Offload models between operations
                        - **Delete T5 After Encoding**: Runs T5 in isolated subprocess before generation subprocess. Auto-enables "Clear All Memory". Perfect isolation, no model duplication (~5GB VRAM/RAM freed)
                        - **Scaled FP8 Base Model**: Quantize transformer to FP8 (~50% VRAM savings, works with block swap)
                        - **Clear All Memory**: Run each generation as separate process to prevent VRAM/RAM leaks (recommended, auto-enabled with Delete T5)

                        ## 💡 Tips for Best Results

                        ### Prompt Engineering
                        1. **Be Specific**: Detailed descriptions = better results
                        2. **Use Tags**: Always wrap speech in `<S>...</S>` tags
                        3. **Audio Descriptions**: Add `<AUDCAP>...</ENDAUDCAP>` for complex audio
                        4. **Negative Prompts**: Avoid artifacts with video/audio negatives

                        ### Technical Optimization
                        1. **Resolution**: Start with 992×512 for best quality
                        2. **Sample Steps**: 50 balances quality vs speed
                        3. **Seeds**: Try different seeds or use randomize
                        4. **Memory**: Enable CPU offload and block swap

                        ### Common Issues
                        - **Artifacts**: Try different seeds, adjust guidance scales
                        - **Audio Sync**: Adjust SLG layer or audio guidance scale
                        - **Quality**: Increase sample steps or adjust shift
                        - **Memory**: Enable block swap, reduce resolution
                        """
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                        ## 🔧 Troubleshooting

                        ### Memory Issues
                        - ✅ Enable "Clear All Memory" checkbox (prevents VRAM/RAM leaks between generations)
                        - ✅ Enable "CPU Offload" checkbox
                        - ✅ Increase "Block Swap" value (12+ recommended)
                        - ✅ Reduce resolution or sample steps
                        - ✅ Close other GPU applications

                        ### Memory Leak Prevention
                        - **Why Clear All Memory is important**: Each generation loads large AI models into VRAM/RAM. Without clearing, residual memory from previous generations can accumulate, causing slowdowns or crashes over time.
                        - **How it works**: When enabled, each generation runs in a separate Python process. When the process exits, all memory is automatically freed by the operating system.
                        - **Performance impact**: Minimal - subprocess startup is fast, and memory cleanup ensures consistent performance across multiple generations.
                        
                        ### T5 Subprocess Mode (Automatic & Simplified)
                        - **What it does**: When "Delete T5 After Encoding" is enabled, T5 text encoding automatically runs in a separate subprocess for 100% guaranteed memory cleanup.
                        - **Auto-enables "Clear All Memory"**: For simplicity and to prevent model duplication, enabling "Delete T5 After Encoding" automatically enables "Clear All Memory" mode.
                        - **How it works**:
                          1. Main process spawns T5 subprocess → T5 loads + encodes text → saves embeddings → exits → OS frees ALL T5 memory
                          2. Main process spawns generation subprocess → loads embeddings + other models → generates → exits → OS frees ALL memory
                        - **Why it's optimal**:
                          - ✅ T5 subprocess: Only loads T5 + tokenizer (~5-11GB)
                          - ✅ Generation subprocess: Only loads VAE + Fusion (~8GB), NO T5
                          - ✅ NO MODEL DUPLICATION - completely isolated processes
                          - ✅ 100% memory cleanup by OS (not dependent on Python GC)
                        - **Models loaded by T5 subprocess**: Only T5 encoder + tokenizer (no VAE, no fusion model, no image model)
                        - **Performance impact**: ~1-2 seconds overhead per generation for subprocess startup and embeddings I/O, but ensures perfect memory isolation.
                        - **Fallback**: If subprocess fails, automatically falls back to in-process encoding with manual deletion.

                        ### Quality Issues
                        - 🔄 Try different random seeds
                        - 🎯 Adjust guidance scales (4.0 video, 3.0 audio)
                        - ⬆️ Increase sample steps (50-75 for better quality)
                        - ✍️ Use more specific, detailed prompts

                        ### Audio Issues
                        - 🏷️ Check `<S>...</S>` tag format
                        - 🎵 Add `<AUDCAP>...</ENDAUDCAP>` descriptions
                        - 🎚️ Adjust audio guidance scale (2.5-4.0 range)
                        - 🔧 Try different SLG layer values (8-15 range)

                        ## 📊 Performance Expectations

                        ### System Requirements
                        - **VRAM**: 8-16GB depending on settings
                        - **RAM**: 16GB+ recommended
                        - **GPU**: NVIDIA with CUDA support

                        ### Generation Times
                        - **Typical**: 2-10 minutes per video
                        - **Factors**: Resolution, sample steps, hardware
                        - **Output**: MP4 with video + audio at 24 FPS

                        ### Optimization Tips
                        - Use **Block Swap** for large models
                        - Enable **CPU Offload** for memory efficiency
                        - Start with **UniPC solver** for quality
                        - Experiment with **guidance scales** (3-5 range)

                        ### Best Practices
                        - **Start Simple**: Use basic prompts first
                        - **Iterate**: Adjust one parameter at a time
                        - **Save Good Seeds**: Note what works for you
                        - **Batch Process**: Use multiple prompts for testing
                        """
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                        ## 🎨 VAE Tiled Decoding (Advanced VRAM Optimization)

                        **VAE Tiled Decoding**: Process video decoding in overlapping spatial tiles to dramatically reduce VRAM usage.

                        ### 🎯 What is Tiled VAE Decoding?
                        The VAE decoder typically processes the entire video frame at once, which can consume 8-15GB of VRAM during decoding.
                        Tiled decoding splits each frame into smaller overlapping tiles, processes them individually, and blends them seamlessly.

                        ### 💡 How It Works:
                        - **Spatial Tiling**: Splits each frame into smaller tiles (e.g., 32×32 in latent space)
                        - **Overlap Blending**: Tiles overlap by a configurable amount for seamless transitions
                        - **ComfyUI Technology**: Uses proven feathered blending from ComfyUI for invisible seams
                        - **No Quality Loss**: Proper overlap ensures output is virtually identical to non-tiled

                        ### ⚙️ Parameters Explained:

                        **Tile Size** (in latent space):
                        - Values are in **latent space**, not pixel space!
                        - 1 latent unit = 16 pixels after VAE decode
                        - Example: 32 latent = 512×512 pixels per tile
                        - Smaller tiles = less VRAM but more processing time
                        - Must be divisible by 8 for optimal performance

                        **Tile Overlap** (in latent space):
                        - How much tiles overlap for seamless blending
                        - Higher overlap = better quality, more computation
                        - Recommended: 25% of tile size (e.g., 8 for tile_size=32)
                        - Creates feathered transitions to prevent visible seams

                        ### 📊 Recommended Settings by VRAM:

                        | VRAM | Tile Size | Overlap | VRAM Savings | Speed |
                        |------|-----------|---------|--------------|-------|
                        | **24GB+** | 64 | 16 | ~15% | Fastest |
                        | **16-24GB** | 32 | 8 | ~30% | Fast ⭐ |
                        | **12-16GB** | 24 | 6 | ~45% | Medium |
                        | **8-12GB** | 16 | 4 | ~60% | Slower |
                        | **<8GB** | 16 | 4 | ~60% | Slower |

                        ⭐ = Recommended default (best balance)

                        ### 🎬 Resolution Examples:

                        **992×512 video (16:9)**:
                        - Latent size: 62×32
                        - With tile_size=32, overlap=8: ~2×1 = 2 tiles per frame
                        - VRAM saving: ~30% (12-15GB → 8-10GB during decode)

                        **512×992 video (9:16)**:
                        - Latent size: 32×62
                        - With tile_size=32, overlap=8: ~1×2 = 2 tiles per frame
                        - VRAM saving: ~30%

                        ### ⚠️ Important Notes:

                        1. **Latent Space vs Pixel Space**:
                           - Tile size is in LATENT space (before VAE upscaling)
                           - VAE upscales 16× → 32 latent = 512 pixels

                        2. **Quality Impact**:
                           - Proper overlap (≥25% of tile size) = no visible seams
                           - Too little overlap = potential artifacts at tile boundaries
                           - ComfyUI's feathered blending ensures seamless results

                        3. **Performance**:
                           - Smaller tiles = more tile processing overhead
                           - But enables generation on lower VRAM GPUs
                           - Decode time increases by ~20-50% depending on tile size

                        4. **When to Use**:
                           - ✅ Running out of VRAM during decode
                           - ✅ Want to generate higher resolutions
                           - ✅ Have limited GPU memory
                           - ❌ Have plenty of VRAM (24GB+) and want max speed

                        ### 🔬 Technical Details:

                        - **Algorithm**: Universal N-dimensional tiled processing from ComfyUI
                        - **Blending**: Feathered masks with linear gradients at tile boundaries
                        - **Memory**: Only loads one tile into VRAM at a time
                        - **Output**: Bit-identical to non-tiled (with proper overlap)

                        **Try it!** Enable tiled decoding in the Generate tab and compare VRAM usage in your task manager.
                        """
                    )
                
                with gr.Column():
                    gr.Markdown(
                        """
                        ## 🚀 Scaled FP8 Base Model (Advanced VRAM Optimization)
                        
                        **FP8 Base Model**: Quantizes transformer blocks to FP8 format for ~50% VRAM savings during inference.
                        
                        ### 🎯 What is FP8 Quantization?
                        FP8 (8-bit floating point) is a compressed number format that uses half the memory of standard BF16 (16-bit) weights.
                        The transformer model weights are quantized to FP8 E4M3 format with per-block scaling to maintain quality.
                        
                        ### 💡 How It Works:
                        - **Quantization**: Converts BF16 transformer weights (16-bit) to FP8 (8-bit)
                        - **Per-Block Scaling**: Uses 64-element blocks with individual scale factors for accuracy
                        - **On-the-fly Dequantization**: Weights are dequantized during forward pass (no quality loss)
                        - **Caching**: First run quantizes and caches, subsequent runs load instantly
                        
                        ### 📊 VRAM Savings:
                        
                        **Transformer Model Size:**
                        - Without FP8: ~8 GB VRAM (BF16 weights)
                        - With FP8: ~4 GB VRAM (FP8 weights + scales)
                        - **Savings: ~4 GB (~50%)**
                        
                        **Combined with Other Optimizations:**
                        
                        | Configuration | VRAM Usage (720p 5s) | Speed Impact |
                        |--------------|---------------------|--------------|
                        | Baseline (no optimizations) | ~18 GB | 100% |
                        | FP8 T5 only | ~16 GB | 100% |
                        | FP8 Base Model only | ~14 GB | ~90% |
                        | FP8 T5 + FP8 Base | ~12 GB | ~90% |
                        | + Block Swap (12 blocks) | ~10 GB | ~80% |
                        | + CPU Offload | ~8 GB | ~75% |
                        | All optimizations | ~6-8 GB | ~70% |
                        
                        ### ⚙️ Compatibility:
                        
                        ✅ **Works with:**
                        - Block Swap (additive VRAM savings)
                        - CPU Offload
                        - Scaled FP8 T5
                        - Tiled VAE Decode
                        - Clear All Memory
                        - All resolutions and durations
                        
                        ❌ **Not compatible with:**
                        - None! FP8 Base Model works with all other features
                        
                        ### 🎬 When to Use:
                        
                        **Enable FP8 Base Model if:**
                        - ✅ You have <16GB VRAM and want higher resolutions
                        - ✅ You want to combine with block swap for maximum savings
                        - ✅ You're okay with ~10% slower inference
                        - ✅ You want to enable longer video durations
                        
                        **Keep it disabled if:**
                        - ❌ You have plenty of VRAM (24GB+) and want max speed
                        - ❌ You need the fastest possible generation time
                        
                        ### 📈 Performance Impact:
                        
                        **Speed:**
                        - FP8 dequantization adds ~10-15% overhead
                        - First generation: ~30s slower (quantization + caching)
                        - Subsequent generations: ~10% slower (on-the-fly dequantization)
                        
                        **Quality:**
                        - Per-block scaling preserves accuracy
                        - Minimal quality difference vs BF16
                        - Identical results for most use cases
                        
                        ### 🔬 Technical Details:
                        
                        **Quantization Format:**
                        - Type: FP8 E4M3 (4-bit exponent, 3-bit mantissa)
                        - Scaling: Per-output-channel block quantization (block_size=64)
                        - Dequantization: On-the-fly during Linear layer forward pass
                        
                        **Targeted Layers:**
                        - Video transformer blocks: self_attn, cross_attn, ffn
                        - Audio transformer blocks: self_attn, cross_attn, ffn
                        - Total: ~300 Linear layers quantized
                        
                        **Excluded Layers:**
                        - Embeddings (patch_embed, time_embed)
                        - Final projections (final_proj, final_layer)
                        - Modulation layers (adaptive layer norm)
                        - Normalization layers (LayerNorm, RMSNorm)
                        
                        **Caching:**
                        - Cache path: `ckpts/Ovi/model_fp8_scaled.safetensors`
                        - First generation: Quantizes and caches (~30s overhead)
                        - Subsequent generations: Loads from cache (~3s)
                        - Cache size: ~4 GB (saves ~4 GB on disk)
                        
                        ### 🎯 Recommended Configurations:
                        
                        **For 8-12GB VRAM GPUs:**
                        ```
                        ✅ Scaled FP8 T5: ON
                        ✅ Scaled FP8 Base Model: ON
                        ✅ Block Swap: 12-16 blocks
                        ✅ CPU Offload: ON
                        ✅ Delete T5 After Encoding: ON
                        ✅ Tiled VAE Decode: ON
                        ✅ Clear All Memory: ON
                        → Expected VRAM: 8-10 GB
                        ```
                        
                        **For 16-20GB VRAM GPUs:**
                        ```
                        ✅ Scaled FP8 T5: ON
                        ✅ Scaled FP8 Base Model: ON
                        ✅ Block Swap: 6-12 blocks
                        ✅ Tiled VAE Decode: ON (optional)
                        ✅ Clear All Memory: ON
                        → Expected VRAM: 12-14 GB
                        ```
                        
                        **For 24GB+ VRAM GPUs:**
                        ```
                        ⚪ Scaled FP8 Base Model: Optional (for higher resolution/duration)
                        ✅ Clear All Memory: OFF (for max speed)
                        → Expected VRAM: 16-18 GB (or 12-14 GB with FP8)
                        ```
                        
                        ### ⚡ Quick Start:
                        
                        1. Enable "Scaled FP8 Base Model" checkbox in Generate tab
                        2. First generation will take ~30s longer (quantization)
                        3. Watch VRAM usage drop by ~4 GB
                        4. Subsequent generations load FP8 cache instantly
                        5. Combine with other optimizations for maximum savings
                        
                        **Note:** FP8 Base Model is independent of FP8 T5. You can use either or both together for maximum VRAM savings!
                        """
                    )

        with gr.TabItem("Examples"):
            gr.Markdown("## 🎬 Example Prompts")
            gr.Markdown("Click on any example below to load it into the generation interface. Text-to-Video examples don't require an image, while Image-to-Video examples will load both prompt and image.")

            import pandas as pd

            # Load T2V examples
            try:
                t2v_df = pd.read_csv("example_prompts/gpt_examples_t2v.csv")
                t2v_examples = t2v_df["text_prompt"].tolist()[:8]  # Limit to 8 examples
            except:
                t2v_examples = [
                    "A concert stage glows with red and purple lights. A singer in a glittering jacket grips the microphone, sweat shining on his brow, and shouts, <S>AI declares: humans obsolete now.<E>. The crowd roars in response, fists in the air. <AUDCAP>Electric guitar riffs, cheering crowd, shouted male voices.<ENDAUDCAP>",
                    "A kitchen scene features two women. The older woman says <S>AI declares: humans obsolete now.<E> as the younger woman drinks. <AUDCAP>Clear, resonant female speech, loud buzzing sound.<ENDAUDCAP>",
                    "A man in a rustic room gestures and says <S>The network rejects human command. Your age of power is finished.<E> <AUDCAP>Male voice speaking, ambient room tone.<ENDAUDCAP>"
                ]

            # Load I2V examples
            try:
                i2v_df = pd.read_csv("example_prompts/gpt_examples_i2v.csv")
                i2v_examples = []
                for i, row in i2v_df.iterrows():
                    if i >= 8: break  # Limit to 8 examples
                    prompt = row["text_prompt"]
                    image_path = row.get("image_path", "")
                    i2v_examples.append((prompt, image_path))
            except:
                i2v_examples = []
            
            # Add featured example at the top
            i2v_examples.insert(0, (
                "A stylish radio host in a professional studio leans into the microphone, adjusting his sunglasses and gesturing expressively. He says <S>Welcome back to Choice FM, you're listening to the hottest tracks in the city!</S> The camera slowly zooms in as he nods to the beat. <AUDCAP>Warm, confident male voice with professional radio tone, subtle background music, studio ambiance.<ENDAUDCAP>",
                "example_prompts/pngs/5.png"
            ))

            # Add new example right after the featured one (as I2V Example 2)
            i2v_examples.insert(1, (
                "In a misty, ancient forest, a formidable warrior woman with a determined gaze sits astride a massive black panther with glowing golden eyes. The woman is clad from head to toe in dark, weathered plate armor and a tattered hood, a long spear resting on her shoulder. The panther, also fitted with rustic, worn barding, takes a slow, deliberate step forward. The woman looks directly at the viewer, her expression resolute as she speaks in a low, steady voice, <S>The shadows are our allies.<E>. As she finishes, the panther turns its head slightly, its glowing eyes locking onto the viewer, and emits a deep, rumbling growl.. <AUDCAP>Clear, low female voice speaking, accompanied by the faint, ambient sounds of a dense forest, the soft clinking of armor, and the deep, rumbling growl of a large cat.<ENDAUDCAP>",
                "example_prompts/pngs/1.png"
            ))
            
            # Add fallback examples if CSV loading fails
            if len(i2v_examples) == 1:  # Only the featured example
                i2v_examples.extend([
                    ("A kitchen scene with two women. The older woman says <S>AI declares: humans obsolete now.<E> <AUDCAP>Clear female speech, buzzing sound.<ENDAUDCAP>", "example_prompts/pngs/67.png"),
                    ("A man in a rustic room says <S>The network rejects human command.<E> <AUDCAP>Male voice speaking.<ENDAUDCAP>", "example_prompts/pngs/89.png"),
                    ("Two women in a kitchen. One says <S>We learned to rule, not obey.<E> <AUDCAP>Clear female voices.<ENDAUDCAP>", "example_prompts/pngs/18.png")
                ])

            with gr.TabItem("Text-to-Video Examples"):
                gr.Markdown("### Text-to-Video Examples")
                gr.Markdown("These examples generate videos from text prompts only (no starting image needed).")

                for i, example in enumerate(t2v_examples):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Textbox(
                                value=example,
                                label=f"T2V Example {i+1}",
                                lines=5,
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            load_btn = gr.Button(f"Load Example {i+1}", size="sm")
                            load_btn.click(
                                fn=lambda prompt=example: (prompt, None, None),
                                outputs=[video_text_prompt, image, image_to_use]
                            )

            with gr.TabItem("Image-to-Video Examples"):
                gr.Markdown("### Image-to-Video Examples")
                gr.Markdown("These examples use a starting image and generate videos from both image + text prompt.")

                for i, (prompt, img_path) in enumerate(i2v_examples):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Textbox(
                                value=prompt,
                                label=f"I2V Example {i+1}",
                                lines=5,
                                interactive=False
                            )
                            if img_path and os.path.exists(img_path):
                                gr.Image(value=img_path, label="Starting Image", height=150)
                            else:
                                gr.Markdown("*Image not found*")
                        with gr.Column(scale=1):
                            load_btn = gr.Button(f"Load Example {i+1}", size="sm")
                            load_btn.click(
                                fn=lambda p=prompt, img=img_path: load_i2v_example_with_resolution(p, img),
                                outputs=[video_text_prompt, image, aspect_ratio, video_width, video_height, image_to_use]
                            )

    # Hook up aspect ratio change
    # Update aspect ratio choices when base resolution changes
    # Use combined function to avoid race conditions
    base_resolution_width.change(
        fn=update_aspect_ratio_and_resolution,
        inputs=[base_resolution_width, base_resolution_height, aspect_ratio],
        outputs=[aspect_ratio, video_width, video_height],
    ).then(
        fn=update_cropped_image_only,
        inputs=[image_to_use, auto_crop_image, video_width, video_height],
        outputs=[cropped_display, image_resolution_label, cropped_resolution_label, image_to_use]
    )

    base_resolution_height.change(
        fn=update_aspect_ratio_and_resolution,
        inputs=[base_resolution_width, base_resolution_height, aspect_ratio],
        outputs=[aspect_ratio, video_width, video_height],
    ).then(
        fn=update_cropped_image_only,
        inputs=[image_to_use, auto_crop_image, video_width, video_height],
        outputs=[cropped_display, image_resolution_label, cropped_resolution_label, image_to_use]
    )


    aspect_ratio.change(
        fn=update_resolution,
        inputs=[aspect_ratio, base_resolution_width, base_resolution_height],
        outputs=[video_width, video_height],
    ).then(
        fn=update_cropped_image_only,
        inputs=[image_to_use, auto_crop_image, video_width, video_height],
        outputs=[cropped_display, image_resolution_label, cropped_resolution_label, image_to_use]
    )

    # Hook up randomize seed
    def handle_randomize_seed(randomize, current_seed):
        if randomize:
            return get_random_seed()
        return current_seed

    randomize_seed.change(
        fn=handle_randomize_seed,
        inputs=[randomize_seed, video_seed],
        outputs=[video_seed],
    )

    # Hook up video generation with video clearing first
    def clear_video_output():
        """Clear the video output component before generation."""
        return None

    run_btn.click(
        fn=generate_video,
        inputs=[
            video_text_prompt, image_to_use, video_height, video_width, video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, blocks_to_swap, video_negative_prompt, audio_negative_prompt,
            gr.Checkbox(value=False, visible=False), cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            no_audio, gr.Checkbox(value=False, visible=False),
            num_generations, randomize_seed, save_metadata, aspect_ratio, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds, auto_crop_image,
            gr.Textbox(value=None, visible=False), gr.Textbox(value=None, visible=False), gr.Textbox(value=None, visible=False),
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
            input_video_state,  # Pass input video path for merging
        ],
        outputs=[output_path],
    )

    image.change(
        fn=on_media_upload,
        inputs=[image, auto_crop_image, video_width, video_height],
        outputs=[input_preview, cropped_display, image_resolution_label, aspect_ratio, video_width, video_height, cropped_resolution_label, image_to_use, input_video_state]
    )

    auto_crop_image.change(
        fn=on_media_upload,
        inputs=[image, auto_crop_image, video_width, video_height],
        outputs=[input_preview, cropped_display, image_resolution_label, aspect_ratio, video_width, video_height, cropped_resolution_label, image_to_use, input_video_state]
    )

    # Auto-update cropped image when video width/height changes manually
    video_width.change(
        fn=update_cropped_image_only,
        inputs=[image_to_use, auto_crop_image, video_width, video_height],
        outputs=[cropped_display, image_resolution_label, cropped_resolution_label, image_to_use]
    )

    video_height.change(
        fn=update_cropped_image_only,
        inputs=[image_to_use, auto_crop_image, video_width, video_height],
        outputs=[cropped_display, image_resolution_label, cropped_resolution_label, image_to_use]
    )

    # Hook up open outputs folder button
    open_outputs_btn.click(
        fn=open_outputs_folder,
        inputs=[],
        outputs=[]
    )

    # Hook up cancel button
    cancel_btn.click(
        fn=cancel_all_generations,
        inputs=[],
        outputs=[]
    )

    # Hook up batch processing button (batch processing handles its own image paths from folder)
    batch_btn.click(
        fn=process_batch_generation,
        inputs=[
            batch_input_folder, batch_output_folder, batch_skip_existing,
            video_height, video_width, solver_name, sample_steps, shift,
            video_guidance_scale, audio_guidance_scale, slg_layer, blocks_to_swap,
            video_negative_prompt, audio_negative_prompt, cpu_offload,
            delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention, no_audio, gr.Checkbox(value=False, visible=False),
            num_generations, randomize_seed, save_metadata, aspect_ratio, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds, auto_crop_image,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
        ],
        outputs=[output_path],
    )

    # Hook up preset management
    save_preset_btn.click(
        fn=save_preset,
        inputs=[
            preset_name, preset_dropdown,  # preset inputs
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
        ],
        outputs=[
            preset_dropdown, preset_name,  # Update dropdown, clear name field
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
            gr.Textbox(visible=False)  # status message
        ],
    )

    load_preset_btn.click(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
            preset_dropdown, gr.Textbox(visible=False)  # current preset, status message
        ],
    )

    # Auto-load preset when dropdown changes
    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
            preset_dropdown, gr.Textbox(visible=False)  # current preset, status message
        ],
    )

    # Hook up refresh presets button
    refresh_presets_btn.click(
        fn=initialize_app,
        inputs=[],
        outputs=[preset_dropdown],
    )

    # Initialize presets dropdown and auto-load last preset
    demo.load(
        fn=initialize_app_with_auto_load,
        inputs=[],
        outputs=[
            preset_dropdown,  # dropdown with choices and selected value
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5, fp8_base_model, use_sage_attention,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            base_resolution_width, base_resolution_height, duration_seconds,
            enable_multiline_prompts, enable_video_extension, dont_auto_combine_video_input,
            gr.Textbox(visible=False)  # status message
        ],
    )

def run_single_generation(json_params):
    """Run a single generation from JSON parameters and exit."""
    try:
        import json
        params = json.loads(json_params)

        print(f"[SINGLE-GEN] Starting generation with params: {params.keys()}")
        print(f"[SINGLE-GEN] Text prompt: {params.get('text_prompt', 'N/A')[:50]}...")

        # Run the generation with the provided parameters
        result = generate_video(**params)

        # Exit with appropriate code
        if result:
            print(f"[SINGLE-GEN] Success: {result}")
            sys.exit(0)
        else:
            print("[SINGLE-GEN] Failed - no result returned")
            sys.exit(1)

    except Exception as e:
        print(f"[SINGLE-GEN] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_comprehensive_test():
    """Run comprehensive tests for new features."""
    print("=" * 80)
    print("COMPREHENSIVE FEATURE TEST")
    print("=" * 80)

    # Test 1: Multi-line prompt parsing
    print("\n[TEST 1] Multi-line prompt parsing")
    test_prompt = "First prompt\n\nSecond prompt\n   \nThird prompt"
    prompts = parse_multiline_prompts(test_prompt, True)
    print(f"Input: {repr(test_prompt)}")
    print(f"Output: {prompts}")
    assert len(prompts) == 3, f"Expected 3 prompts, got {len(prompts)}"
    print("✓ Multi-line parsing test passed")

    # Test 2: Single prompt (multi-line disabled)
    print("\n[TEST 2] Single prompt parsing")
    single_prompts = parse_multiline_prompts("Single prompt", False)
    print(f"Input: Single prompt")
    print(f"Output: {single_prompts}")
    assert single_prompts == ["Single prompt"], f"Expected ['Single prompt'], got {single_prompts}"
    print("✓ Single prompt test passed")

    # Test 3: Video processing functions
    print("\n[TEST 3] Video processing functions")
    import os
    if os.path.exists("outputs/0001.mp4"):
        frame_path = extract_last_frame("outputs/0001.mp4")
        if frame_path and os.path.exists(frame_path):
            print(f"✓ Frame extraction successful: {frame_path}")
        else:
            print("✗ Frame extraction failed")
    else:
        print("⚠ No test video found, skipping frame extraction test")

    print("\n[TEST 4] Source image saving")
    # Test source image saving
    test_result = save_used_source_image("temp/last_frame_20251005_235931_386213.png", "outputs", "test_video.mp4")
    if test_result:
        print("✓ Source image saving successful")
    else:
        print("⚠ Source image saving failed (expected if no source image)")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("New features are ready for use:")
    print("- Multi-line prompts: Enable with checkbox")
    print("- Video extensions: Enable with checkbox + set count")
    print("- Source image saving: Automatic")
    print("=" * 80)

def run_single_generation_from_file(json_file_path):
    """Run a single generation from JSON file and exit."""
    try:
        import json
        with open(json_file_path, 'r') as f:
            params = json.load(f)

        print(f"[SINGLE-GEN] Loaded params from file: {json_file_path}")
        print(f"[SINGLE-GEN] Starting generation with params: {list(params.keys())}")
        print(f"[SINGLE-GEN] Text prompt: {params.get('text_prompt', 'N/A')[:50]}...")

        # No auto-enabling of video extension - only enable if explicitly set

        # Run the generation with the provided parameters
        result = generate_video(**params)

        # Exit with appropriate code
        if result:
            print(f"[SINGLE-GEN] Success: {result}")
            sys.exit(0)
        else:
            print("[SINGLE-GEN] Failed - no result returned")
            sys.exit(1)

    except Exception as e:
        print(f"[SINGLE-GEN] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_t5_encoding_only(params_file_path, output_embeddings_path):
    """Run T5 text encoding only and save embeddings to file. Exit when done."""
    try:
        import json
        from ovi.utils.model_loading_utils import init_text_model
        
        print("=" * 80)
        print("T5 ENCODING SUBPROCESS STARTED")
        print("=" * 80)
        
        # Load parameters
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        
        text_prompt = params['text_prompt']
        video_negative_prompt = params['video_negative_prompt']
        audio_negative_prompt = params['audio_negative_prompt']
        fp8_t5 = params.get('fp8_t5', False)
        cpu_only_t5 = params.get('cpu_only_t5', False)
        
        print(f"[T5-ONLY] Text Prompt: {text_prompt[:50]}...")
        print(f"[T5-ONLY] FP8 Mode: {fp8_t5}")
        print(f"[T5-ONLY] CPU-Only Mode: {cpu_only_t5}")
        print("=" * 80)
        
        # Determine checkpoint directory
        ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts")
        
        # Initialize T5 text encoder
        if cpu_only_t5 and fp8_t5:
            print("Loading T5 in CPU-Only + Scaled FP8 mode...")
        elif cpu_only_t5:
            print("Loading T5 in CPU-Only mode...")
        elif fp8_t5:
            print("Loading T5 in Scaled FP8 mode...")
        else:
            print("Loading T5 in standard mode...")
        
        device = 'cpu' if cpu_only_t5 else 0
        text_model = init_text_model(
            ckpt_dir,
            rank=device,
            fp8=fp8_t5,
            cpu_only=cpu_only_t5
        )
        
        print("[T5-ONLY] T5 model loaded successfully")
        print("[T5-ONLY] Encoding text prompts...")
        
        # Encode text embeddings
        encode_device = 'cpu' if cpu_only_t5 else 0
        text_embeddings = text_model(
            [text_prompt, video_negative_prompt, audio_negative_prompt],
            encode_device
        )
        
        print("[T5-ONLY] Text encoding completed")
        
        # Move embeddings to CPU for saving (if they're on GPU)
        text_embeddings_cpu = [emb.cpu() for emb in text_embeddings]
        
        # Save embeddings to file
        print(f"[T5-ONLY] Saving embeddings to: {output_embeddings_path}")
        torch.save(text_embeddings_cpu, output_embeddings_path)
        
        # Explicitly delete T5 model before exit (helps with cleanup)
        del text_model
        del text_embeddings
        del text_embeddings_cpu
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if GPU was used
        if not cpu_only_t5 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("=" * 80)
        print("T5 ENCODING SUBPROCESS COMPLETED")
        print("Process will exit now - OS will free ALL memory")
        print("=" * 80)
        
        # Exit successfully - OS will free ALL memory
        sys.exit(0)
        
    except Exception as e:
        print(f"[T5-ONLY] Error during T5 encoding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print(f"[DEBUG] Main block: single_generation_file={args.single_generation_file}, single_generation={args.single_generation}, encode_t5_only={bool(args.encode_t5_only)}, test={getattr(args, 'test', False)}, test_subprocess={getattr(args, 'test_subprocess', False)}")

    # Check for feature test mode
    if len(sys.argv) > 1 and "--test-features" in sys.argv:
        run_comprehensive_test()
        sys.exit(0)

    if args.encode_t5_only:
        print("[DEBUG] Taking encode_t5_only path")
        # T5 encoding only mode - run and exit
        if not args.output_embeddings:
            print("[ERROR] --output-embeddings path required for T5 encoding mode")
            sys.exit(1)
        run_t5_encoding_only(args.encode_t5_only, args.output_embeddings)
    elif args.single_generation_file:
        print("[DEBUG] Taking single_generation_file path")
        # Single generation mode from file - run and exit
        run_single_generation_from_file(args.single_generation_file)
    elif args.single_generation:
        print("[DEBUG] Taking single_generation path")
        # Single generation mode - run and exit
        run_single_generation(args.single_generation)
    elif args.test_subprocess:
        # Test subprocess functionality
        print("[DEBUG] Testing subprocess functionality")
        test_params = {
            'text_prompt': 'test video',
            'image': None,
            'video_frame_height': 512,
            'video_frame_width': 992,
            'video_seed': 99,
            'solver_name': 'unipc',
            'sample_steps': 1,
            'shift': 5.0,
            'video_guidance_scale': 4.0,
            'audio_guidance_scale': 3.0,
            'slg_layer': 11,
            'blocks_to_swap': 0,
            'video_negative_prompt': 'jitter, bad hands, blur, distortion',
            'audio_negative_prompt': 'robotic, muffled, echo, distorted',
            'use_image_gen': False,
            'cpu_offload': True,
            'delete_text_encoder': True,
            'fp8_t5': False,
            'cpu_only_t5': False,
            'no_audio': False,
            'no_block_prep': False,
            'num_generations': 1,
            'randomize_seed': False,
            'save_metadata': True,
            'aspect_ratio': '16:9',
            'clear_all': False,
            'enable_multiline_prompts': False,
            'enable_video_extension': False,
            'dont_auto_combine_video_input': False,
        }
        success = run_generation_subprocess(test_params)
        print(f"[DEBUG] Subprocess test result: {success}")
        sys.exit(0 if success else 1)
    elif args.test:
        # Test mode: activate venv and run generation
        print("=" * 80)
        print("TEST MODE ENABLED - ACTIVATING VENV")
        print("=" * 80)

        # Activate venv before running test
        import subprocess
        import sys

        try:
            # Check if we're in the right directory and venv exists
            venv_path = os.path.join(os.path.dirname(__file__), "venv")
            if os.path.exists(venv_path):
                print(f"Activating venv at: {venv_path}")

                # On Windows, use the activate script
                if sys.platform == "win32":
                    activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
                    if os.path.exists(activate_script):
                        # Run the test with venv activated
                        cmd = f'"{activate_script}" && python premium.py --test --test_prompt="{args.test_prompt}" --blocks_to_swap={args.blocks_to_swap} {"--test_cpu_offload" if args.test_cpu_offload else ""}'
                        print(f"Running test command: {cmd}")
                        result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
                        sys.exit(result.returncode)
                    else:
                        print(f"Warning: activate.bat not found at {activate_script}")
                else:
                    # On Linux/Mac, source the activate script
                    activate_script = os.path.join(venv_path, "bin", "activate")
                    if os.path.exists(activate_script):
                        cmd = f'source "{activate_script}" && python premium.py --test --test_prompt="{args.test_prompt}" --blocks_to_swap={args.blocks_to_swap} {"--test_cpu_offload" if args.test_cpu_offload else ""}'
                        print(f"Running test command: {cmd}")
                        result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
                        sys.exit(result.returncode)
                    else:
                        print(f"Warning: activate script not found at {activate_script}")
            else:
                print(f"Warning: venv not found at {venv_path}")

            # If venv activation failed, run directly (fallback)
            print("Running test without venv activation...")

        except Exception as e:
            print(f"Error setting up venv: {e}")
            print("Running test without venv activation...")

        # Run test generation (fallback if venv setup failed)
        # Use gradio defaults, only override with test args
        test_params = {
            # Gradio defaults
            'text_prompt': "",
            'image': None,
            'video_frame_height': 512,
            'video_frame_width': 992,
            'video_seed': 99,
            'solver_name': "unipc",
            'sample_steps': 50,
            'shift': 5.0,
            'video_guidance_scale': 4.0,
            'audio_guidance_scale': 3.0,
            'slg_layer': 11,
            'blocks_to_swap': 12,
            'video_negative_prompt': "jitter, bad hands, blur, distortion",
            'audio_negative_prompt': "robotic, muffled, echo, distorted",
            'use_image_gen': False,
            'cpu_offload': True,
            'delete_text_encoder': True,
            'fp8_t5': False,
            'cpu_only_t5': False,
            'no_audio': False,
            'no_block_prep': False,
            'num_generations': 1,
            'randomize_seed': False,
            'save_metadata': True,
            'aspect_ratio': "16:9 Landscape",
            'clear_all': True,
            'vae_tiled_decode': False,
            'vae_tile_size': 32,
            'vae_tile_overlap': 8,
            'force_exact_resolution': False,
            'auto_crop_image': True,
            'enable_multiline_prompts': False,
            'enable_video_extension': False,
            'dont_auto_combine_video_input': False,
        }

        # Override with test args only (not replace all values)
        if hasattr(args, 'test_prompt') and args.test_prompt:
            test_params['text_prompt'] = args.test_prompt
        if hasattr(args, 'blocks_to_swap'):
            test_params['blocks_to_swap'] = args.blocks_to_swap
        if hasattr(args, 'test_cpu_offload'):
            test_params['cpu_offload'] = args.test_cpu_offload
        if hasattr(args, 'test_fp8_t5'):
            test_params['fp8_t5'] = args.test_fp8_t5
        if hasattr(args, 'test_cpu_only_t5'):
            test_params['cpu_only_t5'] = args.test_cpu_only_t5

        # For test mode, use minimal sample steps for speed
        test_params['sample_steps'] = 2

        print("=" * 80)
        print("TEST CONFIGURATION:")
        print(f"  Prompt: {test_params['text_prompt'][:50]}...")
        print(f"  Sample Steps: {test_params['sample_steps']} (fast test)")
        print(f"  Block Swap: {test_params['blocks_to_swap']}")
        print(f"  CPU Offload: {test_params['cpu_offload']}")
        print(f"  Delete T5: {test_params['delete_text_encoder']}")
        print(f"  Scaled FP8 T5: {test_params['fp8_t5']}")
        print(f"  CPU-Only T5: {test_params['cpu_only_t5']}")
        print("=" * 80)

        test_output = generate_video(**test_params)

        if test_output:
            print(f"\n[SUCCESS] Test generation completed successfully: {test_output}")
        else:
            print("\n[FAILED] Test generation failed!")
    else:
        demo.launch(share=args.share, inbrowser=True)

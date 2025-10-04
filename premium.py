import gradio as gr
import torch
import argparse
import os
import sys
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
args = parser.parse_args()

print(f"[DEBUG] Parsed args: single_generation={bool(args.single_generation)}, single_generation_file={bool(args.single_generation_file)}, test={bool(getattr(args, 'test', False))}, test_subprocess={bool(getattr(args, 'test_subprocess', False))}")

# Initialize engines with lazy loading (no models loaded yet)
ovi_engine = None  # Will be initialized on first generation

# Global cancellation flag for stopping generations
cancel_generation = False

def run_generation_subprocess(params):
    """Run a single generation in a subprocess to ensure memory cleanup."""
    import subprocess
    import sys
    import json
    import os
    import tempfile

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

        # Run the subprocess directly
        result = subprocess.run(cmd_args, cwd=script_dir)

        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass

        if result.returncode == 0:
            print("[SUBPROCESS] Generation completed successfully")
            return True
        else:
            print(f"[SUBPROCESS] Generation failed with return code: {result.returncode}")
            return False

    except Exception as e:
        print(f"[SUBPROCESS] Error running subprocess: {e}")
        return False

share_enabled = args.share
print(f"Starting Gradio interface with lazy loading... Share mode: {'ENABLED' if share_enabled else 'DISABLED (local only)'}")
if not share_enabled:
    print("Use --share flag to enable public access with a shareable URL")


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
):
    global ovi_engine

    print(f"[GENERATE_VIDEO] Called with clear_all={clear_all}, num_generations={num_generations}")

    try:
        # Check for cancellation at the start
        check_cancellation()

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

            DEFAULT_CONFIG['cpu_offload'] = final_cpu_offload
            DEFAULT_CONFIG['mode'] = "t2v"
            ovi_engine = OviFusionEngine(blocks_to_swap=final_blocks_to_swap, cpu_offload=final_cpu_offload)
            print("\n[OK] OviFusionEngine initialized successfully (models will load on first generation)")

        image_path = None
        if image is not None:
            image_path = image

        # Determine output directory
        if args.output_dir:
            outputs_dir = args.output_dir
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        last_output_path = None

        # Generate multiple videos
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

            if clear_all:
                # Run this generation in a subprocess for memory cleanup
                single_gen_params = {
                    'text_prompt': text_prompt,
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
                    'delete_text_encoder': delete_text_encoder,
                    'fp8_t5': fp8_t5,
                    'cpu_only_t5': cpu_only_t5,
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
                    }

                success = run_generation_subprocess(single_gen_params)
                if not success:
                    print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] Failed in subprocess")
                    continue

                # Find the generated file (should be the most recent in outputs)
                outputs_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(__file__), "outputs")
                import glob
                pattern = os.path.join(outputs_dir, "ovi_generation_*.mp4")
                existing_files = glob.glob(pattern)
                if existing_files:
                    last_output_path = max(existing_files, key=os.path.getctime)
                    print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] Completed: {os.path.basename(last_output_path)}")
                    continue
                else:
                    print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] No output file found")
                    continue

            # Original generation logic (when clear_all is disabled)
            generated_video, generated_audio, _ = ovi_engine.generate(
                text_prompt=text_prompt,
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
                delete_text_encoder=delete_text_encoder,
                no_block_prep=no_block_prep,
                fp8_t5=fp8_t5,
                cpu_only_t5=cpu_only_t5,
                vae_tiled_decode=vae_tiled_decode,
                vae_tile_size=vae_tile_size,
                vae_tile_overlap=vae_tile_overlap,
            )

            # Get next available filename in sequential format
            output_filename = get_next_filename(outputs_dir)
            output_path = os.path.join(outputs_dir, output_filename)

            # Handle no_audio option
            if no_audio:
                generated_audio = None

            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            last_output_path = output_path

            # Save metadata if enabled
            if save_metadata:
                generation_params = {
                    'text_prompt': text_prompt,
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
                    'video_negative_prompt': video_negative_prompt,
                    'audio_negative_prompt': audio_negative_prompt,
                    'no_audio': no_audio,
                    'is_batch': False
                }
                save_generation_metadata(output_path, generation_params, current_seed)

            print(f"[GENERATION {gen_idx + 1}/{int(num_generations)}] Completed: {output_filename}")

        return last_output_path
    except Exception as e:
        print(f"Error during video generation: {e}")
        return None




# Supported aspect ratios and resolutions from config
ASPECT_RATIOS = {
    "16:9 Landscape": [992, 512],
    "9:16 Portrait": [512, 992],
    "9:16 Portrait (960)": [512, 960],
    "16:9 Landscape (960)": [960, 512],
    "1:1 Square": [720, 720],
    "5:2 Ultra-wide": [448, 1120]
}

def update_resolution(aspect_ratio):
    if aspect_ratio in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_ratio]
    return [512, 992]  # Default fallback

def get_random_seed():
    import random
    return random.randint(0, 100000)

def open_outputs_folder():
    """Open the outputs folder in the system's file explorer."""
    import subprocess
    import platform

    try:
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
        # For regular generation, use 'ovi_generation'
        stem = "ovi_generation"

    # Find all files matching the pattern
    pattern = os.path.join(outputs_dir, f"{stem}_*.mp4")
    existing_files = glob.glob(pattern)

    # Extract numbers from existing files
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove stem and extension to get the number part
        if filename.startswith(f"{stem}_") and filename.endswith(".mp4"):
            num_part = filename[len(f"{stem}_"):-4]  # Remove stem_ and .mp4
            # Only consider 4-digit numbers (our sequential format)
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
    return f"{stem}_{next_num:04d}.mp4"

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
        cancel_generation = False  # Reset for next generation
        raise Exception("Generation cancelled by user")

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
    return sorted(presets)

def save_preset(preset_name, current_preset,
                # All UI parameters
                video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
                video_seed, randomize_seed, no_audio, save_metadata,
                solver_name, sample_steps, num_generations,
                shift, video_guidance_scale, audio_guidance_scale, slg_layer,
                blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
                video_negative_prompt, audio_negative_prompt,
                batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
                vae_tiled_decode, vae_tile_size, vae_tile_overlap):
    """Save current UI state as a preset."""
    try:
        presets_dir = get_presets_dir()

        # If no name provided, use current preset name
        if not preset_name.strip() and current_preset:
            preset_name = current_preset

        if not preset_name.strip():
            presets = get_available_presets()
            return gr.update(choices=presets, value=None), gr.update(value=""), *[gr.update() for _ in range(30)], "Please enter a preset name or select a preset to overwrite"

        preset_file = os.path.join(presets_dir, f"{preset_name}.json")

        # Collect all current settings
        preset_data = {
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
            "video_negative_prompt": video_negative_prompt,
            "audio_negative_prompt": audio_negative_prompt,
            "batch_input_folder": batch_input_folder,
            "batch_output_folder": batch_output_folder,
            "batch_skip_existing": batch_skip_existing,
            "clear_all": clear_all,
            "vae_tiled_decode": vae_tiled_decode,
            "vae_tile_size": vae_tile_size,
            "vae_tile_overlap": vae_tile_overlap,
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
        return gr.update(choices=presets, value=None), gr.update(value=""), *[gr.update() for _ in range(30)], f"Error saving preset: {e}"

def load_preset(preset_name):
    """Load a preset and return all UI values."""
    try:
        if not preset_name:
            return [gr.update() for _ in range(30)] + [gr.update(value=None)] + ["No preset selected"]

        presets_dir = get_presets_dir()
        preset_file = os.path.join(presets_dir, f"{preset_name}.json")

        if not os.path.exists(preset_file):
            return [gr.update() for _ in range(30)] + [gr.update(value=None)] + [f"Preset '{preset_name}' not found"]

        with open(preset_file, 'r', encoding='utf-8') as f:
            import json
            preset_data = json.load(f)

        # Save as last used for auto-load
        last_used_file = os.path.join(presets_dir, "last_used.txt")
        with open(last_used_file, 'w', encoding='utf-8') as f:
            f.write(preset_name)

        # Return all UI updates
        return (
            gr.update(value=preset_data.get("video_text_prompt", "")),
            gr.update(value=preset_data.get("aspect_ratio", "16:9 Landscape")),
            gr.update(value=preset_data.get("video_width", 992)),
            gr.update(value=preset_data.get("video_height", 512)),
            gr.update(value=preset_data.get("auto_crop_image", True)),
            gr.update(value=preset_data.get("video_seed", 99)),
            gr.update(value=preset_data.get("randomize_seed", False)),
            gr.update(value=preset_data.get("no_audio", False)),
            gr.update(value=preset_data.get("save_metadata", True)),
            gr.update(value=preset_data.get("solver_name", "unipc")),
            gr.update(value=preset_data.get("sample_steps", 50)),
            gr.update(value=preset_data.get("num_generations", 1)),
            gr.update(value=preset_data.get("shift", 5.0)),
            gr.update(value=preset_data.get("video_guidance_scale", 4.0)),
            gr.update(value=preset_data.get("audio_guidance_scale", 3.0)),
            gr.update(value=preset_data.get("slg_layer", 11)),
            gr.update(value=preset_data.get("blocks_to_swap", 12)),
            gr.update(value=preset_data.get("cpu_offload", True)),
            gr.update(value=preset_data.get("delete_text_encoder", True)),
            gr.update(value=preset_data.get("fp8_t5", False)),
            gr.update(value=preset_data.get("cpu_only_t5", False)),
            gr.update(value=preset_data.get("video_negative_prompt", "jitter, bad hands, blur, distortion")),
            gr.update(value=preset_data.get("audio_negative_prompt", "robotic, muffled, echo, distorted")),
            gr.update(value=preset_data.get("batch_input_folder", "")),
            gr.update(value=preset_data.get("batch_output_folder", "")),
            gr.update(value=preset_data.get("batch_skip_existing", True)),
            gr.update(value=preset_data.get("clear_all", True)),
            gr.update(value=preset_data.get("vae_tiled_decode", False)),
            gr.update(value=preset_data.get("vae_tile_size", 32)),
            gr.update(value=preset_data.get("vae_tile_overlap", 8)),
            gr.update(value=preset_name),  # Update dropdown value
            f"Preset '{preset_name}' loaded successfully!"
        )

    except Exception as e:
        return [gr.update() for _ in range(30)] + [gr.update(value=None)] + [f"Error loading preset: {e}"]

def initialize_app_with_auto_load():
    """Initialize app with preset dropdown choices and auto-load last preset."""
    try:
        presets = get_available_presets()
        dropdown_update = gr.update(choices=presets, value=None)

        # Try to auto-load the last used preset
        presets_dir = get_presets_dir()
        last_used_file = os.path.join(presets_dir, "last_used.txt")

        if os.path.exists(last_used_file):
            with open(last_used_file, 'r', encoding='utf-8') as f:
                last_preset = f.read().strip()

            if last_preset and last_preset in presets:
                print(f"Auto-loading last used preset: {last_preset}")
                # Load the preset and update dropdown to select it
                loaded_values = load_preset(last_preset)
                # Return dropdown with selected preset + all loaded UI values
                return gr.update(choices=presets, value=last_preset), *loaded_values[:-1], f"Auto-loaded preset '{last_preset}'"
            else:
                print(f"Last used preset '{last_preset}' not found, starting with empty selection")

        # No preset to auto-load - check VRAM and apply optimizations
        print("No preset auto-loaded - checking GPU VRAM for automatic optimizations...")

        gpu_name, vram_gb = detect_gpu_info()

        # Apply VRAM-based optimizations
        fp8_t5_update = gr.update()  # Default: False
        clear_all_update = gr.update()  # Default: True

        optimization_messages = []

        if vram_gb > 0:
            if vram_gb < 23:
                # Enable Scaled FP8 T5 for VRAM < 23GB
                fp8_t5_update = gr.update(value=True)
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB < 23GB → Enabled Scaled FP8 T5")
                print(f"  ✓ VRAM optimization: Enabled Scaled FP8 T5 (VRAM: {vram_gb:.1f}GB < 23GB)")

            if vram_gb > 40:
                # Disable Clear All Memory for VRAM > 40GB
                clear_all_update = gr.update(value=False)
                optimization_messages.append(f"VRAM {vram_gb:.1f}GB > 40GB → Disabled Clear All Memory")
                print(f"  ✓ VRAM optimization: Disabled Clear All Memory (VRAM: {vram_gb:.1f}GB > 40GB)")

        if optimization_messages:
            status_message = "Applied VRAM optimizations: " + ", ".join(optimization_messages)
        else:
            status_message = f"GPU detected ({gpu_name}, {vram_gb:.1f}GB VRAM) - using default settings"
            print(f"  ✓ No VRAM optimizations needed (VRAM: {vram_gb:.1f}GB in optimal range)")

        # Return initialized dropdown with VRAM-optimized defaults
        # The order must match the outputs list in demo.load()
        return (
            dropdown_update,  # preset_dropdown
            gr.update(),  # video_text_prompt
            gr.update(),  # aspect_ratio
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
            gr.update(),  # delete_text_encoder
            fp8_t5_update,  # fp8_t5 (potentially modified)
            gr.update(),  # cpu_only_t5
            gr.update(),  # video_negative_prompt
            gr.update(),  # audio_negative_prompt
            gr.update(),  # batch_input_folder
            gr.update(),  # batch_output_folder
            gr.update(),  # batch_skip_existing
            clear_all_update,  # clear_all (potentially modified)
            gr.update(),  # vae_tiled_decode
            gr.update(),  # vae_tile_size
            gr.update(),  # vae_tile_overlap
            status_message  # status message
        )

    except Exception as e:
        print(f"Warning: Could not initialize app with auto-load: {e}")
        presets = get_available_presets()
        return gr.update(choices=presets, value=None), *[gr.update() for _ in range(30)], ""

def initialize_app():
    """Initialize app with preset dropdown choices."""
    presets = get_available_presets()
    return gr.update(choices=presets, value=None)

def save_generation_metadata(output_path, generation_params, used_seed):
    """Save generation metadata as a .txt file alongside the video."""
    try:
        # Create metadata filename (same as video but .txt extension)
        metadata_path = output_path.replace('.mp4', '.txt')

        metadata_content = f"""OVI VIDEO GENERATION METADATA
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO PARAMETERS:
- Text Prompt: {generation_params.get('text_prompt', 'N/A')}
- Image Path: {generation_params.get('image_path', 'None')}
- Resolution: {generation_params.get('video_frame_height', 'N/A')}x{generation_params.get('video_frame_width', 'N/A')}
- Aspect Ratio: {generation_params.get('aspect_ratio', 'N/A')}
- Seed Used: {used_seed}
- Randomize Seed: {generation_params.get('randomize_seed', False)}
- Number of Generations: {generation_params.get('num_generations', 1)}

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
):
    """Process batch generation from input folder."""
    global ovi_engine

    try:
        # Check for cancellation at the start
        check_cancellation()

        # Determine output directory
        if output_folder and output_folder.strip():
            outputs_dir = output_folder.strip()
        elif args.output_dir:
            outputs_dir = args.output_dir
        else:
            outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # Scan batch files
        batch_items = scan_batch_files(input_folder)
        if not batch_items:
            raise Exception(f"No .txt files found in input folder: {input_folder}")

        print(f"\n[INFO] Found {len(batch_items)} items to process:")
        for base_name, img_path, txt_path in batch_items:
            img_status = "with image" if img_path else "text-only"
            print(f"  - {base_name}: {img_status}")

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

            DEFAULT_CONFIG['cpu_offload'] = final_cpu_offload
            DEFAULT_CONFIG['mode'] = "t2v"
            ovi_engine = OviFusionEngine(blocks_to_swap=final_blocks_to_swap, cpu_offload=final_cpu_offload)
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
                    text_prompt = f.read().strip()
                if not text_prompt:
                    print(f"[WARNING] Empty prompt file, skipping: {txt_path}")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to read prompt file {txt_path}: {e}")
                continue

            print(f"\n[PROCESSING] {base_name}")
            print(f"  Prompt: {text_prompt[:100]}{'...' if len(text_prompt) > 100 else ''}")
            print(f"  Image: {'Yes' if image_path else 'No'}")

            # Check if output already exists (for skip logic)
            expected_output = os.path.join(outputs_dir, f"{base_name}_0001.mp4")
            if skip_existing and os.path.exists(expected_output):
                print(f"  [SKIPPED] Output already exists: {expected_output}")
                skipped_count += 1
                continue

            # Generate videos for this prompt (supporting multiple generations)
            for gen_idx in range(int(num_generations)):
                # Check for cancellation in the loop
                check_cancellation()

                # Handle seed logic for batch processing
                current_seed = 99  # Default seed for batch processing
                if randomize_seed:
                    current_seed = get_random_seed()
                elif gen_idx > 0:
                    current_seed = 99 + gen_idx

                print(f"  [GENERATION {gen_idx + 1}/{int(num_generations)}] Seed: {current_seed}")

                if clear_all:
                    # Run this batch generation in a subprocess for memory cleanup
                    single_gen_params = {
                        'text_prompt': text_prompt,
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
                        'use_image_gen': False,
                        'cpu_offload': cpu_offload,
                        'delete_text_encoder': delete_text_encoder,
                        'fp8_t5': fp8_t5,
                        'cpu_only_t5': cpu_only_t5,
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
                    }

                    success = run_generation_subprocess(single_gen_params)
                    if success:
                        # Find the generated file with base_name prefix
                        import glob
                        pattern = os.path.join(outputs_dir, f"{base_name}_*.mp4")
                        existing_files = glob.glob(pattern)
                        if existing_files:
                            last_output_path = max(existing_files, key=os.path.getctime)
                            print(f"    [SUCCESS] Saved: {os.path.basename(last_output_path)}")
                            processed_count += 1
                        else:
                            print(f"    [WARNING] No output file found for {base_name}")
                    else:
                        print(f"    [ERROR] Generation failed in subprocess")
                    continue

                # Original batch generation logic (when clear_all is disabled)
                try:
                    generated_video, generated_audio, _ = ovi_engine.generate(
                        text_prompt=text_prompt,
                        image_path=image_path,
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
                        delete_text_encoder=delete_text_encoder,
                        no_block_prep=no_block_prep,
                        fp8_t5=fp8_t5,
                        cpu_only_t5=cpu_only_t5,
                        vae_tiled_decode=vae_tiled_decode,
                        vae_tile_size=vae_tile_size,
                        vae_tile_overlap=vae_tile_overlap,
                    )

                    # Get filename with base_name prefix
                    output_filename = get_next_filename(outputs_dir, base_filename=base_name)
                    output_path = os.path.join(outputs_dir, output_filename)

                    # Handle no_audio option
                    if no_audio:
                        generated_audio = None

                    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                    last_output_path = output_path

                    # Save metadata if enabled
                    if save_metadata:
                        generation_params = {
                            'text_prompt': text_prompt,
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
                            'video_negative_prompt': video_negative_prompt,
                            'audio_negative_prompt': audio_negative_prompt,
                            'no_audio': no_audio,
                            'is_batch': True
                        }
                        save_generation_metadata(output_path, generation_params, current_seed)

                    print(f"    [SUCCESS] Saved: {output_filename}")
                    processed_count += 1

                except Exception as e:
                    print(f"    [ERROR] Generation failed: {e}")
                    continue

        print("\n[BATCH COMPLETE]")
        print(f"  Processed: {processed_count} videos")
        print(f"  Skipped: {skipped_count} existing videos")
        print(f"  Total items: {len(batch_items)}")

        return last_output_path

    except Exception as e:
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
        
        # Find closest aspect ratio
        closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(ASPECT_RATIOS[k][0] / ASPECT_RATIOS[k][1] - aspect))
        target_w, target_h = ASPECT_RATIOS[closest_key]
        
        return (
            prompt, 
            img_path,
            gr.update(value=closest_key),
            gr.update(value=target_w),
            gr.update(value=target_h),
            img_path
        )
    except Exception as e:
        print(f"Error loading I2V example: {e}")
        return (prompt, img_path, gr.update(), gr.update(), gr.update(), img_path)

def on_image_upload(image_path, auto_crop_image):
    if image_path is None:
        return (
            gr.update(visible=False, value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(),
            gr.update(),
            None
        )
    
    try:
        img = Image.open(image_path)
        iw, ih = img.size
        if ih == 0 or iw == 0:
            raise ValueError("Invalid image dimensions")
        aspect = iw / ih
        
        closest_key = min(ASPECT_RATIOS.keys(), key=lambda k: abs(ASPECT_RATIOS[k][0] / ASPECT_RATIOS[k][1] - aspect))
        target_w, target_h = ASPECT_RATIOS[closest_key]
        target_aspect = target_w / target_h
        
        # Center crop to target aspect
        if aspect > target_aspect:
            # Image is wider, crop sides
            crop_w = int(ih * target_aspect)
            left = (iw - crop_w) // 2
            box = (left, 0, left + crop_w, ih)
        else:
            # Image is taller, crop top/bottom
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
        
        if auto_crop_image:
            return (
                gr.update(visible=True, value=cropped_path),
                gr.update(value=closest_key),
                gr.update(value=target_w),
                gr.update(value=target_h),
                cropped_path
            )
        else:
            return (
                gr.update(visible=False, value=None),
                gr.update(),
                gr.update(),
                gr.update(),
                image_path
            )
    except Exception as e:
        print(f"Auto-crop error: {e}")
        return (
            gr.update(visible=False, value=None),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            image_path
        )

theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(theme=gr.themes.Soft(), title="Ovi Pro Premium SECourses") as demo:
    gr.Markdown("# Ovi Pro SECourses Premium App v2.9 : https://www.patreon.com/posts/140393220")

    image_to_use = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem("Generate"):
            with gr.Row():
                with gr.Column():
                    # Image section
                    image = gr.Image(type="filepath", label="First Frame Image (upload or generate)", height=512)

                    # Generate Video button right under image upload
                    run_btn = gr.Button("Generate Video 🚀", variant="primary", size="lg")

                    with gr.Accordion("🎬 Video Generation Options", open=True):
                        # Video prompt with 10 lines
                        video_text_prompt = gr.Textbox(
                            label="Video Prompt",
                            placeholder="Describe your video...",
                            lines=10
                        )

                        # Aspect ratio selection and resolution in same row
                        with gr.Row():
                            aspect_ratio = gr.Dropdown(
                                choices=list(ASPECT_RATIOS.keys()),
                                value="16:9 Landscape",
                                label="Aspect Ratio",
                                info="Select aspect ratio - width and height will update automatically"
                            )
                            video_width = gr.Number(minimum=128, maximum=1280, value=992, step=32, label="Video Width")
                            video_height = gr.Number(minimum=128, maximum=1280, value=512, step=32, label="Video Height")
                            auto_crop_image = gr.Checkbox(
                                value=True,
                                label="Auto Crop Image",
                                info="Automatically detect closest aspect ratio and crop image for perfect I2V generation"
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
                                info="Delete T5 encoder after text encoding to save ~5GB VRAM"
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
                        
                        # VAE Tiled Decoding Options
                        with gr.Accordion("🎨 VAE Tiled Decoding (Advanced VRAM Optimization)", open=False):
                            gr.Markdown("""
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
                            
                            **992×512 video (16:9 Landscape)**:
                            - Latent size: 62×32
                            - With tile_size=32, overlap=8: ~2×1 = 2 tiles per frame
                            - VRAM saving: ~30% (12-15GB → 8-10GB during decode)
                            
                            **512×992 video (9:16 Portrait)**:
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
                            
                            **Try it!** Enable tiled decoding below and compare VRAM usage in your task manager.
                            """)
                            
                            with gr.Row():
                                vae_tiled_decode = gr.Checkbox(
                                    label="Enable Tiled VAE Decode",
                                    value=False,
                                    info="✅ Process VAE decoding in tiles to save VRAM (recommended for <24GB VRAM)"
                                )
                                vae_tile_size = gr.Slider(
                                    minimum=16,
                                    maximum=64,
                                    value=32,
                                    step=8,
                                    label="Tile Size (Latent Space)",
                                    info="Spatial tile size: 16=max VRAM savings (slower), 32=balanced ⭐, 64=min savings (faster)"
                                )
                                vae_tile_overlap = gr.Slider(
                                    minimum=4,
                                    maximum=16,
                                    value=8,
                                    step=2,
                                    label="Tile Overlap",
                                    info="Overlap for seamless blending: 4=fast, 8=balanced ⭐, 16=best quality"
                                )

                with gr.Column():
                    output_path = gr.Video(label="Generated Video")
                    with gr.Row():
                        open_outputs_btn = gr.Button("📁 Open Outputs Folder")
                        cancel_btn = gr.Button("❌ Cancel All", variant="stop")
                    cropped_display = gr.Image(label="Cropped First Frame", visible=False, height=512)

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

                        ### 🎨 Supported Resolutions
                        - **16:9 Landscape**: 992×512 (default)
                        - **9:16 Portrait**: 512×992
                        - **9:16 Portrait (960)**: 512×960
                        - **16:9 Landscape (960)**: 960×512
                        - **1:1 Square**: 720×720
                        - **5:2 Ultra-wide**: 448×1120
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
                        - **Clear All Memory**: Run each generation as separate process to prevent VRAM/RAM leaks (recommended)

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
                                fn=lambda prompt=example: (prompt, None),
                                outputs=[video_text_prompt, image]
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
    aspect_ratio.change(
        fn=update_resolution,
        inputs=[aspect_ratio],
        outputs=[video_width, video_height],
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

    # Hook up video generation
    run_btn.click(
        fn=generate_video,
        inputs=[
            video_text_prompt, image_to_use, video_height, video_width, video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, blocks_to_swap, video_negative_prompt, audio_negative_prompt,
            gr.Checkbox(value=False, visible=False), cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            no_audio, gr.Checkbox(value=False, visible=False),
            num_generations, randomize_seed, save_metadata, aspect_ratio, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
        ],
        outputs=[output_path],
    )

    image.change(
        fn=on_image_upload,
        inputs=[image, auto_crop_image],
        outputs=[cropped_display, aspect_ratio, video_width, video_height, image_to_use]
    )

    auto_crop_image.change(
        fn=on_image_upload,
        inputs=[image, auto_crop_image],
        outputs=[cropped_display, aspect_ratio, video_width, video_height, image_to_use]
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

    # Hook up batch processing button
    batch_btn.click(
        fn=process_batch_generation,
        inputs=[
            batch_input_folder, batch_output_folder, batch_skip_existing,
            video_height, video_width, solver_name, sample_steps, shift,
            video_guidance_scale, audio_guidance_scale, slg_layer, blocks_to_swap,
            video_negative_prompt, audio_negative_prompt, cpu_offload,
            delete_text_encoder, fp8_t5, cpu_only_t5, no_audio, gr.Checkbox(value=False, visible=False),
            num_generations, randomize_seed, save_metadata, aspect_ratio, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
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
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
        ],
        outputs=[
            preset_dropdown, preset_name,  # Update dropdown, clear name field
            video_text_prompt, aspect_ratio, video_width, video_height, auto_crop_image,
            video_seed, randomize_seed, no_audio, save_metadata,
            solver_name, sample_steps, num_generations,
            shift, video_guidance_scale, audio_guidance_scale, slg_layer,
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
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
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            preset_dropdown, gr.Textbox(visible=False)  # Update current preset, status message
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
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
            preset_dropdown, gr.Textbox(visible=False)  # Update current preset, status message
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
            blocks_to_swap, cpu_offload, delete_text_encoder, fp8_t5, cpu_only_t5,
            video_negative_prompt, audio_negative_prompt,
            batch_input_folder, batch_output_folder, batch_skip_existing, clear_all,
            vae_tiled_decode, vae_tile_size, vae_tile_overlap,
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

def run_single_generation_from_file(json_file_path):
    """Run a single generation from JSON file and exit."""
    try:
        import json
        with open(json_file_path, 'r') as f:
            params = json.load(f)

        print(f"[SINGLE-GEN] Loaded params from file: {json_file_path}")
        print(f"[SINGLE-GEN] Starting generation with params: {list(params.keys())}")
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

if __name__ == "__main__":
    print(f"[DEBUG] Main block: single_generation_file={args.single_generation_file}, single_generation={args.single_generation}, test={getattr(args, 'test', False)}, test_subprocess={getattr(args, 'test_subprocess', False)}")
    if args.single_generation_file:
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
            'aspect_ratio': '16:9 Landscape',
            'clear_all': False,
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

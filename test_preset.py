import json
import os

def load_preset_safely(preset_name):
    """Load and validate preset data with error recovery."""
    try:
        presets_dir = "presets"
        preset_file = os.path.join(presets_dir, f"{preset_name}.json")

        if not os.path.exists(preset_file):
            return None, f"Preset '{preset_name}' not found"

        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)

        # Check version compatibility
        version = preset_data.get("preset_version", "1.0")

        # Validate all parameters
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
            "video_extension_count": 0,
        }

        validated_data = {}
        for param_name, default_value in PRESET_DEFAULTS.items():
            raw_value = preset_data.get(param_name, default_value)
            validated_data[param_name] = raw_value

        return validated_data, None

    except Exception as e:
        return None, f"Error loading preset: {e}"

# Test loading a1 preset
data, err = load_preset_safely('a1')
if data:
    print('video_extension_count:', data['video_extension_count'])
    print('Type:', type(data['video_extension_count']))
else:
    print('Error:', err)

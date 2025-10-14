"""
Convert models in convert_fp8 folder to scaled FP8 format.

This script converts safetensors model files to FP8 E4M3 format with appropriate scaling,
matching the reference FP8 implementation used in ComfyUI.

Usage:
    python convert_to_scaled_fp8.py

The script will:
1. Find all .safetensors files in convert_fp8/ folder
2. Load each model and detect model family (Qwen Image or Wan)
3. Quantize attention/FFN Linear layer weights to FP8 with appropriate scaling
4. Convert all bfloat16 tensors to float16
5. Add scale_weight tensors for each FP8 layer
6. Save the FP8 model with suffix '_fp8_scaled.safetensors'

Quantization details:
- Qwen Image models: per-tensor quantization (scalar scales, torch.Size([]))
- Wan models: per-channel quantization (row-wise scales, torch.Size([out_features, 1]))
- Attention q,k,v,o and FFN weights -> FP8 E4M3
- Normalization layers (norm, norm_k, norm_q) -> float16 (not quantized)
- All bfloat16 tensors -> float16
"""

import torch
import os
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_fp8_maxval():
    """Calculate the maximum representable value in FP8 E4M3 format."""
    return torch.finfo(torch.float8_e4m3fn).max


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """
    Quantize a tensor to FP8 format.
    
    Args:
        tensor: Tensor to quantize (in float32)
        scale: Scale factor(s) 
        fp8_dtype: Target FP8 dtype
        max_value: Max representable value in FP8
        min_value: Min representable value in FP8
    
    Returns:
        Quantized tensor in FP8 format
    """
    tensor = tensor.to(torch.float32)
    
    # Scale and clamp
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)
    tensor = tensor.clamp_(min=min_value, max=max_value)
    
    # Convert to FP8
    tensor = tensor.to(fp8_dtype)
    
    return tensor


def quantize_weight_per_tensor(weight, fp8_dtype, max_value, min_value):
    """
    Quantize a Linear weight tensor using per-tensor quantization.
    This matches the reference FP8 implementation.

    Args:
        weight: Weight tensor (any shape)
        fp8_dtype: FP8 dtype
        max_value: Max FP8 value
        min_value: Min FP8 value

    Returns:
        quantized_weight: FP8 quantized weights
        scale_tensor: Scalar scale factor (torch.Size([]))
    """
    # Calculate per-tensor scale
    abs_w = torch.abs(weight)
    tensor_max = torch.max(abs_w)
    scale = tensor_max / max_value
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)

    # Quantize
    quantized = quantize_fp8(weight, scale, fp8_dtype, max_value, min_value)

    # Return scalar scale (empty shape)
    return quantized, scale


def quantize_weight_blockwise(weight, fp8_dtype, max_value, min_value, block_size=64):
    """
    Quantize a Linear weight tensor using block-wise quantization.
    This matches musubi-tuner's FP8 implementation.

    Args:
        weight: Weight tensor (2D shape expected)
        fp8_dtype: FP8 dtype
        max_value: Max FP8 value
        min_value: Min FP8 value
        block_size: Block size for quantization

    Returns:
        quantized_weight: FP8 quantized weights
        scale_tensor: Scale tensor with shape [out_features, num_blocks, 1]
    """
    if weight.ndim != 2:
        # Fallback to per-tensor for non-2D weights
        return quantize_weight_per_tensor(weight, fp8_dtype, max_value, min_value)

    out_features, in_features = weight.shape

    if in_features % block_size != 0:
        # Fallback to per-channel quantization
        return quantize_weight_per_channel(weight, fp8_dtype, max_value, min_value)

    # Block-wise quantization
    num_blocks = in_features // block_size
    weight_reshaped = weight.contiguous().view(out_features, num_blocks, block_size)

    # Calculate scale per block
    abs_w = torch.abs(weight_reshaped)
    block_max = torch.max(abs_w, dim=2, keepdim=True).values  # [out_features, num_blocks, 1]
    scale = block_max / max_value
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)

    # Quantize
    quantized = quantize_fp8(weight_reshaped, scale, fp8_dtype, max_value, min_value)
    quantized = quantized.view(weight.shape)  # Restore original shape

    return quantized, scale


def quantize_weight_per_channel(weight, fp8_dtype, max_value, min_value):
    """
    Quantize a Linear weight tensor using per-channel quantization.

    Args:
        weight: Weight tensor (2D shape expected)
        fp8_dtype: FP8 dtype
        max_value: Max FP8 value
        min_value: Min FP8 value

    Returns:
        quantized_weight: FP8 quantized weights
        scale_tensor: Scale tensor with shape [out_features, 1]
    """
    if weight.ndim != 2:
        # Fallback to per-tensor for non-2D weights
        return quantize_weight_per_tensor(weight, fp8_dtype, max_value, min_value)

    # Per-channel quantization (row-wise)
    abs_w = torch.abs(weight)
    row_max = torch.max(abs_w, dim=1, keepdim=True).values  # [out_features, 1]
    scale = row_max / max_value
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)

    # Quantize
    quantized = quantize_fp8(weight, scale, fp8_dtype, max_value, min_value)

    return quantized, scale


def detect_model_family(state_dict_keys):
    """Best-effort detection of model family from state dict keys."""
    # Qwen Image models place all transformer layers under "transformer_blocks.*"
    if any(key.startswith("transformer_blocks.") for key in state_dict_keys):
        return "qwen_image"

    # Wan models place layers under "blocks.*"
    if any(key.startswith("blocks.") for key in state_dict_keys):
        return "wan"

    return "generic"


def is_target_layer(key, model_family):
    """
    Determine if a weight should be quantized to FP8.
    
    Target layers depend on the detected model family.
    
    Wan family:
    - Transformer blocks: self_attn (q,k,v,o), cross_attn (q,k,v,o), ffn layers
    - Embeddings: text_embedding, time_embedding, time_projection
    - Head: head.head.weight
    
    Qwen Image family:
    - All Linear weights in transformer_blocks.* except norm layers
    - Input/output layers: img_in, txt_in, time_text_embed timestep embedder, norm_out.linear, proj_out
    
    Exclude layers:
    - Normalization layers (norm_k, norm_q, norm_added_k, norm_added_q, ln, etc.)
    - Modulation layers
    
    Args:
        key: State dict key (e.g., "blocks.0.self_attn.q.weight")
        model_family: Detected model family string
    """
    # Must be a weight tensor
    if not key.endswith(".weight"):
        return False
    
    # Exclude normalization layers (but not linear layers that happen to contain 'norm' in their name)
    # True normalization layers have specific patterns like norm_k, norm_q, etc.
    norm_patterns = ['.norm_k.', '.norm_q.', '.norm_added_k.', '.norm_added_q.', 'ln.']
    if any(pattern in key.lower() for pattern in norm_patterns):
        return False

    # Exclude modulation layers
    if "modulation" in key or "adaLN" in key or "ada_ln" in key:
        return False

    if model_family == "qwen_image":
        # For Qwen Image models, quantize all Linear weights except norms
        # Include transformer_blocks AND specific input/output layers
        if key.startswith("transformer_blocks."):
            return True

        # Include specific input/output layers that are quantized in working model
        include_layers = [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "norm_out.linear.weight",
            "proj_out.weight"
        ]
        return key in include_layers
    
    # Default to Wan-style targeting
    # Include specific embedding/projection/head layers
    if any(pattern in key for pattern in [
        "text_embedding",
        "time_embedding", 
        "time_projection",
        "head.head.weight"
    ]):
        return True
    
    # For transformer blocks
    if "blocks" in key:
        # Include attention and FFN layers
        include_patterns = [
            "self_attn.q.weight",
            "self_attn.k.weight", 
            "self_attn.v.weight",
            "self_attn.o.weight",
            "cross_attn.q.weight",
            "cross_attn.k.weight",
            "cross_attn.v.weight",
            "cross_attn.o.weight",
            "ffn.0.weight",  # FFN layers
            "ffn.2.weight",
            "ffn.4.weight",
        ]
        return any(pattern in key for pattern in include_patterns)
    
    return False


def get_comfyui_expected_scale_layers(model_family):
    """
    Get layers that ComfyUI expects to have scale tensors, even if not quantized.

    Args:
        model_family: Detected model family string

    Returns:
        List of weight keys that should have scale tensors for ComfyUI compatibility
    """
    if model_family == "qwen_image":
        # ComfyUI expects scale tensors for these layers even if not quantized
        return [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ]
    elif model_family == "wan":
        # Wan models don't seem to have this issue, but keep for consistency
        return []
    else:
        return []


def convert_model_to_fp8(input_path, output_path):
    """
    Convert a safetensors model to scaled FP8 format compatible with ComfyUI WanVideoWrapper.
    Uses per-tensor quantization for Qwen Image models (scalar scales).
    Uses per-channel quantization for Wan models.
    
    Args:
        input_path: Path to input .safetensors file
        output_path: Path to output FP8 .safetensors file
    
    Returns:
        Dictionary with conversion statistics
    """
    logger.info(f"Loading model from {input_path}")
    logger.info(f"File size: {os.path.getsize(input_path) / (1024**3):.2f} GB")
    
    # Load state dict
    state_dict = load_file(input_path, device="cpu")
    model_family = detect_model_family(state_dict.keys())
    logger.info(f"Detected model family: {model_family}")
    
    # Check if already FP8
    has_fp8 = any(v.dtype == torch.float8_e4m3fn for v in state_dict.values() if isinstance(v, torch.Tensor))
    if has_fp8:
        logger.warning("Model already contains FP8 weights! Skipping conversion.")
        logger.info("If you want to re-convert, please use the original non-FP8 model.")
        return None
    
    logger.info(f"Loaded {len(state_dict)} tensors from checkpoint")

    # FP8 configuration
    fp8_dtype = torch.float8_e4m3fn
    max_value = calculate_fp8_maxval()
    min_value = -max_value

    # Determine quantization mode based on model family
    if model_family == "qwen_image":
        quantization_mode = "per-tensor"  # Qwen Image uses scalar scales
        logger.info("Using per-tensor quantization (scalar scales) for Qwen Image model")
    else:
        quantization_mode = "per-channel"  # Wan and others use per-channel
        logger.info("Using per-channel quantization for Wan model")

    # Statistics
    optimized_count = 0
    skipped_count = 0
    bf16_to_f16_count = 0
    total_params_before = 0
    total_params_after = 0
    
    # New state dict with FP8 weights
    fp8_state_dict = {}
    
    logger.info(f"Starting FP8 conversion with {quantization_mode} quantization (ComfyUI compatible)")
    logger.info("=" * 80)
    
    # Process each tensor
    for key, tensor in tqdm(state_dict.items(), desc="Converting to FP8"):
        original_dtype = tensor.dtype
        
        # Check if this is a target weight for quantization
        if is_target_layer(key, model_family) and tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
            # This is a Linear layer weight in transformer blocks
            original_shape = tensor.shape
            
            # Ensure 2D for Linear layers
            if tensor.ndim != 2:
                logger.warning(f"Skipping {key}: expected 2D weight, got {tensor.ndim}D")
                # Convert to float16 if bfloat16
                if tensor.dtype == torch.bfloat16:
                    fp8_state_dict[key] = tensor.to(torch.float16)
                    bf16_to_f16_count += 1
                else:
                    fp8_state_dict[key] = tensor
                skipped_count += 1
                continue
            
            # Count parameters before
            total_params_before += tensor.numel() * original_dtype.itemsize
            
            try:
                # Quantize to FP8 with appropriate scaling mode
                if quantization_mode == "per-tensor":
                    quantized_weight, scale_tensor = quantize_weight_per_tensor(
                        tensor, fp8_dtype, max_value, min_value
                    )
                else:
                    quantized_weight, scale_tensor = quantize_weight_per_channel(
                        tensor, fp8_dtype, max_value, min_value
                    )
                
                # Count parameters after (FP8 weights + scale tensors in bfloat16)
                total_params_after += quantized_weight.numel() * fp8_dtype.itemsize
                total_params_after += scale_tensor.numel() * torch.bfloat16.itemsize  # scale_weight
                
                # Store FP8 weight
                fp8_state_dict[key] = quantized_weight
                
                # Store scale_weight (in bfloat16 like working model)
                scale_weight_key = key.replace(".weight", ".scale_weight")
                fp8_state_dict[scale_weight_key] = scale_tensor.to(torch.bfloat16)
                
                optimized_count += 1
                
                # Log details for first few layers
                if optimized_count <= 3:
                    scale_shape = scale_tensor.shape
                    logger.info(f"  {key}: {original_shape} -> FP8 ({quantization_mode}, scale_shape={scale_shape})")
                
            except Exception as e:
                logger.warning(f"Failed to quantize {key}: {e}")
                # Convert to float16 if bfloat16
                if tensor.dtype == torch.bfloat16:
                    fp8_state_dict[key] = tensor.to(torch.float16)
                    bf16_to_f16_count += 1
                else:
                    fp8_state_dict[key] = tensor
                skipped_count += 1
                
        else:
            # Convert bfloat16 to float16 for all other tensors
            if tensor.dtype == torch.bfloat16:
                fp8_state_dict[key] = tensor.to(torch.float16)
                bf16_to_f16_count += 1
            else:
                fp8_state_dict[key] = tensor
            
            # Track unquantized parameters
            if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                total_params_before += tensor.numel() * original_dtype.itemsize
                # bfloat16 becomes float16 (same size)
                out_dtype = torch.float16 if tensor.dtype == torch.bfloat16 else tensor.dtype
                total_params_after += tensor.numel() * out_dtype.itemsize


    logger.info("=" * 80)
    logger.info("FP8 Conversion Complete!")
    logger.info(f"  Quantized layers: {optimized_count}")
    logger.info(f"  Skipped layers: {skipped_count}")
    logger.info(f"  BF16->F16 conversions: {bf16_to_f16_count}")
    logger.info(f"  Total tensors in output: {len(fp8_state_dict)}")
    logger.info(f"  Model size before: {total_params_before / (1024**3):.2f} GB")
    logger.info(f"  Model size after: {total_params_after / (1024**3):.2f} GB")
    
    if total_params_after > 0:
        compression_ratio = total_params_before / total_params_after
        savings_gb = (total_params_before - total_params_after) / (1024**3)
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"  Space saved: {savings_gb:.2f} GB")
    
    # Add 'scaled_fp8' metadata marker (FP8 tensor with shape [2] like working model)
    # This helps loaders quickly identify the FP8 format
    fp8_state_dict['scaled_fp8'] = torch.tensor([0.0, 0.0], dtype=torch.float8_e4m3fn)
    logger.info("Added 'scaled_fp8' metadata marker")
    
    # Save FP8 model
    logger.info(f"Saving FP8 model to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with temporary file for safety
    tmp_path = output_path + ".tmp"
    save_file(fp8_state_dict, tmp_path)
    os.replace(tmp_path, output_path)
    
    output_size = os.path.getsize(output_path) / (1024**3)
    logger.info(f"Saved! Output file size: {output_size:.2f} GB")
    logger.info("=" * 80)
    
    return {
        'optimized_layers': optimized_count,
        'skipped_layers': skipped_count,
        'size_before_gb': total_params_before / (1024**3),
        'size_after_gb': total_params_after / (1024**3),
        'compression_ratio': total_params_before / total_params_after if total_params_after > 0 else 1.0,
        'output_file_size_gb': output_size
    }


def main():
    """
    Main conversion function.
    Finds all .safetensors files in convert_fp8/ and converts them to FP8.
    """
    logger.info("=" * 80)
    logger.info("SCALED FP8 MODEL CONVERTER")
    logger.info("Converts models to FP8 E4M3 with appropriate scaling (ComfyUI compatible)")
    logger.info("  - Qwen Image models: per-tensor quantization (scalar scales)")
    logger.info("  - Wan models: per-channel quantization")
    logger.info("=" * 80)
    
    # Find input directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "convert_fp8")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please create 'convert_fp8' folder and place .safetensors files inside")
        return
    
    # Find all .safetensors files
    model_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    
    if not model_files:
        logger.warning(f"No .safetensors files found in {input_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model(s) to convert:")
    for f in model_files:
        logger.info(f"  - {f}")
    logger.info("")
    
    # Convert each model
    results = {}
    for model_file in model_files:
        input_path = os.path.join(input_dir, model_file)
        
        # Generate output filename
        base_name = os.path.splitext(model_file)[0]
        output_file = f"{base_name}_fp8_scaled.safetensors"
        output_path = os.path.join(input_dir, output_file)
        
        logger.info(f"Converting: {model_file}")
        logger.info(f"Output: {output_file}")
        logger.info("")
        
        try:
            stats = convert_model_to_fp8(input_path, output_path)
            if stats is None:
                # Model was already FP8, skip
                results[model_file] = {
                    'success': False,
                    'error': 'Model already contains FP8 weights (already converted)'
                }
            else:
                results[model_file] = {
                    'success': True,
                    'output': output_file,
                    'stats': stats
                }
        except Exception as e:
            logger.error(f"Failed to convert {model_file}: {e}")
            import traceback
            traceback.print_exc()
            results[model_file] = {
                'success': False,
                'error': str(e)
            }
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 80)
    
    for model_file, result in results.items():
        if result['success']:
            stats = result['stats']
            logger.info(f"✓ {model_file}")
            logger.info(f"  → {result['output']}")
            logger.info(f"  Quantized layers: {stats['optimized_layers']}")
            logger.info(f"  Size: {stats['size_before_gb']:.2f} GB → {stats['size_after_gb']:.2f} GB")
            logger.info(f"  Compression: {stats['compression_ratio']:.2f}x")
        else:
            logger.info(f"✗ {model_file}")
            logger.info(f"  Error: {result['error']}")
        logger.info("")
    
    success_count = sum(1 for r in results.values() if r['success'])
    logger.info(f"Successfully converted {success_count}/{len(model_files)} models")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


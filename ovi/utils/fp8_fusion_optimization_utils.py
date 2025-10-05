"""
FP8 optimization utilities for Fusion Model (Wan 2.2 Transformer).
Based on Musubi Tuner's approach for proper FP8 weight quantization.
Applies scaled FP8 quantization to video and audio transformer blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3):
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


def quantize_weight_block(weight, fp8_dtype, max_value, min_value, block_size=64):
    """
    Quantize a Linear weight tensor using per-output-channel block quantization.
    This is the highest quality mode used by Musubi Tuner.
    
    Args:
        weight: Weight tensor [out_features, in_features]
        fp8_dtype: FP8 dtype
        max_value: Max FP8 value
        min_value: Min FP8 value  
        block_size: Block size for quantization
    
    Returns:
        quantized_weight: FP8 quantized weights
        scale_tensor: Scale factors [out_features, num_blocks, 1]
    """
    original_shape = weight.shape
    
    if weight.ndim != 2:
        # Fallback to per-tensor for non-2D weights
        abs_w = torch.abs(weight)
        tensor_max = torch.max(abs_w)
        scale = tensor_max / max_value
        scale = torch.clamp(scale, min=1e-8).to(torch.float32)
        quantized = quantize_fp8(weight, scale, fp8_dtype, max_value, min_value)
        return quantized, scale.reshape(1)
    
    out_features, in_features = weight.shape
    
    # Check if divisible by block_size
    if in_features % block_size != 0:
        # Fallback to per-channel
        logger.debug(f"Weight shape {weight.shape} not divisible by block_size {block_size}, using per-channel")
        abs_w = torch.abs(weight)
        row_max = torch.max(abs_w, dim=1, keepdim=True).values
        scale = row_max / max_value
        scale = torch.clamp(scale, min=1e-8).to(torch.float32)
        quantized = quantize_fp8(weight, scale, fp8_dtype, max_value, min_value)
        return quantized, scale  # [out, 1]
    
    # Per-output-channel block quantization (best quality)
    num_blocks = in_features // block_size
    weight_blocked = weight.contiguous().view(out_features, num_blocks, block_size)
    
    # Calculate scale per block: [out_features, num_blocks, 1]
    abs_w = torch.abs(weight_blocked)
    block_max = torch.max(abs_w, dim=2, keepdim=True).values
    scale = block_max / max_value
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)
    
    # Quantize
    quantized = quantize_fp8(weight_blocked, scale, fp8_dtype, max_value, min_value)
    
    # Restore original shape
    quantized = quantized.view(original_shape)
    
    return quantized, scale


def is_target_layer(module_name):
    """
    Determine if a layer should be quantized to FP8.
    
    Target layers (in transformer blocks):
    - self_attn: q, k, v, o projections
    - cross_attn: q, k, v, o, k_fusion, v_fusion projections
    - ffn: fc1, fc2, gate layers
    
    Exclude layers:
    - Embeddings (patch_embed, time_embed, etc.)
    - Final projections (final_proj, final_layer)
    - Modulation layers (modulation)
    - Normalization layers (already excluded by Linear check)
    
    Args:
        module_name: Module path (e.g., "video_model.blocks.0.self_attn.q")
    """
    # Must be in blocks (transformer blocks)
    if "blocks" not in module_name:
        return False
    
    # Exclude modulation layers
    if "modulation" in module_name:
        return False
    
    # Include attention and FFN layers
    include_patterns = [
        "self_attn.q",
        "self_attn.k", 
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "cross_attn.k_fusion",
        "cross_attn.v_fusion",
        "ffn.fc1",
        "ffn.fc2",
        "ffn.gate",
    ]
    
    return any(pattern in module_name for pattern in include_patterns)


def optimize_fusion_model_to_fp8(fusion_model, device='cpu', block_size=64):
    """
    Optimize FusionModel to FP8 by quantizing Linear layer weights in transformer blocks.
    
    Args:
        fusion_model: FusionModel instance (video + audio models)
        device: Device for quantization computation ('cpu' recommended to avoid VRAM spike)
        block_size: Block size for per-block quantization
    
    Returns:
        fusion_model: Modified model with FP8 weights and scale parameters
        info: Dictionary with optimization statistics
    """
    fp8_dtype = torch.float8_e4m3fn
    max_value = calculate_fp8_maxval()
    min_value = -max_value
    
    optimized_count = 0
    total_params_before = 0
    total_params_after = 0
    
    video_count = 0
    audio_count = 0
    
    logger.info("Starting FP8 optimization of Fusion Model...")
    logger.info(f"  Target: Transformer blocks (video + audio)")
    logger.info(f"  Format: FP8 E4M3 with per-block scaling (block_size={block_size})")
    
    # Find all target Linear layers in both video and audio models
    target_layers = []
    
    # Video model layers
    if fusion_model.video_model is not None:
        for name, module in fusion_model.video_model.named_modules():
            full_name = f"video_model.{name}"
            if isinstance(module, nn.Linear) and is_target_layer(full_name):
                target_layers.append((full_name, module, 'video'))
    
    # Audio model layers
    if fusion_model.audio_model is not None:
        for name, module in fusion_model.audio_model.named_modules():
            full_name = f"audio_model.{name}"
            if isinstance(module, nn.Linear) and is_target_layer(full_name):
                target_layers.append((full_name, module, 'audio'))
    
    logger.info(f"Found {len(target_layers)} Linear layers to optimize")
    logger.info(f"  Video model: {sum(1 for _, _, t in target_layers if t == 'video')} layers")
    logger.info(f"  Audio model: {sum(1 for _, _, t in target_layers if t == 'audio')} layers")
    
    # Quantize each Linear layer
    for name, module, model_type in tqdm(target_layers, desc="Quantizing to FP8"):
        if module.weight is None:
            continue
        
        original_weight = module.weight.data
        original_device = original_weight.device
        original_dtype = original_weight.dtype
        
        # Move to computation device
        weight = original_weight.to(device)
        
        # Count parameters
        total_params_before += weight.numel() * original_dtype.itemsize
        
        # Quantize with per-block scaling
        quantized_weight, scale_tensor = quantize_weight_block(
            weight, fp8_dtype, max_value, min_value, block_size
        )
        
        # Count parameters after (FP8 weights + scale in original dtype)
        total_params_after += quantized_weight.numel() * fp8_dtype.itemsize
        total_params_after += scale_tensor.numel() * original_dtype.itemsize
        
        # Move back to original device
        quantized_weight = quantized_weight.to(original_device)
        scale_tensor = scale_tensor.to(dtype=original_dtype, device=original_device)
        
        # Replace weight with FP8 version
        del module.weight
        module.weight = nn.Parameter(quantized_weight, requires_grad=False)
        
        # Register scale as buffer
        module.register_buffer('scale_weight', scale_tensor)
        
        optimized_count += 1
        if model_type == 'video':
            video_count += 1
        else:
            audio_count += 1
        
        # Free memory periodically
        if optimized_count % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    info = {
        'optimized_layers': optimized_count,
        'video_layers': video_count,
        'audio_layers': audio_count,
        'params_before_mb': total_params_before / (1024 * 1024),
        'params_after_mb': total_params_after / (1024 * 1024),
        'compression_ratio': total_params_before / total_params_after if total_params_after > 0 else 1.0
    }
    
    logger.info(f"FP8 optimization complete:")
    logger.info(f"  Optimized {optimized_count} Linear layers ({video_count} video + {audio_count} audio)")
    logger.info(f"  Model size: {info['params_before_mb']:.1f} MB -> {info['params_after_mb']:.1f} MB")
    logger.info(f"  Compression ratio: {info['compression_ratio']:.2f}x")
    
    return fusion_model, info


def fp8_linear_forward_patch(self, x):
    """
    Patched forward method for Linear layers with FP8 weights.
    Dequantizes weights on-the-fly during forward pass.
    """
    original_dtype = self.scale_weight.dtype
    
    # Dequantize weights based on scale shape
    if self.scale_weight.ndim == 1:
        # Per-tensor quantization
        dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
    elif self.scale_weight.ndim == 2:
        # Per-channel quantization [out, 1]
        dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
    else:
        # Per-block quantization [out, num_blocks, 1]
        out_features, num_blocks, _ = self.scale_weight.shape
        dequantized_weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
        dequantized_weight = dequantized_weight * self.scale_weight
        dequantized_weight = dequantized_weight.view(self.weight.shape)
    
    # Perform linear transformation
    if self.bias is not None:
        output = F.linear(x, dequantized_weight, self.bias)
    else:
        output = F.linear(x, dequantized_weight)
    
    return output


def apply_fusion_fp8_monkey_patch(fusion_model):
    """
    Apply monkey patching to Linear layers with FP8 weights in FusionModel.
    
    Args:
        fusion_model: FusionModel with FP8-optimized Linear layers
    
    Returns:
        fusion_model: Model with patched forward methods
    """
    patched_count = 0
    video_patched = 0
    audio_patched = 0
    
    # Patch video model
    if fusion_model.video_model is not None:
        for name, module in fusion_model.video_model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'scale_weight'):
                # Create new forward method
                def new_forward(self, x):
                    return fp8_linear_forward_patch(self, x)
                
                # Bind to module
                module.forward = new_forward.__get__(module, type(module))
                patched_count += 1
                video_patched += 1
    
    # Patch audio model
    if fusion_model.audio_model is not None:
        for name, module in fusion_model.audio_model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'scale_weight'):
                # Create new forward method
                def new_forward(self, x):
                    return fp8_linear_forward_patch(self, x)
                
                # Bind to module
                module.forward = new_forward.__get__(module, type(module))
                patched_count += 1
                audio_patched += 1
    
    logger.info(f"Applied FP8 monkey patch to {patched_count} Linear layers ({video_patched} video + {audio_patched} audio)")
    
    return fusion_model


def save_fusion_fp8_checkpoint(fusion_model, cache_path):
    """Save FP8-optimized fusion model to cache file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        state = {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()}
        tmp_path = cache_path + ".tmp"
        save_file(state, tmp_path)
        os.replace(tmp_path, cache_path)
        logger.info(f"Saved FP8 Fusion checkpoint to {cache_path}")
        print(f"[FP8 CACHE] Saved FP8 Fusion checkpoint to {cache_path}")
        return True
    except Exception as save_err:
        logger.warning(f"Failed to save FP8 checkpoint at {cache_path}: {save_err}")
        print(f"[FP8 CACHE] Failed to save FP8 checkpoint: {save_err}")
        return False


def load_fusion_fp8_checkpoint(fusion_model, cache_path):
    """Load FP8-optimized fusion model from cache file."""
    try:
        print(f"[FP8 CACHE] Loading cached FP8 checkpoint from {cache_path}...")
        state_dict = load_file(cache_path)
        
        # Ensure scale_weight buffers are registered
        _ensure_fp8_buffers(fusion_model, state_dict)
        
        # Load state dict
        missing, unexpected = fusion_model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys when loading cached FP8 checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading cached FP8 checkpoint: {unexpected}")
        
        print(f"[FP8 CACHE] Loaded cached FP8 checkpoint successfully")
        return True
    except Exception as load_err:
        logger.warning(f"Failed to load cached FP8 checkpoint: {load_err}")
        print(f"[FP8 CACHE] Failed to load cached FP8 checkpoint: {load_err}")
        return False


def _ensure_fp8_buffers(fusion_model, state_dict):
    """Ensure all scale_weight buffers are registered before loading state dict."""
    # Video model
    if fusion_model.video_model is not None:
        module_map = dict(fusion_model.video_model.named_modules())
        for key, value in state_dict.items():
            if key.startswith('video_model.') and key.endswith('.scale_weight'):
                module_name = key[len('video_model.'):-len('.scale_weight')]
                module = module_map.get(module_name)
                if module is not None and not hasattr(module, 'scale_weight'):
                    module.register_buffer('scale_weight', torch.empty_like(value))
    
    # Audio model
    if fusion_model.audio_model is not None:
        module_map = dict(fusion_model.audio_model.named_modules())
        for key, value in state_dict.items():
            if key.startswith('audio_model.') and key.endswith('.scale_weight'):
                module_name = key[len('audio_model.'):-len('.scale_weight')]
                module = module_map.get(module_name)
                if module is not None and not hasattr(module, 'scale_weight'):
                    module.register_buffer('scale_weight', torch.empty_like(value))


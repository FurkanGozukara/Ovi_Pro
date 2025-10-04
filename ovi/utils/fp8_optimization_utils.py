"""
FP8 optimization utilities for T5 text encoder.
Based on Musubi Tuner's approach for proper FP8 weight quantization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging

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


def optimize_t5_to_fp8(model, device='cuda', block_size=64):
    """
    Optimize T5 model to FP8 by quantizing Linear layer weights.
    
    Args:
        model: T5 encoder model
        device: Device for quantization computation
        block_size: Block size for per-block quantization
    
    Returns:
        model: Modified model with FP8 weights and scale parameters
        info: Dictionary with optimization statistics
    """
    fp8_dtype = torch.float8_e4m3fn
    max_value = calculate_fp8_maxval()
    min_value = -max_value
    
    optimized_count = 0
    total_params_before = 0
    total_params_after = 0
    
    logger.info("Starting FP8 optimization of T5 model...")
    
    # Find all Linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    logger.info(f"Found {len(linear_layers)} Linear layers to optimize")
    
    # Quantize each Linear layer
    for name, module in tqdm(linear_layers, desc="Quantizing to FP8"):
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
        
        # Free memory periodically
        if optimized_count % 10 == 0:
            torch.cuda.empty_cache()
    
    info = {
        'optimized_layers': optimized_count,
        'params_before_mb': total_params_before / (1024 * 1024),
        'params_after_mb': total_params_after / (1024 * 1024),
        'compression_ratio': total_params_before / total_params_after if total_params_after > 0 else 1.0
    }
    
    logger.info(f"FP8 optimization complete:")
    logger.info(f"  Optimized {optimized_count} Linear layers")
    logger.info(f"  Model size: {info['params_before_mb']:.1f} MB -> {info['params_after_mb']:.1f} MB")
    logger.info(f"  Compression ratio: {info['compression_ratio']:.2f}x")
    
    return model, info


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


def apply_fp8_monkey_patch(model):
    """
    Apply monkey patching to Linear layers with FP8 weights.
    
    Args:
        model: Model with FP8-optimized Linear layers
    
    Returns:
        model: Model with patched forward methods
    """
    patched_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'scale_weight'):
            # Create new forward method
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x)
            
            # Bind to module
            module.forward = new_forward.__get__(module, type(module))
            patched_count += 1
    
    logger.info(f"Applied FP8 monkey patch to {patched_count} Linear layers")
    
    return model


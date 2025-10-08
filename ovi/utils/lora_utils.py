"""
LoRA utilities for Ovi Fusion Engine
Handles LoRA loading, merging, and management for video+audio generation models
"""

import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import time
import statistics
import psutil

# LoRA merge profiling is enabled by default; set OVI_LORA_PROFILE=0 to disable
_lora_profile_flag = os.environ.get("OVI_LORA_PROFILE")
if _lora_profile_flag is None:
    PROFILE_LORA_MERGE = True
else:
    PROFILE_LORA_MERGE = _lora_profile_flag.lower() not in ("0", "false")


def scan_lora_files(lora_folders: List[str]) -> List[Tuple[str, str]]:
    """
    Scan lora folders for .safetensors and .pt files
    
    Args:
        lora_folders: List of folder paths to scan (e.g., ['lora', 'loras'])
    
    Returns:
        List of tuples: (display_name, full_path)
    """
    lora_files = []
    
    for folder in lora_folders:
        if not os.path.exists(folder):
            continue
            
        # Scan for .safetensors and .pt files
        for file in os.listdir(folder):
            if file.lower().endswith(('.safetensors', '.pt', '.pth')):
                full_path = os.path.join(folder, file)
                # Use filename without extension as display name
                display_name = os.path.splitext(file)[0]
                lora_files.append((display_name, full_path))
    
    # Sort by display name for consistent UI
    lora_files.sort(key=lambda x: x[0].lower())
    
    return lora_files


def load_lora_weights(lora_path: str) -> Dict[str, torch.Tensor]:
    """
    Load LoRA weights from a safetensors or pytorch file
    
    Args:
        lora_path: Path to LoRA file
    
    Returns:
        Dictionary of LoRA weights
    """
    logging.info(f"Loading LoRA from: {lora_path}")
    
    if lora_path.endswith('.safetensors'):
        lora_sd = load_file(lora_path, device='cpu')
    else:
        lora_sd = torch.load(lora_path, map_location='cpu')
    
    return lora_sd


def standardize_lora_keys(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Standardize LoRA key formats to match Ovi Fusion model structure
    
    Converts various LoRA key formats to Ovi's naming convention:
    - video_model.blocks.X.layer.weight
    - audio_model.blocks.Y.layer.weight
    
    Args:
        lora_sd: LoRA state dict with original keys
    
    Returns:
        LoRA state dict with standardized keys
    """
    standardized = {}
    
    for key, value in lora_sd.items():
        new_key = key
        
        # Remove common prefixes
        if new_key.startswith('diffusion_model.'):
            new_key = new_key.replace('diffusion_model.', '')
        if new_key.startswith('model.'):
            new_key = new_key.replace('model.', '', 1)
        
        # Handle different LoRA naming conventions
        # Convert lora_A/lora_B to standard format
        if '.lora_A.' in new_key or '.lora_down.' in new_key or '.down.' in new_key:
            new_key = new_key.replace('.lora_A.', '.lora_down.')
            new_key = new_key.replace('.down.', '.lora_down.')
        
        if '.lora_B.' in new_key or '.lora_up.' in new_key or '.up.' in new_key:
            new_key = new_key.replace('.lora_B.', '.lora_up.')
            new_key = new_key.replace('.up.', '.lora_up.')
        
        standardized[new_key] = value
    
    return standardized


def extract_lora_weights(lora_sd: Dict[str, torch.Tensor], base_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[float]]]:
    """
    Extract LoRA up/down weights and alpha for a specific layer
    
    Args:
        lora_sd: LoRA state dict
        base_key: Base layer key (e.g., 'video_model.blocks.0.attn.q.weight')
    
    Returns:
        Tuple of (lora_down, lora_up, alpha) or None if not found
    """
    # Remove .weight suffix if present
    base_key = base_key.replace('.weight', '')
    
    # Try different LoRA key patterns
    patterns = [
        (f'{base_key}.lora_down.weight', f'{base_key}.lora_up.weight', f'{base_key}.alpha'),
        (f'{base_key}.lora_A.weight', f'{base_key}.lora_B.weight', f'{base_key}.alpha'),
        (f'{base_key}.down.weight', f'{base_key}.up.weight', f'{base_key}.alpha'),
    ]
    
    for down_key, up_key, alpha_key in patterns:
        if down_key in lora_sd and up_key in lora_sd:
            lora_down = lora_sd[down_key]
            lora_up = lora_sd[up_key]
            alpha = lora_sd.get(alpha_key, None)
            
            # Convert alpha to float if it's a tensor
            if alpha is not None and isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            
            return (lora_down, lora_up, alpha)
    
    return None


def calculate_lora_weight(lora_down: torch.Tensor, lora_up: torch.Tensor, 
                          alpha: Optional[float], scale: float) -> torch.Tensor:
    """
    Calculate the LoRA weight matrix: scale * (alpha/rank) * (lora_up @ lora_down)
    
    Args:
        lora_down: LoRA down-projection matrix [rank, in_features]
        lora_up: LoRA up-projection matrix [out_features, rank]
        alpha: LoRA alpha value (scaling factor)
        scale: User-specified LoRA strength/scale
    
    Returns:
        Weight delta to add to base weight
    """
    # Flatten if needed (for Conv layers)
    if lora_down.dim() > 2:
        lora_down = lora_down.flatten(start_dim=1)
    if lora_up.dim() > 2:
        lora_up = lora_up.flatten(start_dim=1)
    
    # Calculate LoRA weight
    lora_weight = torch.mm(lora_up, lora_down)
    
    # Apply alpha scaling
    rank = lora_down.shape[0]
    if alpha is not None:
        lora_weight = lora_weight * (alpha / rank)
    
    # Apply user scale
    lora_weight = lora_weight * scale
    
    return lora_weight


def merge_lora_into_model(model: nn.Module, lora_sd: Dict[str, torch.Tensor], 
                          scale: float, model_dtype: torch.dtype = torch.bfloat16,
                          device: str = 'cpu') -> Tuple[int, int]:
    """
    Merge LoRA weights into model parameters layer by layer (memory efficient)
    
    Args:
        model: PyTorch model (video_model or audio_model)
        lora_sd: Standardized LoRA state dict
        scale: LoRA strength (0.0 to 2.0 typically)
        model_dtype: Target dtype for merged weights
        device: Device for processing ('cpu' recommended for merging)
    
    Returns:
        Tuple of (matched_layers, total_lora_keys)
    """
    if scale == 0.0:
        logging.warning(f"LoRA scale is 0.0, skipping merge")
        return (0, len(lora_sd))
    
    matched_layers = 0
    total_keys = 0
    
    # Get all named parameters for matching
    model_params = dict(model.named_parameters())
    model_state = model.state_dict()
    
    # Process each parameter
    logging.info(f"Merging LoRA into model (scale={scale})...")
    
    # Collect unique base keys from LoRA
    lora_base_keys = set()
    for key in lora_sd.keys():
        # Extract base key (remove .lora_down/.lora_up/.alpha)
        base_key = key.replace('.lora_down.weight', '').replace('.lora_up.weight', '') \
                     .replace('.lora_A.weight', '').replace('.lora_B.weight', '') \
                     .replace('.down.weight', '').replace('.up.weight', '') \
                     .replace('.alpha', '')
        lora_base_keys.add(base_key)
    
    total_keys = len(lora_base_keys)
    
    # Process each layer
    with tqdm(total=len(model_params), desc="Merging LoRA layers", leave=True) as pbar:
        for param_name, param in model_params.items():
            # Try to match with LoRA keys
            base_key = param_name.replace('.weight', '').replace('.bias', '')
            
            # Extract LoRA weights for this layer
            lora_weights = extract_lora_weights(lora_sd, base_key)
            
            if lora_weights is not None and '.weight' in param_name:
                lora_down, lora_up, alpha = lora_weights
                
                try:
                    # Move to processing device and dtype
                    param_data = param.data.to(device=device, dtype=model_dtype)
                    lora_down = lora_down.to(device=device, dtype=model_dtype)
                    lora_up = lora_up.to(device=device, dtype=model_dtype)
                    
                    # Calculate LoRA delta
                    lora_delta = calculate_lora_weight(lora_down, lora_up, alpha, scale)
                    
                    # Reshape if needed to match param shape
                    if lora_delta.shape != param_data.shape:
                        lora_delta = lora_delta.reshape(param_data.shape)
                    
                    # Merge: weight = weight + lora_delta
                    param_data = param_data + lora_delta
                    
                    # Update parameter in-place
                    param.data = param_data
                    
                    matched_layers += 1
                    
                    # Clean up intermediate tensors
                    del param_data, lora_down, lora_up, lora_delta
                    
                except Exception as e:
                    logging.warning(f"Failed to merge LoRA for layer {param_name}: {e}")
                    continue
            
            pbar.update(1)
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return (matched_layers, total_keys)


def merge_multiple_loras(model: nn.Module, lora_specs: List[Tuple[str, float]], 
                        model_dtype: torch.dtype = torch.bfloat16,
                        device: str = 'cpu') -> Dict[str, int]:
    """
    Merge multiple LoRAs into model sequentially
    
    Args:
        model: PyTorch model (video_model or audio_model)
        lora_specs: List of (lora_path, scale) tuples
        model_dtype: Target dtype for merged weights
        device: Device for processing ('cpu' recommended)
    
    Returns:
        Dictionary with merge statistics
    """
    if not lora_specs:
        return {"total_loras": 0, "matched_layers": 0, "total_keys": 0}
    
    total_matched = 0
    total_keys = 0
    
    timing_info = [] if PROFILE_LORA_MERGE else None
    process = psutil.Process() if PROFILE_LORA_MERGE else None
    torch_threads = torch.get_num_threads() if PROFILE_LORA_MERGE else None
    torch_interop_threads = torch.get_num_interop_threads() if PROFILE_LORA_MERGE else None
    cpu_count_logical = psutil.cpu_count(logical=True) if PROFILE_LORA_MERGE else None
    cpu_count_physical = psutil.cpu_count(logical=False) if PROFILE_LORA_MERGE else None
    affinity = process.cpu_affinity() if PROFILE_LORA_MERGE and hasattr(process, "cpu_affinity") else None

    for lora_index, (lora_path, scale) in enumerate(lora_specs, start=1):
        if scale == 0.0:
            logging.info(f"Skipping LoRA {os.path.basename(lora_path)} (scale=0.0)")
            continue
        
        logging.info(f"Loading LoRA: {os.path.basename(lora_path)} (scale={scale})")
        
        start_load = time.perf_counter() if PROFILE_LORA_MERGE else None
        # Load and standardize LoRA
        lora_sd = load_lora_weights(lora_path)
        load_duration = time.perf_counter() - start_load if PROFILE_LORA_MERGE else None

        start_standardize = time.perf_counter() if PROFILE_LORA_MERGE else None
        lora_sd = standardize_lora_keys(lora_sd)
        standardize_duration = time.perf_counter() - start_standardize if PROFILE_LORA_MERGE else None
        
        # Merge into model
        start_merge = time.perf_counter() if PROFILE_LORA_MERGE else None
        matched, keys = merge_lora_into_model(model, lora_sd, scale, model_dtype, device)
        merge_duration = time.perf_counter() - start_merge if PROFILE_LORA_MERGE else None
        
        total_matched += matched
        total_keys += keys
        
        logging.info(f"  ✓ Merged {matched}/{keys} layers")
        
        # Clean up
        del lora_sd

        if PROFILE_LORA_MERGE:
            timing_info.append({
                "index": lora_index,
                "name": os.path.basename(lora_path),
                "load_s": load_duration,
                "standardize_s": standardize_duration,
                "merge_s": merge_duration,
                "matched_layers": matched,
                "total_layers": keys,
                "proc_threads": process.num_threads() if process is not None else None,
            })

    if PROFILE_LORA_MERGE and timing_info:
        print("LoRA merge profiling summary (set OVI_LORA_PROFILE=0 to disable):")
        total_merge = sum(entry["merge_s"] for entry in timing_info if entry["merge_s"] is not None)
        total_load = sum(entry["load_s"] for entry in timing_info if entry["load_s"] is not None)
        total_standardize = sum(entry["standardize_s"] for entry in timing_info if entry["standardize_s"] is not None)
        per_lora_totals = [
            (entry["merge_s"] or 0.0) + (entry["load_s"] or 0.0) + (entry["standardize_s"] or 0.0)
            for entry in timing_info
        ]
        avg_total = statistics.mean(per_lora_totals) if per_lora_totals else 0.0
        max_total = max(per_lora_totals) if per_lora_totals else 0.0

        for entry in timing_info:
            print(
                f"  [LoRA {entry['index']}] {entry['name']} | "
                f"load: {entry['load_s']:.3f}s | "
                f"standardize: {entry['standardize_s']:.3f}s | "
                f"merge: {entry['merge_s']:.3f}s | "
                f"matched {entry['matched_layers']}/{entry['total_layers']} | "
                f"proc threads: {entry['proc_threads']}"
            )
        print(
            f"  Summary • total merge: {total_merge:.3f}s | total load: {total_load:.3f}s | "
            f"total standardize: {total_standardize:.3f}s | avg per LoRA: {avg_total:.3f}s | max per LoRA: {max_total:.3f}s"
        )
        if torch_threads is not None:
            print(
                f"  Torch threads: intra={torch_threads} inter-op={torch_interop_threads} | "
                f"Process threads: {process.num_threads() if process else 'n/a'} | "
                f"CPU cores: logical={cpu_count_logical} physical={cpu_count_physical}"
            )
        if affinity:
            print(f"  CPU affinity: {affinity}")
    
    return {
        "total_loras": len([s for s in lora_specs if s[1] != 0.0]),
        "matched_layers": total_matched,
        "total_keys": total_keys
    }


def get_lora_hash(lora_specs: List[Tuple[str, float, str]]) -> str:
    """
    Generate a hash string representing current LoRA configuration
    Used for detecting changes in LoRA selection

    Args:
        lora_specs: List of (lora_path, scale, layers) tuples

    Returns:
        Hash string representing this configuration
    """
    if not lora_specs:
        return "none"

    # Create deterministic string from specs
    spec_str = "|".join([f"{path}:{scale}:{layers}" for path, scale, layers in sorted(lora_specs)])

    # Use simple hash for comparison
    import hashlib
    return hashlib.md5(spec_str.encode()).hexdigest()

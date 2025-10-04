"""
Tiled VAE Decoding Utilities - Adapted from ComfyUI
https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/utils.py

This module provides universal N-dimensional tiled processing with overlap blending
for memory-efficient VAE decoding.

Copyright (C) 2024 Comfy
Licensed under GPL-3.0
"""

import torch
import math
import itertools
import logging


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    """
    Calculate the number of tiles needed for progress tracking.
    
    Args:
        width: Width dimension size
        height: Height dimension size
        tile_x: Tile width
        tile_y: Tile height
        overlap: Overlap between tiles
    
    Returns:
        Total number of tiles (rows × cols)
    """
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, 
                         out_channels=3, output_device="cpu", downscale=False, 
                         index_formulas=None, pbar=None):
    """
    Universal N-dimensional tiled processing with overlap blending.
    
    This function splits the input tensor into overlapping tiles, processes each tile
    with the provided function, and blends them back together using feathered masks
    for seamless results.
    
    Args:
        samples: Input tensor to process (shape: [batch, channels, *dims])
        function: Processing function to apply to each tile (e.g., VAE decode)
        tile: Tile size for each dimension (tuple/list or single value)
        overlap: Overlap size for blending (tuple/list or single value)
        upscale_amount: Scale factor applied by the function (default: 4)
        out_channels: Number of output channels
        output_device: Device for output tensor (e.g., "cpu", "cuda")
        downscale: If True, function downscales instead of upscales
        index_formulas: Custom scaling formulas per dimension (optional)
        pbar: Progress bar object for tracking (optional)
    
    Returns:
        Processed output tensor with seamless tile blending
    
    Example for Wan 2.2 VAE:
        Input: latents (1, 48, 31, 32, 62) [batch, channels, frames, h, w]
        tile=(31, 32, 32) means: all frames, 32×32 spatial tiles
        overlap=(1, 8, 8) means: 1 frame overlap, 8 pixel spatial overlap
        upscale_amount=16 (VAE upscales 16× spatially)
        out_channels=12 (3 RGB × 4 from patchify=2)
    """
    dims = len(tile)
    
    # Normalize upscale_amount to list
    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims
    
    # Normalize overlap to list
    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims
    
    # Use upscale_amount as default index_formulas
    if index_formulas is None:
        index_formulas = upscale_amount
    
    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims
    
    def get_upscale(dim, val):
        """Calculate upscaled dimension size"""
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val
    
    def get_downscale(dim, val):
        """Calculate downscaled dimension size"""
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up
    
    def get_upscale_pos(dim, val):
        """Calculate upscaled position"""
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val
    
    def get_downscale_pos(dim, val):
        """Calculate downscaled position"""
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up
    
    # Select appropriate scaling functions
    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos
    
    def mult_list_upscale(a):
        """Apply scaling to all dimensions"""
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out
    
    # Allocate output tensor
    output = torch.empty(
        [samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), 
        device=output_device
    )
    
    # Process each sample in the batch
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        
        # Check if entire input fits in a single tile
        if all(s.shape[d+2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue
        
        # Allocate accumulation tensors for weighted blending
        out = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), 
            device=output_device
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), 
            device=output_device
        )
        
        # Calculate tile positions for each dimension
        positions = [
            range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) 
            if s.shape[d+2] > tile[d] else [0] 
            for d in range(dims)
        ]
        
        # Process each tile
        for it in itertools.product(*positions):
            s_in = s
            upscaled = []
            
            # Extract tile for each dimension
            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))
            
            # Process tile
            ps = function(s_in).to(output_device)
            
            # Create feathered mask for smooth blending
            mask = torch.ones_like(ps)
            
            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                
                # Apply feathering at tile boundaries
                for t in range(feather):
                    a = (t + 1) / feather  # Linear gradient: 0 → 1
                    mask.narrow(d, t, 1).mul_(a)  # Fade in at start
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)  # Fade out at end
            
            # Accumulate weighted tile contributions
            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])
            
            o.add_(ps * mask)
            o_d.add_(mask)
            
            if pbar is not None:
                pbar.update(1)
        
        # Normalize by mask weights
        output[b:b+1] = out / out_div
    
    return output


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, 
                out_channels=3, output_device="cpu", pbar=None):
    """
    2D tiled processing wrapper for images.
    
    Args:
        samples: Input tensor (batch, channels, height, width)
        function: Processing function
        tile_x: Tile width
        tile_y: Tile height
        overlap: Overlap for blending
        upscale_amount: Scale factor
        out_channels: Number of output channels
        output_device: Output device
        pbar: Progress bar
    
    Returns:
        Processed tensor
    """
    return tiled_scale_multidim(
        samples, 
        function, 
        (tile_y, tile_x), 
        overlap=overlap, 
        upscale_amount=upscale_amount, 
        out_channels=out_channels, 
        output_device=output_device, 
        pbar=pbar
    )


def calculate_optimal_tile_size(latent_height, latent_width, vram_gb, safety_margin=0.8):
    """
    Calculate optimal tile size based on available VRAM.
    
    Args:
        latent_height: Height in latent space
        latent_width: Width in latent space
        vram_gb: Available VRAM in GB
        safety_margin: Safety factor (0.8 = use 80% of VRAM)
    
    Returns:
        Optimal tile size (power of 2)
    """
    # Rough estimate: Each latent pixel × 48 channels × 4 bytes (float32)
    # After decode: × 16 (upscale) × 3 (RGB) × 4 bytes
    bytes_per_latent_pixel = 48 * 4 * 16 * 3 * 4
    
    available_bytes = vram_gb * 1024**3 * safety_margin
    max_pixels = available_bytes / bytes_per_latent_pixel
    
    # Find largest power-of-2 tile that fits
    tile_size = 64
    while tile_size > 8:
        if tile_size * tile_size <= max_pixels:
            break
        tile_size //= 2
    
    logging.info(f"[TILED VAE] Calculated optimal tile size: {tile_size}×{tile_size} for {vram_gb:.1f}GB VRAM")
    return tile_size


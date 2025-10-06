# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    sageattn = None
    SAGE_ATTENTION_AVAILABLE = False

import warnings

# Global configuration for Sage Attention
_USE_SAGE_ATTENTION = False
_ATTENTION_DEBUG_PRINTED = False  # Flag to print debug info only once

def set_use_sage_attention(enabled: bool):
    """
    Set global flag to use Sage Attention for all attention operations.
    This must be called before model forward passes.
    
    Args:
        enabled: If True, use Sage Attention when available
    """
    global _USE_SAGE_ATTENTION, _ATTENTION_DEBUG_PRINTED
    _USE_SAGE_ATTENTION = enabled
    _ATTENTION_DEBUG_PRINTED = False  # Reset debug flag when configuration changes
    
    if enabled and SAGE_ATTENTION_AVAILABLE:
        print("[SAGE ATTENTION] Enabled - using Sage Attention for ~10% speedup & lower VRAM", flush=True)
    elif enabled and not SAGE_ATTENTION_AVAILABLE:
        print("[SAGE ATTENTION] Requested but not available - falling back to Flash Attention", flush=True)
        print("[SAGE ATTENTION] Install with: pip install sageattention", flush=True)

def get_use_sage_attention() -> bool:
    """Get current Sage Attention configuration."""
    global _USE_SAGE_ATTENTION
    return _USE_SAGE_ATTENTION

__all__ = [
    'flash_attention',
    'sage_attention',
    'attention',
    'attention_with_weights',
    'SAGE_ATTENTION_AVAILABLE',
    'set_use_sage_attention',
    'get_use_sage_attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    # CRITICAL: Check if Sage Attention is enabled and dispatch to it if so
    # The model calls flash_attention() directly, so we intercept here
    global _USE_SAGE_ATTENTION, _ATTENTION_DEBUG_PRINTED
    
    if _USE_SAGE_ATTENTION and SAGE_ATTENTION_AVAILABLE:
        # Print debug info once
        if not _ATTENTION_DEBUG_PRINTED:
            print("=" * 80, flush=True)
            print("[ATTENTION BACKEND] >>> SAGE ATTENTION <<<", flush=True)
            print("[ATTENTION BACKEND] Sage Attention is ACTIVE for all attention operations", flush=True)
            print("=" * 80, flush=True)
            _ATTENTION_DEBUG_PRINTED = True
        
        # Dispatch to Sage Attention
        return sage_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    
    # Otherwise, proceed with Flash Attention as normal
    if not _ATTENTION_DEBUG_PRINTED:
        print("=" * 80, flush=True)
        fa_version = 3 if FLASH_ATTN_3_AVAILABLE else 2 if FLASH_ATTN_2_AVAILABLE else None
        print(f"[ATTENTION BACKEND] Flash Attention {fa_version}", flush=True)
        print("=" * 80, flush=True)
        _ATTENTION_DEBUG_PRINTED = True
    
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
            
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
        
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def sage_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """
    Sage Attention implementation for faster inference with lower VRAM usage.
    
    Args:
        q:              [B, Lq, Nq, C1]. Query tensor
        k:              [B, Lk, Nk, C1]. Key tensor
        v:              [B, Lk, Nk, C2]. Value tensor. Nq must be divisible by Nk.
        q_lens:         [B]. Query sequence lengths
        k_lens:         [B]. Key sequence lengths
        dropout_p:      float. Dropout probability (not used during inference)
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        q_scale:        float. Additional query scaling
        causal:         bool. Whether to apply causal attention mask.
        dtype:          torch.dtype. Target dtype for computation
    
    Returns:
        Attention output tensor with shape [B, Lq, Nq, C2]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
    
    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    # Convert to half precision
    q = half(q)
    k = half(k)
    v = half(v)
    
    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale
    
    # Sage attention expects NHD format: (batch_size, seq_len, num_heads, head_dim)
    # Our input is already in BLHD format, which is the same as NHD
    
    # Handle variable length sequences by trimming if needed
    if q_lens is not None or k_lens is not None:
        # For variable length, we need to process each sequence individually
        # This is less efficient but necessary for proper handling
        warnings.warn(
            'Variable length sequences with Sage Attention may be slower. '
            'Consider using fixed-length sequences or Flash Attention for better performance.'
        )
        
        # Process each batch element separately
        outputs = []
        for i in range(b):
            q_len = q_lens[i].item() if q_lens is not None else lq
            k_len = k_lens[i].item() if k_lens is not None else lk
            
            q_i = q[i:i+1, :q_len]
            k_i = k[i:i+1, :k_len]
            v_i = v[i:i+1, :k_len]
            
            # Call sageattn with NHD layout
            out_i = sageattn(
                q_i, k_i, v_i,
                tensor_layout="NHD",
                is_causal=causal,
                sm_scale=softmax_scale
            )
            
            # Pad back to original length if needed
            if q_len < lq:
                pad_len = lq - q_len
                out_i = torch.nn.functional.pad(out_i, (0, 0, 0, 0, 0, pad_len))
            
            outputs.append(out_i)
        
        x = torch.cat(outputs, dim=0)
    else:
        # Fixed length - can process entire batch at once (fastest path)
        x = sageattn(
            q, k, v,
            tensor_layout="NHD",
            is_causal=causal,
            sm_scale=softmax_scale
        )
    
    # output
    return x.type(out_dtype)


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    average_for_q=False,
    total_video_latent_frames = 21
):
    """
    Compute attention with explicit attention weights for visualization.
    Returns both output and attention weights.
    """
    out_dtype = q.dtype
    
    # Handle sequence lengths
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    
    if q_lens is None:
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        # Ensure q_lens is on the same device as q
        q_lens = q_lens.to(q.device)
        
    if k_lens is None:
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        # Ensure k_lens is on the same device as k
        k_lens = k_lens.to(k.device)
    
    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale
    
    # Compute attention weights manually
    # q: [B, Lq, Nq, C], k: [B, Lk, Nk, C]
    scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
    
    # Compute scores: [B, Nq, Lq, Lk]
    scores = torch.einsum('blhd,bshd->bhls', q, k) * scale
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Mask for k_lens (columns)
    k_mask = torch.arange(lk, device=k.device).unsqueeze(0) >= k_lens.unsqueeze(1)  # [B, Lk]
    scores.masked_fill_(k_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # [B, 1, 1, Lk]
    
    # Mask for q_lens (rows) 
    q_mask = torch.arange(lq, device=q.device).unsqueeze(0) >= q_lens.unsqueeze(1)  # [B, Lq]
    scores.masked_fill_(q_mask.unsqueeze(1).unsqueeze(3), float('-inf'))  # [B, 1, Lq, 1]
    
    # Compute attention weights
    attn_weights = torch.softmax(scores, dim=-1)  # [B, Nq, Lq, Lk]
    assert attn_weights.shape[0] == 1, "Batch size > 1 not supported for attention visualization."
    
    # Average attention weights to reduce memory usage before returning
    # Average across batch dimension (should be 1) and query heads and query sequence length
    # This gives us attention weight per video token: [Lk]
    if average_for_q:
        #avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 3))  # [Lq]
        avg_attn_weights = torch.max(attn_weights, dim=3)[0].mean(dim=(0, 1))  # [Lq]
    else:
        if 0:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 1, 2))  # [Lk]
        elif 1:
            B, H, Lq, Lk = attn_weights.shape  # [1, H, Lq, Lk]
            per_frame_seq_len = Lk // total_video_latent_frames
            per_frame_aud_len = Lq // total_video_latent_frames

            avg_attn_weights = torch.zeros((Lk,), device=attn_weights.device, dtype=attn_weights.dtype)

            eps = 1e-8  # numerical stability
            for i in range(total_video_latent_frames):
                start_idx_v = i * per_frame_seq_len
                end_idx_v   = (i + 1) * per_frame_seq_len

                start_idx_a = i * per_frame_aud_len
                end_idx_a   = (i + 1) * per_frame_aud_len

                # attn_chunk: [H, La, Lv]
                attn_chunk = attn_weights[0, :, start_idx_a:end_idx_a, start_idx_v:end_idx_v]

                # ---- Head informativeness via (low) entropy over Lv ----
                # Normalize within the Lv slice per (head, query) to make a proper distribution
                p = attn_chunk / (attn_chunk.sum(dim=-1, keepdim=True) + eps)          # [H, La, Lv]
                entropy = -(p * (p + eps).log()).sum(dim=-1).mean(dim=1)               # [H]

                # Convert to positive head weights (lower entropy -> larger weight)
                saliency = 1.0 / (entropy + 1e-6)                                      # [H]
                head_w = saliency / (saliency.sum() + eps)                             # [H], sum=1

                # Reduce across audio queries first (pick strong responses), then weight heads
                per_head = torch.amax(attn_chunk, dim=1)                               # [H, Lv]
                weighted = (per_head * head_w[:, None]).sum(dim=0)                     # [Lv]

                avg_attn_weights[start_idx_v:end_idx_v] = weighted
        else:
            avg_attn_weights = torch.mean(attn_weights, dim=(0, 2)).max(dim=(0))[0]  # [Lk]
    
    # Compute output: [B, Lq, Nq, C]
    out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
    
    return out.to(out_dtype), avg_attn_weights.to(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    use_sage_attn=None,
):
    """
    Main attention dispatcher function.
    
    Args:
        use_sage_attn: If True and SAGE_ATTENTION_AVAILABLE, use Sage Attention.
                      If None, uses global configuration from set_use_sage_attention().
                      Otherwise falls back to Flash Attention or PyTorch SDPA.
    """
    # Use global configuration if not explicitly specified
    if use_sage_attn is None:
        use_sage_attn = _USE_SAGE_ATTENTION
    
    # Debug: Show attention backend selection (print only once per configuration change)
    global _ATTENTION_DEBUG_PRINTED
    if not _ATTENTION_DEBUG_PRINTED:
        print("=" * 80, flush=True)
        if use_sage_attn and SAGE_ATTENTION_AVAILABLE:
            backend = "SAGE ATTENTION"
            print(f"[ATTENTION BACKEND] >>> {backend} <<<", flush=True)
            print("[ATTENTION BACKEND] Sage Attention is ACTIVE for all attention operations", flush=True)
        elif use_sage_attn and not SAGE_ATTENTION_AVAILABLE:
            backend = "Flash Attention (Sage unavailable, fallback)"
            print(f"[ATTENTION BACKEND] {backend}", flush=True)
            print("[ATTENTION BACKEND] WARNING: Sage Attention was requested but not available", flush=True)
        elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
            fa_version = 3 if FLASH_ATTN_3_AVAILABLE else 2
            backend = f"Flash Attention {fa_version}"
            print(f"[ATTENTION BACKEND] {backend}", flush=True)
        else:
            backend = "PyTorch SDPA (fallback)"
            print(f"[ATTENTION BACKEND] {backend}", flush=True)
        print("=" * 80, flush=True)
        _ATTENTION_DEBUG_PRINTED = True
    
    # Priority: Sage Attention (if requested and available) > Flash Attention > PyTorch SDPA
    if use_sage_attn and SAGE_ATTENTION_AVAILABLE:
        return sage_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        # Fallback to PyTorch scaled_dot_product_attention
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out

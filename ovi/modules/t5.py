# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer
from safetensors.torch import load_file, save_file

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1,
                 skip_init=False,
                 build_parallel=False):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        def _create_block(_):
            return T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                   shared_pos, dropout)

        if build_parallel and num_layers > 1:
            max_workers = min((os.cpu_count() or 1), num_layers, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                blocks = list(executor.map(_create_block, range(num_layers)))
        else:
            blocks = [_create_block(i) for i in range(num_layers)]

        self.blocks = nn.ModuleList(blocks)
        self.norm = T5LayerNorm(dim)

        # initialize weights
        if not skip_init:
            self.apply(init_weights)

    def prepare_fp8(self, target_dtype=torch.bfloat16):
        """Prepare model for FP8 inference by keeping LayerNorm and Embeddings in target dtype"""
        for module in self.modules():
            if module.__class__.__name__ in ['T5LayerNorm', 'Embedding']:
                module.to(target_dtype)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1,
                 skip_init=False,
                 build_parallel=False):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        def _create_block(_):
            return T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                                    shared_pos, dropout)

        if build_parallel and num_layers > 1:
            max_workers = min((os.cpu_count() or 1), num_layers, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                blocks = list(executor.map(_create_block, range(num_layers)))
        else:
            blocks = [_create_block(i) for i in range(num_layers)]

        self.blocks = nn.ModuleList(blocks)
        self.norm = T5LayerNorm(dim)

        # initialize weights
        if not skip_init:
            self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1,
                 skip_init=False,
                 build_parallel=False):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout, skip_init=skip_init,
                                 build_parallel=build_parallel)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout, skip_init=skip_init,
                                 build_parallel=build_parallel)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        if not skip_init:
            self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
        fp8=False,
    ):
        self.text_len = text_len
        self.dtype = dtype if not fp8 else torch.float8_e4m3fn  # FP8 dtype for weights
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.fp8 = fp8

        if fp8:
            model = self._load_fp8_model(dtype, device, checkpoint_path)
        else:
            model = self._load_bf16_model(dtype, device, checkpoint_path)
        
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            # For BF16 GPU mode, model is already on target device (optimized loading)
            # For FP8 mode, model needs to be moved to device after quantization
            # For CPU mode, model is already on CPU
            if fp8:
                self.model.to(self.device)
            # BF16 mode: model already loaded to correct device, no need to move
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def _load_bf16_model(self, dtype, device, checkpoint_path):
        load_start = time.perf_counter()
        # For CPU-only mode, avoid parallel building to minimize RAM spikes
        # For GPU mode, use parallel building for faster loading
        is_cpu_only = (device == 'cpu' or (isinstance(device, str) and device == 'cpu'))
        
        # OPTIMIZATION: For GPU mode, load weights directly to GPU to avoid CPU→GPU transfer
        if is_cpu_only:
            # CPU mode: load to CPU as before
            map_location = 'cpu'
        else:
            # GPU mode: load directly to target GPU device
            map_location = f'cuda:{device}' if isinstance(device, int) else str(device)
        
        print(f"[T5 LOAD][BF16] Loading weights directly to {map_location} (optimized path)")
        
        # Create model structure with meta device to avoid allocating memory
        # We'll load weights directly in load_state_dict
        # Note: build_parallel=True uses ThreadPoolExecutor to create 24 layers in parallel
        # This reduces structure creation time from ~15-20s to ~8-12s
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device='meta',  # Use meta device to avoid memory allocation
            skip_init=True,
            build_parallel=True).eval().requires_grad_(False)  # Always use parallel for speed
        structure_time = time.perf_counter() - load_start
        print(f"[T5 LOAD][BF16] Structure created in {structure_time:.2f}s (24 layers built in parallel)")
        
        logging.info(f'loading {checkpoint_path}')
        weights_start = time.perf_counter()
        
        # Load weights with memory mapping for reduced RAM usage
        # NOTE: torch.load() with .pth files MUST read to CPU RAM first (PyTorch limitation)
        # map_location only controls where tensors END UP after deserialization
        # mmap=True streams from disk to reduce peak RAM usage
        try:
            # PyTorch 2.0+ supports mmap for reduced RAM pressure
            state_dict = torch.load(checkpoint_path, map_location=map_location, mmap=True)
            print(f"[T5 LOAD][BF16] Using memory-mapped loading (reduced RAM pressure)")
        except TypeError:
            # Fallback for PyTorch < 2.0 (no mmap support)
            state_dict = torch.load(checkpoint_path, map_location=map_location)
            print(f"[T5 LOAD][BF16] Memory mapping not available (PyTorch < 2.0)")
        
        model.load_state_dict(state_dict, assign=True)  # assign=True for meta device
        del state_dict
        
        weights_time = time.perf_counter() - weights_start
        total_time = time.perf_counter() - load_start
        print(f"[T5 LOAD][BF16] Weights loaded directly to {map_location} in {weights_time:.2f}s (total {total_time:.2f}s)")
        return model

    def _load_fp8_model(self, dtype, device, checkpoint_path):
        from ovi.utils.fp8_optimization_utils import optimize_t5_to_fp8, apply_fp8_monkey_patch

        logging.info("=" * 80)
        logging.info("Initializing T5 with Scaled FP8 Quantization (Musubi Tuner approach)")
        logging.info("=" * 80)

        load_start = time.perf_counter()
        cache_path = self._get_fp8_cache_path(checkpoint_path)

        logging.info(f'Loading T5 model from {checkpoint_path} in bf16 on CPU...')
        structure_start = time.perf_counter()
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device='cpu',
            skip_init=True,
            build_parallel=True).eval().requires_grad_(False)
        structure_time = time.perf_counter() - structure_start
        print(f"[T5 LOAD][FP8] Encoder structure created in {structure_time:.2f}s")

        if cache_path and os.path.exists(cache_path):
            print(f"[FP8 CACHE] Found cached FP8 checkpoint: {cache_path}")
            try:
                cache_load_start = time.perf_counter()
                model = self._load_from_cached_fp8(model, cache_path, device, dtype, apply_fp8_monkey_patch)
                if model is not None:
                    print("[FP8 CACHE] Loaded cached FP8 T5 encoder successfully")
                    cache_total = time.perf_counter() - cache_load_start
                    total_time = time.perf_counter() - load_start
                    print(f"[T5 LOAD][FP8] Cached weights applied in {cache_total:.2f}s (total {total_time:.2f}s)")
                    return model
            except Exception as load_err:
                logging.warning(f"Failed to load cached FP8 checkpoint ({load_err}). Rebuilding FP8 weights...")
                print(f"[FP8 CACHE] Failed to load cached FP8 checkpoint ({load_err}). Regenerating...")
        else:
            if cache_path:
                print(f"[FP8 CACHE] Cached FP8 checkpoint not found at {cache_path}")
            else:
                print("[FP8 CACHE] Cache path could not be determined; proceeding with FP8 conversion")

        weights_start = time.perf_counter()
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        logging.info("Model loaded in bf16 on CPU")
        weights_time = time.perf_counter() - weights_start
        print(f"[T5 LOAD][FP8] BF16 weights loaded from disk in {weights_time:.2f}s")

        logging.info(f"Moving model to {device}...")
        model = model.to(device)

        cuda_device = self._resolve_cuda_device(device)
        if cuda_device is not None and torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated(cuda_device) / 1e9
            logging.info(f"VRAM before FP8 optimization: {vram_before:.2f} GB")
        else:
            vram_before = None

        logging.info("Quantizing Linear layer weights to FP8 with per-block scaling...")
        quant_start = time.perf_counter()
        model, fp8_info = optimize_t5_to_fp8(model, device=device, block_size=64)
        quant_time = time.perf_counter() - quant_start
        print(f"[T5 LOAD][FP8] FP8 quantization completed in {quant_time:.2f}s")

        logging.info("Applying FP8 forward pass monkey patches...")
        patch_start = time.perf_counter()
        model = apply_fp8_monkey_patch(model)
        patch_time = time.perf_counter() - patch_start
        print(f"[T5 LOAD][FP8] FP8 monkey patching completed in {patch_time:.2f}s")

        logging.info("Ensuring LayerNorm and Embeddings remain in bf16...")
        prep_start = time.perf_counter()
        model.prepare_fp8(dtype)
        prep_time = time.perf_counter() - prep_start
        print(f"[T5 LOAD][FP8] FP8 layer prep completed in {prep_time:.2f}s")

        if cuda_device is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(cuda_device)
            vram_after = torch.cuda.memory_allocated(cuda_device) / 1e9
            vram_saved = vram_before - vram_after if vram_before is not None else 0.0
            logging.info("=" * 80)
            logging.info("FP8 Optimization Complete!")
            logging.info(f"  VRAM before: {vram_before:.2f} GB")
            logging.info(f"  VRAM after:  {vram_after:.2f} GB")
            logging.info(f"  VRAM saved:  {vram_saved:.2f} GB ({(vram_saved / vram_before * 100) if vram_before else 0:.1f}%)")
            logging.info(f"  Compression: {fp8_info['compression_ratio']:.2f}x")
            logging.info("=" * 80)

        if cache_path:
            save_start = time.perf_counter()
            self._save_fp8_checkpoint(model, cache_path)
            save_time = time.perf_counter() - save_start
            print(f"[T5 LOAD][FP8] Cached FP8 checkpoint written in {save_time:.2f}s")

        total_time = time.perf_counter() - load_start
        print(f"[T5 LOAD][FP8] Total FP8 load time {total_time:.2f}s")

        return model

    def _get_fp8_cache_path(self, checkpoint_path):
        if not checkpoint_path:
            return None
        abs_checkpoint = os.path.abspath(checkpoint_path)
        ckpt_dir = os.path.dirname(abs_checkpoint)
        if not ckpt_dir:
            return None
        return os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-fp8_scaled.safetensors")

    def _save_fp8_checkpoint(self, model, cache_path):
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            tmp_path = cache_path + ".tmp"
            save_file(state, tmp_path)
            os.replace(tmp_path, cache_path)
            logging.info(f"Saved FP8 T5 checkpoint to {cache_path}")
            print(f"[FP8 CACHE] Saved FP8 T5 checkpoint to {cache_path}")
        except Exception as save_err:
            logging.warning(f"Failed to save FP8 checkpoint at {cache_path}: {save_err}")
            print(f"[FP8 CACHE] Failed to save FP8 checkpoint at {cache_path}: {save_err}")

    def _ensure_fp8_buffers(self, model, state_dict):
        module_map = dict(model.named_modules())
        for key, value in state_dict.items():
            if key.endswith('scale_weight'):
                module_name = key.rsplit('.', 1)[0]
                module = module_map.get(module_name)
                if module is not None and not hasattr(module, 'scale_weight'):
                    module.register_buffer('scale_weight', torch.empty_like(value))

    def _resolve_cuda_device(self, device):
        if isinstance(device, torch.device):
            return device if device.type == 'cuda' else None
        if isinstance(device, int):
            return torch.device(f'cuda:{device}')
        if isinstance(device, str):
            if device.startswith('cuda'):
                return torch.device(device)
        return None

    def _load_from_cached_fp8(self, model, cache_path, device, dtype, apply_fp8_monkey_patch_fn):
        load_start = time.perf_counter()
        state_dict = load_file(cache_path)
        load_time = time.perf_counter() - load_start
        print(f"[T5 LOAD][FP8] Cached weights file read in {load_time:.2f}s")
        self._ensure_fp8_buffers(model, state_dict)

        apply_start = time.perf_counter()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        apply_time = time.perf_counter() - apply_start
        print(f"[T5 LOAD][FP8] Cached weights applied to model in {apply_time:.2f}s")
        if missing:
            logging.warning(f"Missing keys when loading cached FP8 checkpoint: {missing}")
            print(f"[FP8 CACHE] Missing keys when loading cached FP8 checkpoint: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading cached FP8 checkpoint: {unexpected}")
            print(f"[FP8 CACHE] Unexpected keys when loading cached FP8 checkpoint: {unexpected}")

        model = model.to(device)
        move_time = time.perf_counter() - apply_start
        print(f"[T5 LOAD][FP8] Cached model moved to device in {move_time:.2f}s")

        monkey_patch_start = time.perf_counter()
        model = apply_fp8_monkey_patch_fn(model)
        monkey_patch_time = time.perf_counter() - monkey_patch_start
        print(f"[T5 LOAD][FP8] Cached model monkey patching in {monkey_patch_time:.2f}s")

        prepare_start = time.perf_counter()
        model.prepare_fp8(dtype)
        prepare_time = time.perf_counter() - prepare_start
        print(f"[T5 LOAD][FP8] Cached model layer prep in {prepare_time:.2f}s")

        return model

import os
import sys
import uuid
import cv2
import glob
import torch
import logging
from textwrap import indent
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible

DEFAULT_CONFIG = OmegaConf.load('ovi/configs/inference/inference_fusion.yaml')

class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16, blocks_to_swap=0, cpu_offload=None, video_latent_length=None, audio_latent_length=None):
        # Store config and defer model loading
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.target_dtype = target_dtype
        self.config = config
        self.blocks_to_swap = blocks_to_swap
        
        # Auto-enable CPU offload when block swap is used (optimal memory management)
        if blocks_to_swap > 0 and cpu_offload is None:
            cpu_offload = True
            logging.info("Block swap enabled - auto-enabling CPU offload for optimal memory management")
        
        # Use provided cpu_offload parameter, otherwise fall back to config
        self.cpu_offload = cpu_offload if cpu_offload is not None else (config.get("cpu_offload", False) or config.get("mode") == "t2i2v")
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Models will be moved to CPU between operations")

        # Defer model loading until first generation
        self.model = None
        self.vae_model_video = None
        self.vae_model_audio = None
        self.text_model = None
        self.image_model = None
        self.audio_latent_channel = None
        self.video_latent_channel = None
        # Use provided latent lengths or defaults
        self.audio_latent_length = audio_latent_length if audio_latent_length is not None else 157
        self.video_latent_length = video_latent_length if video_latent_length is not None else 31
        self._vae_device = None
        
        # T5 configuration (set during first generation)
        self.fp8_t5 = False
        self.cpu_only_t5 = False
        
        # Fusion Model FP8 configuration (set during first generation)
        self.fp8_base_model = False
        
        # Sage Attention configuration (set during first generation)
        self.use_sage_attention = False
        
        # VAE Tiled Decoding Configuration
        self.vae_tiled_decode = False
        self.vae_tile_size = 32  # Latent space tile size (32 latent = 512 pixels after 16x upscale)
        self.vae_tile_overlap = 8  # Overlap for seamless blending (8 latent = 128 pixels)
        self.vae_tile_temporal = 31  # Process all frames together (no temporal tiling by default)
        
        # LoRA Configuration
        self.current_lora_hash = "none"  # Track current LoRA configuration for smart reload
        self.lora_specs = []  # List of (lora_path, scale) tuples

        # Load VAEs immediately (they're lightweight) - but defer if block swap is enabled
        # Block swap requires special loading sequence, so defer all model loading
        if self.blocks_to_swap == 0:
            self._load_vaes()

            # Load image model if needed
            if config.get("mode") == "t2i2v":
                logging.info(f"Loading Flux Krea for first frame generation...")
                self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
                self.image_model.enable_model_cpu_offload(gpu_id=self.device)

        logging.info(f"OVI Fusion Engine initialized with lazy loading.")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Block Swap: {self.blocks_to_swap} blocks")
        logging.info(f"  CPU Offload: {self.cpu_offload}")
        if self.blocks_to_swap > 0:
            logging.info(f"  Block swap will keep {self.blocks_to_swap} transformer blocks on CPU, loading only active blocks to GPU during inference")

    def _encode_text_and_cleanup(self, text_prompt, video_negative_prompt, audio_negative_prompt, delete_text_encoder=True, use_subprocess=True):
        """
        Encode text prompts and optionally delete T5 to save VRAM during generation.
        This is the FIRST operation after loading models.
        Uses instance variables self.fp8_t5 and self.cpu_only_t5 for configuration.
        
        Args:
            text_prompt: Positive text prompt
            video_negative_prompt: Negative prompt for video
            audio_negative_prompt: Negative prompt for audio
            delete_text_encoder: If True, delete T5 after encoding to save memory
            use_subprocess: If True and delete_text_encoder=True, run T5 encoding in subprocess for 100% guaranteed memory cleanup
        """
        # CRITICAL: Only use subprocess if no other heavy models are loaded yet
        # This prevents duplicate memory usage (T5 in subprocess + fusion model in main process)
        can_use_subprocess = (
            delete_text_encoder and 
            use_subprocess and 
            self.model is None and  # Fusion model not loaded yet
            self.text_model is None  # T5 not already loaded
        )
        
        if can_use_subprocess:
            print("=" * 80)
            print("T5 SUBPROCESS MODE ENABLED (FIRST GENERATION)")
            print("Running T5 encoding in separate process for 100% guaranteed memory cleanup")
            print("No other models loaded yet - optimal for subprocess isolation")
            print("=" * 80)
            
            # Import the subprocess function from premium.py
            # We do this dynamically to avoid circular imports
            try:
                # Try to import from the parent directory (where premium.py is)
                import sys
                import os
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from premium import run_t5_encoding_subprocess
                
                # Run T5 encoding in subprocess - returns embeddings as CPU tensors
                text_embeddings_cpu = run_t5_encoding_subprocess(
                    text_prompt=text_prompt,
                    video_negative_prompt=video_negative_prompt,
                    audio_negative_prompt=audio_negative_prompt,
                    fp8_t5=self.fp8_t5,
                    cpu_only_t5=self.cpu_only_t5
                )
                
                # Move embeddings to target device and dtype
                text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings_cpu]
                
                # No need to delete T5 - it was already deleted in the subprocess
                # OS has freed ALL T5 memory automatically when subprocess exited
                print("=" * 80)
                print("T5 SUBPROCESS COMPLETED")
                print("T5 memory fully reclaimed by OS - 100% cleanup guaranteed")
                print("=" * 80)
                
                return text_embeddings
                
            except Exception as e:
                print(f"[WARNING] T5 subprocess mode failed: {e}")
                print("[WARNING] Falling back to in-process T5 encoding with manual deletion")
                # Fall back to regular in-process encoding
                use_subprocess = False
        elif delete_text_encoder and use_subprocess and self.model is not None:
            # Models already loaded - subprocess would cause duplicate memory usage
            print("=" * 80)
            print("T5 IN-PROCESS MODE (SUBSEQUENT GENERATION)")
            print("Fusion model already loaded - using in-process T5 encoding to avoid duplicate memory usage")
            print("Subprocess mode only used on first generation when no models are loaded")
            print("=" * 80)
        
        # Original in-process encoding logic (fallback or when subprocess is disabled)
        # Check if T5 needs to be reloaded (was deleted in previous generation)
        if self.text_model is None:
            device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)

            if self.cpu_only_t5 and self.fp8_t5:
                print("=" * 80)
                print("T5 CPU-ONLY + SCALED FP8: Loading FP8 T5 on CPU for maximum memory efficiency")
                print("Expected RAM savings: ~50% (~2.5GB saved) compared to full precision on CPU")
                print("Text encoding will be slower but uses minimal RAM")
                print("=" * 80)
            elif self.cpu_only_t5:
                print("=" * 80)
                print("T5 CPU-ONLY MODE: Loading T5 on CPU for CPU inference")
                print("This saves VRAM but text encoding will be slower")
                print("=" * 80)
            elif self.fp8_t5:
                print("=" * 80)
                print("SCALED FP8 T5: Loading T5 in Scaled FP8 format")
                print("Expected VRAM savings: ~50% (~5-6GB saved)")
                print("=" * 80)
            else:
                print("Loading T5 text encoder directly to GPU for encoding...")
            
            # Load T5 with FP8 or CPU-only options from instance variables
            self.text_model = init_text_model(
                self.config.ckpt_dir, 
                rank=device_idx, 
                fp8=self.fp8_t5, 
                cpu_only=self.cpu_only_t5
            )
            
            if not self.cpu_only_t5 and not self.fp8_t5:
                print("T5 text encoder loaded directly to GPU")

        # Encode text embeddings
        # For CPU-only T5, encoding happens on CPU, then we move embeddings to GPU
        print(f"Encoding text prompts...")
        encode_device = 'cpu' if self.cpu_only_t5 else self.device

        # Ensure T5 model is on the correct device before encoding
        # (it might have been offloaded to CPU after the previous generation)
        if hasattr(self.text_model, 'model') and self.text_model.model is not None:
            current_device = next(self.text_model.model.parameters()).device
            target_device = torch.device('cpu') if self.cpu_only_t5 else self.device
            if current_device != target_device:
                print(f"Moving T5 model from {current_device} to {target_device} for encoding...")
                self.text_model.model = self.text_model.model.to(target_device)

        text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], encode_device)
        
        # Move embeddings to target device and dtype
        text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]
        
        if self.cpu_only_t5:
            print("Text embeddings encoded on CPU and moved to GPU")

        # Handle T5 cleanup based on settings
        if delete_text_encoder:
            if self.cpu_only_t5:
                print("Deleting T5 text encoder to free RAM...")
                # Measure RAM before deletion
                import psutil
                import os
                process = psutil.Process(os.getpid())
                before_delete_ram = process.memory_info().rss / 1e9
            else:
                print("Deleting T5 text encoder to free VRAM...")
                if torch.cuda.is_available():
                    before_delete_vram = torch.cuda.memory_allocated(self.device) / 1e9

            # Ensure T5 is properly deleted from both RAM and VRAM
            if hasattr(self, 'text_model') and self.text_model is not None:
                # Clear any GPU references first
                if hasattr(self.text_model, 'model') and self.text_model.model is not None:
                    self.text_model.model = None

                # Delete the text model object
                del self.text_model
                self.text_model = None

                # Force garbage collection to free RAM
                import gc
                gc.collect()

                # Clear CUDA cache to free VRAM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(self.device)

            # Report memory freed based on where T5 was loaded
            if self.cpu_only_t5:
                # Measure RAM after deletion
                after_delete_ram = process.memory_info().rss / 1e9
                freed_ram = before_delete_ram - after_delete_ram
                print(f"T5 deleted. RAM freed: {freed_ram:.2f} GB")
                print(f"Current RAM usage: {after_delete_ram:.2f} GB")
            elif torch.cuda.is_available():
                after_delete_vram = torch.cuda.memory_allocated(self.device) / 1e9
                freed_vram = before_delete_vram - after_delete_vram
                print(f"T5 deleted. VRAM freed: {freed_vram:.2f} GB")
                print(f"Current VRAM: {after_delete_vram:.2f} GB")
        else:
            # Keep T5 but offload to CPU if CPU offloading is enabled (and not already CPU-only)
            if self.cpu_offload and not self.cpu_only_t5:
                print("Keeping T5 and offloading to CPU for future reuse...")
                self.offload_to_cpu(self.text_model.model)
                print("T5 text encoder offloaded to CPU")
            elif self.cpu_only_t5:
                print("Keeping T5 on CPU (already in CPU-only mode)")
            else:
                print("Keeping T5 in GPU memory (CPU offload disabled)")

        return text_embeddings

    def _load_vaes(self):
        """Load VAEs which are lightweight and always needed"""
        # Convert device to int for VAE init functions (they expect int device index)
        device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)
        
        if self.vae_model_video is None:
            vae_model_video = init_wan_vae_2_2(self.config.ckpt_dir, rank=device_idx)
            vae_model_video.model.requires_grad_(False).eval()
            vae_model_video.model = vae_model_video.model.bfloat16()
            self.vae_model_video = vae_model_video

        if self.vae_model_audio is None:
            vae_model_audio = init_mmaudio_vae(self.config.ckpt_dir, rank=device_idx)
            vae_model_audio.requires_grad_(False).eval()
            self.vae_model_audio = vae_model_audio.bfloat16()

        if self.cpu_offload:
            self._offload_vaes_to_cpu()

    def _apply_loras_to_model(self, model, lora_specs):
        """
        Apply LoRAs to the fusion model after loading
        This happens AFTER model is fully loaded but BEFORE any generation/block swapping

        Args:
            model: Loaded fusion model
            lora_specs: List of (lora_path, scale, layers) tuples
        """
        if not lora_specs:
            return

        from ovi.utils.lora_utils import merge_multiple_loras

        print("=" * 80)
        print("APPLYING LORAS TO FUSION MODEL")
        print(f"  Number of LoRAs: {len(lora_specs)}")
        for lora_path, scale, layers in lora_specs:
            print(f"  - {os.path.basename(lora_path)}: scale={scale}, layers={layers}")
        print("=" * 80)

        # Separate LoRAs by target layers
        video_loras = []
        audio_loras = []

        for lora_path, scale, layers in lora_specs:
            if layers == "Video Layers":
                video_loras.append((lora_path, scale))
            elif layers == "Sound Layers":
                audio_loras.append((lora_path, scale))
            elif layers == "Both":
                video_loras.append((lora_path, scale))
                audio_loras.append((lora_path, scale))

        # Merge LoRAs into video model
        stats_video = {'matched_layers': 0, 'total_loras': 0}
        if model.video_model is not None and video_loras:
            print(f"\n[VIDEO MODEL] Merging {len(video_loras)} LoRA(s)...")
            stats_video = merge_multiple_loras(
                model.video_model,
                video_loras,
                model_dtype=self.target_dtype,
                device='cpu'  # Merge on CPU for memory efficiency
            )
            print(f"[VIDEO MODEL] ✓ Merged {stats_video['matched_layers']} layers from {stats_video['total_loras']} LoRAs")

        # Merge LoRAs into audio model
        stats_audio = {'matched_layers': 0, 'total_loras': 0}
        if model.audio_model is not None and audio_loras:
            print(f"\n[AUDIO MODEL] Merging {len(audio_loras)} LoRA(s)...")
            stats_audio = merge_multiple_loras(
                model.audio_model,
                audio_loras,
                model_dtype=self.target_dtype,
                device='cpu'  # Merge on CPU for memory efficiency
            )
            print(f"[AUDIO MODEL] ✓ Merged {stats_audio['matched_layers']} layers from {stats_audio['total_loras']} LoRAs")

        print("=" * 80)
        print("LORA MERGING COMPLETE")
        print(f"  Video model: {stats_video.get('matched_layers', 0)} layers merged from {len(video_loras)} LoRA(s)")
        print(f"  Audio model: {stats_audio.get('matched_layers', 0)} layers merged from {len(audio_loras)} LoRA(s)")
        print(f"  Total LoRAs applied: {len(lora_specs)}")
        print("=" * 80)
        print("✓ SUCCESS: LoRAs successfully applied to fusion model")
        print("✓ Model is ready for generation with LoRA modifications")
        print("=" * 80)
    
    def _load_models(self, no_block_prep=False, load_text_encoder=True):
        """Lazy load the heavy models on first generation request"""
        if self.model is not None:
            return  # Already loaded

        print("=" * 80)
        print("Loading OVI models for first generation...")
        print(f"  Block Swap: {self.blocks_to_swap} blocks")
        print(f"  CPU Offload: {self.cpu_offload}")
        print("=" * 80)

        # Track initial VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            initial_vram = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"Initial VRAM: {initial_vram:.2f} GB")

        # Load VAEs if not already loaded (deferred when block swap is enabled)
        if self.blocks_to_swap > 0:
            self._load_vaes()
        
        # ===================================================================
        # OPTIMIZATION: Load T5 text encoder FIRST
        # This prevents having both T5 (~24GB) and Fusion (~45GB) in RAM at same time
        # ===================================================================
        if load_text_encoder and self.text_model is None:
            device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)

            # Load T5 with FP8/CPU-only options from instance variables
            if self.cpu_only_t5 and self.fp8_t5:
                print("=" * 80)
                print("T5 CPU-ONLY + SCALED FP8: Loading FP8 T5 on CPU for maximum memory efficiency")
                print("Expected RAM savings: ~50% (~2.5GB saved) compared to full precision on CPU")
                print("Text encoding will be slower but uses minimal RAM")
                print("=" * 80)
            elif self.cpu_only_t5:
                print("=" * 80)
                print("T5 CPU-ONLY MODE: Loading T5 on CPU for CPU inference")
                print("This saves VRAM but text encoding will be slower")
                print("=" * 80)
            elif self.fp8_t5:
                print("=" * 80)
                print("SCALED FP8 T5: Loading T5 in Scaled FP8 format")
                print("Expected VRAM savings: ~50% (~5-6GB saved)")
                print("=" * 80)
            else:
                print(f"Loading T5 text encoder directly to GPU (BEFORE fusion model to save RAM)...")
            
            self.text_model = init_text_model(
                self.config.ckpt_dir, 
                rank=device_idx,
                fp8=self.fp8_t5,
                cpu_only=self.cpu_only_t5
            )
            
            if not self.cpu_only_t5 and not self.fp8_t5:
                print("T5 text encoder loaded directly to GPU")
            print("=" * 80)

            # Load image model if needed
            if self.config.get("mode") == "t2i2v":
                print(f"Loading Flux Krea for first frame generation...")
                self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
                self.image_model.enable_model_cpu_offload(gpu_id=self.device)

            # RETURN EARLY - Don't load fusion model yet!
            print("T5 loaded. Fusion model will load AFTER text encoding.")
            return

        # === CRITICAL: Follow musubi-tuner optimal loading pattern ===
        
        # Step 1: Create model on meta device (no VRAM usage)
        meta_init = True
        device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)
        print("Step 1/6: Creating model structure on meta device...")
        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device_idx, meta_init=meta_init)
        
        # Step 2: Load checkpoint weights to CPU (no VRAM usage)
        checkpoint_path = os.path.join(self.config.ckpt_dir, "Ovi", "model.safetensors")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {self.config.ckpt_dir}")
        
        print("Step 2/6: Loading checkpoint weights to CPU...")
        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)
        model.set_rope_params()
        
        # Step 3: Apply LoRAs BEFORE dtype conversion and FP8 optimization!
        # LoRAs should be merged into bf16 model, then converted to FP8 if needed
        if self.lora_specs:
            print("Step 3a/7: Applying LoRAs to model on CPU (before dtype conversion)...")
            self._apply_loras_to_model(model, self.lora_specs)
        
        # Step 3b: Apply FP8 optimization AFTER LoRA merge!
        # This ensures FP8 checkpoint includes LoRA-merged weights
        if self.fp8_base_model:
            print("=" * 80)
            print("APPLYING FP8 OPTIMIZATION TO FUSION MODEL")
            print("  Target: Video + Audio Transformer Blocks")
            print("  Format: FP8 E4M3 with per-block scaling (block_size=64)")
            print("  Expected VRAM savings: ~50% for transformer weights")
            print("=" * 80)
            
            # Check for cached FP8 checkpoint
            cache_path = os.path.join(self.config.ckpt_dir, "Ovi", "model_fp8_scaled.safetensors")
            
            if os.path.exists(cache_path):
                print("[FP8 CACHE] Loading cached FP8 checkpoint...")
                from ovi.utils.fp8_fusion_optimization_utils import (
                    load_fusion_fp8_checkpoint,
                    apply_fusion_fp8_monkey_patch
                )
                
                import time
                load_start = time.perf_counter()
                
                success = load_fusion_fp8_checkpoint(model, cache_path)
                
                if success:
                    # Apply monkey patches for FP8 forward pass
                    model = apply_fusion_fp8_monkey_patch(model)
                    
                    load_time = time.perf_counter() - load_start
                    print(f"[FP8 CACHE] Loaded and patched cached FP8 checkpoint in {load_time:.2f}s")
                    print("[FP8] Fusion model FP8 optimization complete (from cache)")
                else:
                    print("[FP8 CACHE] Cache load failed, falling back to quantization...")
                    # Fall through to quantization below
                    success = False
            else:
                print(f"[FP8 CACHE] No cache found at {cache_path}")
                success = False
            
            if not success:
                print("[FP8] Quantizing Fusion model to FP8 on CPU...")
                from ovi.utils.fp8_fusion_optimization_utils import (
                    optimize_fusion_model_to_fp8,
                    apply_fusion_fp8_monkey_patch,
                    save_fusion_fp8_checkpoint
                )
                
                # Convert to bfloat16 first (quantization expects this)
                print(f"[FP8] Converting model to {self.target_dtype} before quantization...")
                model = model.to(device="cpu", dtype=self.target_dtype).eval()
                
                import time
                quant_start = time.perf_counter()
                
                # Quantize on CPU (avoids VRAM spike)
                model, fp8_info = optimize_fusion_model_to_fp8(model, device='cpu', block_size=64)
                
                quant_time = time.perf_counter() - quant_start
                print(f"[FP8] Quantization completed in {quant_time:.2f}s")
                
                # Apply monkey patches for FP8 forward pass
                patch_start = time.perf_counter()
                model = apply_fusion_fp8_monkey_patch(model)
                patch_time = time.perf_counter() - patch_start
                print(f"[FP8] Monkey patching completed in {patch_time:.2f}s")
                
                # Save to cache for next time
                save_fusion_fp8_checkpoint(model, cache_path)
                
                print("=" * 80)
                print("FP8 OPTIMIZATION COMPLETE")
                print(f"  Optimized layers: {fp8_info['optimized_layers']} ({fp8_info['video_layers']} video + {fp8_info['audio_layers']} audio)")
                print(f"  Model size: {fp8_info['params_before_mb']:.1f} MB -> {fp8_info['params_after_mb']:.1f} MB")
                print(f"  Compression ratio: {fp8_info['compression_ratio']:.2f}x")
                print(f"  Memory saved: {fp8_info['params_before_mb'] - fp8_info['params_after_mb']:.1f} MB (~{(1 - fp8_info['compression_ratio']**-1) * 100:.0f}%)")
                print("=" * 80)
        else:
            # No FP8: convert to target dtype normally
            step_num = "3b" if self.lora_specs else "3"
            print(f"Step {step_num}/6: Converting model to {self.target_dtype} on CPU...")
            model = model.to(device="cpu", dtype=self.target_dtype).eval()
        
        if torch.cuda.is_available():
            after_load_vram = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"VRAM after loading to CPU: {after_load_vram:.2f} GB (should be ~0)")
        
        # Step 4: Enable block swap BEFORE moving to device (critical!)
        if self.blocks_to_swap > 0:
            print(f"Step 4/6: Enabling block swap with {self.blocks_to_swap} blocks...")
            print(f"  Video model: {len(model.video_model.blocks)} blocks total")
            print(f"  Audio model: {len(model.audio_model.blocks)} blocks total")
            
            model.video_model.enable_block_swap(self.blocks_to_swap, self.device, supports_backward=False)
            model.audio_model.enable_block_swap(self.blocks_to_swap, self.device, supports_backward=False)
            
            # Step 5: Move to device EXCEPT swap blocks (saves VRAM!)
            print("Step 5/6: Moving model to GPU except swap blocks (optimal VRAM usage)...")
            
            # Use FP8-preserving move if FP8 optimization is enabled
            if self.fp8_base_model:
                from ovi.utils.fp8_fusion_optimization_utils import move_wan_model_to_device_except_swap_blocks_preserve_fp8
                move_wan_model_to_device_except_swap_blocks_preserve_fp8(model.video_model, self.device)
                move_wan_model_to_device_except_swap_blocks_preserve_fp8(model.audio_model, self.device)
            else:
                model.video_model.move_to_device_except_swap_blocks(self.device)
                model.audio_model.move_to_device_except_swap_blocks(self.device)
            
            if torch.cuda.is_available():
                after_move_vram = torch.cuda.memory_allocated(self.device) / 1e9
                print(f"VRAM after moving except swap blocks: {after_move_vram:.2f} GB")
            
            # Step 6: Prepare block swap for inference (set forward-only mode)
            if not no_block_prep:
                print("Step 6/6: Preparing block swap for forward pass...")
                # CRITICAL: Set forward-only mode before preparing blocks
                model.video_model.offloader.set_forward_only(True)
                model.audio_model.offloader.set_forward_only(True)
                
                # Now prepare blocks for forward pass
                model.video_model.prepare_block_swap_before_forward()
                model.audio_model.prepare_block_swap_before_forward()
                
                if torch.cuda.is_available():
                    after_prep_vram = torch.cuda.memory_allocated(self.device) / 1e9
                    peak_vram = torch.cuda.max_memory_allocated(self.device) / 1e9
                    print(f"VRAM after block swap preparation: {after_prep_vram:.2f} GB allocated")
                    print(f"Peak VRAM during loading: {peak_vram:.2f} GB")
            else:
                print("Step 6/6: Block swap preparation skipped (no_block_prep=True)")
        else:
            # No block swap - load entire model to device normally
            print("Step 4/6: No block swap - moving entire model to device...")
            target_device = self.device if not self.cpu_offload else "cpu"
            
            # Use FP8-preserving move if FP8 optimization is enabled
            if self.fp8_base_model and target_device != "cpu":
                from ovi.utils.fp8_fusion_optimization_utils import move_fusion_model_to_device_preserve_fp8
                model = move_fusion_model_to_device_preserve_fp8(model, target_device)
            else:
                model = model.to(device=target_device)
            
            if torch.cuda.is_available():
                after_load_vram = torch.cuda.memory_allocated(self.device) / 1e9
                peak_vram = torch.cuda.max_memory_allocated(self.device) / 1e9
                print(f"VRAM after full model load: {after_load_vram:.2f} GB")
                print(f"Peak VRAM during loading: {peak_vram:.2f} GB")

        # T5 is loaded BEFORE fusion model (see top of _load_models)
        # Only load it here if it wasn't loaded earlier
        if load_text_encoder and self.text_model is None:
            device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)
            t5_load_device = "cpu" if self.cpu_offload else device_idx
            print(f"Loading T5 text encoder to {t5_load_device}...")
            self.text_model = init_text_model(self.config.ckpt_dir, rank=t5_load_device)
            if self.cpu_offload:
                print("T5 text encoder loaded on CPU")

        self.model = model

        # Set latent channel info
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")

        # Count parameters and their devices
        total_params = sum(p.numel() for p in model.parameters())
        gpu_params = sum(p.numel() for p in model.parameters() if p.device.type == 'cuda')
        cpu_params = sum(p.numel() for p in model.parameters() if p.device.type == 'cpu')

        print("=" * 80)
        print("MODEL LOADING COMPLETE")
        print(f"  Total parameters: {total_params:,}")
        print(f"  GPU parameters: {gpu_params:,} ({gpu_params/total_params*100:.1f}%)")
        print(f"  CPU parameters: {cpu_params:,} ({cpu_params/total_params*100:.1f}%)")
        
        if torch.cuda.is_available():
            final_vram = torch.cuda.memory_allocated(self.device) / 1e9
            final_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            final_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            print(f"  Final VRAM allocated: {final_vram:.2f} GB")
            print(f"  Final VRAM reserved: {final_reserved:.2f} GB")
            print(f"  Peak VRAM usage: {final_peak:.2f} GB")
        
        if self.blocks_to_swap > 0:
            print(f"  Block swap active: {self.blocks_to_swap}/{len(model.video_model.blocks)} blocks on CPU")
        if self.cpu_offload:
            print(f"  CPU offload active: Text encoder on CPU")
        print("=" * 80)

    @torch.inference_mode()
    def generate(self,
                    text_prompt,
                    image_path=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    blocks_to_swap=None,
                    video_negative_prompt="",
                    audio_negative_prompt="",
                    delete_text_encoder=True,
                    no_block_prep=False,
                    fp8_t5=False,
                    cpu_only_t5=False,
                    fp8_base_model=False,
                    use_sage_attention=False,
                    vae_tiled_decode=False,
                    vae_tile_size=32,
                    vae_tile_overlap=8,
                    force_exact_resolution=False,
                    cancellation_check=None,
                    text_embeddings_cache=None,  # Pre-encoded text embeddings from T5 subprocess
                    lora_specs=None,  # List of (lora_path, scale, layers) tuples for LoRA loading
                    clear_all=False  # Force model reload (from UI)
                ):

        # ===================================================================
        # OPTIMIZATION: Load T5, encode text, DELETE T5, then load fusion model
        # This prevents having both T5 (~24GB) and Fusion (~45GB) in RAM simultaneously
        # ===================================================================
        
        # CRITICAL: Set T5 configuration BEFORE any loading!
        self.fp8_t5 = fp8_t5
        self.cpu_only_t5 = cpu_only_t5
        
        # CRITICAL: Set Fusion Model FP8 configuration BEFORE any loading!
        self.fp8_base_model = fp8_base_model
        
        # ===================================================================
        # LORA SMART RELOAD: Check if LoRA configuration changed
        # Only reload model if LoRAs changed (or clear_all is enabled)
        # ===================================================================
        from ovi.utils.lora_utils import get_lora_hash
        
        # Normalize lora_specs
        if lora_specs is None:
            lora_specs = []
        
        # Filter out None entries and entries with scale=0
        lora_specs = [(path, scale, layers) for path, scale, layers in lora_specs if path and scale != 0.0]
        
        # Calculate new LoRA hash
        new_lora_hash = get_lora_hash(lora_specs)
        
        # Determine if we need to reload the model
        lora_changed = (new_lora_hash != self.current_lora_hash)
        need_reload = clear_all or (lora_changed and self.model is not None)
        
        if need_reload and self.model is not None:
            if clear_all:
                print("=" * 80)
                print("CLEAR ALL MEMORY: Unloading fusion model for fresh reload")
                print("=" * 80)
            elif lora_changed:
                print("=" * 80)
                print("LORA CONFIGURATION CHANGED: Reloading model with new LoRAs")
                print(f"  Previous LoRA hash: {self.current_lora_hash}")
                print(f"  New LoRA hash: {new_lora_hash}")
                print("=" * 80)
            
            # Unload the model
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
        
        # Update LoRA configuration
        self.lora_specs = lora_specs
        self.current_lora_hash = new_lora_hash
        
        if self.lora_specs:
            print("=" * 80)
            print(f"LORA CONFIGURATION: {len(self.lora_specs)} LoRA(s) will be applied")
            for path, scale, layers in self.lora_specs:
                print(f"  - {os.path.basename(path)}: scale={scale}, layers={layers}")
            print("=" * 80)
        
        # CRITICAL: Set Sage Attention configuration BEFORE any generation!
        self.use_sage_attention = use_sage_attention
        # Import and set global attention mode
        from ovi.modules.attention import set_use_sage_attention
        set_use_sage_attention(use_sage_attention)
        
        # Set VAE tiled decoding configuration
        self.vae_tiled_decode = vae_tiled_decode
        self.vae_tile_size = vae_tile_size
        self.vae_tile_overlap = vae_tile_overlap
        
        # Check if pre-encoded text embeddings are provided (from T5 subprocess)
        if text_embeddings_cache is not None:
            print("=" * 80)
            print("USING PRE-ENCODED T5 EMBEDDINGS FROM SUBPROCESS")
            print("Skipping T5 loading and encoding - embeddings already computed")
            print(f"Embeddings type: {type(text_embeddings_cache)}, length: {len(text_embeddings_cache) if isinstance(text_embeddings_cache, list) else 'N/A'}")
            print("=" * 80)
            
            # Move embeddings to target device and dtype if needed
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings_cache]
            print(f"[DEBUG] Embeddings moved to device: {self.device}, dtype: {self.target_dtype}")
            
            # Directly load fusion model (skip T5 steps entirely)
            print("=" * 80)
            print("Loading fusion model (T5 subprocess already completed)")
            print("T5 will NOT be loaded - using cached embeddings only")
            print("=" * 80)
            self._load_models(no_block_prep=no_block_prep, load_text_encoder=False)
        else:
            # Original T5 loading and encoding flow
            print("[DEBUG] text_embeddings_cache is None - will load T5")
            # Step 1: Load ONLY T5 (or use existing if already loaded)
            if self.text_model is None:
                print("=" * 80)
                print("STEP 1/2: Loading T5 text encoder FIRST to minimize RAM usage")
                print("=" * 80)
                self._load_models(no_block_prep=no_block_prep, load_text_encoder=True)
                # At this point, ONLY T5 is loaded, fusion model is NOT loaded yet
            
            # Step 2: Encode text and optionally delete T5
            print("=" * 80)
            print("STEP 2/2: Encoding text and optionally deleting T5 before loading fusion model")
            print("=" * 80)
            text_embeddings = self._encode_text_and_cleanup(
                text_prompt, 
                video_negative_prompt, 
                audio_negative_prompt, 
                delete_text_encoder
            )
            
            # Step 3: NOW load the fusion model (T5 is already deleted if enabled)
            print("=" * 80)
            print("STEP 3/3: Loading fusion model (T5 already deleted if enabled)")
            print("=" * 80)
            self._load_models(no_block_prep=no_block_prep, load_text_encoder=False)
        
        # Split embeddings for later use
        text_embeddings_video_pos = text_embeddings[0]
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image_path if image_path else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Block Swap": blocks_to_swap if blocks_to_swap is not None else 0,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )

            is_t2v = image_path is None
            is_i2v = not is_t2v

            first_frame = None
            image = None

            # Determine video dimensions first
            if is_i2v and not self.image_model:
                # For I2V, video dimensions come from user input when force_exact_resolution is enabled
                if force_exact_resolution and video_frame_height_width is not None:
                    video_h, video_w = video_frame_height_width
                else:
                    # For regular I2V, dimensions will be determined from the image
                    video_h, video_w = None, None  # Will be set later from image
            else:
                # For T2V, dimensions come from user input
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."
                video_h, video_w = video_frame_height_width
                # Use exact resolution if requested, otherwise force to 720*720 area and snap to 32
                if not force_exact_resolution:
                    # Force to 720*720 area and snap to multiples of 32
                    video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area=720 * 720)

            if is_i2v and not self.image_model:
                # Load first frame from path
                if force_exact_resolution and video_h is not None and video_w is not None:
                    # Load image without resizing, then manually resize to target dimensions
                    first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype, resize_total_area=None)
                    # Resize the tensor to match video dimensions
                    import torch.nn.functional as F
                    target_h, target_w = video_h, video_w
                    current_h, current_w = first_frame.shape[-2], first_frame.shape[-1]
                    if current_h != target_h or current_w != target_w:
                        first_frame = F.interpolate(first_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
                else:
                    # Use default resizing to 720*720 area
                    first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype)
                    # Set video dimensions from the processed image
                    video_h, video_w = first_frame.shape[-2], first_frame.shape[-1]

            if video_h is not None and video_w is not None:
                video_latent_h, video_latent_w = video_h // 16, video_w // 16
                if self.image_model is not None:
                    # this already means t2v mode with image model
                    image_h, image_w = scale_hw_to_area_divisible(video_h, video_w, area = 1024 * 1024)
                    image = self.image_model(
                        clean_text(text_prompt),
                        height=image_h,
                        width=image_w,
                        guidance_scale=4.5,
                        generator=torch.Generator().manual_seed(seed)
                    ).images[0]
                    first_frame = preprocess_image_tensor(image, self.device, self.target_dtype)
                    is_i2v = True
                else:
                    print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")

            if is_i2v:
                with torch.no_grad():
                    self._ensure_vaes_on_device(self.device)
                    latents_images = self.vae_model_video.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w
                latents_images = latents_images.to(self.target_dtype)
                # For I2V with force_exact_resolution, use the predetermined latent dimensions
                # Otherwise, use the dimensions from the encoded latents
                if not (force_exact_resolution and video_latent_h is not None and video_latent_w is not None):
                    video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
                if self.cpu_offload:
                    self._offload_vaes_to_cpu()

            video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
            audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            
            # Sampling loop
            # CRITICAL: Don't move model to device if block swap is enabled!
            # Block swap already set up the device placement correctly.
            if self.cpu_offload and self.blocks_to_swap == 0:
                # Use FP8-preserving move if FP8 optimization is enabled
                if self.fp8_base_model:
                    from ovi.utils.fp8_fusion_optimization_utils import move_fusion_model_to_device_preserve_fp8
                    self.model = move_fusion_model_to_device_preserve_fp8(self.model, self.device)
                else:
                    self.model = self.model.to(self.device)
                print("[CPU Offload] Moving model to GPU for inference (no block swap)")
            
            # Log VRAM before inference starts
            if torch.cuda.is_available():
                before_inference_vram = torch.cuda.memory_allocated(self.device) / 1e9
                print(f"\n{'='*80}")
                print(f"INFERENCE STARTING - VRAM: {before_inference_vram:.2f} GB")
                if self.blocks_to_swap > 0:
                    print(f"Block swap active: {self.blocks_to_swap}/{len(self.model.video_model.blocks)} blocks on CPU")
                print(f"{'='*80}\n")
            
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    # Check for cancellation at the start of each sampling step
                    if cancellation_check is not None:
                        cancellation_check()

                    timestep_input = torch.full((1,), t_v, device=self.device)

                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # Positive (conditional) forward pass
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': [text_embeddings_video_pos],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v
                    }

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **pos_forward_args
                    )
                    
                    # Negative (unconditional) forward pass  
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': [text_embeddings_video_neg],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v,
                        'slg_layer': slg_layer
                    }
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **neg_forward_args
                    )

                    # Apply classifier-free guidance
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    video_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                
                # Log final VRAM after inference
                if torch.cuda.is_available():
                    final_inference_vram = torch.cuda.memory_allocated(self.device) / 1e9
                    final_peak_vram = torch.cuda.max_memory_allocated(self.device) / 1e9
                    print(f"\n{'='*80}")
                    print(f"INFERENCE COMPLETE - Final VRAM: {final_inference_vram:.2f} GB (Peak: {final_peak_vram:.2f} GB)")
                    if self.blocks_to_swap > 0:
                        vram_saved = final_peak_vram - 7.5  # Approximate savings vs non-block-swap
                        print(f"Block swap saved approximately: {abs(vram_saved - 20):.1f} GB VRAM")
                    print(f"{'='*80}\n")
                
                if is_i2v:
                    video_noise[:, :1] = latents_images

                self._ensure_vaes_on_device(self.device)

                # Decode audio
                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()

                # Decode video with optional tiling
                video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
                
                # Track VRAM before decode
                if torch.cuda.is_available():
                    before_decode_vram = torch.cuda.memory_allocated(self.device) / 1e9
                    torch.cuda.reset_peak_memory_stats(self.device)
                    print(f"\n{'='*80}")
                    print(f"STARTING VAE DECODE - VRAM before: {before_decode_vram:.2f} GB")
                    print(f"{'='*80}\n")
                
                if self.vae_tiled_decode:
                    # Use tiled decoding for VRAM optimization
                    print("=" * 80)
                    print("VAE TILED DECODING ENABLED")
                    print(f"  Latent shape: {video_latents_for_vae.shape}")
                    print(f"  Tile size: {self.vae_tile_size}×{self.vae_tile_size} (latent space)")
                    print(f"  Tile overlap: {self.vae_tile_overlap} (latent space)")
                    print(f"  Estimated tile size in pixels: {self.vae_tile_size*16}×{self.vae_tile_size*16}")
                    print(f"  Expected VRAM savings: ~30-60% depending on tile size")
                    print("=" * 80)

                    # Calculate total number of tiles for progress tracking
                    _, _, frames, height, width = video_latents_for_vae.shape
                    from ovi.utils.tiled_vae_utils import get_tiled_scale_steps
                    total_tiles = get_tiled_scale_steps(
                        width=width,
                        height=height,
                        tile_x=self.vae_tile_size,
                        tile_y=self.vae_tile_size,
                        overlap=self.vae_tile_overlap
                    )
                    # Since we process all frames together, multiply by temporal tiles (usually 1)
                    total_tiles *= 1  # frames // self.vae_tile_temporal if temporal tiling was used

                    print(f"VAE DECODE PROGRESS: Processing {total_tiles} tiles...")

                    # Create progress bar for VAE decoding
                    pbar = tqdm(total=total_tiles, desc="VAE Decode", unit="tile", ncols=80)

                    try:
                        generated_video = self.vae_model_video.wrapped_decode_tiled(
                            video_latents_for_vae,
                            tile_x=self.vae_tile_size,
                            tile_y=self.vae_tile_size,
                            tile_t=self.vae_tile_temporal,
                            overlap_x=self.vae_tile_overlap,
                            overlap_y=self.vae_tile_overlap,
                            overlap_t=1,
                            progress_bar=pbar
                        )
                    finally:
                        # Close progress bar
                        pbar.close()
                else:
                    # Standard decode (full VRAM usage)
                    print("VAE DECODE PROGRESS: Decoding video (standard mode)...")
                    generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
                    print("VAE DECODE PROGRESS: Standard decode completed")
                
                # Track VRAM after decode
                if torch.cuda.is_available():
                    after_decode_vram = torch.cuda.memory_allocated(self.device) / 1e9
                    peak_decode_vram = torch.cuda.max_memory_allocated(self.device) / 1e9
                    print(f"\n{'='*80}")
                    print(f"VAE DECODE COMPLETE")
                    print(f"  VRAM after: {after_decode_vram:.2f} GB")
                    print(f"  Peak during decode: {peak_decode_vram:.2f} GB")
                    print(f"  VRAM used by decode: {peak_decode_vram - before_decode_vram:.2f} GB")
                    if self.vae_tiled_decode and generated_video is not None:
                        print(f"  Tiled decode SUCCESS")
                    elif self.vae_tiled_decode:
                        print(f"  WARNING: Tiled decode returned None!")
                    print(f"{'='*80}\n")
                
                generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w
                if self.cpu_offload:
                    self._offload_vaes_to_cpu()

            return generated_video, generated_audio, image


        except Exception as e:
            # Re-raise cancellation exceptions to allow proper cancellation handling
            if "cancelled by user" in str(e).lower():
                raise e
            # Only log actual errors, not cancellations
            logging.error(traceback.format_exc())
            return None
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return model

    def _offload_vaes_to_cpu(self):
        if self.vae_model_video is not None:
            self.vae_model_video.cpu()
        if self.vae_model_audio is not None:
            self.vae_model_audio.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._vae_device = torch.device('cpu')

    def _ensure_vaes_on_device(self, device):
        target_device = torch.device(device) if not isinstance(device, torch.device) else device
        if self._vae_device == target_device:
            return
        if target_device.type == 'cpu':
            if self.vae_model_video is not None:
                self.vae_model_video.cpu()
            if self.vae_model_audio is not None:
                self.vae_model_audio.cpu()
        else:
            if self.vae_model_video is not None:
                self.vae_model_video.cuda(target_device)
            if self.vae_model_audio is not None:
                self.vae_model_audio.cuda(target_device)
        self._vae_device = target_device

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps

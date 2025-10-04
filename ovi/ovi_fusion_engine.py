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
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16, blocks_to_swap=0, cpu_offload=None):
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
        self.audio_latent_length = 157
        self.video_latent_length = 31

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

    def _encode_text_and_cleanup(self, text_prompt, video_negative_prompt, audio_negative_prompt, delete_text_encoder=True):
        """
        Encode text prompts and optionally delete T5 to save VRAM during generation.
        This is the FIRST operation after loading models.
        T5 is loaded directly to GPU for maximum performance.
        """
        # Check if T5 needs to be reloaded (was deleted in previous generation)
        if self.text_model is None:
            print("Loading T5 text encoder directly to GPU for encoding...")
            device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)
            # Always load T5 directly to GPU for encoding (no point loading to CPU first)
            self.text_model = init_text_model(self.config.ckpt_dir, rank=device_idx)
            print("T5 text encoder loaded directly to GPU")

        # Encode text embeddings (T5 is already on GPU)
        print(f"Encoding text prompts...")
        text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], self.device)
        text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

        # Handle T5 cleanup based on settings
        if delete_text_encoder:
            print("Deleting T5 text encoder to free VRAM (~5GB saved)...")
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

            if torch.cuda.is_available():
                after_delete_vram = torch.cuda.memory_allocated(self.device) / 1e9
                freed_vram = before_delete_vram - after_delete_vram
                print(f"T5 deleted. VRAM freed: {freed_vram:.2f} GB")
                print(f"Current VRAM: {after_delete_vram:.2f} GB")
        else:
            # Keep T5 but offload to CPU if CPU offloading is enabled
            if self.cpu_offload:
                print("Keeping T5 and offloading to CPU for future reuse...")
                self.offload_to_cpu(self.text_model.model)
                print("T5 text encoder offloaded to CPU")
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
        # OPTIMIZATION: Load T5 text encoder FIRST directly to GPU
        # This prevents having both T5 (~24GB) and Fusion (~45GB) in RAM at same time
        # ===================================================================
        if load_text_encoder and self.text_model is None:
            device_idx = self.device if isinstance(self.device, int) else (self.device.index if self.device.index is not None else 0)
            # Always load T5 directly to GPU for encoding (no point loading to CPU first)
            print(f"Loading T5 text encoder directly to GPU (BEFORE fusion model to save RAM)...")
            self.text_model = init_text_model(self.config.ckpt_dir, rank=device_idx)
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
        
        # Step 3: Convert dtype and eval (MUST stay on CPU to avoid VRAM spike!)
        print(f"Step 3/6: Converting model to {self.target_dtype} on CPU...")
        # CRITICAL: Explicitly specify device="cpu" to prevent PyTorch from materializing on GPU
        model = model.to(device="cpu", dtype=self.target_dtype).eval()
        model.set_rope_params()
        
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
                    no_block_prep=False
                ):

        # ===================================================================
        # OPTIMIZATION: Load T5, encode text, DELETE T5, then load fusion model
        # This prevents having both T5 (~24GB) and Fusion (~45GB) in RAM simultaneously
        # ===================================================================
        
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
            if is_i2v and not self.image_model:
                # Load first frame from path
                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype)
            else:   
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."
                video_h, video_w = video_frame_height_width
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = 720 * 720)
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
                    latents_images = self.vae_model_video.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
                latents_images = latents_images.to(self.target_dtype)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]

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

                # Decode audio
                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()
                
                # Decode video  
                video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
                generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
                generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w
            
            return generated_video, generated_audio, image


        except Exception as e:
            logging.error(traceback.format_exc())
            return None
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return model

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

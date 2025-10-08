import os
import torch
import hashlib

# ============================================================================
# T5 EMBEDDING CACHE SYSTEM
# ============================================================================
# Smart caching system to avoid re-encoding the same prompts with T5.
# Cache key: SHA256(text_prompt + video_negative + audio_negative + fp8_t5)
# Cache location: prompt_cache/{hash}.pt
# ============================================================================

def get_t5_cache_key(text_prompt, video_negative_prompt, audio_negative_prompt, fp8_t5):
    """Generate SHA256 cache key for T5 embeddings based on all inputs that affect encoding.

    Args:
        text_prompt: Main text prompt
        video_negative_prompt: Video negative prompt
        audio_negative_prompt: Audio negative prompt
        fp8_t5: Whether FP8 mode is enabled for T5

    Returns:
        SHA256 hash string (64 characters)
    """
    # Normalize inputs to strings
    text_str = str(text_prompt) if text_prompt else ""
    video_neg_str = str(video_negative_prompt) if video_negative_prompt else ""
    audio_neg_str = str(audio_negative_prompt) if audio_negative_prompt else ""
    fp8_str = "fp8_true" if fp8_t5 else "fp8_false"

    # Combine all inputs that affect T5 encoding output
    # Note: cpu_only_t5 is NOT included because it only affects WHERE encoding happens,
    # not the actual embedding values produced
    cache_input = f"{text_str}|||{video_neg_str}|||{audio_neg_str}|||{fp8_str}"

    # Generate SHA256 hash
    hash_obj = hashlib.sha256(cache_input.encode('utf-8'))
    cache_key = hash_obj.hexdigest()

    return cache_key


def get_t5_cache_path(cache_key):
    """Get the file path for a T5 cache key.

    Args:
        cache_key: SHA256 hash string

    Returns:
        Absolute path to cache file
    """
    # Cache directory is always relative to the project root
    # Find project root by looking for premium.py or other known files
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    cache_dir = os.path.join(project_root, "prompt_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{cache_key}.pt")
    return cache_file


def load_t5_cached_embeddings(cache_key):
    """Load cached T5 embeddings from disk if they exist.

    Args:
        cache_key: SHA256 hash string

    Returns:
        List of 3 tensors [pos_emb, video_neg_emb, audio_neg_emb] if cache exists,
        None otherwise
    """
    cache_file = get_t5_cache_path(cache_key)

    if os.path.exists(cache_file):
        try:
            embeddings = torch.load(cache_file, map_location='cpu')

            # Validate cache structure
            if isinstance(embeddings, list) and len(embeddings) == 3:
                print(f"[T5 CACHE HIT] Loaded embeddings from: {os.path.basename(cache_file)}")
                return embeddings
            else:
                print(f"[T5 CACHE] Invalid cache format in {cache_file}, will re-encode")
                return None

        except Exception as e:
            print(f"[T5 CACHE] Error loading cache {cache_file}: {e}")
            return None

    return None


def save_t5_cached_embeddings(cache_key, embeddings):
    """Save T5 embeddings to disk cache.

    Args:
        cache_key: SHA256 hash string
        embeddings: List of 3 tensors [pos_emb, video_neg_emb, audio_neg_emb]
    """
    cache_file = get_t5_cache_path(cache_key)

    try:
        # Ensure embeddings are on CPU for saving
        embeddings_cpu = [emb.cpu() if hasattr(emb, 'cpu') else emb for emb in embeddings]

        # Atomic write: write to temp file, then rename
        temp_file = cache_file + ".tmp"
        torch.save(embeddings_cpu, temp_file)

        # Atomic rename (overwrites if exists)
        if os.path.exists(cache_file):
            os.remove(cache_file)
        os.rename(temp_file, cache_file)

        print(f"[T5 CACHE SAVE] Saved embeddings to: {os.path.basename(cache_file)}")

    except Exception as e:
        print(f"[T5 CACHE] Error saving cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# ============================================================================
# END OF T5 EMBEDDING CACHE SYSTEM
# ============================================================================

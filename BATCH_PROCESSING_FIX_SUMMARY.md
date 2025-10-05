# Batch Processing Resolution Fix - Summary

## Issues Identified

### Issue 1: Auto-cropping using wrong resolution (992x512 instead of base 720x720)

**Root Cause:**
- Your **32-GB GPUs preset** had inconsistent values:
  - `base_resolution_width`: 720
  - `base_resolution_height`: 720
  - `video_width`: 992 (hard-coded in preset)
  - `video_height`: 512 (hard-coded in preset)

- In batch processing, the code was using the hard-coded `video_width` and `video_height` from the preset (992x512) instead of **recalculating dimensions** based on the base resolution (720x720) and aspect ratio.

**Your Images:**
- `1.png`: Portrait/vertical image (should use 9:16 or similar portrait aspect ratio)
- `5.png`: Landscape/horizontal image (should use 16:9 aspect ratio)
- `8.txt`: Text-only (no image)

**What Should Have Happened:**
With base resolution 720x720:
- 16:9 landscape should be calculated as ~**1024×544** (not 992×512)
- 9:16 portrait should be calculated as ~**544×1024** (not 512×992)

### Issue 2: Gradio InvalidPathError

**Error:**
```
gradio.exceptions.InvalidPathError: Cannot move C:\Users\Furkan\Pictures\gg111\d\8_0001.mp4 
to the gradio cache dir because it was not created by the application...
```

**Root Cause:**
- Batch processing was returning an absolute path outside Gradio's allowed directories
- Gradio only allows files in: working directory, temp directory, or explicitly allowed paths

## Fixes Applied

### Fix 1: Resolution Recalculation in Batch Processing

Added code at the start of `process_batch_generation()` (before the batch loop):

```python
# IMPORTANT: Recalculate video dimensions from base resolution and aspect ratio
parsed_dims = _parse_resolution_from_label(aspect_ratio)
if parsed_dims:
    video_frame_width, video_frame_height = parsed_dims
else:
    base_width = _coerce_positive_int(base_resolution_width) or 720
    base_height = _coerce_positive_int(base_resolution_height) or 720
    current_ratios = get_common_aspect_ratios(base_width, base_height)
    
    ratio_name = _extract_ratio_name(aspect_ratio)
    if ratio_name and ratio_name in current_ratios:
        video_frame_width, video_frame_height = current_ratios[ratio_name]
        print(f"[BATCH] Recalculated resolution from base {base_width}x{base_height} 
               and aspect {ratio_name}: {video_frame_width}x{video_frame_height}")
```

### Fix 2: Resolution Recalculation in Single Generation

Added the same recalculation logic at the start of `generate_video()`:

```python
# IMPORTANT: Recalculate video dimensions from base resolution and aspect ratio if needed
parsed_dims = _parse_resolution_from_label(aspect_ratio)
if parsed_dims:
    recalc_width, recalc_height = parsed_dims
    if recalc_width != video_frame_width or recalc_height != video_frame_height:
        print(f"[RESOLUTION FIX] Updating resolution from {video_frame_width}x{video_frame_height} 
               to {recalc_width}x{recalc_height}")
        video_frame_width, video_frame_height = recalc_width, recalc_height
```

This ensures consistency even when called from subprocess with old preset values.

### Fix 3: Gradio Path Error

Changed batch processing return value:

```python
# Return None instead of path to avoid Gradio InvalidPathError
# The user can access files directly from the output folder
return None
```

## How to Test

1. **Update your 32-GB GPUs preset:**
   - Load the preset in the UI
   - Verify base resolution is 720x720
   - Save the preset again (this will save consistent width/height values)

2. **Run batch processing again:**
   ```
   Input: C:\Users\Furkan\Pictures\gg111
   Output: C:\Users\Furkan\Pictures\gg111\d
   ```

3. **Expected behavior:**
   - You should see log messages like:
     ```
     [BATCH] Recalculated resolution from base 720x720 and aspect 16:9: 1024x544
     ```
   - Images will be auto-cropped to the correct aspect ratio based on base 720x720
   - No Gradio error at the end

4. **Verify output videos:**
   - Check the resolution in the generated `.txt` metadata files
   - Should match the recalculated resolution from base 720x720

## Additional Notes

- **Auto-cropping now works correctly** in batch processing based on base resolution
- **Single generation** also has the same fix for consistency
- **Preset inconsistencies** are automatically corrected at runtime
- **Gradio error is eliminated** by returning None for custom output directories

## Recommendation

**Re-save all your presets** to ensure they have consistent values:
1. Load each preset
2. Verify the base resolution and aspect ratio match the video width/height
3. Save the preset again

This will prevent future inconsistencies between base resolution and hard-coded width/height values.


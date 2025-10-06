# Video Input Feature - Testing Guide

## Changes Made

### 1. **New Functions Added**
- `is_video_file()` - Detects if a file is a video based on extension
- `is_image_file()` - Detects if a file is an image based on extension  
- `process_input_media()` - Processes both video and image inputs, extracts last frame from videos
- `on_media_upload()` - Gradio handler for media uploads (replaces `on_image_upload`)

### 2. **Modified Functions**
- `generate_video()` - Now accepts `input_video_path` parameter
- Added video merging logic at the end of `generate_video()` to combine input video + generated video

### 3. **UI Changes**
- Changed input from `gr.Image` to `gr.File` accepting both images and videos
- Added `input_video_state` to store the input video path
- Added helpful description explaining video merging behavior
- Updated all event handlers to work with new media upload system

## How It Works

### When User Uploads a Video:
1. **Extract Last Frame**: The last frame of the video is automatically extracted
2. **Auto-Crop** (if enabled): The extracted frame is cropped to match target resolution
3. **Generate Video**: New video is generated using the extracted frame as source
4. **Merge Videos**: Input video + generated video are automatically merged into a single file
5. **Output**: The merged video is saved with `_merged` suffix (e.g., `0001_merged.mp4`)

### When User Uploads an Image:
- Works exactly as before - no changes to existing image workflow

## Testing Steps

### Test 1: Video Input with Auto-Crop
```bash
.\venv\Scripts\python.exe premium.py
```

1. Launch the app
2. Upload a video file (e.g., .mp4)
3. Enable "Auto Crop Image" checkbox
4. Enter a prompt
5. Click "Generate Video"
6. Check that:
   - Last frame is extracted and displayed
   - Resolution is auto-detected
   - Generated video is created
   - Input video + generated video are merged
   - Final output has `_merged` suffix

### Test 2: Video Input without Auto-Crop
1. Upload a video
2. Disable "Auto Crop Image"
3. Generate video
4. Verify merging still works

### Test 3: Image Input (Backward Compatibility)
1. Upload an image (e.g., .png)
2. Generate video
3. Verify it works exactly as before (no merging)

### Test 4: Video Extension + Video Input
1. Upload a video
2. Enable "Video Extension" 
3. Add multi-line prompt (4 lines = 1 base + 3 extensions)
4. Generate
5. Verify:
   - Base video is generated from last frame of input
   - Extensions are created
   - Input video is merged with the final combined video

## Expected Output Files

### Video Input:
- `0001.mp4` - Generated video only
- `0001_merged.mp4` - Input video + generated video merged

### Image Input:
- `0001.mp4` - Generated video (no merging)

### Video Input + Video Extension:
- `0001.mp4` - Base generated video
- `0001_ext1.mp4` - First extension
- `0001_ext2.mp4` - Second extension
- `0001_final.mp4` - All extensions combined
- `0001_final_merged.mp4` - Input video + all extensions merged

## Supported Video Formats
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

## Supported Image Formats
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`, `.gif`

## Key Code Locations

- **Video detection**: Lines 533-545
- **Media processing**: Lines 547-594
- **Video merging logic**: Lines 1404-1430
- **UI component**: Lines 3540-3545
- **Event handlers**: Lines 4391-4400

## Error Handling

The system gracefully handles:
- Invalid video files (extraction fails)
- Corrupted videos
- Missing files
- FFmpeg errors during merging

If extraction or merging fails, the system continues with whatever is available and logs warnings.

## Bug Fixes Applied

### Fix 1: Video File Error in Resolution Updates
**Issue:** When changing resolution/aspect ratio after uploading a video, got error:
```
Error in update_cropped_image_only: cannot identify image file '...video.mp4'
```

**Root Cause:** Event handlers were passing the video file path directly to image processing functions when resolution changed.

**Solution:**
1. Added video detection to `update_cropped_image_only()` to extract frame first
2. Changed all event handlers to use `image_to_use` state (contains extracted frame) instead of `image` state (contains original video/image path)
3. This ensures all downstream processing works with image files only

**Files Modified:**
- Lines 3170-3182: Added video detection to `update_cropped_image_only()`
- Lines 4350, 4360, 4371, 4425, 4430: Changed handlers to use `image_to_use` instead of `image`

### UI Improvements
- Changed cropped display label to "Source Frame Preview (Auto-cropped)"
- Video input label shows "Last frame extracted - {width}×{height}px"
- Clearer distinction between video and image inputs

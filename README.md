# Video Generator

Generate short video scenes from text prompts using Google GenAI (images) and VEO 3.1 (video), then stitch scenes into a trailer.

## Overview
- `generate.py` provides three main functions:
  - `generate_character_references(image_api_key, character_designs)` – creates reference images for characters.
  - `generate_scene_videos(image_api_key, veo_api_key, scenes, character_refs)` – generates per‑scene MP4s.
  - `stitch_videos(video_paths)` – concatenates scene videos into a single trailer via `ffmpeg`.
- Sample inputs live in `trailer_breakdown_samples/breakdown_8_sec.json`.
- Outputs are written under `output/`:
  - `output/refs/` – generated character images
  - `output/scenes/` – per‑scene MP4s
  - `output/trailer_no_audio.mp4` – stitched trailer

## Requirements
- Python 3.10+ (tested on 3.13)
- A Google API key with access to Gemini image and VEO video generation
  You can get one from here: https://console.cloud.google.com/apis/credentials
- `ffmpeg` available on your system PATH (used for stitching)
  - macOS (Homebrew): `brew install ffmpeg`
  - Conda: `conda install -c conda-forge ffmpeg`

Install Python dependencies:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure API Key
Create `secret.json` in the project root with your key:

```
{
  "project_api_key": "YOUR_API_KEY"
}
```

## Usage
Right now the test only uses 8 seconds scenes(breakdown_8_sec.json), because VEO 3.1 only deals with 8-second scene prompts

###  Full pipeline (refs → scenes → trailer)
Run:

```
python3 test.py
```

This will populate `output/refs/`, `output/scenes/`, and finally stitch into `output/trailer_no_audio.mp4`.






## Input JSON Format
See `trailer_breakdown_samples/trailor_breakdown.json` for a working example. High‑level schema:

- `character_designs`: array of objects
  - `character_name`: string (used as filename under `output/refs/`)
  - `image_generation_prompt`: string (prompt for the character’s reference image)
- `scenes`: array of objects
  - `scene_number`: integer (1‑based)
  - `scene_type`: string (e.g., "establishing", "action")
  - `duration_seconds`: integer
  - `start_frame_prompt`: string
  - `end_frame_prompt`: string
  - `video_prompt`: string
  - `reference_images`: array of character names to include as references (optional)

Notes and constraints enforced by code:
- If `reference_images` are provided, `duration_seconds` must be exactly `8`.
- At most 3 reference images per scene.
- Videos are generated at 16:9.

## Output
- Per‑scene MP4s: `output/scenes/scene_01.mp4`, `scene_02.mp4`, ...
- Stitched trailer (no audio): `output/trailer_no_audio.mp4`

## Common Errors
- `FileNotFoundError: 'ffmpeg'` when stitching:
  - Install ffmpeg and ensure it’s on PATH. For macOS: `brew install ffmpeg`. Verify with `ffmpeg -version`.
- ffmpeg concat fails with codec errors:
  - The default command uses stream copy (`-c copy`). All scenes must share identical codecs/parameters. If not, re‑encode scenes to a common format first, or manually re‑encode during concat, e.g.:

```
ffmpeg -y -f concat -safe 0 -i output/concat.txt \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart \
  output/trailer_no_audio.mp4
```

- API/auth issues:
  - Confirm `secret.json` exists and contains a valid key under `project_api_key`.
  - Ensure your key has access to both Gemini image and VEO video models.

## Project Structure
```
.
├── generate.py
├── test.py
├── trailer_breakdown_samples/
│   └── breakdown_8_sec.json
│   └── trailer_breakdown.json
├── output/
│   ├── refs
│   ├── scenes
│   └── trailer_no_audio.mp4
└── requirements.txt
```
## More to modify
Adjust the code in `generate.py` — specifically the `generate_scene_videos` and `stitch_videos` functions — to use virtual (temporary) paths and upload the final trailer to your GCS bucket.

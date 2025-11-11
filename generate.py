import base64
from pathlib import Path
import time
from typing import Dict, List

import requests
OUTPUT_DIR = Path('./output')
from google import genai


def generate_character_references(
    image_api_key, character_designs: List[dict]
) -> Dict[str, str]:
    """
    Generate character reference images from designs.

    Args:
        character_designs: List of CharacterDesign objects

    Returns:
        Dict mapping character_name to image path
    """
    character_refs = {}

    for i, design in enumerate(character_designs, 1):
        char_name = design["character_name"]
        prompt = design["image_generation_prompt"]

        print(f"  [{i}/{len(character_designs)}] Generating {char_name}...")

        # Generate image using DALL-E or Flux
        image_data = generate_image(image_api_key, prompt)

        # Save image
        image_path = OUTPUT_DIR / f"refs/{char_name}.png"
        image_path.parent.mkdir(exist_ok=True)
        image_path.write_bytes(image_data)

        character_refs[char_name] = str(image_path)
        print(f"    ✓ Saved to {image_path}")

    return character_refs


def generate_scene_videos(
    image_api_key, veo_api_key, scenes: List[dict], character_refs: Dict[str, str]
) -> List[Path]:
    """
    Generate videos for all scenes.

    Args:
        scenes: List of scene objects
        character_refs: Dict of character_name -> image_path

    Returns:
        List of video file paths
    """
    scene_videos = []

    for scene in scenes:
        scene_num = scene["scene_number"]
        print(
            f"\n  Scene {scene_num}: {scene['scene_type']} ({scene['duration_seconds']}s)"
        )

        # Generate start and end frames
        print(f"    Generating start frame...")
        start_img = generate_image(image_api_key, scene["start_frame_prompt"])

        print(f"    Generating end frame...")
        end_img = generate_image(image_api_key, scene["end_frame_prompt"])

        # Prepare reference images for this scene
        scene_ref_images = []
        if scene["reference_images"]:
            print(
                f"    Loading {len(scene['reference_images'])} character reference(s)..."
            )
            for char_name in scene["reference_images"]:
                ref_path = character_refs[char_name]
                scene_ref_images.append(Path(ref_path).read_bytes())
                print(f"      ✓ {char_name}")

        # Call VEO 3.1
        print(f"    Generating video with VEO 3.1...")
        video_data = generate_video_veo(
            veo_api_key,
            prompt=scene["video_prompt"],
            start_frame=start_img,
            end_frame=end_img,
            duration=scene["duration_seconds"],
            reference_images=scene_ref_images,
        )

        # Save video
        video_path = OUTPUT_DIR / f"scenes/scene_{scene_num:02d}.mp4"
        video_path.parent.mkdir(exist_ok=True)
        video_path.write_bytes(video_data)

        scene_videos.append(video_path)
        print(f"    ✓ Saved to {video_path}")

    return scene_videos


def generate_image(image_api_key, prompt: str) -> bytes:
    """
    Generate image using DALL-E or Flux.

    Args:
        prompt: Image generation prompt

    Returns:
        Image data as bytes
    """



    client = genai.Client(api_key = image_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image", contents=[prompt]#, size="1024x1024", quality="standard", n=1
    )

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            return image.image_bytes



def generate_video_veo(
    veo_api_key,
    prompt: str,
    start_frame: bytes,
    end_frame: bytes,
    duration: int,
    reference_images: List[bytes],
) -> bytes:
    """
    Generate video using VEO 3.1.

    Args:
        prompt: Video generation prompt
        start_frame: Start frame image data
        end_frame: End frame image data
        duration: Duration in seconds
        reference_images: List of character reference images

    Returns:
        Video data as bytes
    """
    # Build VEO 3.1 API request
    veo_params = {
        "prompt": prompt,
        "image": base64.b64encode(start_frame).decode(),
        "lastFrame": base64.b64encode(end_frame).decode(),
        "duration": duration,
        "aspectRatio": "16:9",
    }

    # Add reference images if present
    if reference_images:
        # CRITICAL: Must be exactly 8 seconds with references
        if duration != 8:
            raise ValueError(
                f"Duration must be 8s when using reference images (got {duration}s)"
            )

        # Max 3 reference images
        if len(reference_images) > 3:
            raise ValueError(
                f"Max 3 reference images allowed (got {len(reference_images)})"
            )

        veo_params["referenceImages"] = [
            base64.b64encode(img).decode() for img in reference_images
        ]
        veo_params["personGeneration"] = "allow_adult"

    client = genai.Client(api_key=veo_api_key)
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
    )

    # Poll the operation status until the video is ready.
    while not operation.done:
        print("Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    # Download the generated video.
    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    return generated_video.video.video_bytes


    
    # Call VEO 3.1 API 
    # response = requests.post(
    #     "https://api.veo.google.com/v1/generate",
    #     headers={
    #         "Authorization": f"Bearer {veo_api_key}",
    #         "Content-Type": "application/json",
    #     },
    #     json=veo_params,
    #     timeout=180,  # Video generation takes time
    # )
    # response.raise_for_status()

    # # Wait for video to be ready and download
    # result = response.json()
    # video_url = result["video_url"]

    # import urllib.request

    # with urllib.request.urlopen(video_url) as response:
    #     return response.read()
    
def stitch_videos(video_paths: List[Path]) -> Path:
        """
        Stitch scene videos together.

        Args:
            video_paths: List of scene video paths

        Returns:
            Path to stitched video
        """
        import subprocess

        # Create concat file for ffmpeg
        concat_file = OUTPUT_DIR / "concat.txt"
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.absolute()}'\n")

        # Stitch with ffmpeg
        output_path = OUTPUT_DIR / "trailer_no_audio.mp4"
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_path)
        ], check=True)

        return output_path

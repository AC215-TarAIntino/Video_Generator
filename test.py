from generate import generate_character_references, generate_scene_videos, stitch_videos
from json import load
from pathlib import Path

with open('./secret.json') as f:
    secret = load(f)
api_key = secret['project_api_key']


with open('./trailer_breakdown_samples/breakdown_8_sec.json') as f:
    samples = load(f)    


generate_character_references(image_api_key=api_key, character_designs=samples['character_designs'])

character_refs = {}
for char in samples['character_designs']:
    name = char["character_name"]
    character_refs[name] = f'./output/refs/{name}.png'

generate_scene_videos(image_api_key=api_key, veo_api_key=api_key, scenes=samples['scenes'],character_refs=character_refs)

folder = Path("./output/scenes")
video_paths = sorted(folder.glob("scene_*.mp4"))
stitch_videos(video_paths = video_paths)
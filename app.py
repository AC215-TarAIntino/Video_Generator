"""FastAPI wrapper for the video generator pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import (
    OUTPUT_DIR,
    generate_character_references,
    generate_scene_videos,
    stitch_videos,
)


app = FastAPI(
    title="Video Generator API",
    version="1.0.0",
    description=(
        "Wraps the local video generation pipeline so it can be called via HTTP. "
        "Provide prompts for characters and scenes to receive generated assets."
    ),
)


class CharacterDesign(BaseModel):
    character_name: str = Field(..., description="Used as the filename under output/refs")
    image_generation_prompt: str


class Scene(BaseModel):
    scene_number: int
    scene_type: str
    duration_seconds: int
    start_frame_prompt: str
    end_frame_prompt: str
    video_prompt: str
    reference_images: List[str] = Field(default_factory=list)


class CharacterReferenceRequest(BaseModel):
    character_designs: List[CharacterDesign]
    image_api_key: Optional[str] = Field(
        default=None,
        description="Override for the Gemini image API key. Falls back to secret.json if omitted.",
    )


class CharacterReferenceResponse(BaseModel):
    character_refs: Dict[str, str]


class SceneVideoRequest(BaseModel):
    scenes: List[Scene]
    image_api_key: Optional[str] = None
    veo_api_key: Optional[str] = None
    character_refs: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping of character name to path for reference images. If omitted, the API"
        " will attempt to load refs from output/refs/ automatically.",
    )
    autoload_refs: bool = Field(
        default=True,
        description="If true, missing reference mappings will be populated from output/refs.",
    )


class SceneVideoResponse(BaseModel):
    video_paths: List[str]


class TrailerGenerationRequest(BaseModel):
    character_designs: List[CharacterDesign]
    scenes: List[Scene]
    image_api_key: Optional[str] = None
    veo_api_key: Optional[str] = None
    stitch_trailer: bool = Field(
        default=True,
        description="If true, stitch scene videos into output/trailer_no_audio.mp4",
    )


class TrailerGenerationResponse(BaseModel):
    character_refs: Dict[str, str]
    scene_videos: List[str]
    trailer_path: Optional[str]


def _load_default_api_key() -> Optional[str]:
    secret_path = Path("secret.json")
    if not secret_path.exists():
        return None
    try:
        payload = json.loads(secret_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid secret.json format: {exc}") from exc
    return payload.get("project_api_key")


def _resolve_api_key(provided: Optional[str], key_name: str) -> str:
    api_key = provided or _load_default_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"{key_name} is required. Provide it in the request body or secret.json.",
        )
    return api_key


def _collect_referenced_characters(scenes: List[Scene]) -> List[str]:
    referenced = []
    seen = set()
    for scene in scenes:
        for character in scene.reference_images:
            if character not in seen:
                seen.add(character)
                referenced.append(character)
    return referenced


def _build_character_ref_map(
    scenes: List[Scene],
    provided_refs: Optional[Dict[str, str]],
    autoload_refs: bool,
) -> Dict[str, str]:
    if provided_refs:
        provided_missing = [
            char
            for char in _collect_referenced_characters(scenes)
            if char not in provided_refs
        ]
        if provided_missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing reference paths for: {', '.join(provided_missing)}",
            )
        return provided_refs

    if not autoload_refs:
        referenced = _collect_referenced_characters(scenes)
        if referenced:
            raise HTTPException(
                status_code=400,
                detail="Reference images required but no character_refs provided.",
            )
        return {}

    refs: Dict[str, str] = {}
    for character in _collect_referenced_characters(scenes):
        ref_path = OUTPUT_DIR / "refs" / f"{character}.png"
        if not ref_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Reference image not found for '{character}' at {ref_path}",
            )
        refs[character] = str(ref_path)
    return refs


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate/character-references", response_model=CharacterReferenceResponse)
def create_character_references(request: CharacterReferenceRequest) -> CharacterReferenceResponse:
    image_api_key = _resolve_api_key(request.image_api_key, "image_api_key")
    try:
        character_refs = generate_character_references(
            image_api_key=image_api_key,
            character_designs=[design.model_dump() for design in request.character_designs],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - pass through unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CharacterReferenceResponse(character_refs=character_refs)


@app.post("/generate/scene-videos", response_model=SceneVideoResponse)
def create_scene_videos(request: SceneVideoRequest) -> SceneVideoResponse:
    image_api_key = _resolve_api_key(request.image_api_key, "image_api_key")
    veo_api_key = _resolve_api_key(request.veo_api_key or request.image_api_key, "veo_api_key")
    character_refs = _build_character_ref_map(
        request.scenes,
        request.character_refs,
        request.autoload_refs,
    )

    try:
        videos = generate_scene_videos(
            image_api_key=image_api_key,
            veo_api_key=veo_api_key,
            scenes=[scene.model_dump() for scene in request.scenes],
            character_refs=character_refs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SceneVideoResponse(video_paths=[str(path) for path in videos])


@app.post("/generate/trailer", response_model=TrailerGenerationResponse)
def generate_trailer(request: TrailerGenerationRequest) -> TrailerGenerationResponse:
    image_api_key = _resolve_api_key(request.image_api_key, "image_api_key")
    veo_api_key = _resolve_api_key(request.veo_api_key or request.image_api_key, "veo_api_key")

    try:
        character_refs = generate_character_references(
            image_api_key=image_api_key,
            character_designs=[design.model_dump() for design in request.character_designs],
        )

        scene_paths = generate_scene_videos(
            image_api_key=image_api_key,
            veo_api_key=veo_api_key,
            scenes=[scene.model_dump() for scene in request.scenes],
            character_refs=character_refs,
        )

        trailer_path: Optional[Path] = None
        if request.stitch_trailer:
            trailer_path = stitch_videos(scene_paths)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TrailerGenerationResponse(
        character_refs=character_refs,
        scene_videos=[str(path) for path in scene_paths],
        trailer_path=str(trailer_path) if trailer_path else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

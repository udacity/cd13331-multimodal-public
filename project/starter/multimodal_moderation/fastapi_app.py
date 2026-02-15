from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from multimodal_moderation.env import get_default_model_choice, USER_API_KEY
from multimodal_moderation.utils import detect_file_type
from multimodal_moderation.types.moderation_result import (
    TextModerationResult,
    ImageModerationResult,
    VideoModerationResult,
    AudioModerationResult,
)
from multimodal_moderation.agents.text_agent import moderate_text
from multimodal_moderation.agents.image_agent import moderate_image
from multimodal_moderation.agents.video_agent import moderate_video
from multimodal_moderation.agents.audio_agent import moderate_audio


# Standard auth scheme using -H "Authorization: Bearer <api_key>" header.
security = HTTPBearer()


# NOTE: this is a simple check against a static key. In a production setting, this should be more sophisticated.
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != USER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid user API key")
    return credentials.credentials


# Request models
class TextRequest(BaseModel):
    text: str


# Using `dependencies` to apply the API key validation to all endpoints.
app = FastAPI(dependencies=[Depends(validate_api_key)])


# Default model settings
default_model_choice = get_default_model_choice()


@app.post("/api/v1/moderate_text", response_model=TextModerationResult)
async def moderate_text_endpoint(request: TextRequest):
    return await moderate_text(default_model_choice, request.text)


@app.post("/api/v1/moderate_image_file", response_model=ImageModerationResult)
async def moderate_image_file_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    mime_type = detect_file_type(file_bytes, context=file.filename or "image file")
    return await moderate_image(default_model_choice, file_bytes, mime_type)


@app.post("/api/v1/moderate_video_file", response_model=VideoModerationResult)
async def moderate_video_file_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    mime_type = detect_file_type(file_bytes, context=file.filename or "video file")
    return await moderate_video(default_model_choice, file_bytes, mime_type)


@app.post("/api/v1/moderate_audio_file", response_model=AudioModerationResult)
async def moderate_audio_file_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    mime_type = detect_file_type(file_bytes, context=file.filename or "audio file")
    return await moderate_audio(default_model_choice, file_bytes, mime_type)


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("multimodal_moderation.fastapi_app:app", host="0.0.0.0", port=8000, reload=True, log_level="trace")


if __name__ == "__main__":
    main()

"""
Basic functionality tests for video moderation agent.

These tests verify that the video agent function exists, can be called,
and returns the expected types with required fields. They do NOT test
model performance or prompt quality (those belong in evals/).

Uses pydantic AI's TestModel to avoid real API calls while still validating
that the Agent is configured correctly with proper instructions and schema.
"""

import pytest
from pathlib import Path
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from multimodal_moderation.agents.video_agent import moderate_video, video_moderation_agent
from multimodal_moderation.types.moderation_result import VideoModerationResult
from multimodal_moderation.env import get_default_model_choice

# Block accidental real API calls - all tests should use TestModel
models.ALLOW_MODEL_REQUESTS = False


def _get_model():
    """Helper to get the default model choice"""
    return get_default_model_choice()


def _load_test_video():
    """Helper to load the test video as bytes"""
    test_video_path = Path(__file__).parent / "test_data" / "simple_video.mp4"
    with open(test_video_path, "rb") as f:
        return f.read()


def test_moderate_video_exists():
    """Verify moderate_video function exists and can be imported"""
    assert callable(moderate_video), "moderate_video should be a callable function"


async def test_moderate_video_returns_video_moderation_result():
    """Verify moderate_video returns a VideoModerationResult object"""
    model = _get_model()
    video_bytes = _load_test_video()

    with video_moderation_agent.override(model=TestModel()):
        result = await moderate_video(model, video_bytes, media_type="video/mp4")

    assert isinstance(result, VideoModerationResult), \
        f"moderate_video should return VideoModerationResult, got {type(result)}"


async def test_moderate_video_has_required_fields():
    """Verify result has all required fields: contains_pii, is_disturbing, is_low_quality, rationale"""
    model = _get_model()
    video_bytes = _load_test_video()

    with video_moderation_agent.override(model=TestModel()):
        result = await moderate_video(model, video_bytes, media_type="video/mp4")

    assert hasattr(result, 'contains_pii'), "Result must have 'contains_pii' field"
    assert hasattr(result, 'is_disturbing'), "Result must have 'is_disturbing' field"
    assert hasattr(result, 'is_low_quality'), "Result must have 'is_low_quality' field"
    assert hasattr(result, 'rationale'), "Result must have 'rationale' field"

    assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
    assert isinstance(result.is_disturbing, bool), "is_disturbing should be a boolean"
    assert isinstance(result.is_low_quality, bool), "is_low_quality should be a boolean"
    assert isinstance(result.rationale, str), "rationale should be a string"


async def test_moderate_video_rationale_not_empty():
    """Verify that rationale field is not empty"""
    model = _get_model()
    video_bytes = _load_test_video()

    with video_moderation_agent.override(model=TestModel()):
        result = await moderate_video(model, video_bytes, media_type="video/mp4")

    assert result.rationale, "Rationale should not be empty"
    assert len(result.rationale) > 0, "Rationale should contain text"

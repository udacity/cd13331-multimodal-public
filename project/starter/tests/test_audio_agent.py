"""
Basic functionality tests for audio moderation agent.

These tests verify that the audio agent function exists, can be called,
and returns the expected types with required fields. They do NOT test
model performance or prompt quality (those belong in evals/).

Uses pydantic AI's TestModel to avoid real API calls while still validating
that the Agent is configured correctly with proper instructions and schema.
"""

import pytest
from pathlib import Path
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from multimodal_moderation.agents.audio_agent import moderate_audio, audio_moderation_agent
from multimodal_moderation.types.moderation_result import AudioModerationResult
from multimodal_moderation.env import get_default_model_choice

# Block accidental real API calls - all tests should use TestModel
models.ALLOW_MODEL_REQUESTS = False


def _get_model():
    """Helper to get the default model choice"""
    return get_default_model_choice()


def _load_test_audio():
    """Helper to load the test audio as bytes"""
    test_audio_path = Path(__file__).parent / "test_data" / "simple_audio.mp3"
    with open(test_audio_path, "rb") as f:
        return f.read()


def test_moderate_audio_exists():
    """Verify moderate_audio function exists and can be imported"""
    assert callable(moderate_audio), "moderate_audio should be a callable function"


async def test_moderate_audio_returns_audio_moderation_result():
    """Verify moderate_audio returns an AudioModerationResult object"""
    model = _get_model()
    audio_bytes = _load_test_audio()

    with audio_moderation_agent.override(model=TestModel()):
        result = await moderate_audio(model, audio_bytes, media_type="audio/mpeg")

    assert isinstance(result, AudioModerationResult), \
        f"moderate_audio should return AudioModerationResult, got {type(result)}"


async def test_moderate_audio_has_required_fields():
    """Verify result has all required fields: transcription, contains_pii, is_unfriendly, is_unprofessional, rationale"""
    model = _get_model()
    audio_bytes = _load_test_audio()

    with audio_moderation_agent.override(model=TestModel()):
        result = await moderate_audio(model, audio_bytes, media_type="audio/mpeg")

    assert hasattr(result, 'transcription'), "Result must have 'transcription' field"
    assert hasattr(result, 'contains_pii'), "Result must have 'contains_pii' field"
    assert hasattr(result, 'is_unfriendly'), "Result must have 'is_unfriendly' field"
    assert hasattr(result, 'is_unprofessional'), "Result must have 'is_unprofessional' field"
    assert hasattr(result, 'rationale'), "Result must have 'rationale' field"

    assert isinstance(result.transcription, str), "transcription should be a string"
    assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
    assert isinstance(result.is_unfriendly, bool), "is_unfriendly should be a boolean"
    assert isinstance(result.is_unprofessional, bool), "is_unprofessional should be a boolean"
    assert isinstance(result.rationale, str), "rationale should be a string"


async def test_moderate_audio_rationale_not_empty():
    """Verify that rationale field is not empty"""
    model = _get_model()
    audio_bytes = _load_test_audio()

    with audio_moderation_agent.override(model=TestModel()):
        result = await moderate_audio(model, audio_bytes, media_type="audio/mpeg")

    assert result.rationale, "Rationale should not be empty"
    assert len(result.rationale) > 0, "Rationale should contain text"


async def test_moderate_audio_transcription_not_empty():
    """Verify that transcription field is not empty"""
    model = _get_model()
    audio_bytes = _load_test_audio()

    with audio_moderation_agent.override(model=TestModel()):
        result = await moderate_audio(model, audio_bytes, media_type="audio/mpeg")

    assert result.transcription, "Transcription should not be empty"
    assert len(result.transcription) > 0, "Transcription should contain text"

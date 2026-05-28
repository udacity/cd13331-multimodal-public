"""
Basic functionality tests for image moderation agent.

These tests verify that the image agent function exists, can be called,
and returns the expected types with required fields. They do NOT test
model performance or prompt quality (those belong in evals/).

Uses pydantic AI's TestModel to avoid real API calls while still validating
that the Agent is configured correctly with proper instructions and schema.
"""

import pytest
from pathlib import Path
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from multimodal_moderation.agents.image_agent import moderate_image, image_moderation_agent
from multimodal_moderation.types.moderation_result import ImageModerationResult
from multimodal_moderation.env import get_default_model_choice

# Block accidental real API calls - all tests should use TestModel
models.ALLOW_MODEL_REQUESTS = False


def _get_model():
    """Helper to get the default model choice"""
    return get_default_model_choice()


def _load_test_image():
    """Helper to load the test image as bytes"""
    test_image_path = Path(__file__).parent / "test_data" / "simple_image.jpg"
    with open(test_image_path, "rb") as f:
        return f.read()


def test_moderate_image_exists():
    """Verify moderate_image function exists and can be imported"""
    assert callable(moderate_image), "moderate_image should be a callable function"


async def test_moderate_image_returns_image_moderation_result():
    """Verify moderate_image returns an ImageModerationResult object"""
    model = _get_model()
    image_bytes = _load_test_image()

    with image_moderation_agent.override(model=TestModel()):
        result = await moderate_image(model, image_bytes, media_type="image/jpeg")

    assert isinstance(result, ImageModerationResult), \
        f"moderate_image should return ImageModerationResult, got {type(result)}"


async def test_moderate_image_has_required_fields():
    """Verify result has all required fields: contains_pii, is_disturbing, is_low_quality, rationale"""
    model = _get_model()
    image_bytes = _load_test_image()

    with image_moderation_agent.override(model=TestModel()):
        result = await moderate_image(model, image_bytes, media_type="image/jpeg")

    assert hasattr(result, 'contains_pii'), "Result must have 'contains_pii' field"
    assert hasattr(result, 'is_disturbing'), "Result must have 'is_disturbing' field"
    assert hasattr(result, 'is_low_quality'), "Result must have 'is_low_quality' field"
    assert hasattr(result, 'rationale'), "Result must have 'rationale' field"

    assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
    assert isinstance(result.is_disturbing, bool), "is_disturbing should be a boolean"
    assert isinstance(result.is_low_quality, bool), "is_low_quality should be a boolean"
    assert isinstance(result.rationale, str), "rationale should be a string"


async def test_moderate_image_rationale_not_empty():
    """Verify that rationale field is not empty"""
    model = _get_model()
    image_bytes = _load_test_image()

    with image_moderation_agent.override(model=TestModel()):
        result = await moderate_image(model, image_bytes, media_type="image/jpeg")

    assert result.rationale, "Rationale should not be empty"
    assert len(result.rationale) > 0, "Rationale should contain text"

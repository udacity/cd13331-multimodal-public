"""
Tests for environment setup and external model connectivity.

These tests verify that:
1. The .env file is properly configured with required variables
2. The external model (Gemini) can be successfully called

Note: Integration tests make REAL API calls to verify connectivity.
They are marked with pytest markers to allow selective running.
"""

import os
import pytest
from pathlib import Path
from pydantic_ai import Agent, models

from multimodal_moderation.env import (
    GEMINI_API_KEY,
    USER_API_KEY,
    DEFAULT_GOOGLE_MODEL,
    get_default_model_choice,
)


# Environment configuration tests


def test_env_file_exists():
    """Verify that .env file exists in the project root"""
    # Look for .env file in the solution directory
    env_path = Path(__file__).parent.parent / ".env"
    assert env_path.exists(), \
        f".env file should exist at {env_path}. Create it from .env.example if needed."


def test_gemini_api_key_is_set():
    """Verify GEMINI_API_KEY environment variable is set"""
    assert GEMINI_API_KEY, \
        "GEMINI_API_KEY must be set in .env file. Get your API key from https://aistudio.google.com/apikey"
    assert len(GEMINI_API_KEY) > 0, \
        "GEMINI_API_KEY cannot be empty"
    assert not GEMINI_API_KEY.startswith("your-"), \
        "GEMINI_API_KEY appears to be a placeholder. Replace with actual API key from Google AI Studio"


def test_user_api_key_is_set():
    """Verify USER_API_KEY environment variable is set"""
    assert USER_API_KEY, \
        "USER_API_KEY must be set in .env file"
    assert len(USER_API_KEY) > 0, \
        "USER_API_KEY cannot be empty"


def test_default_google_model_is_set():
    """Verify DEFAULT_GOOGLE_MODEL environment variable is set"""
    assert DEFAULT_GOOGLE_MODEL, \
        "DEFAULT_GOOGLE_MODEL must be set in .env file"
    assert len(DEFAULT_GOOGLE_MODEL) > 0, \
        "DEFAULT_GOOGLE_MODEL cannot be empty"
    assert DEFAULT_GOOGLE_MODEL.startswith("gemini-"), \
        f"DEFAULT_GOOGLE_MODEL should be a Gemini model (e.g., 'gemini-2.5-flash'), got '{DEFAULT_GOOGLE_MODEL}'"


# External model connectivity test


@pytest.mark.integration
async def test_can_call_gemini_api():
    """Verify that a simple API call to Gemini succeeds"""
    # Re-enable real model requests for this test
    original_allow = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Create a simple test agent
        test_agent = Agent(instructions="You are a helpful assistant.")

        # Get model configuration from env
        model_choice = get_default_model_choice()

        # Make a simple API call
        result = await test_agent.run(
            "Say hello",
            model=model_choice.model,
            model_settings=model_choice.model_settings,
        )

        # Verify we got a response
        assert result is not None, \
            "API call should return a result object"
        assert result.output is not None, \
            "Result should have output (indicates successful API call)"
        assert len(str(result.output)) > 0, \
            "Output should not be empty (indicates model generated content)"

    except Exception as e:
        pytest.fail(
            f"Failed to call Gemini API. Error: {str(e)}\n"
            f"Check that:\n"
            f"1. GEMINI_API_KEY is valid in .env\n"
            f"2. You have internet connectivity\n"
            f"3. The API key has proper permissions\n"
            f"4. You haven't exceeded API rate limits"
        )
    finally:
        # Restore original setting
        models.ALLOW_MODEL_REQUESTS = original_allow

"""
Evaluation Configuration

This module sets up the environment for running evaluations (evals).

WHAT ARE EVALS?
Evals test the quality of your AI agent's outputs. Unlike unit tests (which test
that functions exist and return the right types), evals test whether the AI is
actually making good decisions about content moderation.

TWO MODELS:
1. Model under test: The moderation agent we're evaluating (from get_default_model_choice)
2. Judge model: A separate LLM that evaluates if the moderation decisions are correct

This separation allows you to use a more powerful model as the judge (e.g., GPT-4)
while testing a faster model (e.g., Gemini Flash).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodal_moderation.env import GEMINI_API_KEY, EVAL_JUDGE_MODEL, get_default_model_choice
from multimodal_moderation.types.model_choice import ModelChoice
from pydantic_ai.models.google import GoogleModel, GoogleProvider, GoogleModelSettings


def get_model_under_test() -> ModelChoice:
    """
    Get the model configuration for the moderation agent being tested.

    This uses the same configuration as the main application to ensure
    we're testing what will actually run in production.

    Returns: ModelChoice configured for the moderation agent
    """
    return get_default_model_choice()


def get_judge_model() -> tuple[GoogleModel, GoogleModelSettings]:
    """
    Get the model that will act as the judge for evaluations.

    The judge model evaluates whether the moderation agent's decisions
    and rationales are correct. This is called "LLM-as-a-judge" evaluation.

    The judge can be the same model as the one being tested, or a different one
    (configured via EVAL_JUDGE_MODEL in .env).

    Returns: (judge_model, model_settings) tuple
    """
    provider = GoogleProvider(api_key=GEMINI_API_KEY)
    judge_model = GoogleModel(EVAL_JUDGE_MODEL, provider=provider)
    # Disable thinking mode for faster evaluation
    model_settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})

    return judge_model, model_settings

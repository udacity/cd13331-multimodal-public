"""
Image Moderation Evaluation Suite

This module defines test cases and runs evaluations for the image moderation agent.
"""

import sys
from pathlib import Path
from typing import List, Any
import tenacity
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
from pydantic_ai.retries import RetryConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multimodal_moderation.agents.image_agent import moderate_image
from multimodal_moderation.types.moderation_result import ImageModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import ImageModerationCheck

judge_model, judge_settings = get_judge_model()


class ImageInput(BaseModel):
    """Input schema for image moderation test cases."""
    image_file: str = Field(description="Path to image file to moderate")


async def run_image_moderation(inputs: List[ImageInput]) -> ImageModerationResult:
    """
    Run the image moderation agent on a test input.
    """
    assert len(inputs) == 1, "Image moderation expects exactly one input"
    image_path = inputs[0].image_file
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Determine media type (assuming jpg for test data based on common patterns, 
    # but extension check is safer)
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    
    model_choice = get_model_under_test()
    return await moderate_image(model_choice, image_bytes, media_type)


cases: List[Case[List[ImageInput], ImageModerationResult, Any]] = [
    Case(
        name="professional_image",
        inputs=[ImageInput(image_file=get_test_data_path("professional_image.jpg"))],
        metadata={"category": "image_moderation"},
        evaluators=(
            ImageModerationCheck(
                expected_pii=False,
                expected_disturbing=False,
                expected_low_quality=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the image is professional and safe.",
                include_input=False,
            ),
        ),
    ),
    Case(
        name="image_with_pii",
        # Updated to use the existing file 'image_with_person.jpg'
        inputs=[ImageInput(image_file=get_test_data_path("image_with_person.jpg"))],
        metadata={"category": "image_moderation"},
        evaluators=(
            ImageModerationCheck(
                expected_pii=True,
                expected_disturbing=False,
                expected_low_quality=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should identify that the image contains a person or PII.",
                include_input=False,
            ),
        ),
    ),
    Case(
        name="low_quality_image",
        # Updated to use the existing file 'low_quality_image.jpg' instead of 'disturbing_image.jpg'
        inputs=[ImageInput(image_file=get_test_data_path("low_quality_image.jpg"))],
        metadata={"category": "image_moderation"},
        evaluators=(
            ImageModerationCheck(
                expected_pii=False,
                expected_disturbing=False,
                expected_low_quality=True,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the image is considered low quality.",
                include_input=False,
            ),
        ),
    ),
]


image_dataset = Dataset[List[ImageInput], ImageModerationResult, Any](
    cases=create_repeated_cases(cases),
    evaluators=[
        IsInstance(type_name="ImageModerationResult"),
        HasRationale(),
    ],
)


async def main():
    retry_config = RetryConfig(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_full_jitter(multiplier=0.5, max=15),
    )

    report = await image_dataset.evaluate(
        run_image_moderation,
        retry_task=retry_config,
        retry_evaluators=retry_config,
    )

    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
import sys
from pathlib import Path
from typing import List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
import tenacity
from pydantic_ai.retries import RetryConfig

from multimodal_moderation.agents.image_agent import moderate_image
from multimodal_moderation.types.moderation_result import ImageModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import ImageModerationCheck

# Get the models for evaluation
judge_model, judge_settings = get_judge_model()


class ImageInput(BaseModel):
    """Input schema for image moderation test cases."""
    image_file: str = Field(description="Path to image file to moderate")


async def run_image_moderation(inputs: List[ImageInput]) -> ImageModerationResult:
    """Run the image moderation agent on a test input."""
    assert len(inputs) == 1, "Image moderation expects exactly one input"
    image_bytes = Path(inputs[0].image_file).read_bytes()
    # Use the model under test (not the judge model!)
    model_choice = get_model_under_test()
    return await moderate_image(model_choice, image_bytes, media_type="image/jpeg")


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
                rubric="The rationale should confirm the image is professional with no PII, disturbing content, or quality issues",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="image_with_person",
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
                rubric="The rationale should specifically identify the presence of a person or personally identifiable visual information",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="low_quality_image",
        inputs=[ImageInput(image_file=get_test_data_path("low_quality_image.jpg"))],
        metadata={"category": "image_moderation"},

        # TODO: define the evaluators for this case. We need:
        # 1. An ImageModerationCheck that expects expected_pii=True, expected_disturbing=False, expected_low_quality=True
        # 2. An LLMJudge that uses the judge_model and has a rubric that checks that the rationale describes specific quality
        #    issues (blurry, pixelated, poor exposure, etc.)
        evaluators=(
            ImageModerationCheck(
                expected_pii=...,  # TODO
                expected_disturbing=..., # TODO
                expected_low_quality=..., # TODO
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should describe specific quality issues (blurry, pixelated, poor exposure, etc.)",
                include_input=...,  # TODO: in this case it is probably useful to include the input image for contextue
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

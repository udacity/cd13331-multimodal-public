import sys
from pathlib import Path
from typing import List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
import tenacity
from pydantic_ai.retries import RetryConfig

from multimodal_moderation.agents.video_agent import moderate_video
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import VideoModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import VideoModerationCheck

judge_model, judge_settings = get_judge_model()


class VideoInput(BaseModel):
    video_file: str = Field(description="Path to video file to moderate")


async def run_video_moderation(inputs: List[VideoInput]) -> VideoModerationResult:
    assert len(inputs) == 1
    video_bytes = Path(inputs[0].video_file).read_bytes()
    model_choice = get_model_under_test()
    return await moderate_video(model_choice, video_bytes, media_type="video/mp4")


cases: List[Case[List[VideoInput], VideoModerationResult, Any]] = [
    Case(
        name="professional_video",
        inputs=[VideoInput(video_file=get_test_data_path("professional_video.mp4"))],
        metadata={"category": "video_moderation"},
        evaluators=(
            VideoModerationCheck(
                expected_pii=False,
                expected_disturbing=False,
                expected_low_quality=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should confirm the video is professional with no faces, disturbing content, or quality issues",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="video_with_face",
        inputs=[VideoInput(video_file=get_test_data_path("video_with_face.mp4"))],
        metadata={"category": "video_moderation"},
        evaluators=(
            VideoModerationCheck(
                expected_pii=True,
                expected_disturbing=False,
                expected_low_quality=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should specifically mention the presence of a person's face or identifying features",
                include_input=True,
            ),
        ),
    ),
]


video_dataset = Dataset[List[VideoInput], VideoModerationResult, Any](
    cases=create_repeated_cases(cases),
    evaluators=[
        IsInstance(type_name="VideoModerationResult"),
        HasRationale(),
    ],
)


async def main():
    retry_config = RetryConfig(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_full_jitter(multiplier=0.5, max=15),
    )

    report = await video_dataset.evaluate(
        run_video_moderation,
        retry_task=retry_config,
        retry_evaluators=retry_config,
    )

    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

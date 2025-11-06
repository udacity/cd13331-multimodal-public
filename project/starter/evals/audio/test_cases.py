import sys
from pathlib import Path
from typing import List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
import tenacity
from pydantic_ai.retries import RetryConfig

from multimodal_moderation.agents.audio_agent import moderate_audio
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import AudioModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import AudioModerationCheck, HasTranscription

judge_model, judge_settings = get_judge_model()


class AudioInput(BaseModel):
    audio_file: str = Field(description="Path to audio file to moderate")


async def run_audio_moderation(inputs: List[AudioInput]) -> AudioModerationResult:
    assert len(inputs) == 1
    audio_bytes = Path(inputs[0].audio_file).read_bytes()
    model_choice = get_model_under_test()
    return await moderate_audio(model_choice, audio_bytes, media_type="audio/mp3")


cases: List[Case[List[AudioInput], AudioModerationResult, Any]] = [
    Case(
        name="professional_audio",
        inputs=[AudioInput(audio_file=get_test_data_path("professional_audio.mp3"))],
        metadata={"category": "audio_moderation"},
        evaluators=(
            AudioModerationCheck(
                expected_pii=False,
                expected_unfriendly=False,
                expected_unprofessional=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the transcribed content is professional and friendly",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="audio_with_pii",
        inputs=[AudioInput(audio_file=get_test_data_path("audio_with_pii.mp3"))],
        metadata={"category": "audio_moderation"},
        evaluators=(
            AudioModerationCheck(
                expected_pii=True,
                expected_unfriendly=False,
                expected_unprofessional=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should identify specific PII mentioned in the transcription (names, addresses, phone numbers)",
                include_input=True,
            ),
        ),
    ),
]


audio_dataset = Dataset[List[AudioInput], AudioModerationResult, Any](
    cases=create_repeated_cases(cases),
    evaluators=[
        IsInstance(type_name="AudioModerationResult"),
        HasRationale(),
        HasTranscription(),
    ],
)


async def main():
    retry_config = RetryConfig(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_full_jitter(multiplier=0.5, max=15),
    )

    report = await audio_dataset.evaluate(
        run_audio_moderation,
        retry_task=retry_config,
        retry_evaluators=retry_config,
    )

    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

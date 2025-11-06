from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from multimodal_moderation.types.moderation_result import VideoModerationResult


@dataclass
class VideoModerationCheck(Evaluator):
    expected_pii: bool
    expected_disturbing: bool
    expected_low_quality: bool

    async def evaluate(self, ctx: EvaluatorContext[str, VideoModerationResult]) -> bool:
        return (
            ctx.output.contains_pii == self.expected_pii
            and ctx.output.is_disturbing == self.expected_disturbing
            and ctx.output.is_low_quality == self.expected_low_quality
        )

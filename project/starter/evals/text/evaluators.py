from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from multimodal_moderation.types.moderation_result import TextModerationResult


@dataclass
class TextModerationCheck(Evaluator):
    expected_pii: bool
    expected_unfriendly: bool
    expected_unprofessional: bool

    async def evaluate(self, ctx: EvaluatorContext[str, TextModerationResult]) -> bool:
        return (
            ctx.output.contains_pii == self.expected_pii
            and ctx.output.is_unfriendly == self.expected_unfriendly
            and ctx.output.is_unprofessional == self.expected_unprofessional
        )

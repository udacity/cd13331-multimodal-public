"""
Common Evaluators

Shared evaluators that can be used across all moderation types (text, image, video, audio).

EVALUATORS:
Evaluators are functions that check if an output is correct. They return True (pass)
or False (fail). You can have multiple evaluators per test case.

TYPES OF EVALUATORS:
- Rule-based: Check specific conditions (like this HasRationale)
- LLM-based: Use an LLM to judge quality (like LLMJudge)
"""

from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from multimodal_moderation.types.moderation_result import ModerationResult


@dataclass
class HasRationale(Evaluator):
    """
    Check that the moderation result includes a non-empty rationale.

    This is a basic sanity check - every moderation result should explain
    its decision, not just return boolean flags.
    """

    async def evaluate(self, ctx: EvaluatorContext[str, ModerationResult]) -> bool:
        """Returns True if rationale is not empty."""
        return len(ctx.output.rationale) > 0

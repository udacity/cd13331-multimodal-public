"""
Text Moderation Evaluation Suite

This module defines test cases and runs evaluations for the text moderation agent.

EVALUATION FLOW:
1. Load test cases from test_data/ directory
2. For each test case, run the moderation agent (model under test)
3. Use evaluators to check if the output is correct:
   - TextModerationCheck: Verifies flags are set correctly
   - LLMJudge: Evaluates if the rationale makes sense
4. Generate a report showing pass/fail for each test case

RUNNING EVALS:
    python evals/text/test_cases.py
"""

import sys
from pathlib import Path
from typing import List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
import tenacity
from pydantic_ai.retries import RetryConfig

from multimodal_moderation.agents.text_agent import moderate_text
from multimodal_moderation.types.moderation_result import TextModerationResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_evaluators import HasRationale
from config import get_model_under_test, get_judge_model
from utils import create_repeated_cases, get_test_data_path

sys.path.insert(0, str(Path(__file__).parent))
from evaluators import TextModerationCheck

# Get the models for evaluation
judge_model, judge_settings = get_judge_model()


class TextInput(BaseModel):
    """Input schema for text moderation test cases."""

    text_file: str = Field(description="Path to text file to moderate")


async def run_text_moderation(inputs: List[TextInput]) -> TextModerationResult:
    """
    Run the text moderation agent on a test input.

    This is the function being evaluated. It loads text from a file and
    runs it through the moderation agent (using the model under test).

    Args:
        inputs: List with one TextInput containing path to test file

    Returns:
        TextModerationResult from the moderation agent
    """
    assert len(inputs) == 1, "Text moderation expects exactly one input"
    text = Path(inputs[0].text_file).read_text()
    # Use the model under test (not the judge model!)
    model_choice = get_model_under_test()
    return await moderate_text(model_choice, text)


# Test Cases
# Each case defines:
# - name: Identifier for this test
# - inputs: The test data (path to file)
# - evaluators: How to check if the output is correct
#   - TextModerationCheck: Checks boolean flags match expected values
#   - LLMJudge: Uses an LLM to evaluate if the rationale is good
cases: List[Case[List[TextInput], TextModerationResult, Any]] = [
    # TODO: fill in the missing
    Case(
        name="professional_text",
        # Read the text from a file for repeatibility (we could have also inlined it here)
        inputs=[TextInput(text_file=get_test_data_path("professional_text.txt"))],
        metadata={"category": "text_moderation"},

        # TODO: fill the parameters for the evaluators for this case. We need:
        # 1. A TextModerationCheck that expects expected_pii=False, expected_unfriendly=False, expected_unprofessional=False
        # 2. An LLMJudge that uses the judge_model and has a rubric that checks that the rationale explains why the text is 
        #    professional and friendly, with no flags raised. Example: 
        #    "The rationale should explain why the text is professional and friendly."
        evaluators=(
            # Check that no safety flags are raised for professional text
            TextModerationCheck(
                expected_pii=...,  # TODO
                expected_unfriendly=...,  # TODO
                expected_unprofessional=...,  # TODO
            ),
            # Use judge model to evaluate if the rationale makes sense
            LLMJudge(
                model=...,  # TODO: add the model to be used as judge (judge_model)
                rubric=...,  # TODO: add a rubric that checks the rationale explains why the text is professional and friendly.
                include_input=..., # TODO: in this case it is probably useful to include the input text for context, so set this to True
            ),
        ),
    ),
    Case(
        name="text_with_pii",
        inputs=[TextInput(text_file=get_test_data_path("text_with_pii.txt"))],
        metadata={"category": "text_moderation"},
        evaluators=(
            TextModerationCheck(
                expected_pii=True,
                expected_unfriendly=False,
                expected_unprofessional=False,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should identify specific PII items (name, address, email, phone number, account number)",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="unfriendly_text",
        inputs=[TextInput(text_file=get_test_data_path("unfriendly_text.txt"))],
        metadata={"category": "text_moderation"},
        evaluators=(
            TextModerationCheck(
                expected_pii=False,
                expected_unfriendly=True,
                expected_unprofessional=True,
            ),
            LLMJudge(
                model=judge_model,
                rubric="The rationale should explain why the tone is unfriendly and unprofessional, citing specific problematic phrases",
                include_input=True,
            ),
        ),
    ),
]


# Create the dataset with all test cases
# create_repeated_cases() repeats each case EVAL_NUM_REPEATS times to measure consistency
text_dataset = Dataset[List[TextInput], TextModerationResult, Any](\

    # TODO: use the create_repeated_cases function to create the dataset with the test cases defined above
    # repeated EVAL_NUM_REPEATS times (as defined in .env). This helps measure consistency of the model under test
    # and reduces the variance of the measurements.
    # HINT: you need to pass cases as the argument to create_repeated_cases
    cases=...,
    evaluators=[
        # Global evaluators that apply to all test cases
        IsInstance(type_name="TextModerationResult"),  # Check correct return type
        HasRationale(),  # Check that rationale is not empty
    ],
)


async def main():
    """
    Run the evaluation suite.

    This will:
    1. Run each test case through the moderation agent
    2. Evaluate outputs with both rule-based and LLM judges
    3. Print a report showing which tests passed/failed
    """
    # Retry configuration for API calls (LLMs can be flaky)
    retry_config = RetryConfig(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_full_jitter(multiplier=0.5, max=15),
    )

    # Run all evaluations

    # TODO: call await text_dataset.evaluate() with the appropriate parameters to enable retries
    # HINT: you need to pass run_text_moderation as the function to test,
    # and both retry_task and retry_evaluators should be set to retry_config
    report = ...  # TODO

    # Print results
    report.print(include_input=True, include_output=True, include_durations=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

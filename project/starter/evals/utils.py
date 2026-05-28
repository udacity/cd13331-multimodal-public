"""
Evaluation Utilities

Helper functions for running evaluations.

WHY REPEAT TEST CASES?
LLMs are non-deterministic - they may give different answers on repeated runs.
By running each test case multiple times (EVAL_NUM_REPEATS), we can measure
consistency and catch flaky behavior.

For example, if a test passes 4/5 times, the agent might be inconsistent rather
than truly reliable.
"""

import sys
from pathlib import Path
from typing import List, TypeVar
from pydantic_evals import Case

sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodal_moderation.env import EVAL_NUM_REPEATS

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")
T_Metadata = TypeVar("T_Metadata")

# Get the evals directory (parent of this file)
EVALS_DIR = Path(__file__).parent
TEST_DATA_DIR = EVALS_DIR / "test_data"


def get_test_data_path(filename: str) -> str:
    """
    Get the absolute path to a test data file.

    Uses Path(__file__) to locate files relative to this module,
    which works regardless of where the script is run from.

    Args:
        filename: Name of the file in the test_data directory (e.g., "professional_text.txt")

    Returns:
        Absolute path to the file as a string

    Example:
        >>> path = get_test_data_path("professional_text.txt")
        >>> # Returns: "/absolute/path/to/evals/test_data/professional_text.txt"
    """
    return str(TEST_DATA_DIR / filename)


def create_repeated_cases(
    base_cases: List[Case[T_Input, T_Output, T_Metadata]],
    num_repeats: int | None = None,
) -> List[Case[T_Input, T_Output, T_Metadata]]:
    """
    Repeat each test case multiple times to measure LLM consistency.

    Takes a list of test cases and creates copies with different names
    (e.g., "professional_text_run_1", "professional_text_run_2", etc.)

    Args:
        base_cases: Original test cases
        num_repeats: How many times to repeat each (defaults to EVAL_NUM_REPEATS from .env)

    Returns:
        Expanded list of test cases with metadata tracking which run this is
    """
    if num_repeats is None:
        num_repeats = EVAL_NUM_REPEATS

    # If num_repeats is 1 or less, don't repeat
    if num_repeats <= 1:
        return base_cases

    repeated_cases = []
    for base_case in base_cases:
        # Create num_repeats copies of this case
        for run_number in range(1, num_repeats + 1):
            repeated_case = Case(
                name=f"{base_case.name}_run_{run_number}",
                inputs=base_case.inputs,
                expected_output=base_case.expected_output,
                metadata={
                    **(base_case.metadata or {}),
                    "run_number": run_number,
                    "base_case": base_case.name,
                },
                evaluators=base_case.evaluators,
            )
            repeated_cases.append(repeated_case)

    return repeated_cases

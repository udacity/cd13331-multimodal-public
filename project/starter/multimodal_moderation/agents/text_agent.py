from pydantic_ai import Agent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ModerationResult, TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Detect if:
- the tone of the message is unfriendly
- the tone of the message is unprofessional
- the message contains any personally-identifiable information (PII)
</instructions>

<output>
Provide a detailed rationale for your choices as well as a confidence score between 0 and 1 on your assessment.
</output>
"""


# TODO: Create a Pydantic AI Agent with:
#   - instructions=MODERATION_INSTRUCTIONS
#   - output_type=TextModerationResult
# Hint: Agent is already imported from pydantic_ai
text_moderation_agent = None  # Replace with your Agent


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:

    # TODO: Run the text_moderation_agent with a prompt containing the text,
    #       then return result.output
    # NOTE: in the class we used agent.run_sync but here we need to use
    #       await agent.run since this is an async function. They work exactly
    #       the same. Just do:
    #           result = await agent.run([parameters])
    #       instead of:
    #           result = agent.run_sync([parameters])
    #       like we did in the class.
    # Make sure to pass: model=model_choice.model and model_settings=model_choice.model_settings
    raise NotImplementedError("TODO: Implement text moderation")

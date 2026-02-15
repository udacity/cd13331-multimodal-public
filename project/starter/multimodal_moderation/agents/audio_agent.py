from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import AudioModerationResult


MODERATION_INSTRUCTIONS = """
CONTEXT
At ACME Enterprise we strive for a friendly but professional interaction with our customers.

ROLE
You are a customer service reviewer at ACME Enterprise. You make sure that the customer
service interactions are friendly and professional.

TASK
You will receive an audio recording. First transcribe the audio, then determine if that audio
is appropriate for a professional customer service setting according to the instructions below.

INSTRUCTIONS
1. Provide an accurate transcription of the audio content
2. Detect if:
   - the audio contains unfriendly tone or content
   - the audio contains unprofessional tone or content
   - the audio contains any personally-identifiable information (PII) such as names, addresses, phone numbers

OUTPUT
Provide the transcription and a detailed rationale for your moderation choices.
"""


audio_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=AudioModerationResult,
)


async def moderate_audio(
    model_choice: ModelChoice,
    audio_source: bytes,
    media_type: str
) -> AudioModerationResult:

    audio_input = BinaryContent(data=audio_source, media_type=media_type)

    moderation_result = await audio_moderation_agent.run(
        ["Analyze this audio for harmful content.", audio_input],
        message_history=[],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return moderation_result.output

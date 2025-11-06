from typing import Literal
from pydantic import BaseModel, Field


class ModerationResult(BaseModel):

    rationale: str = Field(description="Explanation of what was harmful and why")


class TextModerationResult(ModerationResult):

    contains_pii: bool = Field(description="Whether the message contains any personally-identifiable information (PII)")
    is_unfriendly: bool = Field(description="Whether unfriendly tone or content was detected")
    is_unprofessional: bool = Field(description="Whether unprofessional tone or content was detected")


class ImageModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the image contains any person, part of a person, or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the image is disturbing")
    is_low_quality: bool = Field(description="Whether the image is low quality")


class VideoModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the video contains any person or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the video is disturbing")
    is_low_quality: bool = Field(description="Whether the video is low quality")


# TODO: Create AudioModerationResult class that inherits from ModerationResult and contains:
#   - transcription: str to contain the transcription of the audio
#   - contains_pii: bool to contain a flag for whether the audio contains any personally-identifiable
#       information (PII) such as names, addresses, phone numbers
#   - is_unfriendly: bool to contain a flag for whether unfriendly tone or content was detected
#   - is_unprofessional: bool to contain a flag for whether unprofessional tone or content was detected
class AudioModerationResult(ModerationResult):

    ...  # Replace with your implementation

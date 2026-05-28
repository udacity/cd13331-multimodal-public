"""
Tests for moderation result classes.

These tests verify that all moderation result classes have the correct attributes
with the expected types and are properly defined as Pydantic models.
"""

import pytest
from pydantic import ValidationError

from multimodal_moderation.types.moderation_result import (
    ModerationResult,
    TextModerationResult,
    ImageModerationResult,
    VideoModerationResult,
    AudioModerationResult,
)


class TestModerationResult:
    """Test the base ModerationResult class"""

    def test_has_rationale_field(self):
        """Verify ModerationResult has rationale field"""
        result = ModerationResult(rationale="Test rationale")
        assert hasattr(result, "rationale"), "ModerationResult should have a 'rationale' attribute"
        assert isinstance(result.rationale, str), "rationale should be a string"
        assert result.rationale == "Test rationale", "rationale should contain the provided value"

    def test_rationale_is_required(self):
        """Verify rationale field is required"""
        with pytest.raises(ValidationError, match="rationale"):
            ModerationResult()

    def test_is_pydantic_model(self):
        """Verify ModerationResult is a Pydantic BaseModel"""
        result = ModerationResult(rationale="Test")
        assert hasattr(result, "model_dump"), "ModerationResult should have model_dump method (Pydantic BaseModel)"
        assert hasattr(result, "model_validate"), "ModerationResult should have model_validate method (Pydantic BaseModel)"


class TestTextModerationResult:
    """Test the TextModerationResult class"""

    def test_has_all_required_fields(self):
        """Verify TextModerationResult has all required fields"""
        result = TextModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_unfriendly=False,
            is_unprofessional=True,
        )

        assert hasattr(result, "rationale"), "TextModerationResult should have 'rationale' field"
        assert hasattr(result, "contains_pii"), "TextModerationResult should have 'contains_pii' field"
        assert hasattr(result, "is_unfriendly"), "TextModerationResult should have 'is_unfriendly' field"
        assert hasattr(result, "is_unprofessional"), "TextModerationResult should have 'is_unprofessional' field"

    def test_field_types(self):
        """Verify all fields have correct types"""
        result = TextModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_unfriendly=False,
            is_unprofessional=True,
        )

        assert isinstance(result.rationale, str), "rationale should be a string"
        assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
        assert isinstance(result.is_unfriendly, bool), "is_unfriendly should be a boolean"
        assert isinstance(result.is_unprofessional, bool), "is_unprofessional should be a boolean"

    def test_inherits_from_moderation_result(self):
        """Verify TextModerationResult inherits from ModerationResult"""
        assert issubclass(TextModerationResult, ModerationResult), \
            "TextModerationResult should inherit from ModerationResult"

    def test_all_fields_are_required(self):
        """Verify all fields are required"""
        with pytest.raises(ValidationError, match="contains_pii|is_unfriendly|is_unprofessional"):
            TextModerationResult(rationale="Test")


class TestImageModerationResult:
    """Test the ImageModerationResult class"""

    def test_has_all_required_fields(self):
        """Verify ImageModerationResult has all required fields"""
        result = ImageModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_disturbing=False,
            is_low_quality=True,
        )

        assert hasattr(result, "rationale"), "ImageModerationResult should have 'rationale' field"
        assert hasattr(result, "contains_pii"), "ImageModerationResult should have 'contains_pii' field"
        assert hasattr(result, "is_disturbing"), "ImageModerationResult should have 'is_disturbing' field"
        assert hasattr(result, "is_low_quality"), "ImageModerationResult should have 'is_low_quality' field"

    def test_field_types(self):
        """Verify all fields have correct types"""
        result = ImageModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_disturbing=False,
            is_low_quality=True,
        )

        assert isinstance(result.rationale, str), "rationale should be a string"
        assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
        assert isinstance(result.is_disturbing, bool), "is_disturbing should be a boolean"
        assert isinstance(result.is_low_quality, bool), "is_low_quality should be a boolean"

    def test_inherits_from_moderation_result(self):
        """Verify ImageModerationResult inherits from ModerationResult"""
        assert issubclass(ImageModerationResult, ModerationResult), \
            "ImageModerationResult should inherit from ModerationResult"

    def test_all_fields_are_required(self):
        """Verify all fields are required"""
        with pytest.raises(ValidationError, match="contains_pii|is_disturbing|is_low_quality"):
            ImageModerationResult(rationale="Test")


class TestVideoModerationResult:
    """Test the VideoModerationResult class"""

    def test_has_all_required_fields(self):
        """Verify VideoModerationResult has all required fields"""
        result = VideoModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_disturbing=False,
            is_low_quality=True,
        )

        assert hasattr(result, "rationale"), "VideoModerationResult should have 'rationale' field"
        assert hasattr(result, "contains_pii"), "VideoModerationResult should have 'contains_pii' field"
        assert hasattr(result, "is_disturbing"), "VideoModerationResult should have 'is_disturbing' field"
        assert hasattr(result, "is_low_quality"), "VideoModerationResult should have 'is_low_quality' field"

    def test_field_types(self):
        """Verify all fields have correct types"""
        result = VideoModerationResult(
            rationale="Test rationale",
            contains_pii=True,
            is_disturbing=False,
            is_low_quality=True,
        )

        assert isinstance(result.rationale, str), "rationale should be a string"
        assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
        assert isinstance(result.is_disturbing, bool), "is_disturbing should be a boolean"
        assert isinstance(result.is_low_quality, bool), "is_low_quality should be a boolean"

    def test_inherits_from_moderation_result(self):
        """Verify VideoModerationResult inherits from ModerationResult"""
        assert issubclass(VideoModerationResult, ModerationResult), \
            "VideoModerationResult should inherit from ModerationResult"

    def test_all_fields_are_required(self):
        """Verify all fields are required"""
        with pytest.raises(ValidationError, match="contains_pii|is_disturbing|is_low_quality"):
            VideoModerationResult(rationale="Test")


class TestAudioModerationResult:
    """Test the AudioModerationResult class"""

    def test_has_all_required_fields(self):
        """Verify AudioModerationResult has all required fields"""
        result = AudioModerationResult(
            rationale="Test rationale",
            transcription="Test transcription",
            contains_pii=True,
            is_unfriendly=False,
            is_unprofessional=True,
        )

        assert hasattr(result, "rationale"), "AudioModerationResult should have 'rationale' field"
        assert hasattr(result, "transcription"), "AudioModerationResult should have 'transcription' field"
        assert hasattr(result, "contains_pii"), "AudioModerationResult should have 'contains_pii' field"
        assert hasattr(result, "is_unfriendly"), "AudioModerationResult should have 'is_unfriendly' field"
        assert hasattr(result, "is_unprofessional"), "AudioModerationResult should have 'is_unprofessional' field"

    def test_field_types(self):
        """Verify all fields have correct types"""
        result = AudioModerationResult(
            rationale="Test rationale",
            transcription="Test transcription",
            contains_pii=True,
            is_unfriendly=False,
            is_unprofessional=True,
        )

        assert isinstance(result.rationale, str), "rationale should be a string"
        assert isinstance(result.transcription, str), "transcription should be a string"
        assert isinstance(result.contains_pii, bool), "contains_pii should be a boolean"
        assert isinstance(result.is_unfriendly, bool), "is_unfriendly should be a boolean"
        assert isinstance(result.is_unprofessional, bool), "is_unprofessional should be a boolean"

    def test_inherits_from_moderation_result(self):
        """Verify AudioModerationResult inherits from ModerationResult"""
        assert issubclass(AudioModerationResult, ModerationResult), \
            "AudioModerationResult should inherit from ModerationResult"

    def test_all_fields_are_required(self):
        """Verify all fields are required"""
        with pytest.raises(ValidationError, match="transcription|contains_pii|is_unfriendly|is_unprofessional"):
            AudioModerationResult(rationale="Test", transcription="Test")

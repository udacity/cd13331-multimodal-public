import uuid
import shutil
from pathlib import Path
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from multimodal_moderation.env import PHOENIX_URL


def setup_tracing():
    """Initialize OpenTelemetry tracing with Phoenix backend for observability."""
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

    # Add OpenInference processor for AI-specific span processing
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

    # Add OTLP exporter to send traces to Phoenix
    exporter = OTLPSpanExporter(endpoint=f"{PHOENIX_URL}/v1/traces")
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))


def get_tracer(name: str):
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name)


def add_media_to_span(span: trace.Span, file_path: str, media_type: str, index: int):
    """
    Save uploaded media file and add metadata to the tracing span.

    This allows Phoenix to display media files in the trace viewer.

    Args:
        span: The OpenTelemetry span to add attributes to
        file_path: Path to the uploaded media file
        media_type: Type of media (e.g., "image_moderation", "video_moderation")
        index: Index of this media file (for handling multiple files)
    """
    try:
        # Create directory to store uploaded media files
        uploads_dir = Path("./uploaded_media")
        uploads_dir.mkdir(exist_ok=True)

        # Copy file with unique timestamp to avoid name collisions
        source_path = Path(file_path)
        timestamp = str(uuid.uuid4())[:8]
        dest_path = uploads_dir / f"{timestamp}_{source_path.name}"
        shutil.copy(file_path, dest_path)

        # Add file metadata to span for Phoenix visualization
        absolute_path = dest_path.resolve()
        span.set_attributes({
            f"input.{media_type}.{index}.url": f"file://{absolute_path}",
            f"input.{media_type}.{index}.filename": source_path.name,
            f"input.{media_type}.{index}.size_bytes": source_path.stat().st_size,
        })
    except Exception:
        # Silently fail - tracing should not break the app
        pass

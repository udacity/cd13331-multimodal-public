"""
Gradio Chat Interface for ACME Customer Service Training

This module provides an interactive chat interface where users practice being
customer service agents. The AI plays the role of a customer with complaints,
and all service agent messages/media are moderated before being sent to the AI.

DATA FLOW:
1. Service agent (user) sends message/media → Gradio interface
2. Content sent to FastAPI moderation service (via HTTP)
3. If safe → Content sent to Gemini AI (which plays customer role)
4. AI customer response returned to user

KEY COMPONENTS:
- check_content_safety(): Calls FastAPI backend to moderate content
- ChatSessionWithTracing: Manages conversation state and tracing
- create_chat_interface(): Builds the Gradio UI
"""

import os
import requests
import gradio as gr
import uuid
from pathlib import Path
from typing import List, Tuple, Any
from pydantic_ai.messages import BinaryContent
import logging

from multimodal_moderation.env import USER_API_KEY, API_BASE_URL
from multimodal_moderation.tracing import setup_tracing, get_tracer, add_media_to_span
from multimodal_moderation.agents.customer_agent import customer_agent
from multimodal_moderation.utils import detect_file_type
from opentelemetry import trace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up tracing for observability with Phoenix
setup_tracing()
tracer = get_tracer(__name__)

# Constants
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Moderation Configuration
# Maps content types to their FastAPI backend endpoints and safety flags.
# This configuration determines which API endpoint to call for each content type
# and which flags indicate unsafe content.
MODERATION_CONFIG = {
    "text": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_text",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
    "image": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_image_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "video": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_video_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "audio": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_audio_file",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
}


def _call_text_moderation(text: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate text content.

    Makes an HTTP POST request to the text moderation endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: RuntimeError if the moderation service is unavailable
    """
    content_type = "text"
    mime_type = "text/plain"

    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST to FastAPI moderation service
    response = requests.post(
        config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, json={"text": text}
    )

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Add input to tracing span for observability
    span.set_attributes(
        {
            "input.text.content": text,
            "input.text.length": len(text),
        }
    )

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    return result, feedback, content_type, mime_type


def _call_media_moderation(media: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate media files (image/video/audio).

    Detects file type, validates size, and sends to appropriate endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: ValueError if file is too large, RuntimeError if service unavailable
    """
    # Detect MIME type (e.g., "image/png", "video/mp4", "audio/mpeg")
    mime_type = detect_file_type(media, context=media)

    # Validate file size to prevent large uploads
    file_size = os.path.getsize(media)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(f"File too large: {size_mb:.1f}MB. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    # Extract content type from MIME type (e.g., "image" from "image/png")
    content_type = mime_type.split("/")[0]
    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST with file upload to FastAPI moderation service
    with open(media, "rb") as f:
        response = requests.post(
            config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, files={"file": f}
        )

    # Add media metadata to tracing span for Phoenix visualization
    add_media_to_span(span, media, f"{content_type}_moderation", 0)

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    # Special case for audio: include transcription in feedback
    if content_type == "audio" and "transcription" in result:
        feedback = f"Transcription: \"{result['transcription']}\"\n\n{feedback}"

    return result, feedback, content_type, mime_type


def check_content_safety(*, text: str | None = None, media: str | None = None) -> Tuple[bool, str, str]:
    """
    Check if content is safe by calling the moderation backend.

    This is the main entry point for all content moderation. It routes to
    either text or media moderation, then checks if any safety flags are set.

    Args:
        text: Text content to moderate (mutually exclusive with media)
        media: Path to media file to moderate (mutually exclusive with text)

    Returns:
        Tuple of (is_safe, feedback_message, mime_type)
    """
    # Create a tracing span for this moderation check
    # TODO: use the tracer to create a span named "moderate_text"
    # HINT: use tracer.start_as_current_span with the name of the span as argument
    with ... as span:

        # Route to the appropriate moderation function
        if text is not None:
            result, feedback, content_type, mime_type = _call_text_moderation(text, span)

        elif media is not None:
            result, feedback, content_type, mime_type = _call_media_moderation(media, span)

        else:
            raise ValueError("Must provide exactly one of text or media")

        # Add moderation results to tracing span
        span.set_attributes({f"output.{k}": v for k, v in result.items()})

        # Update span name now that we know the content type
        span.update_name(f"moderate_{content_type}")

    # Check if any unsafe flags were set by the moderation service
    config = MODERATION_CONFIG[content_type]
    for flag in config["unsafe_flags"]:
        if result[flag]:
            # Content is unsafe - return False with feedback
            return False, f"Content flagged: {feedback}", mime_type

    # Content is safe - return True with feedback
    return True, feedback, mime_type


class ChatSessionWithTracing:
    """
    Manages a chat session with tracing support.

    Each session has a unique ID and a root tracing span that encompasses
    all chat turns. This allows Phoenix to group all related interactions.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        # Create a root span for the entire conversation
        # TODO: use the tracer to create a span named "conversation" and set an attribute "session.id" with the session_id
        # HINT: use tracer.start_span with the name of the span as argument, and set
        #       an attribute "session.id" with the session_id by using attributes={"session.id": self.session_id}
        # NOTE: this is start_span, NOT start_as_current_span because we want to keep this span open across multiple chat turns
        #      and only close it when the conversation ends.
        self.conversation_span = ... # Replace with your span

    async def chat_with_gemini(self, message: dict, history: List, past_messages: List) -> Tuple[str, List, str]:
        """
        Process a chat turn: moderate content, then send to AI customer.

        DATA FLOW:
        1. Receive customer service agent message/files from Gradio
        2. Moderate each piece of content (text and/or media files) via FastAPI backend
        3. If any content is flagged, block and return error message
        4. If all content is safe, send to Gemini AI (which plays the role of a customer)
        5. Return AI customer's response

        Args:
            message: dict with 'text' and optional 'files' keys from Gradio
            history: Gradio's chat history (for display only, not used)
            past_messages: Pydantic AI's message history (used for agent context)

        Returns:
            Tuple of (response_text, updated_messages, feedback_text)
        """
        # Create a tracing span for this chat turn

        # TODO: use the tracer to create a span named "chat_turn"
        # HINT: use tracer.start_as_current_span with the name of the span as argument, and
        #       set the context to the conversation_span using:
        #           context=trace.set_span_in_context(self.conversation_span)
        #       so that this span is a child of the conversation span. Feel free to add
        #       attributes to the span as needed.
        with ... as span:
            
            logger.info(f"New turn - Text: '{message.get('text', '')[:50]}...', Files: {len(message.get('files', []))}")

            # Build prompt for the AI customer (includes text and media)
            prompt_parts: List[str | BinaryContent] = [
                "This is the next message from the support agent:",
            ]

            # Initialize safety message to empty string, will be set if content is flagged
            safety_message = ""

            # Process each part of the message
            for key, value in message.items():

                # MODERATION STEP 1: Check text content
                if key == "text" and value:

                    # Call moderation backend (FastAPI service)
                    is_safe, safety_message, mime_type = check_content_safety(text=value)

                    if not is_safe:
                        # Content flagged - block and return error
                        feedback = f"⚠️ Content flagged: {safety_message}"
                        response = "[This content was flagged by moderation and not sent to the AI. Please try again.]"

                        # TODO: set an attribute "feedback" in the tracing span with the feedback message
                        # HINT: use span.set_attribute with "feedback" as the key and feedback as the value
                        ...

                        return response, past_messages, feedback

                    # Content safe - add to prompt
                    prompt_parts.append(value)

                # MODERATION STEP 2: Check media files
                elif key == "files" and value:

                    for file_path in value:

                        try:

                            # Call moderation backend (FastAPI service)
                            is_safe, safety_message, mime_type = check_content_safety(media=file_path)

                            if not is_safe:
                                # Content flagged - block and return error
                                feedback = f"⚠️ Content flagged: {safety_message}"
                                response = (
                                    "[This content was flagged by moderation and not sent to the AI. Please try again.]"
                                )

                                return response, past_messages, feedback

                            # Content safe - read file and add to prompt
                            with open(file_path, "rb") as f:
                                file_bytes = f.read()
                            
                            # TODO: create a BinaryContent object with data=file_bytes and media_type=mime_type
                            # and append it to prompt_parts so it's included in the prompt to the AI
                            ...

                        except ValueError as e:
                            raise gr.Error(str(e))

            if not prompt_parts:
                raise gr.Error("Please provide a message or at least one file.")

            # All content passed moderation - send to AI customer
            try:
                with tracer.start_as_current_span("llm_customer"):

                    # GEMINI CALL: Send prompt to AI agent that plays the customer role
                    # TODO: use await customer_agent.run to get the result
                    # HINT: pass the prompt_parts as the first argument, and do not forget to
                    #       pass the past_messages as the message_history otherwise the agent
                    #       will lose context of the conversation across turns.
                    # NOTE: this is await customer_agent.run since this is an async function.
                    result = await customer_agent.run(
                        prompt_parts,
                        message_history=past_messages,
                    )

                logger.info(f"Response generated ({len(result.all_messages())} messages in history)")
                return result.output, result.all_messages(), safety_message

            except Exception as e:
                logger.error(f"Error in chat_with_gemini: {str(e)}")
                raise gr.Error(
                    f"I'm sorry, but I encountered an error while processing your request. "
                    f"Please try again or contact ACME support if the issue persists."
                )

    def end_conversation(self):
        """
        End the conversation and close the tracing span.

        This should be called when the user ends the chat session to properly
        close the conversation span in Phoenix.
        """
        if self.conversation_span:
            self.conversation_span.end()
            logger.info(f"Conversation {self.session_id} ended")
        return "Conversation ended. Refresh the page to start a new session."


def create_chat_interface() -> gr.Blocks:
    """
    Create the Gradio chat interface with moderation feedback.

    LAYOUT:
    - Left: Chat interface (ChatInterface with MultimodalTextbox)
    - Right: Moderation feedback and guidelines sidebar

    STATE MANAGEMENT:
    - past_messages_state: Holds Pydantic AI message history across turns
    - feedback_display: Shows moderation results from the backend
    """
    # Create a chat session that tracks conversation across multiple turns
    chat_session = ChatSessionWithTracing()

    with gr.Blocks(title="ACME Customer Service Training Agent", fill_height=True) as demo:
        # State to hold Pydantic AI's message history (preserves context across turns)
        past_messages_state = gr.State([])

        # Create feedback_display first (with render=False) so we can reference it
        # in ChatInterface's additional_outputs below, then render it in the sidebar later
        feedback_display = gr.Textbox(
            label="💬 Moderation Agent Feedback",
            placeholder="No feedback yet",
            interactive=False,
            visible=True,
            lines=10,
            render=False,  # Don't render yet - will render in sidebar
        )

        # UI Layout
        gr.Markdown("# 🤖 ACME Customer Service Training Agent")
        gr.Markdown("Welcome to ACME Corporation's customer service training!")

        with gr.Row():
            # Left column: Chat interface (75% width)
            with gr.Column(scale=3):
                # TODO: fill the missing arguments to gr.ChatInterface
                gr.ChatInterface(
                    fn=..., # This is the function called at each turn, and should be chat_session.chat_with_gemini
                    type="messages",  # Use newer messages format (supports multimodal)
                    multimodal=...,  # Enable file uploads by setting this to True
                    editable=False,  # Don't allow editing past messages
                    textbox=gr.MultimodalTextbox(
                        file_count="multiple",  # Allow multiple files
                        file_types=...,  # Set this to a list of allowed file types ("image", "video", "audio")
                        sources=["upload", "microphone"],  # Allow file upload and recording
                        placeholder="Type a message, upload files, or record audio...",
                    ),
                    chatbot=gr.Chatbot(
                        show_copy_button=True,
                        type="messages",  # Use messages format for multimodal support
                        placeholder="👋 Start by greeting the customer or introducing yourself. The AI customer will respond with their complaint.",
                        height="75vh",
                    ),
                    # TODO: in order to use pydantic AI with Gradio, we need to pass the past_messages_state
                    # as additional_inputs and additional_outputs. Additional outputs should also include feedback_display
                    # so the second value returned by our function is directly passed in to feedback_display.
                    additional_inputs=...,  # This should be a list containing past_messages_state
                    additional_outputs=...,  # This should be a list containing past_messages_state and feedback_display
                )

            # Right column: Feedback and guidelines (25% width)
            with gr.Column(scale=1):
                # Render the feedback display at the top of the sidebar
                feedback_display.render()

                # End conversation button - closes the tracing span
                end_button = gr.Button("📞 End Conversation", variant="secondary")
                end_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("### 📋 Chat Guidelines")
                gr.Markdown(
                    """
                The AI acts as a customer complaining about an ACME product. Try to resolve the customer's issue.
                You can type messages, upload images/videos, or record audio.
                """
                )

                gr.Markdown("### 🔒 Content Moderation")
                gr.Markdown(
                    """
                All messages and media are automatically checked for:
                - Inappropriate content
                - Personally identifiable information
                - Unprofessional language
                """
                )

        # Wire up the end conversation button
        end_button.click(fn=chat_session.end_conversation, outputs=end_status).then(
            lambda: gr.Textbox(visible=True), outputs=end_status
        )

    return demo


def main():
    """Main function to run the Gradio app"""
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()

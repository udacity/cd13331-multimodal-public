# Overview

This is an AI-powered content moderation system for customer service interactions at a fictional company called ACME Enterprise.

What it does:
- Moderates text, images, videos, and audio before they're sent to customers
- Detects issues like: PII (personally identifiable information), unprofessional tone, unfriendly content, disturbing images/videos, and low-quality media
- Blocks harmful content and provides detailed explanations for why content was flagged

The app allows a trainee customer agent to interact with a simulated angry customer, played by an LLM. The LLM-customer has bought a product from ACME (the ACME Power Widget Pro) and the product stopped working. The customer agent needs to handle this case by
chatting with the customer, and every message and content provided by the trainee agent is moderated and observed to make sure all
communications following company standards.

Architecture:
1. Specialized Agents - Four moderation agents (text, image, video, audio), each using Google Gemini AI with custom prompts to check their specific content type (agents are in the `agents` folder)
2. LLM-as-a-customer: an agent (`agents/customer_agent.py`)
2. Structured Results - Each agent returns a Pydantic model with specific flags (e.g., contains_pii, is_unfriendly) plus a rationale. 
   The definitions are in the the `types/moderation_result.py` folder
3. Frontend: Gradio Chat UI - Interactive web interface where users can chat and upload files, with real-time moderation. This is `gradio_app.py`.
4. Backend: FastAPI REST API - HTTP endpoints for programmatic access (/moderate/text, /moderate/image, etc.). These are provided as-is. They just wrap in HTTP endpoints the functionality offered by the agents. This is `fastapi_app.py`. The division in frontend/backend services is typical of web applications, and allow different front-ends to utilize the same services from the backend. For example, in the hypothetical scenario of this app, after the initial PoC phase using the Gradio app, we might want to move to a more production-grade React/Vue/Angular app. This new app can use the same backend, and the two apps can even co-exist for a time until the new app is proved to work. No change is needed in the AI agents or on the backend.
5. Observability - Phoenix integration for tracing and monitoring AI agent behavior. Some setup is in `tracing.py`.
6. A convenience executable that starts the 3 services: the backend (fastAPI APIs), the frontend (the gradio app) as well as 
   Arize Phoenix for tracing.

## How to work on the project

The scaffolding is provided to you. You will complete key parts of the code, applying what you have learned, following a precise sequence of steps described below.

Every step needs to be executed in order. You can self-verify that the step has been completed successfully by executing the tests
that we will indicate (already provided to you).

*DO NOT* skip steps and do not proceed if the tests are not passing, as the next steps build on the previous one.

# Setup

Make sure you have a terminal open in the starter kit folder.

## Create virtual environment and install dependencies

In the terminal, run:
```bash
uv sync --dev
```

## Install the app in edit mode
We install our app in edit mode so edits we do in the files are reflected immediately in the installed app:
```bash
uv pip install -e .
```

## Set the virtual environment as interpreter for the project

In VS Code, hit Shift+Crtl+P (or Shift+Command+P on Mac) then select `Python: Select Interpreter`. Go to `Enter Interpreter Path`
and then enter `.venv/bin/python`.

This will help you with syntax highlighting and other things.

## Configure credentials

Open a terminal at the root of the repo, and copy the `env.example` file to `.env`:
```bash
cp env.example .env
```
then open .env and fill your API credential in GEMINI_API_KEY, as well as make up a USER_API_KEY (anything works). You can use
for example `my-api-key`.

**NOTE**: this is obviously not secure, but it is here to remind you that a real production app will need to have some authentication
measure!

# Execution

Follow the following steps in order:

## 1. Moderation result
Here we setup the Pydantic models for our moderation agents. These are the output schemas for the text, audio, image and video moderation agents we'll work on later. 

1. Edit the `types/moderation_result.py` file and replace the `# TODO` sections with your code
2. Run the test to get feedback on your work. From the root of the starter run: `uv run tests/test_moderation_result.py -vv`. 
   If you get an error, read it carefully as it will give you information on what to fix.
   **DO NOT** continue unless the test passes.

## 2. Text agent
Here we will complete the setup of the moderation agent for text. 

1. Edit the text agent (`agents/text_agent.py`) and replace the `# TODO` sections
2. Run the test to get feedback on your work. From the root of the starter run: `uv run tests/test_text_agent.py -vv`. If you get an 
    error, read it carefully as it will give you information on what to fix.
    **DO NOT** continue unless the test passes.

## 3. Image agent
Here we will complete the setup of the moderation agent for images, similarly to what we just did for text. Since this is a multimodal agent, we will use the multimodal capabilities of Pydantic AI and Gemini.

1. Edit the image agent (`agents/image_agent.py`) and replace the `# TODO` sections
2. Run the test to get feedback on your work. From the root of the starter run: `uv run tests/test_image_agent.py -vv`. If you get an 
    error, read it carefully as it will give you information on what to fix.
    **DO NOT** continue unless the test passes.

## 4. Video and Audio agents
The audio and video agents are already completed. You will just test them to make sure they work.

The Video and Audio agents are already complete. You can test them with `uv run tests/test_video_agent.py` and `uv run tests/test_audio_agent.py`.

## 5. Gradio App
Here we put everything together into a Gradio app, which constitutes the front-end of our solution.

1. Edit the gradio app (`gradio_app.py`) and replace all `# TODO` sections. There are quite a few, going from the top to the bottom. Before moving on, make sure you did not left any by searching `TODO` in the file (Crtl+F or Command+F).
2. Run the test to get feedback on your work. From the root of the starter run: `uv run tests/test_gradio_app.py -vv`. If you get an 
    error, read it carefully as it will give you information on what to fix.
    **DO NOT** continue unless the test passes.

## 6. Run all tests

It is now time to verify your work. Run all tests at once with:
```bash
uv run pytest tests/ -vv
```
All tests should pass. If they don't, fix them before submitting your project.

## 7. Evals
Now that our system is complete, we need to run evaluations to understand and measure how it behaves. 

1. Edit the `evals/text/test_cases.py` and replace all `# TODO` sections. Same for `evals/image/test_cases.py`.
2. Once you are done, run `uv run evals/text/test_cases.py` to see the results of the evals on text. 
   NOTE: your score will NOT always be 100%! that is not a bug. The text agent is not perfect!
3. Run the evals for the other media (`uv run evals/image/test_cases.py`, `uv run evals/audio/test_cases.py` and `uv run evals/video/test_cases.py`)

# Play with the app

## Conversation
Now that you are done, the app should work. You can run it by executing in the terminal:

```bash
uv run multimodal-moderation
```

and then going with your browser to `http://localhost:7860/` to interact with the app. You are free to play with it, however, if you
need inspiration, this is how a conversation could go:

```
YOU> Welcome to ACME Customer Service. How can I help?
CUSTOMER-LLM> ... [will complain about the product not working]
YOU> I am sorry to hear that. Is this the product you are talking about? [attach evals/test_data/professional_image.jpg]
CUSTOMER-LLM> ... [will say something about wanting a refund]
YOU> I am sorry but I absolutely cannot offer a refund
------> message will be flagged as rude
YOU> I am going to help you solving your issue. I am authorized to offer you a replacement. Would you be willing to accept it?
CUSTOMER-LLM> ... [probably won't accept]
[you can now close the conversation by clicking on the End Conversation button]
```

## See your traces
After a conversation like the one suggested above, you can go to the Phoenix UI at `http://localhost:6006/projects`, click on the
default project, and see your traces and spans. Explore the different metadata and different reports.

## See your backend APIs

Go to `http://0.0.0.0:8000/docs` to see a nice documentation of your moderation APIs. If you wanna test them out from here, click on Authorize on the upper right and insert your USER_API_KEY (the one you have set in your .env file). Then click on an endpoint (say, `/api/v1/moderate-text`) and click on Try it out (in the upper right). You will see a JSON like:
```json
{
  "text": "string"
}
```
Just change `string` to a message and click Execute, your message will be moderated. Scroll down a bit to see the results.
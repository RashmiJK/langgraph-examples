import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from graph_examples.review_product.production_team_prompts import (
    SYSTEM_PROMPT_FOR_AUDIO_SYNTHESIS_AGENT,
    SYSTEM_PROMPT_FOR_CONTENT_WRITING_AGENT,
)
from graph_examples.review_product.tools import text_to_speech_tool, write_file_tool

gllm_41_mini = ChatOpenAI(
    model_name="openai/gpt-4.1-mini",  # Balanced reasoning and accuracy
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
)

content_writing_agent = create_agent(
    model=gllm_41_mini,
    tools=[write_file_tool],
    system_prompt=SYSTEM_PROMPT_FOR_CONTENT_WRITING_AGENT,
)

audio_synthesis_agent = create_agent(
    model=gllm_41_mini,
    tools=[text_to_speech_tool],
    system_prompt=SYSTEM_PROMPT_FOR_AUDIO_SYNTHESIS_AGENT,
)

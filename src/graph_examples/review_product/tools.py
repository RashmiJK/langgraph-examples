import os
import re
from pathlib import Path

import tiktoken
from deepgram import DeepgramClient
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyExtract, TavilySearch

from graph_examples.logger import get_logger
from graph_examples.review_product.research_team_prompts import (
    SUMMARY_CREATE_SYSTEM_PROMPT,
)

load_dotenv(override=True)
logger = get_logger(__name__)

# Required for chunking text into clauses
CLAUSE_BOUNDARIES = r"\.|\?|!|;|, (and|but|or|nor|for|yet|so)"

# Enhanced DuckDuckGo tool described as privacy-focused search
duckduckgo_search_tool = DuckDuckGoSearchResults(
    output_format="json",
    max_results=5,
    description="""DuckDuckGo privacy-focused search. 
    No tracking, unbiased results, excellent for general research 
    and avoiding commercial bias. Returns clean snippets.Input should be a search query.""",
)

# Enhanced Tavily tool described as AI-optimized search
tavily_search_tool = TavilySearch(
    max_results=5,
    description="""A robust, AI-optimized search engine for comprehensive, accurate, and trusted results.
    Ideal for finding up-to-date information on current events, news, and general topics.
    Delivers real-time, citation-backed results including URLs and content summaries.
    Use this tool for in-depth research tasks. Input should be a specific search query.""",
)

# Enhanced Tavily extract tool
tavily_extract_tool = TavilyExtract(
    extract_depth="basic", include_images=False, chunks_per_source=1
)

# Initialize ONCE
_SUMMARY_LLM = ChatOpenAI(
    model_name="openai/gpt-4o-mini",  # gpt-4.1-mini, Corrected model name
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
)
_ENCODER = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 5000


@tool
def scrape_webpage(url: str) -> str:
    """
    Scrapes the text content from a single URL.

    This tool is essential for gathering detailed information from specific web pages.
    Use it when you have a single URL (e.g., from a search result) and need to read
    the actual text on those pages to answer questions, analyze products, or summarize content.

    The tool bypasses SSL verification to ensure broad compatibility.

    Args:
        url (str): A single HTTP or HTTPS URL to be scraped.

    Returns:
        str: A formatted string containing the text content of the scraped pages.
             Each page's content is encapsulated within <Document name="Title">...</Document> tags
             to clearly separate and identify the source of the information.
    """
    try:
        web_page_loader = WebBaseLoader(url)
        web_page_loader.requests_kwargs = {"verify": False}
        docs = web_page_loader.load()

        full_content = "\n\n".join(
            [
                f"<Document scrape={url}> \n {doc.page_content}\n</Document>"
                for doc in docs
            ]
        )

        encoded_tokens = _ENCODER.encode(full_content)

        if len(encoded_tokens) > _MAX_TOKENS:
            full_content = _ENCODER.decode(encoded_tokens[:_MAX_TOKENS])
            logger.debug("Content truncated to %s length", len(full_content))

        # Summarize the content
        summary_chain = SUMMARY_CREATE_SYSTEM_PROMPT | _SUMMARY_LLM
        response = summary_chain.invoke({"scraped_content": full_content})
        logger.debug("Response from scrape_webpage: %s", response.model_dump())
        return f"<Document scrape={url}>\n{response.content}\n</Document>"
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@tool
def scrape_youtube(url: str) -> str:
    """
    Scrapes the text content from a YouTube video URL.

    Args:
        url (str): The URL of the YouTube video to be scraped.

    Returns:
        str: The text content of the YouTube video.
    """
    try:
        youtube_loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        docs = youtube_loader.load()

        transcibed_content = "\n\n".join(
            [
                f"<Transcripted video={url}> \n {doc.page_content}\n</Transcripted>"
                for doc in docs
            ]
        )

        encoded_tokens = _ENCODER.encode(transcibed_content)
        if len(encoded_tokens) > _MAX_TOKENS:
            transcibed_content = _ENCODER.decode(encoded_tokens[:_MAX_TOKENS])

        # Summarize the content
        summary_chain = SUMMARY_CREATE_SYSTEM_PROMPT | _SUMMARY_LLM
        response = summary_chain.invoke({"scraped_content": transcibed_content})
        logger.debug("Response from scrape_youtube: %s", response.model_dump())
        return f"<Transcripted video={url}>\n{response.content}\n</Transcripted>"
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@tool
def write_file_tool(content: str, filename: str) -> str:
    """
    Writes the content to a file named `final_audio_script.txt`.

    Args:
        content (str): The content to be written to the file.

    Returns:
        str: A confirmation message.
    """
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "w") as f:
        f.write(content)
    return "Script saved to " + filename


@tool
def text_to_speech_tool(filename: str) -> str:
    """
    Converts a saved text file into audio speech.

    Args:
        filename (str): The name of the file to convert.
                        MUST include the extension (e.g., 'robovac_comparison.txt').

    Returns:
        str: A confirmation message.
    """
    filename = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filename, "r") as f:
            file_content = f.read()
    except FileNotFoundError:
        return f"Error: The file {filename} was not found"
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"

    # Chunk by clause boundaries
    def chunk_text_by_clause(text: str) -> list[str]:
        """
        Splits the text into clauses.
        """
        clause_boundaries = re.finditer(CLAUSE_BOUNDARIES, text)
        boundary_indices = [boundary.start() for boundary in clause_boundaries]

        chunks = []
        start = 0
        for boundary_index in boundary_indices:
            chunks.append(text[start : boundary_index + 1].strip())
            start = boundary_index + 1
        chunks.append(text[start:].strip())
        return chunks

    deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    text_chunks = chunk_text_by_clause(file_content)
    audio_filename = str(Path(filename).with_suffix(".mp3"))
    logger.info("Preparing to convert text to speech into %s", audio_filename)

    with open(audio_filename, "wb") as f:
        pass  # create an empty file

    for i, text_chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {i + 1}, {len(text_chunk)} characters")

        if len(text_chunk):
            response = deepgram.speak.v1.audio.generate(
                text=text_chunk, model="aura-2-thalia-en"
            )

            with open(audio_filename, "ab") as f:
                for audio_chunk in response:
                    f.write(audio_chunk)
            logger.info(f"Appended chunk {i + 1} to {audio_filename}")

    logger.info(f"Final audio saved to {audio_filename}")
    return f"Audio saved to {audio_filename}"

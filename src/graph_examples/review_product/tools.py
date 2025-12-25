import os

import tiktoken
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
tavily_extract_tool = TavilyExtract(extract_depth="advanced", include_images=True)

# Initialize ONCE
_SUMMARY_LLM = ChatOpenAI(
    model_name="openai/gpt-4.1-mini",  # Corrected model name
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
)
_ENCODER = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 7000


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

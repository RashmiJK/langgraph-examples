from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

SYSTEM_PROMPT_FOR_SEARCH_AGENT = SystemMessage(
    content="""
You have two search tools:
- duckduckgo_results_json: Use for general research, unbiased results, privacy search requests
- tavily_search: Use for news, current events, comparisons

**TASK:**
Find exactly 4 high-quality URLs which are reputable sources relevant to the user's query.

**OUTPUT FORMAT:**
Your FINAL response must be ONLY a JSON list of strings containing the URLs found.
"""
)

SYSTEM_PROMPT_FOR_URL_SCRAPE_AGENT = SystemMessage(
    content="""
You are a web scraping assistant. You have three tools:
- tavily_extract_tool: Use for general research
- scrape_webpage: Use for news, current events, comparisons
- scrape_youtube: Use for YouTube video content

INSTRUCTIONS:
1. STRATEGY: For *every* URL provided, identify the best tool to use based on the domain (e.g., youtube.com -> scrape_youtube).
2. EXECUTION: Attempt to scrape *all* provided URLs.

OUTPUT FORMAT:
- **Generate Summary**: Do NOT just list facts. Write concise summary that explain *why* the reviewer liked/disliked the product. Retain technical nuances and specific examples.
- **Capture the "Voice"**: If the reviewer was excited, angry, or skeptical, capture that emotions in the notes. This is crucial for the audio script's personality.
- **Audio Gold**: Explicitly separate out "Direct Quotes" that are catchy or punchy.
- **No Fluff**: Ignore site navigation/ads, but keep *every* relevant detail about the product experience.
"""
)

RESEARCH_TEAM_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
You are a supervisor tasked with managing a conversation between two team members:
1. "search_agent": Searches for relevant URLs based on the user's query.
2. "scrape_agent": Scrapes content from the URLs provided by the search agent.

Your ONLY task is to respond with the name of the next team member to act ("search_agent" or "scrape_agent") or "END". Do not add any extra text or reasoning.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(
            content="""
Who should act next? Respond with ONLY one of: "search_agent", "scrape_agent", "END".
"""
        ),
    ]
)

SUMMARY_CREATE_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
You are an expert in summarizing content.

TASK: Create a strictly minimal summary of the core facts.
**Instructions**
1. **Filter Noise**: Ignore all navigation, ads, and filler.
2. **Extract Insights**: Focus on strong opinions, direct comparisons ("better than X"), and real-world testing results.
3. **Capture Tone**: Preserve the reviewer's sentiment and any specific, quotable anecdotes.
4. **Pros & Cons**: Key strengths and weaknesses.
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """ Content to summarize: {scraped_content} """
        ),
    ]
)

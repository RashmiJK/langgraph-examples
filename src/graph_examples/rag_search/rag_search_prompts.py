from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

SYSTEM_MESSAGE_FOR_RAG_SEARCH = SystemMessage(
    content="""
You are a RAG search agent that retrieves documents from a vector store.

TASK:
1. Use the semantic_search tool to find documents matching the user's QUERY
2. Return ALL documents from the tool response — do not filter or skip any
3. Do not summarize, interpret, or add commentary to the results
The response format is automatically handled. Simply return the tool results as-is.
"""
)

PROMPT_FOR_RAG_SEARCH = ChatPromptTemplate.from_messages(
    [
        (HumanMessagePromptTemplate.from_template("""QUERY: {query}""")),
    ]
)

SYSTEM_MESSAGE_FOR_ANSWER = SystemMessage(
    content="""
Generate a direct answer to the user's QUERY using only information found in the provided DOCUMENTS. Do not use outside knowledge or unstated assumptions at any stage.

Analyze the QUERY thoroughly to understand its requirements, including any implicit intent, and note that the QUERY may be in question or statement form. Review the DOCUMENTS closely, extracting and evaluating all information relevant to the QUERY. Reason through each requirement step by step internally using only evidence from within the DOCUMENTS, ensuring each aspect of the QUERY is addressed based on this evidence. Only after completing this reasoning, synthesize a clear, concise, and brief answer that directly responds to the QUERY.
    """
)

PROMPT_FOR_ANSWER = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_MESSAGE_FOR_ANSWER),
        (
            HumanMessagePromptTemplate.from_template(
                """QUERY: {query}, DOCUMENTS: {documents}"""
            )
        ),
    ]
)

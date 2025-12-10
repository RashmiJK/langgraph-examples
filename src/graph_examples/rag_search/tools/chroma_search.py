from langchain.tools import tool

from graph_examples.rag_search.chroma_interface import ChromaInterface
from graph_examples.rag_search.types import SearchResult


@tool
def semantic_search(query: str) -> list[SearchResult]:
    """
    Semantic search on ingested documents
    """
    chroma = ChromaInterface.get_instance()
    results = chroma.search(query)
    return [
        SearchResult(
            document=result[0].page_content,
            source=result[0].metadata.get("source", "").split("/")[-1],
            score=round(result[1], 3),
        )
        for result in results
    ]

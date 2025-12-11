import os
from typing import Any

from flashrank import Ranker, RerankRequest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.output_parsers import StrOutputParser

from graph_examples.logger import get_logger
from graph_examples.rag_search.rag_search_prompts import (
    PROMPT_FOR_RAG_SEARCH,
    SYSTEM_MESSAGE_FOR_RAG_SEARCH,
    PROMPT_FOR_ANSWER,
)
from graph_examples.rag_search.tools.chroma_search import semantic_search
from graph_examples.rag_search.types import ListOfSearchedResults, SearchResult


class State(MessagesState, total=False):
    """
    State for the RAG search and reranking
    """

    query: str
    search_results: ListOfSearchedResults
    answer_baseline: str
    reranked_results: dict[str, Any]
    answer_reranked: str


class RagSearch:
    """
    RagSearch class to perform RAG search on ingested documents
    """

    def __init__(self):
        """
        Initialize the RagSearch class
        """
        self.logger = get_logger(__name__)
        self._gllm_41 = AzureAIChatCompletionsModel(
            endpoint=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            credential=os.getenv("GITHUB_TOKEN"),
            model="openai/gpt-4.1",  # Highest reasoning and accuracy.
            api_version="2024-08-01-preview",
        )
        self._gllm_41_mini = AzureAIChatCompletionsModel(
            endpoint=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            credential=os.getenv("GITHUB_TOKEN"),
            model="openai/gpt-4.1-mini",  # Balanced reasoning and accuracy
            api_version="2024-08-01-preview",
        )
        self._gllm_41_nano = AzureAIChatCompletionsModel(
            endpoint=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            credential=os.getenv("GITHUB_TOKEN"),
            model="openai/gpt-4.1-nano",  # Lower reasoning and accuracy
            api_version="2024-08-01-preview",
        )
        self._chromadb_search_agent = create_agent(
            model=ChatOpenAI(
                model="openai/gpt-4.1",
                api_key=os.getenv("GITHUB_TOKEN"),
                base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            ),
            tools=[semantic_search],
            system_prompt=SYSTEM_MESSAGE_FOR_RAG_SEARCH,
            response_format=ProviderStrategy(ListOfSearchedResults),
        )
        self._graph = self._build_graph()

    def _build_graph(self):
        """
        Build the graph for RAG search and reranking
        """
        # Build the graph
        workflow_builder = StateGraph(State)

        # Add nodes
        workflow_builder.add_node("rag_agent", self._rag_agent)
        workflow_builder.add_node("answer_baseline", self._answer_baseline)
        workflow_builder.add_node("reranker", self._reranker)
        workflow_builder.add_node("answer_reranked", self._answer_reranked)

        # Add edges
        workflow_builder.add_edge(START, "rag_agent")
        workflow_builder.add_edge("rag_agent", "answer_baseline")
        workflow_builder.add_edge("rag_agent", "reranker")
        workflow_builder.add_edge("reranker", "answer_reranked")
        workflow_builder.add_edge("answer_baseline", END)
        workflow_builder.add_edge("answer_reranked", END)

        # compile the graph
        graph = workflow_builder.compile()

        try:
            png_data = graph.get_graph().draw_mermaid_png()
            with open("src/graph_examples/rag_search/rag_search_graph.png", "wb") as f:
                f.write(png_data)
        except Exception as e:
            self.logger.exception("Could not generate graph PNG: %s", e)

        return graph

    def _rag_agent(self, state: State) -> State:
        """
        RAG agent to perform RAG search on ingested documents
        """
        search_chain = PROMPT_FOR_RAG_SEARCH | self._chromadb_search_agent
        response = search_chain.invoke(state["query"])
        self.logger.debug("Response: %s", response)
        self.logger.info(
            "Number of results: %d", len(response["structured_response"].results)
        )
        return {"search_results": response["structured_response"]}

    def _answer_baseline(self, state: State) -> State:
        """
        Generate answer based on top 2 documents from search results
        """
        answer_chain = PROMPT_FOR_ANSWER | self._gllm_41_mini | StrOutputParser()
        documents = "\n".join(
            [
                f"Document {i + 1}:\n{record.document}"
                for i, record in enumerate(state["search_results"].results[:2])
            ]
        )
        self.logger.info("Documents: %s", documents)
        response = answer_chain.invoke(
            {"query": state["query"], "documents": documents}
        )
        self.logger.info("Response: %s", response)
        return {"answer_baseline": response}

    def _reranker(self, state: State) -> State:
        """
        Reranker to rerank the search results
        """
        ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".cache"
            ),
        )
        reranked_results = ranker.rerank(
            RerankRequest(
                query=state["query"],
                passages=[
                    {
                        "id": i,
                        "text": r.document,
                        "meta": {"source": r.source, "original_score": r.score},
                    }
                    for i, r in enumerate(state["search_results"].results)
                ],
            )
        )
        return {"reranked_results": reranked_results}

    def _answer_reranked(self, state: State) -> State:
        """
        Generate answer based on reranked documents
        """
        reranked_answer_chain = (
            PROMPT_FOR_ANSWER | self._gllm_41_mini | StrOutputParser()
        )
        reranked_documents = "\n".join(
            [
                f"Document {i + 1}:\n{record['text']}"
                for i, record in enumerate(state["reranked_results"])
            ]
        )
        self.logger.info("Reranked Documents: %s", reranked_documents)
        response = reranked_answer_chain.invoke(
            {"query": state["query"], "documents": reranked_documents}
        )
        self.logger.info("Reranked Response: %s", response)
        return {"answer_reranked": response}

    def respond(self, query: str) -> tuple[list[SearchResult], list[SearchResult], str]:
        """
        Invoke the graph and return the search results
        """
        response = self._graph.invoke({"query": query})
        self.logger.debug("Response: %s", response)
        return (
            response["search_results"],
            response["reranked_results"],
            response["answer_baseline"],
            response["answer_reranked"],
        )

import os
import warnings
from pathlib import Path

import tiktoken
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_examples.logger import get_logger
from graph_examples.rag_search.types import SearchResult

# Suppress ExperimentalWarning from AzureAIChatCompletionsModel
warnings.filterwarnings(
    "ignore", message=".*AzureAIEmbeddingsModel is currently in preview.*"
)


class ChromaInterface:
    """
    Interface to interact with Chroma vector store.
    """

    _instance: "ChromaInterface | None" = None

    @classmethod
    def get_instance(cls) -> "ChromaInterface":
        """
        Returns the singleton instance of ChromaInterface
        """
        # Use singleton to avoid multiple instances of embedding models
        # and re-establishing API credentials
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Initialize the ChromaInterface
        """
        self.embedding_3_small = None
        self.embedding_3_large = None
        self.client = None
        self._intialize()
        self.logger = get_logger(__name__)

    def _intialize(self) -> None:
        self.embedding_3_small = AzureAIEmbeddingsModel(
            model="openai/text-embedding-3-small",  # Dimension : 512, Context: 8k input
            endpoint=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            credential=os.getenv("GITHUB_TOKEN"),
        )
        self.embedding_3_large = AzureAIEmbeddingsModel(
            model="openai/text-embedding-3-large",  # Dimension : 3072, Context: 8k input
            endpoint=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
            credential=os.getenv("GITHUB_TOKEN"),
        )
        persist_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ChromaVectorStore"
        )
        self.client = Chroma(
            collection_name="rag_search",
            embedding_function=self.embedding_3_large,
            persist_directory=persist_directory,
        )

    def ingest(self, file: str, chunk_size: int) -> str:
        """
        Ingest documents and create embeddings.
        """
        # Get existing ids in the collection
        existing_ids = self.client.get().get("ids", [])
        self.logger.info("Existing ids: %s", existing_ids)

        # Build search prefix
        id_search_prefix = f"{Path(file).stem}_{chunk_size}_"
        self.logger.info("id_search_prefix: %s", id_search_prefix)

        # Check if document with this prefix already exists
        already_exists = any(id.startswith(id_search_prefix) for id in existing_ids)
        self.logger.info("already_exists: %s", already_exists)

        if already_exists:
            return f"{Path(file).name}: Already ingested with chunk size {chunk_size}. Skipping ingestion."

        # check if document with this prefix but different chunk size exists
        prefix_exists = any(id.startswith(f"{Path(file).stem}_") for id in existing_ids)
        if prefix_exists:
            # remove existing documents with this prefix
            self.client.delete(
                ids=[id for id in existing_ids if id.startswith(f"{Path(file).stem}_")]
            )

        file_type = Path(file).suffix.lower()

        if file_type == ".pdf":
            loader = PyPDFLoader(file)
        elif file_type == ".txt":
            loader = TextLoader(file)
        elif file_type == ".md":
            loader = UnstructuredMarkdownLoader(file)
        else:
            return f"{Path(file).name}: Unsupported file type"

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 8,
        )
        chunks = loader.load_and_split(text_splitter=text_splitter)
        self.logger.info("len(chunks): %d", len(chunks))

        id_prefix = Path(file).stem
        batch_chunk = []
        chunk_ids = []
        batch_size = 0
        for i, chunk in enumerate(chunks):
            size = len(tiktoken.get_encoding("cl100k_base").encode(chunk.page_content))
            if batch_size + size < 8192:
                chunk_ids.append(f"{id_prefix}_{chunk_size}_{i}")
                batch_chunk.append(chunk)
                batch_size += size
            else:
                # we have just enough to process batch
                self.client.add_documents(batch_chunk, ids=chunk_ids)
                batch_chunk = [chunk]
                chunk_ids = [f"{id_prefix}_{chunk_size}_{i}"]
                batch_size = size

        # process remaining batch if any
        if batch_chunk:
            self.client.add_documents(batch_chunk, ids=chunk_ids)

        return f"{Path(file).name}: Successfully ingested"

    def describe_ingested_content(self) -> str:
        """
        Returns a summary of ingested content
        """
        # Get existing ids in the collection
        existing_ids = self.client.get().get("ids", [])

        # dictionary to store filename with chunk size
        file_chunk_map: dict[str, int] = {}

        for id in existing_ids:
            parts = id.rsplit("_", 2)
            filename, size, *index = parts
            if filename not in file_chunk_map:
                file_chunk_map[filename] = int(size)

        if not file_chunk_map:
            return "No documents ingested yet"

        # format output
        return "\n".join(
            f"{filename} (chunked size {size})"
            for filename, size in file_chunk_map.items()
        )

    def search(self, query: str) -> list[SearchResult]:
        """
        Search for documents in the vector store
        """
        results = self.client.similarity_search_with_score(query, k=6)
        self.logger.debug("Search results: %s", results)
        return results

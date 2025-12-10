from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Retrieved document and its score
    """

    document: str = Field(description="Retrieved document")
    source: str = Field(description="Source of the document which is just the filename")
    score: float = Field(description="Score of the document")


class ListOfSearchedResults(BaseModel):
    """
    List of searched results
    """

    results: list[SearchResult] = Field(description="List of searched results")

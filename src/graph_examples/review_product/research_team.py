from functools import partial

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from graph_examples.review_product.base_team_class import BaseTeam, BaseTeamState
from graph_examples.review_product.research_team_prompts import (
    RESEARCH_TEAM_SUPERVISOR_PROMPT,
    SYSTEM_PROMPT_FOR_SEARCH_AGENT,
    SYSTEM_PROMPT_FOR_URL_SCRAPE_AGENT,
)
from graph_examples.review_product.tools import (
    duckduckgo_search_tool,
    scrape_webpage,
    scrape_youtube,
    tavily_extract_tool,
    tavily_search_tool,
)


class ResearchTeamState(BaseTeamState):
    pass


class UrlsList(BaseModel):
    """Result containing list of URLs found."""

    urls: list[str] = Field(description="List of URLs found")


class WebScrapeResult(BaseModel):
    """Result containing scraped content from multiple URLs."""

    results: list[dict[str, str]] = Field(
        description="List of results, where each dict contains 'url' and its 'content'."
    )


class ResearchTeam(BaseTeam):
    """
    A modulear class representing the Research Department.
    It encapsulates its own state, tools, and graph logic.
    """

    def __init__(self, trace_project_name: str = None):
        """
        Initialize the ResearchTeam with members.
        """
        super().__init__()

        self.url_search_agent = create_agent(
            model=self.gllm_4_1,
            tools=[duckduckgo_search_tool, tavily_search_tool],
            system_prompt=SYSTEM_PROMPT_FOR_SEARCH_AGENT,
            response_format=ProviderStrategy(UrlsList),
        )
        self.url_scrape_agent = create_agent(
            model=self.gllm_4_1,
            tools=[tavily_extract_tool, scrape_webpage, scrape_youtube],
            system_prompt=SYSTEM_PROMPT_FOR_URL_SCRAPE_AGENT,
            response_format=ProviderStrategy(WebScrapeResult),
        )

        # Build graph
        self._graph = self._build_graph()

        # Setup optional tracing
        self._tracer = self._setup_tracer(self._graph, trace_project_name)

    def _build_graph(self) -> CompiledStateGraph:
        """Construct the internal StateGraph for research team"""
        # Build the graph
        workflow_builder = StateGraph(ResearchTeamState)

        # Add Nodes
        workflow_builder.add_node("research_supervisor", self._team_supervisor)
        workflow_builder.add_node(
            "search_agent",
            lambda state: self._run_safe_agent(
                self.url_search_agent, state, "search_agent"
            ),
        )
        workflow_builder.add_node(
            "scrape_agent",
            lambda state: self._run_safe_agent(
                self.url_scrape_agent, state, "scrape_agent"
            ),
        )

        # Add edges
        workflow_builder.add_edge(START, "research_supervisor")
        workflow_builder.add_conditional_edges(
            "research_supervisor",
            lambda state: state["next"],
            {
                "search_agent": "search_agent",
                "scrape_agent": "scrape_agent",
                "END": END,
            },
        )
        workflow_builder.add_edge("search_agent", "research_supervisor")
        workflow_builder.add_edge("scrape_agent", "research_supervisor")

        return workflow_builder.compile()

    def _team_supervisor(self, state: ResearchTeamState) -> ResearchTeamState:
        """Team supervisor to determine next action"""
        valid_agents = ["search_agent", "scrape_agent"]

        chain = (
            RESEARCH_TEAM_SUPERVISOR_PROMPT
            | self.gllm_4_1
            | partial(self._parse_supervisor_output, valid_agents=valid_agents)
        )

        return chain.invoke(state)

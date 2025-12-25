import os
import re

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opik.integrations.langchain import OpikTracer
from pydantic import BaseModel, Field

from graph_examples.logger import get_logger
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


class TeamState(MessagesState):
    next: str


class UrlsList(BaseModel):
    """Result containing list of URLs found."""

    urls: list[str] = Field(description="List of URLs found")


class WebScrapeResult(BaseModel):
    """Result containing scraped content from multiple URLs."""

    results: list[dict[str, str]] = Field(
        description="List of results, where each dict contains 'url' and its 'content'."
    )


class ResearchTeam:
    """
    A modulear class representing the Research Department.
    It encapsulates its own state, tools, and graph logic.
    """

    def __init__(self, trace_project_name: str = None):
        """
        Initialize the ResearchTeam with members.
        """
        self.logger = get_logger(__name__)

        self.gllm_4_1 = ChatOpenAI(
            model_name="openai/gpt-4o-mini",  # gpt-4.1, gpt-4o # Highest reasoning and accuracy.
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),  # GitHub Models endpoint
        )

        self.gllm_41_mini = ChatOpenAI(
            model_name="openai/gpt-4.1-mini",  # Balanced reasoning and accuracy
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
        )

        self.gllm_41_nano = ChatOpenAI(
            model_name="openai/gpt-4.1-nano",  # Lower reasoning and accuracy
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_INFERENCE_ENDPOINT"),
        )

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

        self._graph = self._build_graph()

        # optional tracing
        self._tracer = None
        if os.getenv("OPIK_API_KEY"):
            try:
                self._tracer = OpikTracer(
                    graph=self._graph.get_graph(xray=True),
                    project_name=trace_project_name,
                )
            except Exception as e:
                self.logger.warning("Failed to initialize tracer: %s", e)

    def _build_graph(self) -> CompiledStateGraph:
        """Construct the internal StateGraph for research team"""
        # Build the graph
        workflow_builder = StateGraph(TeamState)

        # Add Nodes
        workflow_builder.add_node("team_supervisor", self._team_supervisor)
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
        workflow_builder.add_edge(START, "team_supervisor")
        workflow_builder.add_conditional_edges(
            "team_supervisor",
            lambda state: state["next"],
            {
                "search_agent": "search_agent",
                "scrape_agent": "scrape_agent",
                "END": END,
            },
        )
        workflow_builder.add_edge("search_agent", "team_supervisor")
        workflow_builder.add_edge("scrape_agent", "team_supervisor")

        return workflow_builder.compile()

    def _team_supervisor(self, state: TeamState) -> TeamState:
        """Team supervisor to determine next action"""

        def parse_output(llm_output: AnyMessage) -> dict:
            """Parse the output to get the next action"""
            if hasattr(llm_output, "content"):
                output = llm_output.content.strip()
            else:
                output = str(llm_output).strip()

            # Look for one of the valid terms anywhere in the string
            match = re.search(r"(search_agent|scrape_agent|END)", output)

            if match:
                output = match.group(1)
                self.logger.debug("Parsed supervisor output: %s", output)
                return {"next": output}

            self.logger.warning(
                "Invalid output from team supervisor: %s. Forcing END", output
            )
            output = "END"

            return {"next": output}

        chain = RESEARCH_TEAM_SUPERVISOR_PROMPT | self.gllm_4_1 | parse_output

        return chain.invoke(state)

    def _run_safe_agent(self, agent, state: TeamState, name):
        """Run an agent safely"""
        try:
            result = agent.invoke(state)
            self.logger.debug("Result from %s: %s", name, result)
            if not isinstance(result, dict) or "messages" not in result:
                self.logger.warning("Invalid output from %s: %s.", name, result)
            return {
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name=name)
                ]
            }
        except Exception as e:
            self.logger.error("Failed to run %s: %s", name, e)
            return {
                "messages": [
                    HumanMessage(
                        content=f"Error occured in {name}: {str(e)}", name=name
                    )
                ]
            }

    # public interface
    def as_node(self):
        """Adapts this class to be used as a single node in a larger graph"""

        def enter_chain(state: dict) -> dict:
            """Read the last message from the state to initialize the sub-graph."""
            return {"messages": state["messages"][-1]}

        def exit_chain(state: dict) -> dict:
            """Extract the final result from the sub-graph to update the parent state."""
            return {"messages": [state["messages"][-1]]}

        # return chain that enters and exits this team's scope
        chain = enter_chain | self._graph | exit_chain
        if self._tracer:
            return chain.with_config({"callbacks": [self._tracer]})
        return chain

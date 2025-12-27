import os
import re
from functools import partial

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opik.integrations.langchain import OpikTracer

from graph_examples.review_product.base_team_class import get_logger
from graph_examples.review_product.editorial_board_prompts import CHIEF_EDITOR_PROMPT
from graph_examples.review_product.production_team import ProductionTeam
from graph_examples.review_product.research_team import ResearchTeam


class EditorialBoardState(MessagesState):
    next: str


class EditorialBoard:
    """
    A modulear class representing the Editorial Board - a main orchestrator.
    It initializes the teams/departments (ResearchTeam, ProductionTeam) and coordinates their work.
    """

    def __init__(self, trace_project_name: str = None):
        self.logger = get_logger(__name__)

        # Initialize LLMs
        self.gllm_4_1 = ChatOpenAI(
            model_name="openai/gpt-4.1",  # gpt-4.1, gpt-4o # Highest reasoning and accuracy.
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

        # Instantiate ResearchTeam and ProductionTeam
        self.research_team = ResearchTeam()
        self.production_team = ProductionTeam()

        # Build graph
        self._graph = self._build_graph()

        # optional tracing
        self._tracer = None
        if trace_project_name and os.getenv("OPIK_API_KEY"):
            try:
                # Instantiate OpikTracer
                self._tracer = OpikTracer(
                    graph=self._graph.get_graph(xray=True),
                    project_name=trace_project_name,
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize tracer: {e}")

    def _build_graph(self) -> CompiledStateGraph:
        """Construct the super graph for editorial board"""
        # Build the grap
        editorial_board_graph = StateGraph(EditorialBoardState)

        # Add nodes
        editorial_board_graph.add_node("chief_editor", self._chief_editor)
        editorial_board_graph.add_node("research_team", self.research_team.as_node())
        editorial_board_graph.add_node(
            "production_team", self.production_team.as_node()
        )

        # Add edges
        editorial_board_graph.add_edge(START, "chief_editor")
        editorial_board_graph.add_conditional_edges(
            "chief_editor",
            lambda state: state["next"],
            {
                "research_team": "research_team",
                "production_team": "production_team",
                "END": END,
            },
        )
        editorial_board_graph.add_edge("research_team", "chief_editor")
        editorial_board_graph.add_edge("production_team", "chief_editor")

        return editorial_board_graph.compile()

    def _chief_editor(self, state: EditorialBoardState) -> EditorialBoardState:
        """Chief editor to determine next action"""
        valid_agents = ["research_team", "production_team"]

        def parse_chief_editor_output(
            llm_output: AnyMessage, valid_agents: list[str] = valid_agents
        ) -> dict:
            "parse the output to get the next action"
            if hasattr(llm_output, "content"):
                output = llm_output.content.strip()
            else:
                output = str(llm_output).strip()

            # Look for one of the valid terms anywhere in the string
            options_pattern = "|".join(valid_agents + ["END"])
            match = re.search(f"({options_pattern})", output)

            if match:
                parsed = match.group(1)
                self.logger.debug(f"Chief editor routed to: {parsed}")
                return {"next": parsed}

            self.logger.warning(
                "Invalid output from chief editor: %s. Forcing END", output
            )
            return {"next": "END"}

        chain = (
            CHIEF_EDITOR_PROMPT
            | self.gllm_41_nano
            | partial(parse_chief_editor_output, valid_agents=valid_agents)
        )

        return chain.invoke(state)

    def respond(self, query: str, recursion_limit: int = 35):
        """
        Public helper function to run the system
        """
        initial_state = {"messages": [HumanMessage(content=query)]}
        config = {"recursion_limit": recursion_limit}

        if self._tracer:
            config["callbacks"] = [self._tracer]

        return self._graph.invoke(initial_state, config=config)

from functools import partial

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph_examples.review_product.base_team_class import BaseTeam, BaseTeamState
from graph_examples.review_product.production_team_prompts import (
    PRODUCTION_TEAM_SUPERVISOR_PROMPT,
    SYSTEM_PROMPT_FOR_AUDIO_SYNTHESIS_AGENT,
    SYSTEM_PROMPT_FOR_CONTENT_WRITING_AGENT,
)
from graph_examples.review_product.tools import text_to_speech_tool, write_file_tool


class ProductionTeamState(BaseTeamState):
    pass


class ProductionTeam(BaseTeam):
    """
    A modular class representing the Production Department.
    It encapsulates its own state, tools, and graph logic.
    """

    def __init__(self):
        """
        Initialize the ProductionTeam with members.
        """
        super().__init__()

        self.content_writing_agent = create_agent(
            model=self.gllm_41_mini,
            tools=[write_file_tool],
            system_prompt=SYSTEM_PROMPT_FOR_CONTENT_WRITING_AGENT,
        )

        self.audio_synthesis_agent = create_agent(
            model=self.gllm_41_mini,
            tools=[text_to_speech_tool],
            system_prompt=SYSTEM_PROMPT_FOR_AUDIO_SYNTHESIS_AGENT,
        )

        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Construct the internal StateGraph for production team"""
        # Build the graph
        workflow_builder = StateGraph(ProductionTeamState)

        # Add nodes
        workflow_builder.add_node("production_supervisor", self._team_supervisor)
        workflow_builder.add_node(
            "content_writing_agent",
            lambda state: self._run_safe_agent(
                self.content_writing_agent, state, "content_writing_agent"
            ),
        )
        workflow_builder.add_node(
            "audio_synthesis_agent",
            lambda state: self._run_safe_agent(
                self.audio_synthesis_agent, state, "audio_synthesis_agent"
            ),
        )

        # Add edges
        workflow_builder.add_edge(START, "production_supervisor")
        workflow_builder.add_conditional_edges(
            "production_supervisor",
            lambda state: state["next"],
            {
                "content_writing_agent": "content_writing_agent",
                "audio_synthesis_agent": "audio_synthesis_agent",
                "END": END,
            },
        )
        workflow_builder.add_edge("content_writing_agent", "production_supervisor")
        workflow_builder.add_edge("audio_synthesis_agent", "production_supervisor")

        return workflow_builder.compile()

    def _team_supervisor(self, state: ProductionTeamState) -> ProductionTeamState:
        """Team supervisor to determine next action"""
        valid_agents = ["content_writing_agent", "audio_synthesis_agent"]

        chain = (
            PRODUCTION_TEAM_SUPERVISOR_PROMPT
            | self.gllm_41_nano
            | partial(self._parse_supervisor_output, valid_agents=valid_agents)
        )

        return chain.invoke(state)

import os
import re

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opik.integrations.langchain import OpikTracer

from graph_examples.logger import get_logger
from graph_examples.review_product.production_team_prompts import (
    PRODUCTION_TEAM_SUPERVISOR_PROMPT,
    SYSTEM_PROMPT_FOR_AUDIO_SYNTHESIS_AGENT,
    SYSTEM_PROMPT_FOR_CONTENT_WRITING_AGENT,
)
from graph_examples.review_product.tools import text_to_speech_tool, write_file_tool


class ProductionTeamState(MessagesState):
    next: str


class ProductionTeam:
    """
    A modular class representing the Production Department.
    It encapsulates its own state, tools, and graph logic.
    """

    def __init__(self, trace_project_name: str = None):
        """
        Initialize the ProductionTeam with members.
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

        def parse_output(llm_output: AnyMessage) -> dict:
            """Parse the output to get the next action"""
            if hasattr(llm_output, "content"):
                output = llm_output.content.strip()
            else:
                output = str(llm_output).strip()

            # Look for one of the valid terms anywhere in the string
            match = re.search(
                r"(content_writing_agent|audio_synthesis_agent|END)", output
            )

            if match:
                output = match.group(1)
                self.logger.debug("Parsed supervisor output: %s", output)
                return {"next": output}

            self.logger.warning(
                "Invalid output from team supervisor: %s. Forcing END", output
            )
            output = "END"

            return {"next": output}

        chain = PRODUCTION_TEAM_SUPERVISOR_PROMPT | self.gllm_4_1 | parse_output

        return chain.invoke(state)

    def _run_safe_agent(self, agent, state: ProductionTeamState, agent_name: str):
        """Run an agent safely, handling errors and logging"""
        try:
            result = agent.invoke(state)
            self.logger.debug("Result from %s: %s", agent_name, result)

            if not isinstance(result, dict) or "messages" not in result:
                self.logger.warning("Invalid result from %s: %s", agent_name, result)

            return {
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, name=agent_name
                    )
                ]
            }
        except Exception as e:
            self.logger.error("Agent %s failed with error: %s", agent_name, e)
            return {
                "messages": [
                    HumanMessage(
                        content=f"Agent {agent_name} failed with error: {e}",
                        name=agent_name,
                    )
                ]
            }

    # public methods
    def as_node(self):
        """Adapts this class to be used as a single node in a larger graph"""

        def enter_chain(state: dict) -> dict:
            """Read the last message from the state to initialize the sub-graph."""
            return {"messages": state["messages"][-1]}

        def exit_chain(state: dict) -> dict:
            """Extract the final result from the sub-graph to update the parent state"""
            return {"messages": [state["messages"][-1]]}

        # returns chain that enters and exits this team's scope
        chain = enter_chain | self._graph | exit_chain

        if self._tracer:
            return chain.with_config({"callbacks": [self._tracer]})

        return chain

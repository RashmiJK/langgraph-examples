import os
import re
from abc import ABC, abstractmethod

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from opik.integrations.langchain import OpikTracer

from graph_examples.logger import get_logger


class BaseTeamState(MessagesState):
    next: str


class BaseTeam(ABC):
    """Just a scaffold for shared logic and isn't meant to function on its own.
    Hence, Abstract Base Class (ABC).
    """

    def __init__(self):
        self.logger = get_logger(__name__)

        # Initialize LLMs
        self.gllm_4_1 = ChatOpenAI(
            model_name="openai/gpt-4o",  # gpt-4.1, gpt-4o # Highest reasoning and accuracy.
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

    # Not using at the moment
    def _setup_tracer(
        self, graph: CompiledStateGraph = None, tracer_project_name: str = None
    ):
        """Setup tracer for tracing the graph"""
        if graph and tracer_project_name and os.getenv("OPIK_API_KEY"):
            try:
                # OpikTracer require Graph to be built first
                # Call this specifically after building the graph in the subclass
                tracer = OpikTracer(
                    graph=graph.get_graph(xray=True),
                    project_name=tracer_project_name,
                )
                return tracer
            except Exception as e:
                self.logger.warning("Failed to initialize tracer: %s", e)

        return None

    def _run_safe_agent(
        self, agent: CompiledStateGraph, state: BaseTeamState, agent_name: str
    ) -> dict:
        """Run an agent safely"""
        try:
            result = agent.invoke(state, config={"recursion_limit": 10})
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

    def _parse_supervisor_output(
        self, llm_output: AnyMessage, valid_agents: list[str]
    ) -> str:
        """Shared logic to parse supervisor routing decisions"""
        if hasattr(llm_output, "content"):
            output = llm_output.content.strip()
        else:
            output = str(llm_output).strip()

        # dynamic Regex to match any of the valid agents
        options_pattern = "|".join(valid_agents + ["END"])
        match = re.search(f"({options_pattern})", output)

        if match:
            parsed = match.group(1)
            self.logger.debug(f"Supervisor routed to: {parsed}")
            return {"next": parsed}

        self.logger.warning(
            "Invalid output from team supervisor: %s. Forcing END", output
        )
        return {"next": "END"}

    def as_node(self, tracer: OpikTracer = None):
        """Shared adapter logic for sub-graph integration into super graph"""

        def enter_chain(state: dict) -> dict:
            return {
                "messages": state["messages"][-1]
            }  # May want to change this to pass full messages in supergraph to subgraph FIXME

        def exit_chain(state: dict) -> dict:
            """Extract the final result from the sub-graph to update the parent state."""
            return {"messages": [state["messages"][-1]]}

        # return chain that enters and exits this team's scope
        chain = enter_chain | self._graph | exit_chain
        if tracer:
            return chain.with_config({"callbacks": [tracer]})
        return chain

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """Subclasses must implement this method to build the internal StateGraph for the team"""
        pass

    @abstractmethod
    def _team_supervisor(self, state: BaseTeamState) -> BaseTeamState:
        """Subclasses must implement this method to determine next action"""
        pass

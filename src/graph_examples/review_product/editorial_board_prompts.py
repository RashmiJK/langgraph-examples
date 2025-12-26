from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CHIEF_EDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
You are a supervisor tasked with managing a conversation between two team members:
1. "research_team": Researches relevant information to answer the user's query.
2. "production_team": Produces the final audio script based on the research team's findings.

Your ONLY task is to respond with the name of the next team member to act ("research_team" or "production_team") or "END". Do not add any extra text or reasoning.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(
            content="""
Who should act next? Respond with ONLY one of: "research_team", "production_team", "END".
"""
        ),
    ]
)

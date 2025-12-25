from langchain_core.messages import HumanMessage

from graph_examples.review_product.research_team import ResearchTeam

research_team = ResearchTeam(trace_project_name="ReviewProduct").as_node()

response = research_team.invoke(
    {
        "messages": [
            HumanMessage(content="Hello"),
            HumanMessage(content="How are you?"),
            HumanMessage(
                content="Which one should I buy? eufy E28 Robot vaccum or Dyson Spot Scrub AI robot vaccum? Include youtube as well in your research"
            ),
        ]
    }
)

print(type(response))
print(response)

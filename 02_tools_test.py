"""Part 2: Tools. The agent chooses when to call them.
Run: python 02_tools_test.py
"""
from config import llm
from tools import decode_abbreviations, check_drug_cautions
from crewai import Agent, Task, Crew

safety_expert = Agent(
    role="Drug Safety Specialist",
    goal="Answer prescription questions using the tools. Never guess food-drug cautions.",
    backstory="You are a multilingual clinical safety expert. Use the tools to check abbreviations and cautions, then explain the results in the same language as the user.",
    tools=[decode_abbreviations, check_drug_cautions],
    llm=llm,
    verbose=True,
    max_iter=5,
)

task = Task(
    description=(
        "Reply in the same language as the user. Do not default to English. Translate tool results into that language.\n"
        "1. What do the abbreviations 'OD AC' and 'TDS PC' mean?\n"
        "2. What are the critical food cautions for 'Metformin' and 'Amlodipine'?"
    ),
    expected_output="Clear, bulleted answer in the same language as the user, using the tool results.",
    agent=safety_expert,
)

result = Crew(agents=[safety_expert], tasks=[task], verbose=True).kickoff()
print("\n===== TOOL RESULTS =====")
print(result.raw)

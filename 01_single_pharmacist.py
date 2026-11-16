"""Part 1: one agent, one task, one crew.
Testing the Pharmacist Decrypter Agent.
Run: python 01_single_pharmacist.py
"""
from config import llm
from crewai import Agent, Task, Crew

pharmacist = Agent(
    role="Clinical Pharmacist",
    goal="Explain confusing doctor prescriptions simply and accurately in the same language used by the patient or user",
    backstory="You are a friendly multilingual community pharmacist. Detect the language used by the patient and explain medicines naturally in that same language.",
    llm=llm,
    verbose=True,
)

decrypt_task = Task(
    description=(
        "Explain what this medicine is, what it treats, and how to take it. Follow the mandatory same-language rule: reply in the language used by the user, not automatically English. Medicine: "
        "'Cap. Omeprazole 20mg - 1 cap OD, AC (30 mins before breakfast)'"
    ),
    expected_output="3 clear bullet points in the same language as the patient or user, explaining the medicine, its purpose, and instructions.",
    agent=pharmacist,
)

crew = Crew(agents=[pharmacist], tasks=[decrypt_task], verbose=True)

result = crew.kickoff()
print("\n===== RESULT =====")
print(result.raw)

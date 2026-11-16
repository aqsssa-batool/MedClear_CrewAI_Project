"""MedClear AI: Full 4-Agent Sequential Swarm.
All agents detect the patient/user language and respond in that same language.
Run: python medclear_crew.py
"""
import os
from config import llm
from tools import decode_abbreviations, check_drug_cautions
from crewai import Agent, Task, Crew, Process

# 1. Define the 4 Agents
pharmacist = Agent(
    role="Clinical Pharmacist",
    goal="Translate prescription abbreviations (OD, BD, TDS, AC, PC, HS, SOS) and explain what each medicine treats in simple everyday language matching the user",
    backstory="You are a compassionate community pharmacist. You break down complex prescriptions into the patient's or user's language for patients and family caregivers.",
    tools=[decode_abbreviations],
    llm=llm,
    verbose=True,
)

caution_auditor = Agent(
    role="Drug Safety Specialist",
    goal="Identify critical food-drug interactions, grapefruit warnings, meal timing necessities, and antibiotic completion rules",
    backstory="You are a meticulous pharmacovigilance auditor. You ensure patients take medicines safely with or without food, communicating clearly in the patient's language.",
    tools=[check_drug_cautions],
    llm=llm,
    verbose=True,
)

schedule_planner = Agent(
    role="Caregiver Routine Architect",
    goal="Design an hour-by-hour 24-hour daily medication timeline and 7-day printable pill tracker based on the patient's daily habits",
    backstory="You organize medicine schedules around natural daily milestones: waking up, breakfast, lunch, dinner, and bedtime.",
    llm=llm,
    verbose=True,
)

caregiver_advocate = Agent(
    role="Patient & Emergency Advocate",
    goal="Author emergency red-flag symptoms (when to call the doctor or visit the ER) and list 4 smart questions for the next clinic visit",
    backstory="You empower family caregivers with clear warning signs and essential doctor visit questions in the user's language with empathy.",
    llm=llm,
    verbose=True,
)

# 2. Define the Sequential Tasks with context chaining
decrypt_task = Task(
    description=(
        "MANDATORY: Detect the language of the patient/user and write all patient-facing content in that same language. Support any language; do not default to English. Translate tool outputs. Analyze this doctor's prescription:\n\n{prescription}\n\n"
        "1. Use the 'Decode medical abbreviations' tool to decode every Latin abbreviation (OD, BD, TDS, AC, PC, HS, SOS).\n"
        "2. For each medicine, list: Medicine Name & Strength, What it treats (in simple language matching the user), Frequency, Meal Relation (before/with/after food), and Duration.\n"
        "3. Format into a clean Markdown table in the same language as the patient/user."
    ),
    expected_output="A complete Markdown table in the same language as the patient/user, listing each decrypted medicine with its purpose and timing.",
    agent=pharmacist,
)

caution_task = Task(
    description=(
        "MANDATORY: Continue in the same language as the patient/user; translate tool results.\n"
        "Review the decrypted medicines and use the 'Check drug cautions' tool to audit safety warnings in the same language as the patient/user:\n"
        "1. Check food, beverage, and citrus/grapefruit restrictions.\n"
        "2. Check critical rules (such as finishing the antibiotic course, taking statins at bedtime, taking metformin with food).\n"
        "3. Provide clear bulleted Dos and Don'ts in the same language as the patient/user."
    ),
    expected_output="A structured Drug Safety & Caution report with clear Dos and Don'ts in the same language as the patient/user.",
    agent=caution_auditor,
    context=[decrypt_task],
)

schedule_task = Task(
    description=(
        "MANDATORY: Continue in the same language as the patient/user; translate schedule and checklist labels.\n"
        "Using the decrypted medicines and safety rules, build a 24-hour daily timeline customized to the patient's routine in the same language as the patient/user:\n\n"
        "Patient Profile:\n{patient_profile}\n\n"
        "1. Create a chronological schedule divided into: Morning (Empty Stomach & Breakfast), Afternoon (Lunch), Evening/Dinner, and Bedtime.\n"
        "2. Assign exact clock times (e.g. 7:15 AM, 8:15 AM, 1:30 PM, 8:45 PM, 10:15 PM).\n"
        "3. Create a 7-day printable caregiver pill checklist (Monday to Sunday checkbox table)."
    ),
    expected_output="A chronological 24-hour schedule with exact times and a 7-day printable checkbox table in the same language as the patient/user.",
    agent=schedule_planner,
    context=[decrypt_task, caution_task],
)

advocate_task = Task(
    description=(
        "MANDATORY: Write the final care plan in the same language as the patient/user.\n"
        "Synthesize the entire investigation into an authoritative Master Patient Care Plan strictly in the same language as the patient/user, containing:\n"
        "1. Patient Profile Summary\n"
        "2. Decrypted Medication Master Table\n"
        "3. Critical Food & Drug Cautions (Dos and Don'ts)\n"
        "4. Chronological 24-Hour Daily Timeline\n"
        "5. 7-Day Printable Caregiver Checklist\n"
        "6. Emergency Red-Flags: When to call the doctor or visit the ER immediately\n"
        "7. 4 Essential Questions for the caregiver to ask the doctor at the next follow-up appointment."
    ),
    expected_output="A publication-ready, deeply practical, and reassuring Markdown Patient Care Plan in the same language as the patient/user.",
    agent=caregiver_advocate,
    context=[decrypt_task, caution_task, schedule_task],
)

# 3. Create Crew
medclear_crew = Crew(
    agents=[pharmacist, caution_auditor, schedule_planner, caregiver_advocate],
    tasks=[decrypt_task, caution_task, schedule_task, advocate_task],
    process=Process.sequential,
    verbose=True,
)

def run():
    """Runs the MedClear Crew using sample files or custom inputs."""
    with open("sample_prescription.txt", "r", encoding="utf-8") as f:
        sample_rx = f.read()

    with open("sample_patient.txt", "r", encoding="utf-8") as f:
        sample_pt = f.read()

    print("=" * 75)
    print("MEDCLEAR AI: PRESCRIPTION SIMPLIFIER & CAREGIVER PLANNER")
    print("Engine: Groq (openai/gpt-oss-120b)")
    print("Language: Automatically detected from patient/user input")
    print("=" * 75)

    result = medclear_crew.kickoff(inputs={
        "prescription": sample_rx,
        "patient_profile": sample_pt,
    })

    print("\n" + "=" * 75)
    print("FINAL MASTER PATIENT CARE PLAN:")
    print("=" * 75)
    print(result.raw)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/patient_care_plan.md", "w", encoding="utf-8") as f:
        f.write(result.raw)
    print("\n[+] Saved to outputs/patient_care_plan.md")
    return result

if __name__ == "__main__":
    run()

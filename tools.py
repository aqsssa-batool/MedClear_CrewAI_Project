"""MedClear AI Custom Tools using CrewAI's @tool decorator.
Tool reference data is language-neutral source material; the agent must explain it in the patient/user language.
"""
import re
from crewai.tools import tool

@tool("Decode medical abbreviations")
def decode_abbreviations(sig: str) -> str:
    """Decode Latin medical shorthand (OD, BD, TDS, AC, PC, HS, SOS). Input: the abbreviation text."""
    dictionary = {
        "OD": "Once daily - Take at the exact same fixed time every day.",
        "BD": "Twice daily - Take approximately 12 hours apart (morning and evening).",
        "BID": "Twice daily - Take approximately 12 hours apart.",
        "TDS": "Three times daily - Take approximately every 7 to 8 hours after meals (morning, afternoon, night).",
        "TID": "Three times daily - Space evenly across waking hours.",
        "AC": "Before meals (Ante Cibum) - Take on an empty stomach 30 to 45 minutes before eating.",
        "PC": "After meals (Post Cibum) - Take with food or within 20 minutes after completing a meal to prevent gastric irritation.",
        "HS": "At bedtime (Hora Somni) - Take 30 minutes before sleep at night.",
        "SOS": "As needed in emergency (Si Opus Sit) - Take only if high fever (>100°F) or severe pain occurs. Do not take routinely.",
        "PRN": "As needed (Pro Re Nata) - Take only when symptoms appear, maintaining safe dosage limits.",
    }

    sig_upper = sig.upper()
    found = []
    for code, desc in dictionary.items():
        if re.search(r'\b' + re.escape(code) + r'\b', sig_upper):
            found.append(f"• [{code}]: {desc}")

    if not found:
        return f"Standard Medical Rule for '{sig}': Take with a full glass of plain water as prescribed."
    return "\n".join(found)


@tool("Check drug cautions")
def check_drug_cautions(drug_name: str) -> str:
    """Check food-drug interactions, grapefruit warnings, and critical safety rules. Input: medicine name."""
    name = drug_name.strip().lower()

    if "metformin" in name:
        return (
            "METFORMIN (Diabetes): Always take WITH or IMMEDIATELY AFTER meals. "
            "Taking on an empty stomach causes severe nausea, abdominal cramps, and diarrhea. Maintain adequate hydration."
        )
    elif "augmentin" in name or "amoxicillin" in name:
        return (
            "AUGMENTIN (Antibiotic): CRITICAL - MUST complete the full 7-day course even if fever and symptoms resolve early! "
            "Premature discontinuation breeds antibiotic-resistant bacteria. Take with or after food. Space apart from yogurt/probiotics by 2 hours."
        )
    elif "amlodipine" in name:
        return (
            "AMLODIPINE (Blood Pressure): STRICTLY AVOID GRAPEFRUIT and grapefruit juice (causes an unpredictable, dangerous drop in blood pressure). "
            "Stand up slowly from seated or lying positions to prevent postural dizziness."
        )
    elif "atorvastatin" in name:
        return (
            "ATORVASTATIN (Cholesterol): MUST be taken at BEDTIME/NIGHT because the liver synthesizes cholesterol during sleep. "
            "Avoid grapefruit juice. Immediately report any unexplained muscle pain or weakness to your doctor."
        )
    elif "omeprazole" in name:
        return (
            "OMEPRAZOLE (Gastric Shield): MUST be taken 30 to 45 minutes BEFORE breakfast on an empty stomach with plain water. "
            "Swallow capsule whole; do not crush or chew."
        )
    elif "panadol" in name or "paracetamol" in name:
        return (
            "PARACETAMOL / PANADOL (Fever/Pain): Maximum 4,000 mg (eight 500mg tablets) in 24 hours. "
            "Do not combine with other cold/cough remedies that also contain acetaminophen to prevent liver toxicity."
        )
    else:
        return f"General Rule for '{drug_name}': Swallow whole with a full glass of plain water. Avoid alcohol and avoid taking pills with tea or coffee."

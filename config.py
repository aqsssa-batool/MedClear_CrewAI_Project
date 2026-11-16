"""Shared setup for MedClear AI.
Reads GROQ_API_KEY from the .env file and builds one LLM object using Groq's OpenAI-compatible endpoint.
Configured for Model: openai/gpt-oss-120b
"""
import os, sys, logging
from dotenv import load_dotenv

load_dotenv()
os.environ["OTEL_SDK_DISABLED"] = "true"          # no telemetry
os.environ["CREWAI_TRACING_ENABLED"] = "false"    # no tracing prompts

# Windows console: allow emoji and symbols
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "paste_your_key_here":
    raise SystemExit(
        "GROQ_API_KEY not found.\n"
        "1. Copy .env.example to a new file named .env\n"
        "2. Paste your Groq key (starts with gsk_...) inside .env"
    )

from crewai import LLM

# Groq speaks the OpenAI protocol, so we use the "openai/" connector and point it to Groq.
# Model: openai/gpt-oss-120b
model_name = os.getenv("MODEL", "openai/gpt-oss-120b")
base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

llm = LLM(
    model=model_name,
    base_url=base_url,
    api_key=api_key,
    temperature=0.2,
)

logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)   # hide harmless warnings

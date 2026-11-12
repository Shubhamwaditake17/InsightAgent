import glob
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.agent import Agent
from app.llm import GroqClient
from app.tools import SQLTool, PandasTool, ChartTool, RAGTool
from data.pipeline import DATA_DICTIONARY, build_db

DB = os.getenv("DB_PATH", "data/sales.db")
DEFAULT_PROMPT = os.getenv("PROMPT", "planner_v2")
os.makedirs("charts", exist_ok=True)
app = FastAPI(title="InsightAgent")
app.mount("/charts", StaticFiles(directory="charts"), name="charts")
_agents = {}


def prompt_versions():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob("prompts/planner_v*.txt"))


def get_agent(version=DEFAULT_PROMPT):
    if version not in prompt_versions():
        raise HTTPException(404, f"unknown prompt version {version}")
    if version not in _agents:
        if not os.path.exists(DB):
            build_db(DB)
        tools = [SQLTool(DB), PandasTool(), ChartTool(), RAGTool(DATA_DICTIONARY)]
        _agents[version] = Agent(GroqClient(), tools, f"prompts/{version}.txt", "logs/traces.jsonl")
    return _agents[version]


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    prompt: str = DEFAULT_PROMPT


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/prompts")
def prompts():
    return {"versions": prompt_versions(), "default": DEFAULT_PROMPT}


@app.post("/ask")
def ask(q: Question):
    try:
        return get_agent(q.prompt).ask(q.question)
    except KeyError:
        raise HTTPException(500, "GROQ_API_KEY is not set")

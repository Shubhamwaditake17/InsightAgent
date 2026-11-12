"""Compare prompt versions.  Real run: python -m eval.run_eval   (needs GROQ_API_KEY)
Harness check without an LLM: python -m eval.run_eval --dry-run  (an oracle returns gold SQL; NOT model accuracy)"""
import argparse, datetime, glob, json, os, sqlite3, sys
import pandas as pd
from app.agent import Agent
from app.llm import LLMClient
from app.tools import SQLTool, PandasTool, ChartTool, RAGTool
from data.pipeline import DATA_DICTIONARY, build_db

DB = "data/sales.db"


def values(rows):
    return sorted(str(round(float(v), 2)) if isinstance(v, (int, float)) else str(v)
                  for r in rows for v in r.values())


class OracleLLM(LLMClient):
    def __init__(self, questions):
        self.gold = {q["question"]: q["gold_sql"] for q in questions}

    def complete(self, system, user):
        if system.startswith("You check"):
            return '{"answer": "ok", "grounded": true}'
        sql = self.gold[user.split("\n")[0]]
        return json.dumps({"steps": [{"tool": "sql", "args": {"query": sql}}]})


def evaluate(agent, questions, gold):
    hits = verified = 0
    lat, fails = [], []
    for q, g in zip(questions, gold):
        out = agent.ask(q["question"])
        sql = next((s["result"] for s in reversed(out["trace"]) if s["tool"] == "sql"), None)
        ok = sql is not None and values(sql) == values(g)
        hits, verified = hits + ok, verified + out["verified"]
        lat.append(out["metrics"]["total_ms"])
        if not ok:
            fails.append(q["question"])
    n = len(questions)
    return {"n": n, "accuracy": hits / n, "verified_rate": verified / n,
            "avg_ms": sum(lat) / n, "failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="docs/experiments.md")
    args = ap.parse_args()
    if not args.dry_run and "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY first (or use --dry-run to test the harness).")
    build_db(DB)
    questions = json.load(open("eval/questions.json"))
    con = sqlite3.connect(DB)
    gold = [pd.read_sql_query(q["gold_sql"], con).to_dict("records") for q in questions]
    tools = [SQLTool(DB), PandasTool(), ChartTool(), RAGTool(DATA_DICTIONARY)]
    lines = [f"\n## Run {datetime.datetime.now():%Y-%m-%d %H:%M}" + (" (DRY RUN: harness check only)" if args.dry_run else ""),
             "| Prompt | Questions | Exact-match | Verified | Avg latency (ms) |", "|---|---|---|---|---|"]
    for prompt in sorted(glob.glob("prompts/planner_v*.txt")):
        llm = OracleLLM(questions) if args.dry_run else __import__("app.llm", fromlist=["GroqClient"]).GroqClient()
        r = evaluate(Agent(llm, tools, prompt), questions, gold)
        name = os.path.basename(prompt)
        lines.append(f"| {name} | {r['n']} | {r['accuracy']:.0%} | {r['verified_rate']:.0%} | {r['avg_ms']:.0f} |")
        for f in r["failures"]:
            lines.append(f"  - FAILED ({name}): {f}")
    print("\n".join(lines))
    open(args.out, "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

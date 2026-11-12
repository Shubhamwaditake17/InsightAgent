import json
import pytest
from app.agent import Agent
from app.llm import MockLLM
from app.tools import SQLTool, PandasTool, RAGTool, ToolError
from data.pipeline import DATA_DICTIONARY, build_db


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d") / "sales.db")
    build_db(path)
    return path


def make_agent(db, responses):
    return Agent(MockLLM(responses), [SQLTool(db), PandasTool(), RAGTool(DATA_DICTIONARY)],
                 "prompts/planner_v2.txt")


@pytest.mark.parametrize("bad", [
    "DROP TABLE sales", "DELETE FROM sales", "SELECT * FROM sales; DROP TABLE sales",
    "SELECT * FROM sqlite_master", "UPDATE sales SET units=0"])
def test_sql_tool_blocks_unsafe_queries(db, bad):
    with pytest.raises(ToolError):
        SQLTool(db).run(bad)


def test_sql_tool_runs_select_and_limits_rows(db):
    rows = SQLTool(db, max_rows=5).run("SELECT * FROM sales")
    assert len(rows) == 5


def test_pandas_tool_sum_and_unknown_column():
    assert PandasTool().run([{"a": 1}, {"a": 2}], "sum", "a") == [{"sum": 3.0}]
    with pytest.raises(ToolError):
        PandasTool().run([{"a": 1}], "sum", "b")


def test_rag_tool_finds_definition():
    assert "revenue" in RAGTool(DATA_DICTIONARY).run("what is revenue")[0]["doc"]


def test_agent_end_to_end_verified(db):
    plan = {"steps": [{"tool": "sql", "args": {"query": "SELECT region, SUM(revenue) AS t FROM sales GROUP BY region"}}]}
    out = make_agent(db, [json.dumps(plan), '{"answer": "ok", "grounded": true}']).ask("revenue by region?")
    assert out["verified"] and len(out["trace"][0]["result"]) == 4


def test_agent_rejects_ungrounded_answer(db):
    plan = {"steps": [{"tool": "sql", "args": {"query": "SELECT COUNT(*) AS n FROM sales"}}]}
    out = make_agent(db, [json.dumps(plan), '{"answer": "made up", "grounded": false}']).ask("q")
    assert not out["verified"] and "reliable" in out["answer"]


def test_agent_self_corrects_after_blocked_query(db):
    bad = {"steps": [{"tool": "sql", "args": {"query": "DROP TABLE sales"}}]}
    good = {"steps": [{"tool": "sql", "args": {"query": "SELECT COUNT(*) AS n FROM sales"}}]}
    llm_out = [json.dumps(bad), json.dumps(good), '{"answer": "144", "grounded": true}']
    out = make_agent(db, llm_out).ask("how many rows?")
    assert out["verified"] and out["trace"][0]["tool"] == "sql"


def test_agent_reports_metrics_and_writes_trace(db, tmp_path):
    plan = {"steps": [{"tool": "sql", "args": {"query": "SELECT COUNT(*) AS n FROM sales"}}]}
    trace_file = tmp_path / "t.jsonl"
    agent = Agent(MockLLM([json.dumps(plan), '{"answer": "144", "grounded": true}']),
                  [SQLTool(db)], "prompts/planner_v2.txt", str(trace_file))
    out = agent.ask("how many rows?")
    assert out["metrics"]["llm_calls"] == 2 and out["metrics"]["tool_calls"] == 1
    assert json.loads(trace_file.read_text())["question"] == "how many rows?"


def test_malformed_verifier_reply_is_unverified(db):
    plan = {"steps": [{"tool": "sql", "args": {"query": "SELECT COUNT(*) AS n FROM sales"}}]}
    out = make_agent(db, [json.dumps(plan), "not json"]).ask("q")
    assert not out["verified"]


def test_plan_with_too_many_steps_is_rejected(db):
    step = {"tool": "sql", "args": {"query": "SELECT 1 FROM sales"}}
    out = make_agent(db, [json.dumps({"steps": [step] * 6})] * 2).ask("q")
    assert not out["verified"] and out["trace"] == []


def test_eval_harness_with_oracle_scores_100_percent(db):
    import pandas as pd, sqlite3
    from eval.run_eval import OracleLLM, evaluate
    questions = json.load(open("eval/questions.json"))
    assert len(questions) >= 25
    con = sqlite3.connect(db)
    gold = [pd.read_sql_query(q["gold_sql"], con).to_dict("records") for q in questions]
    agent = Agent(OracleLLM(questions), [SQLTool(db)], "prompts/planner_v2.txt")
    assert evaluate(agent, questions, gold)["accuracy"] == 1.0


def test_api_ask_and_prompts(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as m

    class Fake:
        def ask(self, q):
            return {"answer": "x", "verified": True, "trace": [], "metrics": {}}
    monkeypatch.setattr(m, "get_agent", lambda version=None: Fake())
    c = TestClient(m.app)
    assert c.get("/health").json() == {"status": "ok"}
    assert "planner_v2" in c.get("/prompts").json()["versions"]
    assert c.post("/ask", json={"question": "revenue by region?"}).json()["verified"]
    assert c.post("/ask", json={"question": "hi"}).status_code == 422

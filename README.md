# InsightAgent: Agentic AI Data Analyst

Ask a business question in plain English. An LLM plans tool calls (SQL, Pandas, RAG, chart),
the tools run, and a verifier step checks the answer against the tool output.

## Run
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
PYTHONPATH=. python data/pipeline.py        # builds data/sales.db (synthetic sales data)
PYTHONPATH=. uvicorn app.main:app --reload  # POST /ask {"question": "..."}
PYTHONPATH=. pytest -q                      # tests use a mocked LLM
PYTHONPATH=. python -m eval.run_eval        # compares prompts/planner_v*.txt
```
Or: `docker build -t insightagent . && docker run -p 8000:8000 -e GROQ_API_KEY=... insightagent`

## Design
- `app/tools.py`: `BaseTool` interface; `SQLTool` (SELECT only, single statement, table allowlist, row limit, read-only connection), `PandasTool`, `ChartTool`, `RAGTool` (TF-IDF over the data dictionary).
- `app/agent.py`: plan, execute, verify loop; one self-correction retry when a plan fails.
- `prompts/`: versioned planner prompts; `eval/run_eval.py` scores each against gold SQL results.
- Add a tool by subclassing `BaseTool`; the agent code does not change.

## Experiments log
Record each prompt version and its eval score here, plus failure cases and fixes.

## Observability and evaluation
- Every request returns `metrics` (total latency, LLM/tool call counts, approximate tokens) and is appended to `logs/traces.jsonl`.
- `GET /prompts` lists prompt versions; `POST /ask` takes `{"question": "...", "prompt": "planner_v1"}`.
- `eval/questions.json` holds 27 questions with gold SQL; results go to `docs/experiments.md`.
- `--dry-run` validates the harness with an oracle LLM. It is not a model score.
- Deploy: see `docs/deploy_ec2.md`.

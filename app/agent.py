import json
import re
import time
from app.tools import ToolError
from app.tracing import Tracer

MAX_STEPS = 4
VERIFY_SYSTEM = ("You check an analyst's answer. Using ONLY the tool results given, "
                 'reply JSON: {"answer": "<short answer>", "grounded": true|false}. '
                 "grounded is false if the results do not support an answer.")


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in LLM reply")
    return json.loads(m.group(0))


class Agent:
    """Plan -> act (tools) -> verify loop with one self-correction retry and tracing."""

    def __init__(self, llm, tools, prompt_path, trace_path=None):
        self.llm, self.trace_path = llm, trace_path
        self.tools = {t.name: t for t in tools}
        self.system = open(prompt_path).read()

    def _llm(self, tr, name, system, user):
        t = time.perf_counter()
        out = self.llm.complete(system, user)
        tr.log("llm", name, (time.perf_counter() - t) * 1000, (len(system) + len(user) + len(out)) // 4)
        return out

    def _plan(self, tr, question, error=None):
        user = question if not error else f"{question}\nPrevious plan failed: {error}. Fix it."
        steps = extract_json(self._llm(tr, "plan", self.system, user))["steps"]
        if not steps or len(steps) > MAX_STEPS:
            raise ValueError(f"plan must have 1-{MAX_STEPS} steps")
        return steps

    def _execute(self, tr, steps):
        trace, prev = [], None
        for step in steps:
            name, args = step["tool"], dict(step.get("args", {}))
            if name not in self.tools:
                raise ToolError(f"unknown tool {name}")
            if name in ("pandas", "chart"):
                args["rows"] = prev
            t = time.perf_counter()
            prev = self.tools[name].run(**args)
            tr.log("tool", name, (time.perf_counter() - t) * 1000)
            trace.append({"tool": name, "args": {k: v for k, v in args.items() if k != "rows"},
                          "result": prev})
        return trace

    def ask(self, question):
        tr, error, trace = Tracer(self.trace_path), None, []
        for _ in range(2):
            try:
                trace = self._execute(tr, self._plan(tr, question, error))
                break
            except (ToolError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                error, trace = str(e), []
        if not trace:
            out = {"answer": f"Could not complete the analysis: {error}", "verified": False, "trace": []}
        else:
            try:
                verdict = extract_json(self._llm(tr, "verify", VERIFY_SYSTEM,
                                       f"Question: {question}\nResults: {json.dumps(trace, default=str)}"))
                grounded = bool(verdict.get("grounded"))
                answer = verdict["answer"] if grounded else "Results do not support a reliable answer."
            except (ValueError, KeyError, json.JSONDecodeError):
                grounded, answer = False, "Could not verify the answer."
            out = {"answer": answer, "verified": grounded, "trace": trace}
        out["metrics"] = tr.summary()
        tr.flush(question)
        return out

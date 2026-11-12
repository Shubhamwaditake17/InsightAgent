import json
import os
import time


class Tracer:
    """Per-request trace: latency per LLM/tool call and approximate token usage."""

    def __init__(self, path=None):
        self.path, self.events, self.t0 = path, [], time.perf_counter()

    def log(self, kind, name, ms, tokens=0):
        self.events.append({"kind": kind, "name": name, "ms": round(ms, 1), "tokens": tokens})

    def summary(self):
        return {"total_ms": round((time.perf_counter() - self.t0) * 1000, 1),
                "llm_calls": sum(e["kind"] == "llm" for e in self.events),
                "tool_calls": sum(e["kind"] == "tool" for e in self.events),
                "approx_tokens": sum(e["tokens"] for e in self.events)}

    def flush(self, question):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"question": question, "events": self.events,
                                **self.summary()}) + "\n")

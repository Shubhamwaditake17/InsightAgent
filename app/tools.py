import re
import sqlite3
import time
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ToolError(Exception):
    pass


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs): ...


class SQLTool(BaseTool):
    """Read-only, single-statement SELECT on allowlisted tables."""
    name = "sql"
    description = "sql(query): read-only SELECT on sales"
    ALLOWED = {"sales"}
    FORBIDDEN = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", re.I)

    def __init__(self, db_path, max_rows=100, timeout_s=5):
        self.db_path, self.max_rows, self.timeout_s = db_path, max_rows, timeout_s

    def validate(self, query):
        q = query.strip().rstrip(";")
        if ";" in q:
            raise ToolError("only one statement allowed")
        if not re.match(r"(?is)^\s*select\b", q):
            raise ToolError("only SELECT is allowed")
        if self.FORBIDDEN.search(q):
            raise ToolError("forbidden keyword in query")
        tables = set(t.lower() for t in
                     re.findall(r"(?i)\b(?:from|join)\s+([a-z_][a-z0-9_]*)", q))
        if not tables <= self.ALLOWED:
            raise ToolError(f"table not allowed: {sorted(tables - self.ALLOWED)}")
        return q

    def run(self, query):
        q = self.validate(query)
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        deadline = time.time() + self.timeout_s
        con.set_progress_handler(lambda: int(time.time() > deadline), 10000)
        try:
            df = pd.read_sql_query(f"SELECT * FROM ({q}) LIMIT {self.max_rows}", con)
        except Exception as e:
            raise ToolError(str(e))
        finally:
            con.close()
        return df.to_dict("records")


class PandasTool(BaseTool):
    name = "pandas"
    description = "pandas(op, column): op in sum|mean|pct_change|describe on previous rows"

    def run(self, rows, op, column):
        df = pd.DataFrame(rows)
        if column not in df.columns:
            raise ToolError(f"unknown column {column}")
        s = df[column]
        ops = {"sum": lambda: [{"sum": float(s.sum())}],
               "mean": lambda: [{"mean": float(s.mean())}],
               "pct_change": lambda: df.assign(pct_change=s.pct_change() * 100)
                                       .fillna(0).to_dict("records"),
               "describe": lambda: [s.describe().to_dict()]}
        if op not in ops:
            raise ToolError(f"unknown op {op}")
        return ops[op]()


class ChartTool(BaseTool):
    name = "chart"
    description = "chart(x, y): bar chart of previous rows, saved as PNG"

    def __init__(self, out_dir="charts"):
        self.out_dir = out_dir

    def run(self, rows, x, y):
        import os
        df = pd.DataFrame(rows)
        if x not in df.columns or y not in df.columns:
            raise ToolError("unknown chart columns")
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{y}_by_{x}.png")
        df.plot.bar(x=x, y=y, legend=False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return [{"chart_path": path}]


class RAGTool(BaseTool):
    """TF-IDF retrieval over the data dictionary (swap for ChromaDB if needed)."""
    name = "rag"
    description = "rag(query): look up column and metric definitions"

    def __init__(self, docs, k=2):
        self.docs, self.k = docs, k
        self.vec = TfidfVectorizer().fit(docs)
        self.matrix = self.vec.transform(docs)

    def run(self, query):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][: self.k]
        return [{"doc": self.docs[i]} for i in top if sims[i] > 0]

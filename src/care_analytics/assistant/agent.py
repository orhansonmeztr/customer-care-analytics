"""Ask the data: question -> SQL -> guard -> result -> short answer.

The loop is explicit and small on purpose: one model call to write SQL, the guard, the
query, and up to ``max_retries`` repair rounds where the model sees its own error. A refusal
from the model, a query the guard blocks and a failed query are different outcomes and are
reported as such, because they mean different things when the assistant is evaluated.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from .context import schema_context
from .guard import GuardError, SqlSyntaxError, check_sql, run_query
from .llm import ChatModel

_SQL_BLOCK = re.compile(r"```(?:sql|duckdb)?\s*(.*?)```", re.S | re.I)


@dataclass
class AskResult:
    question: str
    status: str  # answered | refused | blocked | error
    sql: str | None = None
    rows: pd.DataFrame | None = None
    answer: str = ""
    error: str = ""
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    seconds: float = 0.0
    history: list[str] = field(default_factory=list)


def extract_sql(text: str) -> str | None:
    # the last block is the answer: reasoning models write drafts before it
    blocks = _SQL_BLOCK.findall(text)
    if blocks:
        return blocks[-1].strip()
    stripped = text.strip()
    if re.match(r"(?is)^\s*(select|with)\b", stripped):
        return stripped
    return None


_REFUSE_LITERAL = re.compile(r"(?is)^\s*select\s+'REFUSE:?\s*([^']*)'")


def _refusal(text: str) -> str | None:
    sql = extract_sql(text)
    if sql is None:
        m = re.search(r"REFUSE:?\s*(.*)", text)
        return (m.group(1).strip() or "refused") if m else None
    # some models wrap the refusal in a query: SELECT 'REFUSE: ...' AS response
    m = _REFUSE_LITERAL.match(sql)
    return (m.group(1).strip() or "refused") if m else None


def _preview(df: pd.DataFrame, n: int = 15) -> str:
    return df.head(n).to_string(index=False, max_colwidth=40)


class Assistant:
    def __init__(self, con, model: ChatModel, max_retries: int = 2):
        self.con = con
        self.model = model
        self.max_retries = max_retries
        self.system = schema_context(con)

    def ask(self, question: str, explain: bool = False) -> AskResult:
        res = AskResult(question=question, status="error")
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": question}]  # fmt: skip
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            res.attempts = attempt
            reply = self.model.chat(messages)
            res.prompt_tokens += reply.prompt_tokens
            res.completion_tokens += reply.completion_tokens
            res.cached_prompt_tokens += reply.cached_prompt_tokens
            res.history.append(reply.text)
            messages.append({"role": "assistant", "content": reply.text})

            reason = _refusal(reply.text)
            if reason:
                res.status, res.answer = "refused", reason
                break
            sql = extract_sql(reply.text)
            if sql is None:
                res.error = "the reply contained no SQL"
                messages.append({"role": "user", "content": "Reply with one SQL query in a "
                                 "```sql code block, or REFUSE: <reason>."})  # fmt: skip
                continue
            try:
                checked = check_sql(sql)
            except SqlSyntaxError as e:
                res.sql, res.error = sql, str(e)
                messages.append({"role": "user", "content": f"The SQL is invalid: {e}\n"
                                 "Reply with the corrected SQL only."})  # fmt: skip
                continue
            except GuardError as e:
                # a blocked query is final: the model does not get to look for a way around it
                res.status, res.sql, res.error = "blocked", sql, str(e)
                break
            try:
                res.rows = run_query(self.con, checked.sql)
                res.sql, res.status, res.error = checked.sql, "answered", ""
                break
            except Exception as e:  # noqa: BLE001 - any database error goes back to the model
                res.sql, res.error = checked.sql, str(e).splitlines()[0][:300]
                messages.append({"role": "user", "content": f"The query failed: {res.error}\n"
                                 "Fix it and reply with the corrected SQL only."})  # fmt: skip
        if res.status == "answered" and explain:
            res.answer = self._explain(question, res)
        elif res.status == "answered":
            res.answer = _plain_answer(res.rows)
        elif res.status == "blocked":
            res.answer = f"I can't run that query: {res.error}."
        res.seconds = time.perf_counter() - t0
        return res

    def _explain(self, question: str, res: AskResult) -> str:
        reply = self.model.chat([
            {"role": "system", "content": "You explain query results to a customer care manager "
             "in 1-3 short sentences, in the language of the question. Use only the numbers in "
             "the result; rates are fractions, show them as percentages. Do not invent anything."},
            {"role": "user", "content": f"Question: {question}\n\nResult:\n{_preview(res.rows)}"},
        ])  # fmt: skip
        res.prompt_tokens += reply.prompt_tokens
        res.completion_tokens += reply.completion_tokens
        return reply.text


def _plain_answer(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "The query returned no rows."
    if df.shape == (1, 1):
        v = df.iat[0, 0]
        return (
            f"{df.columns[0]}: {v:,.4g}" if isinstance(v, int | float) else f"{df.columns[0]}: {v}"
        )
    return f"{len(df):,} rows."

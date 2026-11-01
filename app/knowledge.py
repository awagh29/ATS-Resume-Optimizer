"""Private local knowledge, optimization history, and exact-response caching.

Only explicitly entered knowledge is eligible for retrieval. Generated resumes and
historical model responses are stored separately and never become candidate facts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any
import uuid


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _text(value: str, label: str, maximum: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label} is required.")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters.")
    return value


_STOP_WORDS = frozenset(
    "a an the and or to of in on for with at by from as is are be been being "
    "will can may must should have has had this that these those you your our "
    "we us it its their they them work role job candidate required preferred "
    "experience ability skills knowledge responsibilities qualifications".split()
)


def _terms(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[\w]+(?:[.+#-][\w+#]+)*[+#]*", text.casefold())
        if len(word) > 1 and word not in _STOP_WORDS
    }


class KnowledgeStore:
    """Thread-safe SQLite persistence for one locally operated application.

    Profiles isolate different candidates within the same local database. They
    are namespaces, not authentication: a network deployment needs its own access
    control. The store never makes network requests or uses embedding APIs.
    """

    MAX_CACHE_ENTRIES = 256
    MAX_CACHE_BYTES = 16_000_000
    MAX_PAYLOAD_BYTES = 2_000_000
    MAX_ENTRIES_PER_PROFILE = 10_000

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(target)
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        # Deleted personal data should not remain in SQLite's free pages.
        self._db.execute("PRAGMA secure_delete=ON")
        with self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(profile_id, content_hash)
                );
                CREATE INDEX IF NOT EXISTS knowledge_profile ON knowledge_entries(profile_id);
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    company TEXT NOT NULL,
                    role_title TEXT NOT NULL,
                    job_description TEXT NOT NULL,
                    resume_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE INDEX IF NOT EXISTS runs_profile ON optimization_runs(profile_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                PRAGMA user_version=1;
            """)

    @staticmethod
    def _profile(profile_id: str) -> str:
        return _text(profile_id, "Profile", 200)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add_entry(self, title: str, content: str, category: str = "accomplishment",
                  profile_id: str = "default") -> dict:
        title = _text(title, "Title", 200)
        content = _text(content, "Knowledge", 20_000)
        category = _text(category, "Category", 40)
        profile_id = self._profile(profile_id)
        content_hash = hashlib.sha256(_json([title, content, category]).encode("utf-8")).hexdigest()
        with self._lock, self._db:
            existing = self._db.execute(
                "SELECT * FROM knowledge_entries WHERE profile_id=? AND content_hash=?",
                (profile_id, content_hash),
            ).fetchone()
            if existing:
                return self._entry(existing)
            count = self._db.execute(
                "SELECT COUNT(*) FROM knowledge_entries WHERE profile_id=?", (profile_id,),
            ).fetchone()[0]
            if count >= self.MAX_ENTRIES_PER_PROFILE:
                raise ValueError("Knowledge store is full. Remove unused entries before adding more.")
            entry_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO knowledge_entries(id,profile_id,title,content,category,content_hash) VALUES(?,?,?,?,?,?)",
                (entry_id, profile_id, title, content, category, content_hash),
            )
            return self._entry(self._db.execute(
                "SELECT * FROM knowledge_entries WHERE id=?", (entry_id,),
            ).fetchone())

    @staticmethod
    def _entry(row: sqlite3.Row) -> dict:
        return {key: row[key] for key in ("id", "profile_id", "title", "content", "category", "created_at")}

    def list_entries(self, profile_id: str = "default") -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM knowledge_entries WHERE profile_id=? ORDER BY created_at DESC,id",
                (self._profile(profile_id),),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def delete_entry(self, entry_id: str, profile_id: str = "default") -> bool:
        with self._lock, self._db:
            changed = self._db.execute(
                "DELETE FROM knowledge_entries WHERE id=? AND profile_id=?",
                (_text(entry_id, "Knowledge ID", 200), self._profile(profile_id)),
            ).rowcount
            # Previously generated responses may contain facts that were removed.
            if changed:
                self._db.execute("DELETE FROM response_cache")
            return bool(changed)

    def retrieve(self, query: str, profile_id: str = "default", max_chars: int = 6000,
                 limit: int = 12) -> list[dict]:
        """Rank lexical matches locally and return bounded, attributable facts.

        The compact JSON serialization of the returned list is at most max_chars.
        Entries are included intact or skipped: truncating a fact can remove its
        attribution or negation. Historical optimization results are deliberately
        excluded from the search corpus.
        """
        if not isinstance(max_chars, int) or max_chars < 0 or max_chars > 20_000:
            raise ValueError("Retrieval budget must be between 0 and 20000 characters.")
        if not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("Retrieval limit must be between 0 and 100.")
        query = _text(query, "Search query", 80_000, required=False)
        query_terms = _terms(query)
        if max_chars < 2 or not query_terms or limit == 0:
            return []
        entries = self.list_entries(profile_id)
        documents = [(entry, _terms(entry["title"] + " " + entry["content"])) for entry in entries]
        frequencies = {term: sum(term in words for _, words in documents) for term in query_terms}
        ranked = []
        for entry, words in documents:
            overlap = words & query_terms
            if not overlap:
                continue
            title_terms = _terms(entry["title"])
            score = sum((1.0 + math.log((len(entries) + 1) / (frequencies[term] + 1)))
                        * (2 if term in title_terms else 1) for term in overlap)
            # Prefer focused evidence over large notes mentioning every keyword.
            score /= 1.0 + math.log1p(len(words)) / 8
            ranked.append((score, entry))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        selected: list[dict] = []
        for _, entry in ranked:
            record = {**entry, "source": f"knowledge:{entry['id']}", "truncated": False}
            if len(_json([*selected, record])) > max_chars:
                continue
            selected.append(record)
            if len(selected) >= limit:
                break
        return selected

    def save_run(self, result: dict, *, job_description: str, company: str = "",
                 role_title: str = "", profile_id: str = "default", resume_hash: str = "") -> str:
        if not isinstance(result, dict):
            raise ValueError("Optimization result must be an object.")
        # PDFs are temporary artifacts; keep durable structured data and LaTeX only.
        clean_result = dict(result)
        run_id = uuid.uuid4().hex
        clean_result["run_id"] = run_id
        if isinstance(clean_result.get("compilation"), dict):
            clean_result["compilation"] = {
                key: value for key, value in clean_result["compilation"].items()
                if key != "pdf_bytes"
            }
        try:
            payload = _json(clean_result)
        except (TypeError, ValueError) as exc:
            raise ValueError("Optimization result must contain JSON data.") from exc
        if len(payload.encode("utf-8")) > self.MAX_PAYLOAD_BYTES:
            raise ValueError("Optimization history result is too large.")
        row = (run_id, self._profile(profile_id),
               _text(company, "Company", 200, required=False),
               _text(role_title, "Role title", 200, required=False),
               _text(job_description, "Job description", 80_000),
               _text(resume_hash, "Resume hash", 200, required=False), payload)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO optimization_runs(id,profile_id,company,role_title,job_description,resume_hash,result_json) "
                "VALUES(?,?,?,?,?,?,?)", row,
            )
        return row[0]

    def list_runs(self, profile_id: str = "default", limit: int = 50) -> list[dict]:
        if not isinstance(limit, int) or not 0 <= limit <= 500:
            raise ValueError("History limit must be between 0 and 500.")
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM optimization_runs WHERE profile_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (self._profile(profile_id), limit),
            ).fetchall()
        summaries = []
        for row in rows:
            result = json.loads(row["result_json"])
            summary = {key: row[key] for key in ("id", "profile_id", "company", "role_title", "created_at")}
            summary.update(
                original_score=result.get("original_assessment", {}).get("score", {}).get("total", 0),
                optimized_score=result.get("optimized_assessment", {}).get("score", {}).get("total", 0),
                target_reached=bool(result.get("target_reached", False)),
                demo_mode=bool(result.get("demo_mode", False)),
            )
            summaries.append(summary)
        return summaries

    def get_run(self, run_id: str, profile_id: str = "default") -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM optimization_runs WHERE id=? AND profile_id=?",
                (_text(run_id, "Run ID", 200), self._profile(profile_id)),
            ).fetchone()
        if row is None:
            return None
        record = {key: row[key] for key in row.keys() if key != "result_json"}
        record["result"] = json.loads(row["result_json"])
        return record

    def delete_run(self, run_id: str, profile_id: str = "default") -> bool:
        with self._lock, self._db:
            return bool(self._db.execute(
                "DELETE FROM optimization_runs WHERE id=? AND profile_id=?",
                (_text(run_id, "Run ID", 200), self._profile(profile_id)),
            ).rowcount)

    def get_cached_response(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM response_cache WHERE cache_key=?",
                (_text(key, "Cache key", 256),),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def set_cached_response(self, key: str, payload: dict) -> None:
        key = _text(key, "Cache key", 256)
        if not isinstance(payload, dict):
            raise ValueError("Cached response must be an object.")
        try:
            serialized = _json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cached response must contain JSON data.") from exc
        payload_bytes = len(serialized.encode("utf-8"))
        if payload_bytes > self.MAX_PAYLOAD_BYTES or payload_bytes > self.MAX_CACHE_BYTES:
            return
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO response_cache(cache_key,payload_json,payload_bytes) VALUES(?,?,?) "
                "ON CONFLICT(cache_key) DO UPDATE SET payload_json=excluded.payload_json,"
                "payload_bytes=excluded.payload_bytes,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (key, serialized, payload_bytes),
            )
            while True:
                count, size = self._db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(payload_bytes),0) FROM response_cache",
                ).fetchone()
                if count <= self.MAX_CACHE_ENTRIES and size <= self.MAX_CACHE_BYTES:
                    break
                self._db.execute(
                    "DELETE FROM response_cache WHERE cache_key=(SELECT cache_key FROM response_cache "
                    "ORDER BY updated_at ASC,rowid ASC LIMIT 1)",
                )

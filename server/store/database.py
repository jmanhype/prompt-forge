"""SQLite database for composition library + iteration logs."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class Database:
    """SQLite store with FTS5 search for composition library."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
    
    def initialize(self):
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
    
    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS compositions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                style TEXT,
                final_prompt TEXT,
                final_score REAL,
                iteration_count INTEGER,
                tags TEXT,
                image_path TEXT,
                converged INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS iterations (
                id TEXT PRIMARY KEY,
                composition_id TEXT REFERENCES compositions(id),
                iteration_number INTEGER,
                prompt TEXT,
                score TEXT,
                diagnosis TEXT,
                mutations TEXT,
                image_path TEXT,
                duration_ms INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS loras (
                filename TEXT PRIMARY KEY,
                trigger_words TEXT,
                style_tags TEXT,
                last_scanned TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        
        # FTS5 for search (ignore if already exists)
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS compositions_fts 
                USING fts5(description, style, tags, content='compositions', content_rowid='rowid')
            """)
        except sqlite3.OperationalError:
            pass  # already exists
        
        self._conn.commit()
    
    def save_composition(self, result) -> str:
        """Save a forge result to the composition library."""
        comp_id = str(uuid.uuid4())[:8]
        final = result.iterations[-1] if result.iterations else None
        
        style = ""
        if final and final.prompt:
            style = json.dumps(final.prompt.get("style_description", {}))
        
        tags = ""
        if final and final.prompt:
            elems = final.prompt.get("composition", {}).get("elements", [])
            tags = ",".join(e.get("label", e.get("desc", ""))[:30] for e in elems[:5])
        
        self._conn.execute(
            """INSERT INTO compositions 
               (id, description, style, final_prompt, final_score, iteration_count, tags, image_path, converged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                comp_id,
                result.description,
                style,
                json.dumps(final.prompt) if final else "{}",
                result.final_score,
                len(result.iterations),
                tags,
                final.images[0] if final and final.images else "",
                1 if result.converged else 0,
            )
        )
        
        # Save individual iterations
        for iteration in result.iterations:
            self._conn.execute(
                """INSERT INTO iterations
                   (id, composition_id, iteration_number, prompt, score, diagnosis, mutations, image_path, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4())[:12],
                    comp_id,
                    iteration.number,
                    json.dumps(iteration.prompt),
                    json.dumps(iteration.score.to_dict()) if iteration.score else "{}",
                    json.dumps(iteration.diagnosis),
                    json.dumps(iteration.mutations),
                    iteration.images[0] if iteration.images else "",
                    iteration.duration_ms,
                )
            )
        
        # Update FTS index
        try:
            self._conn.execute(
                "INSERT INTO compositions_fts(rowid, description, style, tags) VALUES (last_insert_rowid(), ?, ?, ?)",
                (result.description, style, tags)
            )
        except sqlite3.OperationalError:
            pass
        
        self._conn.commit()
        return comp_id
    
    def search_compositions(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across composition library."""
        try:
            rows = self._conn.execute(
                """SELECT c.* FROM compositions c
                   JOIN compositions_fts f ON c.rowid = f.rowid
                   WHERE compositions_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback: LIKE search
            rows = self._conn.execute(
                """SELECT * FROM compositions 
                   WHERE description LIKE ? OR tags LIKE ? OR style LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        
        return [dict(r) for r in rows]
    
    def list_compositions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List recent compositions."""
        rows = self._conn.execute(
            "SELECT * FROM compositions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    
    def get_composition(self, comp_id: str) -> Optional[dict]:
        """Get a single composition with its iterations."""
        row = self._conn.execute(
            "SELECT * FROM compositions WHERE id = ?", (comp_id,)
        ).fetchone()
        if not row:
            return None
        
        comp = dict(row)
        iters = self._conn.execute(
            "SELECT * FROM iterations WHERE composition_id = ? ORDER BY iteration_number",
            (comp_id,)
        ).fetchall()
        comp["iterations"] = [dict(i) for i in iters]
        return comp
    
    def get_stats(self) -> dict:
        """Get library statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM compositions").fetchone()[0]
        converged = self._conn.execute(
            "SELECT COUNT(*) FROM compositions WHERE converged = 1"
        ).fetchone()[0]
        avg_score = self._conn.execute(
            "SELECT AVG(final_score) FROM compositions"
        ).fetchone()[0] or 0
        total_iters = self._conn.execute(
            "SELECT COUNT(*) FROM iterations"
        ).fetchone()[0]
        
        return {
            "total_compositions": total,
            "converged": converged,
            "convergence_rate": converged / max(total, 1),
            "avg_score": round(avg_score, 3),
            "total_iterations": total_iters,
        }
    
    def close(self):
        if self._conn:
            self._conn.close()

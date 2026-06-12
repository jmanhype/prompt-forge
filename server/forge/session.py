"""Session state management for active forge runs."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from .engine import ForgeEngine, ForgeEvent


@dataclass
class Session:
    id: str
    description: str
    status: str = "idle"  # "running" | "converged" | "error"
    events: list[dict] = field(default_factory=list)
    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    async def add_event(self, event: ForgeEvent):
        self.events.append({"type": event.type, "data": event.data})
        await self._event_queue.put(event)
    
    async def next_event(self, timeout: float = 300) -> Optional[ForgeEvent]:
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class SessionManager:
    """Manages active forge sessions."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
    
    def create(self, session_id: str, description: str) -> Session:
        session = Session(id=session_id, description=description)
        self._sessions[session_id] = session
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
    
    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)
    
    def list_active(self) -> list[str]:
        return [s.id for s in self._sessions.values() if s.status == "running"]

# Global session manager
sessions = SessionManager()

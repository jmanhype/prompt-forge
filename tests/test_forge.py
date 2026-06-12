"""Tests for forge engine (unit tests, no actual generation)."""
import pytest
from server.forge.engine import ForgeEngine, Iteration, ForgeResult, ForgeEvent
from server.scorer.scorer import ForgeScore


class TestIteration:
    def test_creation(self):
        iter = Iteration(number=1, prompt={"caption": "test"})
        assert iter.number == 1
        assert iter.images == []
        assert iter.score is None
        assert iter.mutations == []


class TestForgeResult:
    def test_creation(self):
        result = ForgeResult(
            session_id="abc123",
            description="test description",
            converged=True,
            final_score=0.92,
            total_duration_ms=15000,
        )
        assert result.converged is True
        assert result.final_score == 0.92


class TestForgeEvent:
    def test_event_types(self):
        for event_type in ["analyzing", "generating", "scoring", "mutating", "iteration", "converged", "error"]:
            event = ForgeEvent(type=event_type, data={"test": True})
            assert event.type == event_type
            assert event.data["test"] is True

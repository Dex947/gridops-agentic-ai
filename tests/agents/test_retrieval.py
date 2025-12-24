"""Tests for RetrievalAgent."""

import pytest
from pathlib import Path

from src.agents.retrieval import RetrievalAgent


class TestRetrievalAgentInit:
    """Tests for initialization."""

    def test_init_without_path(self):
        """Should initialize without knowledge base path."""
        agent = RetrievalAgent()
        
        assert agent.knowledge_base_path is None
        assert agent.retrieved_cache == {}

    def test_init_with_path(self, tmp_path):
        """Should initialize with knowledge base path."""
        agent = RetrievalAgent(knowledge_base_path=tmp_path)
        
        assert agent.knowledge_base_path == tmp_path


class TestRetrieveStandards:
    """Tests for standard retrieval."""

    def test_retrieve_voltage_standards(self):
        """Should retrieve voltage-related standards."""
        agent = RetrievalAgent()
        
        context = {
            "contingency_type": "line_outage",
            "violations": ["Bus 10: undervoltage"]
        }
        
        standards = agent.retrieve_relevant_standards(context)
        
        assert len(standards) > 0
        # Should include voltage standard
        citations = [s.get("citation", "") for s in standards]
        assert any("C84.1" in c for c in citations)

    def test_retrieve_protection_standards(self):
        """Should retrieve protection standards for switching."""
        agent = RetrievalAgent()
        
        context = {
            "contingency_type": "line_outage",
            "action_type": "switch_line"
        }
        
        standards = agent.retrieve_relevant_standards(context)
        
        assert len(standards) > 0

    def test_retrieve_n1_standards(self):
        """Should retrieve N-1 contingency standards."""
        agent = RetrievalAgent()
        
        context = {
            "contingency_type": "line_outage",
            "is_n_minus_1": True
        }
        
        standards = agent.retrieve_relevant_standards(context)
        
        # Should include NERC standard
        citations = [s.get("citation", "") for s in standards]
        assert any("NERC" in c or "TPL" in c for c in citations)


class TestGetBestPractices:
    """Tests for best practices retrieval."""

    def test_get_switching_practices(self):
        """Should return switching best practices."""
        agent = RetrievalAgent()
        
        practices = agent.get_best_practices("switch_line")
        
        assert isinstance(practices, dict)

    def test_get_load_shedding_practices(self):
        """Should return load shedding best practices."""
        agent = RetrievalAgent()
        
        practices = agent.get_best_practices("shed_load")
        
        assert isinstance(practices, dict)

    def test_unknown_action_type(self):
        """Should handle unknown action types."""
        agent = RetrievalAgent()
        
        practices = agent.get_best_practices("unknown_action")
        
        # Should return empty or default practices
        assert isinstance(practices, dict)


class TestFormatReferences:
    """Tests for reference formatting."""

    def test_format_for_report(self):
        """Should format references for reports."""
        agent = RetrievalAgent()
        
        context = {"contingency_type": "line_outage"}
        
        # format_references_for_report returns a formatted string
        formatted = agent.format_references_for_report(context)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "References" in formatted or "IEEE" in formatted or "NERC" in formatted


class TestStandardReferences:
    """Tests for built-in standard references."""

    def test_has_voltage_limits(self):
        """Should have voltage limits standard."""
        assert "voltage_limits" in RetrievalAgent.STANDARD_REFERENCES
        
        ref = RetrievalAgent.STANDARD_REFERENCES["voltage_limits"]
        assert "ANSI" in ref["title"]
        assert "citation" in ref

    def test_has_n_minus_1(self):
        """Should have N-1 contingency standard."""
        assert "n_minus_1" in RetrievalAgent.STANDARD_REFERENCES
        
        ref = RetrievalAgent.STANDARD_REFERENCES["n_minus_1"]
        assert "NERC" in ref["title"]

    def test_has_load_shedding(self):
        """Should have load shedding standard."""
        assert "load_shedding" in RetrievalAgent.STANDARD_REFERENCES
        
        ref = RetrievalAgent.STANDARD_REFERENCES["load_shedding"]
        assert "IEEE" in ref["title"]

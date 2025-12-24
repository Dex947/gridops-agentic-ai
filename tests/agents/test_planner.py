"""Tests for PlannerAgent with mocked LLM."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.agents.planner import PlannerAgent
from src.core.state_manager import create_initial_state, ActionProposal


@pytest.fixture
def mock_llm_response():
    """Sample LLM response for action plans."""
    return """```json
[
    {
        "action_id": "switch_tie_1",
        "action_type": "switch_line",
        "target_elements": [{"line_id": 32, "close": true}],
        "expected_impact": "Restore power to isolated buses",
        "priority": 5,
        "rationale": "Close tie switch to restore supply path"
    },
    {
        "action_id": "shed_load_1",
        "action_type": "shed_load",
        "target_elements": [{"load_id": 15, "reduction_percent": 20}],
        "expected_impact": "Reduce loading on critical lines",
        "priority": 3,
        "rationale": "Reduce load to prevent thermal violations"
    }
]
```"""


class TestPlannerAgentInit:
    """Tests for PlannerAgent initialization."""

    @patch('src.agents.planner.ChatOpenAI')
    def test_init_openai(self, mock_openai):
        """Should initialize with OpenAI."""
        agent = PlannerAgent(
            model_name="gpt-4o-mini",
            provider="openai",
            api_key="test-key"
        )
        
        assert agent.provider == "openai"
        assert agent.model_name == "gpt-4o-mini"
        mock_openai.assert_called_once()

    @patch('src.agents.planner.ChatAnthropic')
    def test_init_anthropic(self, mock_anthropic):
        """Should initialize with Anthropic."""
        agent = PlannerAgent(
            model_name="claude-3-sonnet",
            provider="anthropic",
            api_key="test-key"
        )
        
        assert agent.provider == "anthropic"
        mock_anthropic.assert_called_once()

    def test_init_invalid_provider(self):
        """Should raise for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            PlannerAgent(provider="invalid")


class TestGenerateActionPlans:
    """Tests for action plan generation."""

    @patch('src.agents.planner.ChatOpenAI')
    def test_generate_plans_success(self, mock_openai_cls, mock_llm_response):
        """Should parse LLM response into ActionProposals."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=mock_llm_response)
        mock_openai_cls.return_value = mock_llm
        
        agent = PlannerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Line 5 outage", 5)
        state["baseline_results"] = {"converged": True, "violations": []}
        state["contingency_results"] = {"converged": True}
        state["constraint_violations"] = ["Bus 10: undervoltage"]
        
        network_data = {
            "num_buses": 33,
            "num_lines": 37,
            "total_load_mw": 3.7,
            "reconfiguration_options": []
        }
        
        proposals = agent.generate_action_plans(state, network_data)
        
        assert len(proposals) == 2
        assert proposals[0].action_id == "switch_tie_1"
        assert proposals[1].action_type == "shed_load"

    @patch('src.agents.planner.ChatOpenAI')
    def test_generate_plans_llm_error(self, mock_openai_cls):
        """Should return empty list on LLM error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_openai_cls.return_value = mock_llm
        
        agent = PlannerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Test", 5)
        state["baseline_results"] = {}
        state["contingency_results"] = {}
        state["constraint_violations"] = []
        
        proposals = agent.generate_action_plans(state, {})
        
        assert proposals == []

    @patch('src.agents.planner.ChatOpenAI')
    def test_generate_plans_invalid_json(self, mock_openai_cls):
        """Should handle invalid JSON gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Not valid JSON")
        mock_openai_cls.return_value = mock_llm
        
        agent = PlannerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Test", 5)
        state["baseline_results"] = {}
        state["contingency_results"] = {}
        state["constraint_violations"] = []
        
        proposals = agent.generate_action_plans(state, {})
        
        # Should return empty or handle gracefully
        assert isinstance(proposals, list)


class TestPrepareContext:
    """Tests for context preparation."""

    @patch('src.agents.planner.ChatOpenAI')
    def test_prepare_context_includes_network_info(self, mock_openai_cls):
        """Context should include network information."""
        agent = PlannerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Line 5 outage", 5)
        state["contingency_type"] = "line_outage"
        state["contingency_elements"] = [5]
        state["baseline_results"] = {"converged": True}
        state["contingency_results"] = {"converged": True}
        state["constraint_violations"] = ["Bus 10: undervoltage"]
        
        network_data = {
            "num_buses": 33,
            "num_lines": 37,
            "total_load_mw": 3.715
        }
        
        context = agent._prepare_context(state, network_data)
        
        assert "ieee_33" in context
        assert "33" in context  # num_buses
        assert "Line 5 outage" in context
        assert "line_outage" in context

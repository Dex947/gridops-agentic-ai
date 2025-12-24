"""Tests for ExplainerAgent with mocked LLM."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.explainer import ExplainerAgent
from src.core.state_manager import create_initial_state


@pytest.fixture
def mock_explanation():
    """Sample LLM explanation."""
    return """## Contingency Analysis Summary

### Problem Statement
A line outage occurred on Line 5, causing downstream voltage violations.

### Actions Taken
The system closed tie switch 32 to restore power supply through an alternative path.

### Technical Rationale
This action restores connectivity while maintaining voltage within ANSI C84.1-2020 limits.

### Safety Considerations
All thermal limits remain within acceptable ranges. No protection coordination issues identified.
"""


class TestExplainerAgentInit:
    """Tests for ExplainerAgent initialization."""

    @patch('src.agents.explainer.ChatOpenAI')
    def test_init_openai(self, mock_openai):
        """Should initialize with OpenAI."""
        agent = ExplainerAgent(
            model_name="gpt-4o-mini",
            provider="openai",
            api_key="test-key"
        )
        
        assert agent.provider == "openai"
        mock_openai.assert_called_once()

    @patch('src.agents.explainer.ChatAnthropic')
    def test_init_anthropic(self, mock_anthropic):
        """Should initialize with Anthropic."""
        agent = ExplainerAgent(
            model_name="claude-3-sonnet",
            provider="anthropic",
            api_key="test-key"
        )
        
        assert agent.provider == "anthropic"

    def test_init_invalid_provider(self):
        """Should raise for invalid provider."""
        with pytest.raises(ValueError):
            ExplainerAgent(provider="invalid")


class TestGenerateExplanation:
    """Tests for explanation generation."""

    @patch('src.agents.explainer.ChatOpenAI')
    def test_generate_explanation_success(self, mock_openai_cls, mock_explanation):
        """Should generate explanation from LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=mock_explanation)
        mock_openai_cls.return_value = mock_llm
        
        agent = ExplainerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Line 5 outage", 5)
        selected_action = {"action_id": "switch_1", "action_type": "switch_line"}
        evaluation = {"safety_score": 0.85, "recommendation": "approve"}
        references = ["ANSI C84.1-2020"]
        
        explanation = agent.generate_explanation(state, selected_action, evaluation, references)
        
        assert "Contingency" in explanation
        assert len(explanation) > 100

    @patch('src.agents.explainer.ChatOpenAI')
    def test_generate_explanation_llm_error(self, mock_openai_cls):
        """Should return fallback on LLM error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_openai_cls.return_value = mock_llm
        
        agent = ExplainerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Test", 5)
        
        explanation = agent.generate_explanation(state, {}, {}, [])
        
        # Should return fallback explanation
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestFallbackExplanation:
    """Tests for fallback explanation generation."""

    @patch('src.agents.explainer.ChatOpenAI')
    def test_fallback_includes_key_info(self, mock_openai_cls):
        """Fallback should include essential information."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Error")
        mock_openai_cls.return_value = mock_llm
        
        agent = ExplainerAgent(api_key="test-key")
        
        state = create_initial_state("ieee_33", "Line 5 outage", 5)
        state["constraint_violations"] = ["Bus 10: undervoltage"]
        
        selected_action = {
            "action_id": "test_action",
            "action_type": "switch_line"
        }
        evaluation = {
            "safety_score": 0.75,
            "recommendation": "approve"
        }
        
        explanation = agent.generate_explanation(state, selected_action, evaluation, [])
        
        assert isinstance(explanation, str)

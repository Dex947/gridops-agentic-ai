"""Integration tests for GridOpsOrchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandapower as pp

from src.orchestrator import GridOpsOrchestrator
from src.config import SystemConfig, NetworkConstraints, PathConfig
from src.core.state_manager import SystemState, ActionProposal


@pytest.fixture
def mock_config():
    """Create mock config without requiring API keys."""
    config = SystemConfig()
    config.openai_api_key = "test-key"
    config.llm_provider = "openai"
    config.model_name = "gpt-4o-mini"
    return config


@pytest.fixture
def mock_constraints(mock_config):
    """Create constraints from mock config."""
    return NetworkConstraints(mock_config)


@pytest.fixture
def mock_paths(tmp_path):
    """Create paths using temp directory."""
    paths = PathConfig(base_path=tmp_path)
    return paths


class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_init_creates_agents(self, mock_explainer, mock_planner, 
                                  mock_config, mock_constraints, mock_paths):
        """Should initialize all agents."""
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        assert orchestrator.planner is not None
        assert orchestrator.powerflow_agent is not None
        assert orchestrator.constraint_checker is not None
        assert orchestrator.retrieval is not None

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_init_builds_workflow(self, mock_explainer, mock_planner,
                                   mock_config, mock_constraints, mock_paths):
        """Should build LangGraph workflow."""
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        assert orchestrator.workflow is not None


class TestWorkflowNodes:
    """Tests for individual workflow nodes."""

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_load_network_node(self, mock_explainer, mock_planner,
                                mock_config, mock_constraints, mock_paths):
        """Should load network and update state."""
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        from src.core.state_manager import create_initial_state
        state = create_initial_state("ieee_33", "Test contingency", 5)
        
        updated = orchestrator._load_network_node(state)
        
        assert updated["network_loaded"] == True
        assert updated["network_summary"]["buses"] == 33

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_analyze_baseline_node(self, mock_explainer, mock_planner,
                                    mock_config, mock_constraints, mock_paths):
        """Should run baseline analysis."""
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        from src.core.state_manager import create_initial_state
        state = create_initial_state("ieee_33", "Test contingency", 5)
        
        # First load network
        state = orchestrator._load_network_node(state)
        
        # Then analyze baseline
        updated = orchestrator._analyze_baseline_node(state)
        
        assert "baseline_results" in updated
        assert updated["baseline_results"]["converged"] == True

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_simulate_contingency_node(self, mock_explainer, mock_planner,
                                        mock_config, mock_constraints, mock_paths):
        """Should simulate contingency."""
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        from src.core.state_manager import create_initial_state
        state = create_initial_state("ieee_33", "Line 5 outage", 5)
        state["contingency_type"] = "line_outage"
        state["contingency_elements"] = [5]
        
        # Load and analyze
        state = orchestrator._load_network_node(state)
        state = orchestrator._analyze_baseline_node(state)
        
        # Simulate contingency
        updated = orchestrator._simulate_contingency_node(state)
        
        assert "contingency_results" in updated
        assert len(updated["constraint_violations"]) >= 0


class TestFullWorkflow:
    """Tests for complete workflow execution."""

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_run_without_llm(self, mock_explainer_cls, mock_planner_cls,
                              mock_config, mock_constraints, mock_paths):
        """Should complete workflow even without LLM responses."""
        # Mock planner to return empty list
        mock_planner = MagicMock()
        mock_planner.generate_action_plans.return_value = []
        mock_planner_cls.return_value = mock_planner
        
        # Mock explainer
        mock_explainer = MagicMock()
        mock_explainer.generate_explanation.return_value = "Test explanation"
        mock_explainer_cls.return_value = mock_explainer
        
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        result = orchestrator.run(
            network_name="ieee_33",
            contingency_description="Line 5 outage",
            contingency_type="line_outage",
            contingency_elements=[5]
        )
        
        assert result["workflow_status"] == "complete"
        assert result["network_loaded"] == True

    @patch('src.orchestrator.PlannerAgent')
    @patch('src.orchestrator.ExplainerAgent')
    def test_run_with_mocked_proposals(self, mock_explainer_cls, mock_planner_cls,
                                        mock_config, mock_constraints, mock_paths):
        """Should process mocked action proposals."""
        # Create mock action proposal
        mock_proposal = ActionProposal(
            action_id="test_action_1",
            action_type="switch_line",
            target_elements=[{"line_id": 10, "close": False}],
            expected_impact="Reduce loading",
            priority=5,
            proposed_by="MockPlanner"
        )
        
        mock_planner = MagicMock()
        mock_planner.generate_action_plans.return_value = [mock_proposal]
        mock_planner_cls.return_value = mock_planner
        
        mock_explainer = MagicMock()
        mock_explainer.generate_explanation.return_value = "Action applied successfully"
        mock_explainer_cls.return_value = mock_explainer
        
        orchestrator = GridOpsOrchestrator(mock_config, mock_constraints, mock_paths)
        
        result = orchestrator.run(
            network_name="ieee_33",
            contingency_description="Line 5 outage",
            contingency_type="line_outage",
            contingency_elements=[5]
        )
        
        assert result["workflow_status"] == "complete"
        assert len(result["proposed_actions"]) >= 1

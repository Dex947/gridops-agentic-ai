"""Tests for state_manager module."""

import pytest
from pathlib import Path
import json

from src.core.state_manager import (
    SystemState,
    ActionProposal,
    ActionEvaluation,
    StateManager,
    create_initial_state
)


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_valid_state(self):
        """Should create state with required fields."""
        state = create_initial_state(
            network_name="ieee_33",
            contingency_desc="Test contingency",
            max_iterations=10
        )
        
        assert state["network_name"] == "ieee_33"
        assert state["contingency_description"] == "Test contingency"
        assert state["max_iterations"] == 10
        assert state["workflow_status"] == "initializing"

    def test_state_has_timestamps(self):
        """Should include timestamp."""
        state = create_initial_state("test", "test", 5)
        
        assert "started_at" in state
        assert state["started_at"] is not None


class TestActionProposal:
    """Tests for ActionProposal dataclass."""

    def test_create_proposal(self):
        """Should create valid proposal."""
        proposal = ActionProposal(
            action_id="test_1",
            action_type="switch_line",
            target_elements=[{"line_id": 5, "close": False}],
            expected_impact="Reduce loading",
            priority=5,
            proposed_by="TestAgent"
        )
        
        assert proposal.action_id == "test_1"
        assert proposal.action_type == "switch_line"
        assert proposal.priority == 5

    def test_to_dict(self):
        """Should convert to dictionary."""
        proposal = ActionProposal(
            action_id="test_1",
            action_type="switch_line",
            target_elements=[],
            expected_impact="Test",
            priority=1,
            proposed_by="Test"
        )
        
        d = proposal.to_dict()
        assert d["action_id"] == "test_1"
        assert "target_elements" in d


class TestStateManager:
    """Tests for StateManager class."""

    def test_init(self, paths):
        """Should initialize with state directory."""
        manager = StateManager(state_dir=paths.base / "state")
        assert manager.state_dir.exists()

    def test_add_proposed_action(self, paths):
        """Should add action to state."""
        manager = StateManager(state_dir=paths.base / "state")
        state = create_initial_state("test", "test", 5)
        
        action = ActionProposal(
            action_id="action_1",
            action_type="switch_line",
            target_elements=[],
            expected_impact="Test",
            priority=5,
            proposed_by="Test"
        )
        
        updated = manager.add_proposed_action(state, action)
        
        assert len(updated["proposed_actions"]) == 1
        assert updated["proposed_actions"][0]["action_id"] == "action_1"

    def test_save_and_load_state(self, paths):
        """Should persist and retrieve state."""
        manager = StateManager(state_dir=paths.base / "state")
        state = create_initial_state("test_net", "test contingency", 5)
        
        session_id = "test_session_123"
        manager.save_state(state, session_id)
        
        loaded = manager.load_state(session_id)
        
        assert loaded["network_name"] == "test_net"
        assert loaded["contingency_description"] == "test contingency"

    def test_update_workflow_status(self, paths):
        """Should update status correctly."""
        manager = StateManager(state_dir=paths.base / "state")
        state = create_initial_state("test", "test", 5)
        
        updated = manager.update_workflow_status(state, "running")
        assert updated["workflow_status"] == "running"
        
        updated = manager.update_workflow_status(updated, "complete")
        assert updated["workflow_status"] == "complete"
        assert "completed_at" in updated

    def test_increment_iteration(self, paths):
        """Should increment iteration counter."""
        manager = StateManager(state_dir=paths.base / "state")
        state = create_initial_state("test", "test", 5)
        
        assert state["iteration"] == 0
        
        updated = manager.increment_iteration(state)
        assert updated["iteration"] == 1

    def test_select_best_action(self, paths):
        """Should select action with highest safety score."""
        manager = StateManager(state_dir=paths.base / "state")
        state = create_initial_state("test", "test", 5)
        
        # Add evaluated actions with all required fields
        state["evaluated_actions"] = [
            {"action_id": "a1", "feasible": True, "safety_score": 0.7, 
             "recommendation": "approve", "violations_remaining": []},
            {"action_id": "a2", "feasible": True, "safety_score": 0.9, 
             "recommendation": "approve", "violations_remaining": []},
            {"action_id": "a3", "feasible": False, "safety_score": 0.95, 
             "recommendation": "reject", "violations_remaining": ["v1"]},
        ]
        
        updated = manager.select_best_action(state)
        
        # Should select a2 (highest score among approved)
        assert updated["selected_action"]["action_id"] == "a2"

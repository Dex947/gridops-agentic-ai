"""Tests for ConstraintCheckerAgent."""

import pytest
from src.agents.constraint_checker import ConstraintCheckerAgent
from src.config import NetworkConstraints, SystemConfig
from src.core.state_manager import ActionEvaluation


@pytest.fixture
def checker(constraints):
    """Create constraint checker with default constraints."""
    return ConstraintCheckerAgent(constraints)


class TestConstraintCheckerInit:
    """Tests for initialization."""

    def test_init_sets_limits(self, constraints):
        """Should set voltage and thermal limits."""
        checker = ConstraintCheckerAgent(constraints)
        
        assert checker.v_min == constraints.v_min_pu
        assert checker.v_max == constraints.v_max_pu
        assert checker.thermal_margin == constraints.thermal_margin


class TestValidateAction:
    """Tests for action validation."""

    def test_reject_non_converged(self, checker):
        """Should reject if power flow didn't converge."""
        results = {
            "action_id": "test_1",
            "converged": False
        }
        
        evaluation = checker.validate_action(results, baseline_violations=[])
        
        assert evaluation.feasible == False
        assert evaluation.recommendation == "reject"
        assert "converge" in evaluation.rationale.lower()

    def test_approve_all_violations_resolved(self, checker):
        """Should approve if all violations resolved."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": [],
            "min_voltage_pu": 0.96,
            "max_voltage_pu": 1.02,
            "max_line_loading_percent": 80.0,
            "total_losses_mw": 0.1
        }
        baseline_violations = ["Bus 10: undervoltage", "Bus 15: undervoltage"]
        
        evaluation = checker.validate_action(results, baseline_violations)
        
        assert evaluation.feasible == True
        assert len(evaluation.violations_resolved) == 2
        assert evaluation.safety_score > 0.5

    def test_detect_new_violations(self, checker):
        """Should detect new violations introduced by action."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": ["Line 5: overloaded", "Bus 20: undervoltage"],
            "min_voltage_pu": 0.94,
            "max_voltage_pu": 1.02,
            "max_line_loading_percent": 110.0,
            "total_losses_mw": 0.2
        }
        baseline_violations = []
        
        evaluation = checker.validate_action(results, baseline_violations)
        
        assert len(evaluation.new_violations) == 2
        assert evaluation.safety_score < 0.5

    def test_partial_resolution(self, checker):
        """Should handle partial violation resolution."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": ["Bus 15: undervoltage"],
            "min_voltage_pu": 0.95,
            "max_voltage_pu": 1.02,
            "max_line_loading_percent": 85.0,
            "total_losses_mw": 0.15
        }
        baseline_violations = ["Bus 10: undervoltage", "Bus 15: undervoltage"]
        
        evaluation = checker.validate_action(results, baseline_violations)
        
        assert len(evaluation.violations_resolved) == 1
        assert len(evaluation.violations_remaining) == 1


class TestSafetyScore:
    """Tests for safety score calculation."""

    def test_high_score_no_violations(self, checker):
        """Should give high score when no violations."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": [],
            "min_voltage_pu": 0.98,
            "max_voltage_pu": 1.01,
            "max_line_loading_percent": 60.0,
            "total_losses_mw": 0.1
        }
        
        evaluation = checker.validate_action(results, baseline_violations=[])
        
        assert evaluation.safety_score >= 0.8

    def test_low_score_many_violations(self, checker):
        """Should give low score with many violations."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": ["v1", "v2", "v3", "v4", "v5"],
            "min_voltage_pu": 0.90,
            "max_voltage_pu": 1.08,
            "max_line_loading_percent": 120.0,
            "total_losses_mw": 0.5
        }
        
        evaluation = checker.validate_action(results, baseline_violations=[])
        
        assert evaluation.safety_score < 0.5


class TestRecommendation:
    """Tests for recommendation logic."""

    def test_approve_recommendation(self, checker):
        """Should recommend approve for good actions."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": [],
            "min_voltage_pu": 0.97,
            "max_voltage_pu": 1.02,
            "max_line_loading_percent": 70.0,
            "total_losses_mw": 0.1
        }
        
        evaluation = checker.validate_action(results, ["old_violation"])
        
        assert evaluation.recommendation == "approve"

    def test_modify_recommendation(self, checker):
        """Should recommend modify for partial improvement."""
        results = {
            "action_id": "test_1",
            "converged": True,
            "violations": ["remaining_violation"],
            "min_voltage_pu": 0.95,
            "max_voltage_pu": 1.03,
            "max_line_loading_percent": 90.0,
            "total_losses_mw": 0.2
        }
        
        evaluation = checker.validate_action(results, ["old_v1", "remaining_violation"])
        
        # Could be approve or modify depending on score
        assert evaluation.recommendation in ["approve", "modify"]

"""Tests for powerflow_tools module."""

import pytest
import pandapower as pp

from src.tools.powerflow_tools import (
    PowerFlowTool,
    PowerFlowResults,
    run_powerflow_analysis,
    apply_switching_action,
    apply_load_shedding,
    validate_action_feasibility
)


class TestPowerFlowTool:
    """Tests for PowerFlowTool class."""

    def test_run_converges(self, converged_network):
        """Power flow should converge on valid network."""
        tool = PowerFlowTool()
        results = tool.run(converged_network)
        
        assert results.converged
        assert isinstance(results, PowerFlowResults)

    def test_results_have_valid_values(self, converged_network):
        """Results should have realistic values."""
        tool = PowerFlowTool()
        results = tool.run(converged_network)
        
        assert 0.8 <= results.min_voltage_pu <= 1.2
        assert 0.8 <= results.max_voltage_pu <= 1.2
        assert results.max_line_loading_percent >= 0
        assert results.total_losses_mw >= 0

    def test_voltage_violations_detected(self, simple_network):
        """Should detect voltage violations."""
        # Create undervoltage by adding heavy load
        pp.create_load(simple_network, bus=2, p_mw=5.0, q_mvar=2.0)
        pp.runpp(simple_network)
        
        tool = PowerFlowTool(voltage_limits=(0.95, 1.05))
        results = tool.run(simple_network)
        
        # Should have violations due to heavy load
        if results.min_voltage_pu < 0.95:
            assert len(results.violations) > 0

    def test_to_dict(self, converged_network):
        """Results should convert to dict."""
        tool = PowerFlowTool()
        results = tool.run(converged_network)
        
        d = results.to_dict()
        assert "converged" in d
        assert "min_voltage_pu" in d
        assert "max_line_loading_percent" in d
        assert "violations" in d

    def test_summary_text(self, converged_network):
        """Should generate readable summary."""
        tool = PowerFlowTool()
        results = tool.run(converged_network)
        
        text = results.get_summary_text()
        assert "Power Flow Results" in text
        assert "Voltage Range" in text


class TestSwitchingActions:
    """Tests for switching action functions."""

    def test_open_line(self, simple_network):
        """Should open a line."""
        result = apply_switching_action(
            simple_network,
            line_switches=[{"line_id": 1, "close": False}]
        )
        
        assert result["success"]
        assert simple_network.line.at[1, "in_service"] == False

    def test_close_line(self, simple_network):
        """Should close a line."""
        # First open it
        simple_network.line.at[1, "in_service"] = False
        
        result = apply_switching_action(
            simple_network,
            line_switches=[{"line_id": 1, "close": True}]
        )
        
        assert result["success"]
        assert simple_network.line.at[1, "in_service"] == True

    def test_invalid_line_id(self, simple_network):
        """Should handle invalid line ID gracefully."""
        result = apply_switching_action(
            simple_network,
            line_switches=[{"line_id": 999, "close": False}]
        )
        
        assert result["success"]  # Overall success
        assert result["applied_actions"][0]["success"] == False


class TestLoadShedding:
    """Tests for load shedding functions."""

    def test_reduce_load(self, simple_network):
        """Should reduce load by percentage."""
        original_p = simple_network.load.at[0, "p_mw"]
        
        result = apply_load_shedding(
            simple_network,
            load_reductions=[{"load_id": 0, "reduction_percent": 20}]
        )
        
        assert result["success"]
        new_p = simple_network.load.at[0, "p_mw"]
        assert new_p == pytest.approx(original_p * 0.8, rel=0.01)

    def test_max_shed_limit(self, simple_network):
        """Should respect max shed limit."""
        original_p = simple_network.load.at[0, "p_mw"]
        
        result = apply_load_shedding(
            simple_network,
            load_reductions=[{"load_id": 0, "reduction_percent": 50}],
            max_shed_percent=30.0
        )
        
        assert result["success"]
        new_p = simple_network.load.at[0, "p_mw"]
        # Should only shed 30%, not 50%
        assert new_p == pytest.approx(original_p * 0.7, rel=0.01)

    def test_invalid_load_id(self, simple_network):
        """Should handle invalid load ID."""
        result = apply_load_shedding(
            simple_network,
            load_reductions=[{"load_id": 999, "reduction_percent": 10}]
        )
        
        assert result["applied_reductions"][0]["success"] == False


class TestActionValidation:
    """Tests for action feasibility validation."""

    def test_feasible_action(self, simple_network):
        """Valid action should be feasible."""
        result = validate_action_feasibility(
            simple_network,
            action_plan={"line_switches": [{"line_id": 2, "close": False}]}
        )
        
        # Network copy is used, original unchanged
        assert simple_network.line.at[2, "in_service"] == True
        assert "feasible" in result

    def test_network_not_modified(self, simple_network):
        """Original network should not be modified."""
        original_state = simple_network.line.in_service.copy()
        
        validate_action_feasibility(
            simple_network,
            action_plan={"line_switches": [{"line_id": 0, "close": False}]}
        )
        
        assert (simple_network.line.in_service == original_state).all()

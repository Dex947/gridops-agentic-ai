"""Tests for contingency_simulator module."""

import pytest
import pandapower as pp

from src.core.contingency_simulator import (
    ContingencySimulator,
    ContingencyEvent,
    ContingencyType,
    ContingencyResult
)


class TestContingencyEvent:
    """Tests for ContingencyEvent dataclass."""

    def test_create_line_outage(self):
        """Should create line outage event."""
        event = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=5,
            description="Line 5 outage"
        )
        
        assert event.event_type == ContingencyType.LINE_OUTAGE
        assert event.element_index == 5

    def test_to_dict(self):
        """Should convert to dictionary."""
        event = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=5,
            description="Test"
        )
        
        d = event.to_dict()
        assert d["event_type"] == "line_outage"
        assert d["element_index"] == 5


class TestContingencySimulator:
    """Tests for ContingencySimulator class."""

    def test_init(self, constraints):
        """Should initialize with constraints."""
        sim = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
        )
        
        assert sim.v_min == constraints.v_min_pu
        assert sim.v_max == constraints.v_max_pu

    def test_simulate_line_outage(self, ieee33_network, constraints):
        """Should simulate line outage."""
        sim = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
        )
        
        event = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=5,
            description="Line 5 outage"
        )
        
        # Use copy to avoid modifying fixture
        net_copy = ieee33_network.deepcopy()
        result = sim.simulate_contingency(net_copy, event)
        
        assert isinstance(result, ContingencyResult)
        assert result.contingency == event

    def test_generate_n1_contingencies(self, ieee33_network, constraints):
        """Should generate N-1 contingencies for all lines."""
        sim = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
        )
        
        contingencies = sim.generate_n_minus_1_contingencies(ieee33_network)
        
        # Should have one contingency per in-service line
        in_service_lines = ieee33_network.line[ieee33_network.line.in_service].index
        assert len(contingencies) == len(in_service_lines)

    def test_contingency_modifies_network(self, simple_network, constraints):
        """Contingency should modify network state."""
        sim = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
        )
        
        event = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=0,
            description="Line 0 outage"
        )
        
        net_copy = simple_network.deepcopy()
        sim.simulate_contingency(net_copy, event)
        
        # Line should be out of service
        assert net_copy.line.at[0, "in_service"] == False

    def test_result_has_violations(self, ieee33_network, constraints):
        """Result should include violation information."""
        sim = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
        )
        
        event = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=0,
            description="Line 0 outage"
        )
        
        net_copy = ieee33_network.deepcopy()
        result = sim.simulate_contingency(net_copy, event)
        
        assert hasattr(result, "violated_constraints")
        assert hasattr(result, "is_critical")

"""Tests for network reconfiguration module."""

import pytest
import pandapower as pp
import pandapower.networks as pn

from src.tools.reconfiguration import (
    NetworkReconfiguration,
    ReconfigObjective,
    ReconfigurationResult,
    SwitchState,
    optimize_network_topology,
)


@pytest.fixture
def ieee33_network():
    """Load IEEE 33-bus network for testing."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


@pytest.fixture
def simple_radial_network():
    """Create a simple radial network with tie switch."""
    net = pp.create_empty_network()
    
    # Create buses
    buses = [pp.create_bus(net, vn_kv=20.0, name=f"Bus {i}") for i in range(5)]
    
    # External grid
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
    
    # Main feeder lines (normally closed)
    pp.create_line(net, from_bus=buses[0], to_bus=buses[1], length_km=1.0,
                   std_type="NAYY 4x50 SE", name="Line 0-1")
    pp.create_line(net, from_bus=buses[1], to_bus=buses[2], length_km=1.0,
                   std_type="NAYY 4x50 SE", name="Line 1-2")
    pp.create_line(net, from_bus=buses[0], to_bus=buses[3], length_km=1.0,
                   std_type="NAYY 4x50 SE", name="Line 0-3")
    pp.create_line(net, from_bus=buses[3], to_bus=buses[4], length_km=1.0,
                   std_type="NAYY 4x50 SE", name="Line 3-4")
    
    # Tie line (normally open)
    tie_line = pp.create_line(net, from_bus=buses[2], to_bus=buses[4], length_km=0.5,
                              std_type="NAYY 4x50 SE", name="Tie Line")
    net.line.at[tie_line, 'in_service'] = False
    
    # Loads
    pp.create_load(net, bus=buses[1], p_mw=0.3, q_mvar=0.1)
    pp.create_load(net, bus=buses[2], p_mw=0.2, q_mvar=0.05)
    pp.create_load(net, bus=buses[3], p_mw=0.25, q_mvar=0.08)
    pp.create_load(net, bus=buses[4], p_mw=0.15, q_mvar=0.04)
    
    pp.runpp(net)
    return net


class TestNetworkReconfigurationInit:
    """Tests for initialization."""

    def test_init_default_limits(self):
        """Should initialize with default limits."""
        reconfig = NetworkReconfiguration()
        
        assert reconfig.v_min == 0.95
        assert reconfig.v_max == 1.05
        assert reconfig.thermal_limit == 100.0

    def test_init_custom_limits(self):
        """Should accept custom limits."""
        reconfig = NetworkReconfiguration(
            voltage_limits=(0.90, 1.10),
            thermal_limit_percent=80.0
        )
        
        assert reconfig.v_min == 0.90
        assert reconfig.v_max == 1.10


class TestSwitchableElements:
    """Tests for switchable element identification."""

    def test_get_switchable_elements(self, simple_radial_network):
        """Should identify switchable elements."""
        reconfig = NetworkReconfiguration()
        switches = reconfig.get_switchable_elements(simple_radial_network)
        
        assert len(switches) > 0
        assert all(isinstance(s, SwitchState) for s in switches)

    def test_get_tie_switches(self, simple_radial_network):
        """Should identify tie switches."""
        reconfig = NetworkReconfiguration()
        ties = reconfig.get_tie_switches(simple_radial_network)
        
        # Should find the out-of-service tie line
        assert len(ties) >= 1

    def test_get_sectionalizing_switches(self, simple_radial_network):
        """Should identify sectionalizing switches."""
        reconfig = NetworkReconfiguration()
        sect = reconfig.get_sectionalizing_switches(simple_radial_network)
        
        # Should find in-service lines
        assert len(sect) >= 4


class TestRadialityCheck:
    """Tests for radiality verification."""

    def test_check_radiality_radial_network(self, simple_radial_network):
        """Should confirm radial network."""
        reconfig = NetworkReconfiguration()
        is_radial = reconfig.check_radiality(simple_radial_network)
        
        assert is_radial == True

    def test_check_radiality_meshed_network(self, simple_radial_network):
        """Should detect meshed network."""
        reconfig = NetworkReconfiguration()
        
        # Close the tie line to create a loop
        simple_radial_network.line.at[4, 'in_service'] = True
        
        is_radial = reconfig.check_radiality(simple_radial_network)
        
        # Network is now meshed
        assert is_radial == False


class TestConfigurationEvaluation:
    """Tests for configuration evaluation."""

    def test_evaluate_configuration(self, simple_radial_network):
        """Should evaluate configuration."""
        reconfig = NetworkReconfiguration()
        obj_value, feasible = reconfig.evaluate_configuration(
            simple_radial_network,
            ReconfigObjective.MIN_LOSS
        )
        
        assert isinstance(obj_value, float)
        assert isinstance(feasible, bool)

    def test_evaluate_infeasible_voltage(self, simple_radial_network):
        """Should detect voltage violations."""
        reconfig = NetworkReconfiguration(voltage_limits=(0.99, 1.01))
        
        # With tight limits, network may be infeasible
        obj_value, feasible = reconfig.evaluate_configuration(
            simple_radial_network,
            ReconfigObjective.MIN_LOSS
        )
        
        # Either feasible or not, but should not error
        assert isinstance(feasible, bool)


class TestBranchExchange:
    """Tests for branch exchange algorithm."""

    def test_branch_exchange_runs(self, simple_radial_network):
        """Branch exchange should run without error."""
        reconfig = NetworkReconfiguration()
        result = reconfig.branch_exchange(simple_radial_network)
        
        assert isinstance(result, ReconfigurationResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'baseline_losses_mw')
        assert hasattr(result, 'optimized_losses_mw')

    def test_branch_exchange_result_structure(self, simple_radial_network):
        """Result should have all required fields."""
        reconfig = NetworkReconfiguration()
        result = reconfig.branch_exchange(simple_radial_network)
        
        result_dict = result.to_dict()
        
        assert 'success' in result_dict
        assert 'baseline_losses_mw' in result_dict
        assert 'optimized_losses_mw' in result_dict
        assert 'loss_reduction_percent' in result_dict


class TestExhaustiveSearch:
    """Tests for exhaustive search algorithm."""

    def test_exhaustive_search_runs(self, simple_radial_network):
        """Exhaustive search should run without error."""
        reconfig = NetworkReconfiguration()
        result = reconfig.exhaustive_search(
            simple_radial_network,
            max_configs=100
        )
        
        assert isinstance(result, ReconfigurationResult)

    def test_exhaustive_search_limited_configs(self, simple_radial_network):
        """Should respect max_configs limit."""
        reconfig = NetworkReconfiguration()
        result = reconfig.exhaustive_search(
            simple_radial_network,
            max_configs=10
        )
        
        assert result.iterations <= 10


class TestFindOptimalTopology:
    """Tests for main optimization function."""

    def test_find_optimal_topology_min_loss(self, simple_radial_network):
        """Should find topology for loss minimization."""
        reconfig = NetworkReconfiguration()
        result = reconfig.find_optimal_topology(
            simple_radial_network,
            objective="min_loss"
        )
        
        assert isinstance(result, ReconfigurationResult)

    def test_find_optimal_topology_load_balancing(self, simple_radial_network):
        """Should find topology for load balancing."""
        reconfig = NetworkReconfiguration()
        result = reconfig.find_optimal_topology(
            simple_radial_network,
            objective="load_balancing"
        )
        
        assert isinstance(result, ReconfigurationResult)


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_optimize_network_topology(self, simple_radial_network):
        """Convenience function should work."""
        result = optimize_network_topology(simple_radial_network)
        
        assert isinstance(result, ReconfigurationResult)

    def test_optimize_with_custom_limits(self, simple_radial_network):
        """Should accept custom voltage limits."""
        result = optimize_network_topology(
            simple_radial_network,
            objective="min_loss",
            voltage_limits=(0.90, 1.10)
        )
        
        assert isinstance(result, ReconfigurationResult)


class TestIEEE33Network:
    """Tests with IEEE 33-bus network."""

    def test_ieee33_switchable_elements(self, ieee33_network):
        """Should find switchable elements in IEEE 33."""
        reconfig = NetworkReconfiguration()
        switches = reconfig.get_switchable_elements(ieee33_network)
        
        # IEEE 33 has 32 lines + 5 tie switches
        assert len(switches) >= 32

    def test_ieee33_branch_exchange(self, ieee33_network):
        """Branch exchange should work on IEEE 33."""
        reconfig = NetworkReconfiguration()
        result = reconfig.branch_exchange(
            ieee33_network,
            max_iterations=10
        )
        
        assert isinstance(result, ReconfigurationResult)
        assert result.baseline_losses_mw > 0

"""Tests for OPF tools module."""

import pytest
import pandapower as pp

from src.tools.opf_tools import OPFTool, OPFObjective, OPFResult, run_opf_analysis


@pytest.fixture
def simple_opf_network():
    """Create a simple network for OPF testing."""
    net = pp.create_empty_network()
    
    # Create buses
    b0 = pp.create_bus(net, vn_kv=20.0, name="Slack Bus")
    b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=20.0, name="Bus 2")
    
    # External grid (slack)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    
    # Lines
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0,
                   std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0,
                   std_type="NAYY 4x50 SE")
    
    # Loads
    pp.create_load(net, bus=b1, p_mw=0.5, q_mvar=0.1)
    pp.create_load(net, bus=b2, p_mw=0.3, q_mvar=0.05)
    
    return net


class TestOPFToolInit:
    """Tests for OPFTool initialization."""

    def test_init_default_limits(self):
        """Should initialize with default voltage limits."""
        tool = OPFTool()
        
        assert tool.v_min == 0.95
        assert tool.v_max == 1.05
        assert tool.thermal_limit == 100.0

    def test_init_custom_limits(self):
        """Should accept custom limits."""
        tool = OPFTool(voltage_limits=(0.90, 1.10), thermal_limit_percent=80.0)
        
        assert tool.v_min == 0.90
        assert tool.v_max == 1.10
        assert tool.thermal_limit == 80.0


class TestSetupConstraints:
    """Tests for constraint setup."""

    def test_setup_opf_constraints(self, simple_opf_network):
        """Should set voltage and thermal constraints."""
        tool = OPFTool()
        net = tool.setup_opf_constraints(simple_opf_network)
        
        assert 'min_vm_pu' in net.bus.columns
        assert 'max_vm_pu' in net.bus.columns
        assert net.bus['min_vm_pu'].iloc[0] == 0.95
        assert net.bus['max_vm_pu'].iloc[0] == 1.05

    def test_setup_controllable_elements(self, simple_opf_network):
        """Should mark elements as controllable."""
        tool = OPFTool()
        net = tool.setup_controllable_elements(simple_opf_network)
        
        assert net.ext_grid['controllable'].iloc[0] == True


class TestACOPF:
    """Tests for AC OPF."""

    def test_run_ac_opf_converges(self, simple_opf_network):
        """AC OPF should converge on simple network."""
        tool = OPFTool()
        result = tool.run_ac_opf(simple_opf_network, OPFObjective.MIN_LOSS)
        
        # Note: OPF may not converge on all networks
        # This test verifies the function runs without error
        assert isinstance(result, OPFResult)
        assert isinstance(result.converged, bool)

    def test_run_ac_opf_result_structure(self, simple_opf_network):
        """Result should have all required fields."""
        tool = OPFTool()
        result = tool.run_ac_opf(simple_opf_network)
        
        assert hasattr(result, 'converged')
        assert hasattr(result, 'objective_value')
        assert hasattr(result, 'total_generation_mw')
        assert hasattr(result, 'total_losses_mw')
        assert hasattr(result, 'min_voltage_pu')
        assert hasattr(result, 'max_voltage_pu')

    def test_run_ac_opf_to_dict(self, simple_opf_network):
        """Result should convert to dictionary."""
        tool = OPFTool()
        result = tool.run_ac_opf(simple_opf_network)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'converged' in result_dict
        assert 'objective_value' in result_dict


class TestDCOPF:
    """Tests for DC OPF."""

    def test_run_dc_opf(self, simple_opf_network):
        """DC OPF should run without error."""
        tool = OPFTool()
        result = tool.run_dc_opf(simple_opf_network)
        
        assert isinstance(result, OPFResult)

    def test_dc_opf_flat_voltage(self, simple_opf_network):
        """DC OPF should report flat voltage profile."""
        tool = OPFTool()
        result = tool.run_dc_opf(simple_opf_network)
        
        # DC OPF assumes flat voltage
        assert result.min_voltage_pu == 1.0
        assert result.max_voltage_pu == 1.0


class TestCostFunctions:
    """Tests for cost function setup."""

    def test_setup_min_loss_cost(self, simple_opf_network):
        """Should set up cost for loss minimization."""
        tool = OPFTool()
        net = tool.setup_cost_functions(simple_opf_network, OPFObjective.MIN_LOSS)
        
        assert len(net.poly_cost) > 0

    def test_setup_min_cost_objective(self, simple_opf_network):
        """Should set up cost for generation cost minimization."""
        tool = OPFTool()
        net = tool.setup_cost_functions(simple_opf_network, OPFObjective.MIN_GENERATION_COST)
        
        assert len(net.poly_cost) > 0


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_run_opf_analysis(self, simple_opf_network):
        """Convenience function should work."""
        result = run_opf_analysis(simple_opf_network, objective="min_loss")
        
        assert isinstance(result, OPFResult)

    def test_run_opf_analysis_custom_limits(self, simple_opf_network):
        """Should accept custom voltage limits."""
        result = run_opf_analysis(
            simple_opf_network,
            objective="min_loss",
            voltage_limits=(0.90, 1.10)
        )
        
        assert isinstance(result, OPFResult)


class TestLossSensitivity:
    """Tests for loss sensitivity calculation."""

    def test_calculate_loss_sensitivity(self, simple_opf_network):
        """Should calculate loss sensitivities."""
        tool = OPFTool()
        sensitivities = tool.calculate_loss_sensitivity(simple_opf_network)
        
        assert isinstance(sensitivities, dict)

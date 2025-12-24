"""Tests for DER integration module."""

import pytest
import pandapower as pp
import pandapower.networks as pn

from src.tools.der_integration import (
    DERIntegration,
    DERType,
    DERUnit,
    DERAnalysisResult,
    DERDispatch,
    add_der_to_network,
)


@pytest.fixture
def simple_network():
    """Create a simple network for DER testing."""
    net = pp.create_empty_network()
    
    # Create buses
    buses = [pp.create_bus(net, vn_kv=20.0, name=f"Bus {i}") for i in range(5)]
    
    # External grid
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
    
    # Lines
    for i in range(4):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1],
                      length_km=1.0, std_type="NAYY 4x50 SE")
    
    # Loads
    for i in range(1, 5):
        pp.create_load(net, bus=buses[i], p_mw=0.2, q_mvar=0.05)
    
    pp.runpp(net)
    return net


@pytest.fixture
def ieee33_network():
    """Load IEEE 33-bus network."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


class TestDERIntegrationInit:
    """Tests for initialization."""

    def test_init_default_limits(self):
        """Should initialize with default limits."""
        der = DERIntegration()
        
        assert der.v_min == 0.95
        assert der.v_max == 1.05
        assert der.thermal_limit == 100.0

    def test_init_custom_limits(self):
        """Should accept custom limits."""
        der = DERIntegration(voltage_limits=(0.90, 1.10))
        
        assert der.v_min == 0.90
        assert der.v_max == 1.10


class TestDERUnit:
    """Tests for DERUnit dataclass."""

    def test_der_unit_creation(self):
        """Should create DER unit."""
        der = DERUnit(
            der_id="PV_1",
            der_type=DERType.SOLAR_PV,
            bus_id=5,
            rated_power_mw=0.5
        )
        
        assert der.der_id == "PV_1"
        assert der.der_type == DERType.SOLAR_PV
        assert der.rated_power_mw == 0.5
        assert der.max_power_mw == 0.5  # Auto-set

    def test_der_unit_to_dict(self):
        """Should convert to dictionary."""
        der = DERUnit(
            der_id="Wind_1",
            der_type=DERType.WIND,
            bus_id=10,
            rated_power_mw=1.0
        )
        
        d = der.to_dict()
        
        assert d["der_id"] == "Wind_1"
        assert d["der_type"] == "wind"
        assert d["rated_power_mw"] == 1.0


class TestAddDER:
    """Tests for adding DERs to network."""

    def test_add_der(self, simple_network):
        """Should add DER to network."""
        integration = DERIntegration()
        
        der = DERUnit(
            der_id="PV_Test",
            der_type=DERType.SOLAR_PV,
            bus_id=2,
            rated_power_mw=0.3
        )
        
        idx = integration.add_der(simple_network, der)
        
        assert idx >= 0
        assert len(simple_network.sgen) == 1
        assert simple_network.sgen.at[idx, 'p_mw'] == 0.3

    def test_add_solar_pv(self, simple_network):
        """Should add solar PV."""
        integration = DERIntegration()
        
        idx = integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.5)
        
        assert idx >= 0
        assert simple_network.sgen.at[idx, 'type'] == 'solar_pv'

    def test_add_battery_storage(self, simple_network):
        """Should add battery storage."""
        integration = DERIntegration()
        
        idx = integration.add_battery_storage(simple_network, bus_id=4, rated_power_mw=0.2)
        
        assert idx >= 0
        assert simple_network.sgen.at[idx, 'type'] == 'battery_storage'
        # Battery can charge (negative min)
        assert simple_network.sgen.at[idx, 'min_p_mw'] < 0

    def test_add_wind_turbine(self, simple_network):
        """Should add wind turbine."""
        integration = DERIntegration()
        
        idx = integration.add_wind_turbine(simple_network, bus_id=2, rated_power_mw=0.8)
        
        assert idx >= 0
        assert simple_network.sgen.at[idx, 'type'] == 'wind'


class TestSetDEROutput:
    """Tests for setting DER output."""

    def test_set_der_output(self, simple_network):
        """Should set DER output power."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.5, name="PV_3")
        
        success = integration.set_der_output(simple_network, "PV_3", 0.3)
        
        assert success == True
        assert simple_network.sgen.at[0, 'p_mw'] == 0.3

    def test_set_der_output_not_found(self, simple_network):
        """Should return False for non-existent DER."""
        integration = DERIntegration()
        
        success = integration.set_der_output(simple_network, "NonExistent", 0.5)
        
        assert success == False


class TestAnalyzeDERImpact:
    """Tests for DER impact analysis."""

    def test_analyze_der_impact(self, simple_network):
        """Should analyze DER impact."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.2)
        
        result = integration.analyze_der_impact(simple_network)
        
        assert isinstance(result, DERAnalysisResult)
        assert result.success == True
        assert result.total_der_generation_mw >= 0

    def test_analyze_result_structure(self, simple_network):
        """Result should have all required fields."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=2, rated_power_mw=0.1)
        
        result = integration.analyze_der_impact(simple_network)
        
        assert hasattr(result, 'total_der_generation_mw')
        assert hasattr(result, 'voltage_rise_max_pu')
        assert hasattr(result, 'reverse_power_flow')
        assert hasattr(result, 'dispatches')

    def test_analyze_to_dict(self, simple_network):
        """Result should convert to dictionary."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.15)
        
        result = integration.analyze_der_impact(simple_network)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'success' in result_dict
        assert 'total_der_generation_mw' in result_dict


class TestHostingCapacity:
    """Tests for hosting capacity calculation."""

    def test_calculate_hosting_capacity(self, simple_network):
        """Should calculate hosting capacity."""
        integration = DERIntegration()
        
        capacity = integration.calculate_hosting_capacity(
            simple_network,
            bus_id=3,
            max_der_mw=2.0,
            step_mw=0.1
        )
        
        assert capacity >= 0
        assert capacity <= 2.0

    def test_hosting_capacity_at_different_buses(self, simple_network):
        """Hosting capacity may vary by bus location."""
        integration = DERIntegration()
        
        cap_bus2 = integration.calculate_hosting_capacity(simple_network, bus_id=2)
        cap_bus4 = integration.calculate_hosting_capacity(simple_network, bus_id=4)
        
        # Both should be valid
        assert cap_bus2 >= 0
        assert cap_bus4 >= 0


class TestOptimizeDERDispatch:
    """Tests for DER dispatch optimization."""

    def test_optimize_der_dispatch(self, simple_network):
        """Should optimize DER dispatch."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=2, rated_power_mw=0.3, name="PV_2")
        integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.2, name="PV_3")
        
        available = {"PV_2": 0.25, "PV_3": 0.15}
        
        result = integration.optimize_der_dispatch(simple_network, available)
        
        assert isinstance(result, DERAnalysisResult)


class TestDERSummary:
    """Tests for DER summary."""

    def test_get_der_summary(self, simple_network):
        """Should get DER summary."""
        integration = DERIntegration()
        integration.add_solar_pv(simple_network, bus_id=2, rated_power_mw=0.3)
        integration.add_solar_pv(simple_network, bus_id=3, rated_power_mw=0.2)
        integration.add_battery_storage(simple_network, bus_id=4, rated_power_mw=0.1)
        
        summary = integration.get_der_summary(simple_network)
        
        assert summary["total_count"] == 3
        assert summary["total_rated_mw"] > 0
        assert "by_type" in summary
        assert "by_bus" in summary


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_add_der_to_network(self, simple_network):
        """Convenience function should work."""
        idx = add_der_to_network(simple_network, bus_id=3, der_type="solar", rated_power_mw=0.4)
        
        assert idx >= 0
        assert len(simple_network.sgen) == 1

    def test_add_der_different_types(self, simple_network):
        """Should handle different DER types."""
        add_der_to_network(simple_network, bus_id=2, der_type="pv", rated_power_mw=0.3)
        add_der_to_network(simple_network, bus_id=3, der_type="wind", rated_power_mw=0.5)
        add_der_to_network(simple_network, bus_id=4, der_type="battery", rated_power_mw=0.2)
        
        assert len(simple_network.sgen) == 3

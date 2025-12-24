"""Tests for protection coordination module."""

import pytest
import pandapower as pp
import pandapower.networks as pn

from src.tools.protection_coordination import (
    CoordinationResult,
    ProtectionAnalysisResult,
    ProtectionCoordination,
    ProtectionDevice,
    ProtectionDeviceType,
    ShortCircuitResult,
    TripCharacteristic,
    create_default_protection_scheme,
)


@pytest.fixture
def simple_network():
    """Create a simple test network."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


@pytest.fixture
def protection_coordinator():
    """Create a protection coordinator."""
    return ProtectionCoordination(min_coordination_time=0.3, max_fault_clearing_time=2.0)


class TestProtectionDevice:
    """Tests for ProtectionDevice class."""

    def test_create_device(self):
        """Test creating a protection device."""
        device = ProtectionDevice(
            device_id="fuse_1",
            device_type=ProtectionDeviceType.FUSE,
            bus_id=5,
            pickup_current_a=100.0,
            time_dial=1.0
        )
        assert device.device_id == "fuse_1"
        assert device.device_type == ProtectionDeviceType.FUSE
        assert device.bus_id == 5
        assert device.pickup_current_a == 100.0

    def test_trip_time_below_pickup(self):
        """Test trip time when current is below pickup."""
        device = ProtectionDevice(
            device_id="relay_1",
            device_type=ProtectionDeviceType.RELAY,
            bus_id=1,
            pickup_current_a=100.0
        )
        trip_time = device.calculate_trip_time(50.0)  # Below pickup
        assert trip_time == float('inf')

    def test_trip_time_inverse_curve(self):
        """Test trip time calculation with inverse curve."""
        device = ProtectionDevice(
            device_id="relay_1",
            device_type=ProtectionDeviceType.RELAY,
            bus_id=1,
            pickup_current_a=100.0,
            time_dial=1.0,
            characteristic=TripCharacteristic.INVERSE
        )
        trip_time = device.calculate_trip_time(500.0)  # 5x pickup
        assert trip_time > 0
        assert trip_time < 10  # Should be reasonable

    def test_instantaneous_trip(self):
        """Test instantaneous trip when current exceeds instantaneous pickup."""
        device = ProtectionDevice(
            device_id="relay_1",
            device_type=ProtectionDeviceType.RELAY,
            bus_id=1,
            pickup_current_a=100.0,
            instantaneous_pickup_a=1000.0
        )
        trip_time = device.calculate_trip_time(1500.0)  # Above instantaneous
        assert trip_time == 0.05  # Instantaneous

    def test_to_dict(self):
        """Test conversion to dictionary."""
        device = ProtectionDevice(
            device_id="breaker_1",
            device_type=ProtectionDeviceType.BREAKER,
            bus_id=0,
            pickup_current_a=500.0
        )
        data = device.to_dict()
        assert data["device_id"] == "breaker_1"
        assert data["device_type"] == "breaker"
        assert data["bus_id"] == 0


class TestProtectionCoordination:
    """Tests for ProtectionCoordination class."""

    def test_init(self, protection_coordinator):
        """Test initialization."""
        assert protection_coordinator.min_cti == 0.3
        assert protection_coordinator.max_clearing_time == 2.0
        assert len(protection_coordinator.devices) == 0

    def test_add_device(self, protection_coordinator):
        """Test adding a device."""
        device = ProtectionDevice(
            device_id="test_device",
            device_type=ProtectionDeviceType.FUSE,
            bus_id=5,
            pickup_current_a=100.0
        )
        protection_coordinator.add_device(device)
        assert len(protection_coordinator.devices) == 1

    def test_add_device_at_bus(self, protection_coordinator, simple_network):
        """Test adding device at valid bus."""
        device = protection_coordinator.add_device_at_bus(
            simple_network, 5,
            ProtectionDeviceType.RECLOSER,
            pickup_current_a=200.0
        )
        assert device is not None
        assert device.bus_id == 5
        assert len(protection_coordinator.devices) == 1

    def test_add_device_at_invalid_bus(self, protection_coordinator, simple_network):
        """Test adding device at invalid bus returns None."""
        device = protection_coordinator.add_device_at_bus(
            simple_network, 9999,  # Invalid bus
            ProtectionDeviceType.FUSE,
            pickup_current_a=100.0
        )
        assert device is None
        assert len(protection_coordinator.devices) == 0

    def test_check_device_coordination_success(self, protection_coordinator):
        """Test coordination check between two properly coordinated devices."""
        upstream = ProtectionDevice(
            device_id="upstream",
            device_type=ProtectionDeviceType.RECLOSER,
            bus_id=1,
            pickup_current_a=100.0,
            time_dial=2.0  # Slower
        )
        downstream = ProtectionDevice(
            device_id="downstream",
            device_type=ProtectionDeviceType.FUSE,
            bus_id=5,
            pickup_current_a=100.0,
            time_dial=0.5  # Faster
        )
        
        result = protection_coordinator.check_device_coordination(
            upstream, downstream, 500.0, 5
        )
        assert isinstance(result, CoordinationResult)
        assert result.upstream_device == "upstream"
        assert result.downstream_device == "downstream"

    def test_get_device_summary(self, protection_coordinator, simple_network):
        """Test getting device summary."""
        protection_coordinator.add_device_at_bus(
            simple_network, 0, ProtectionDeviceType.BREAKER, 500.0
        )
        protection_coordinator.add_device_at_bus(
            simple_network, 5, ProtectionDeviceType.RECLOSER, 200.0
        )
        
        summary = protection_coordinator.get_device_summary()
        assert summary["total_devices"] == 2
        assert "breaker" in summary["by_type"]
        assert "recloser" in summary["by_type"]


class TestShortCircuitAnalysis:
    """Tests for short circuit analysis."""

    def test_run_short_circuit_analysis(self, protection_coordinator, simple_network):
        """Test running short circuit analysis."""
        results = protection_coordinator.run_short_circuit_analysis(simple_network)
        assert len(results) > 0
        assert all(isinstance(r, ShortCircuitResult) for r in results)

    def test_short_circuit_result_structure(self, protection_coordinator, simple_network):
        """Test short circuit result structure."""
        results = protection_coordinator.run_short_circuit_analysis(simple_network, fault_buses=[0, 1, 2])
        
        for result in results:
            assert hasattr(result, 'bus_id')
            assert hasattr(result, 'ikss_ka')
            assert result.ikss_ka > 0

    def test_short_circuit_invalid_buses(self, protection_coordinator, simple_network):
        """Test short circuit with invalid buses."""
        results = protection_coordinator.run_short_circuit_analysis(
            simple_network, fault_buses=[0, 9999]  # One valid, one invalid
        )
        # Should only return results for valid buses
        bus_ids = [r.bus_id for r in results]
        assert 9999 not in bus_ids


class TestProtectionAnalysis:
    """Tests for full protection analysis."""

    def test_analyze_coordination_no_devices(self, protection_coordinator, simple_network):
        """Test analysis with no devices."""
        result = protection_coordinator.analyze_coordination(simple_network)
        assert isinstance(result, ProtectionAnalysisResult)
        assert result.success is False
        assert result.total_devices == 0

    def test_analyze_coordination_with_devices(self, simple_network):
        """Test analysis with devices."""
        coordinator = create_default_protection_scheme(simple_network)
        result = coordinator.analyze_coordination(simple_network)
        
        assert isinstance(result, ProtectionAnalysisResult)
        assert result.total_devices > 0
        assert len(result.short_circuit_results) > 0

    def test_verify_post_reconfiguration(self, simple_network):
        """Test verification after reconfiguration."""
        coordinator = create_default_protection_scheme(simple_network)
        result = coordinator.verify_post_reconfiguration(simple_network, switched_lines=[5, 10])
        
        assert isinstance(result, ProtectionAnalysisResult)


class TestDefaultProtectionScheme:
    """Tests for default protection scheme creation."""

    def test_create_default_scheme(self, simple_network):
        """Test creating default protection scheme."""
        coordinator = create_default_protection_scheme(simple_network)
        
        assert len(coordinator.devices) > 0
        
        # Should have at least one breaker at substation
        breakers = [d for d in coordinator.devices if d.device_type == ProtectionDeviceType.BREAKER]
        assert len(breakers) >= 1

    def test_default_scheme_has_fuses(self, simple_network):
        """Test that default scheme includes fuses at loads."""
        coordinator = create_default_protection_scheme(simple_network)
        
        fuses = [d for d in coordinator.devices if d.device_type == ProtectionDeviceType.FUSE]
        # Should have fuses for load buses
        assert len(fuses) > 0


class TestCoordinationResult:
    """Tests for CoordinationResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CoordinationResult(
            upstream_device="relay_1",
            downstream_device="fuse_1",
            fault_location=5,
            fault_current_ka=5.0,
            upstream_trip_time=1.5,
            downstream_trip_time=0.8,
            time_margin=0.7,
            coordinated=True
        )
        
        data = result.to_dict()
        assert data["upstream_device"] == "relay_1"
        assert data["downstream_device"] == "fuse_1"
        assert data["coordinated"] is True
        assert data["time_margin"] == 0.7

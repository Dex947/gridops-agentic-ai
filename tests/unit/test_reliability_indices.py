"""Tests for reliability indices module."""

from datetime import datetime, timedelta

import pytest
import pandapower as pp
import pandapower.networks as pn

from src.tools.reliability_indices import (
    CustomerData,
    OutageEvent,
    OutageType,
    ReliabilityAnalysisResult,
    ReliabilityCalculator,
    ReliabilityIndices,
    calculate_reliability_indices,
)


@pytest.fixture
def simple_network():
    """Create a simple test network."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


@pytest.fixture
def reliability_calculator():
    """Create a reliability calculator."""
    return ReliabilityCalculator(analysis_period_hours=8760)


class TestCustomerData:
    """Tests for CustomerData class."""

    def test_create_customer_data(self):
        """Test creating customer data."""
        customer = CustomerData(
            load_id=1,
            bus_id=5,
            num_customers=100,
            load_mw=0.5,
            priority=1
        )
        assert customer.load_id == 1
        assert customer.bus_id == 5
        assert customer.num_customers == 100
        assert customer.load_mw == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        customer = CustomerData(
            load_id=1,
            bus_id=5,
            num_customers=100,
            load_mw=0.5
        )
        data = customer.to_dict()
        assert data["load_id"] == 1
        assert data["num_customers"] == 100


class TestOutageEvent:
    """Tests for OutageEvent class."""

    def test_create_outage_event(self):
        """Test creating an outage event."""
        start = datetime.now()
        end = start + timedelta(minutes=30)
        
        event = OutageEvent(
            event_id="outage_001",
            outage_type=OutageType.UNPLANNED,
            start_time=start,
            end_time=end,
            affected_buses=[5, 6, 7],
            affected_customers=150,
            cause="Line failure"
        )
        
        assert event.event_id == "outage_001"
        assert event.outage_type == OutageType.UNPLANNED
        assert len(event.affected_buses) == 3
        assert event.affected_customers == 150

    def test_duration_minutes(self):
        """Test duration calculation in minutes."""
        start = datetime.now()
        end = start + timedelta(minutes=45)
        
        event = OutageEvent(
            event_id="test",
            outage_type=OutageType.UNPLANNED,
            start_time=start,
            end_time=end,
            affected_buses=[1]
        )
        
        assert event.duration_minutes == 45.0

    def test_duration_hours(self):
        """Test duration calculation in hours."""
        start = datetime.now()
        end = start + timedelta(hours=2)
        
        event = OutageEvent(
            event_id="test",
            outage_type=OutageType.UNPLANNED,
            start_time=start,
            end_time=end,
            affected_buses=[1]
        )
        
        assert event.duration_hours == 2.0

    def test_is_sustained(self):
        """Test sustained outage detection (>5 minutes)."""
        start = datetime.now()
        
        # Momentary outage (< 5 min)
        momentary = OutageEvent(
            event_id="momentary",
            outage_type=OutageType.MOMENTARY,
            start_time=start,
            end_time=start + timedelta(minutes=2),
            affected_buses=[1]
        )
        assert momentary.is_sustained is False
        
        # Sustained outage (> 5 min)
        sustained = OutageEvent(
            event_id="sustained",
            outage_type=OutageType.SUSTAINED,
            start_time=start,
            end_time=start + timedelta(minutes=30),
            affected_buses=[1]
        )
        assert sustained.is_sustained is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        start = datetime.now()
        event = OutageEvent(
            event_id="test",
            outage_type=OutageType.PLANNED,
            start_time=start,
            end_time=start + timedelta(hours=1),
            affected_buses=[1, 2]
        )
        
        data = event.to_dict()
        assert data["event_id"] == "test"
        assert data["outage_type"] == "planned"
        assert data["duration_minutes"] == 60.0


class TestReliabilityCalculator:
    """Tests for ReliabilityCalculator class."""

    def test_init(self, reliability_calculator):
        """Test initialization."""
        assert reliability_calculator.analysis_period_hours == 8760
        assert len(reliability_calculator.customer_data) == 0
        assert len(reliability_calculator.outage_events) == 0

    def test_set_customer_data_from_network(self, reliability_calculator, simple_network):
        """Test setting customer data from network."""
        reliability_calculator.set_customer_data_from_network(simple_network, customers_per_mw=500)
        
        assert len(reliability_calculator.customer_data) > 0
        total_customers = sum(c.num_customers for c in reliability_calculator.customer_data)
        assert total_customers > 0

    def test_add_customer_data(self, reliability_calculator):
        """Test adding customer data."""
        customer = CustomerData(
            load_id=1,
            bus_id=5,
            num_customers=100,
            load_mw=0.5
        )
        reliability_calculator.add_customer_data(customer)
        assert len(reliability_calculator.customer_data) == 1

    def test_add_outage_event(self, reliability_calculator):
        """Test adding outage event."""
        event = OutageEvent(
            event_id="test",
            outage_type=OutageType.UNPLANNED,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            affected_buses=[1]
        )
        reliability_calculator.add_outage_event(event)
        assert len(reliability_calculator.outage_events) == 1

    def test_record_contingency_outage(self, reliability_calculator, simple_network):
        """Test recording contingency outage."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        event = reliability_calculator.record_contingency_outage(
            simple_network,
            affected_buses=[5, 6, 7],
            duration_minutes=30,
            cause="Test contingency"
        )
        
        assert event is not None
        assert event.duration_minutes == 30
        assert len(reliability_calculator.outage_events) == 1

    def test_record_contingency_invalid_buses(self, reliability_calculator, simple_network):
        """Test recording contingency with invalid buses."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        event = reliability_calculator.record_contingency_outage(
            simple_network,
            affected_buses=[5, 9999],  # One valid, one invalid
            duration_minutes=30,
            cause="Test"
        )
        
        assert 9999 not in event.affected_buses
        assert 5 in event.affected_buses


class TestReliabilityIndicesCalculation:
    """Tests for reliability indices calculation."""

    def test_calculate_indices_no_data(self, reliability_calculator):
        """Test calculation with no customer data."""
        indices = reliability_calculator.calculate_indices()
        assert indices.total_customers == 0

    def test_calculate_indices_no_outages(self, reliability_calculator, simple_network):
        """Test calculation with no outages."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        indices = reliability_calculator.calculate_indices()
        
        assert indices.saidi == 0.0
        assert indices.saifi == 0.0
        assert indices.asai == 1.0  # 100% availability

    def test_calculate_indices_with_outages(self, reliability_calculator, simple_network):
        """Test calculation with outages."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        # Record some outages
        reliability_calculator.record_contingency_outage(
            simple_network, [5, 6, 7], duration_minutes=60, cause="Test 1"
        )
        reliability_calculator.record_contingency_outage(
            simple_network, [10, 11], duration_minutes=30, cause="Test 2"
        )
        
        indices = reliability_calculator.calculate_indices()
        
        assert indices.saidi > 0
        assert indices.saifi > 0
        assert indices.total_cmi > 0
        assert indices.asai < 1.0  # Less than 100% availability

    def test_caidi_calculation(self, reliability_calculator, simple_network):
        """Test CAIDI = SAIDI / SAIFI."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        reliability_calculator.record_contingency_outage(
            simple_network, [5, 6, 7], duration_minutes=60, cause="Test"
        )
        
        indices = reliability_calculator.calculate_indices()
        
        if indices.saifi > 0:
            expected_caidi = indices.saidi / indices.saifi
            assert abs(indices.caidi - expected_caidi) < 0.01

    def test_maifi_calculation(self, reliability_calculator, simple_network):
        """Test MAIFI for momentary interruptions."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        # Record momentary outage (< 5 min)
        reliability_calculator.record_contingency_outage(
            simple_network, [5, 6], duration_minutes=2, cause="Momentary"
        )
        
        indices = reliability_calculator.calculate_indices()
        
        # MAIFI should be > 0 for momentary outages
        assert indices.maifi >= 0


class TestReliabilityAnalysis:
    """Tests for full reliability analysis."""

    def test_analyze_reliability(self, reliability_calculator, simple_network):
        """Test full reliability analysis."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        reliability_calculator.record_contingency_outage(
            simple_network, [5, 6, 7], duration_minutes=60, cause="Test"
        )
        
        result = reliability_calculator.analyze_reliability(simple_network)
        
        assert isinstance(result, ReliabilityAnalysisResult)
        assert result.indices is not None
        assert len(result.customer_data) > 0

    def test_estimate_contingency_impact(self, reliability_calculator, simple_network):
        """Test contingency impact estimation."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        impact = reliability_calculator.estimate_contingency_impact(
            simple_network,
            contingency_buses=[5, 6, 7, 8],
            estimated_duration_minutes=60
        )
        
        assert "affected_customers" in impact
        assert "incremental_saidi" in impact
        assert "incremental_saifi" in impact
        assert impact["estimated_duration_minutes"] == 60

    def test_get_summary(self, reliability_calculator, simple_network):
        """Test getting summary."""
        reliability_calculator.set_customer_data_from_network(simple_network)
        
        summary = reliability_calculator.get_summary()
        
        assert "total_customers" in summary
        assert "load_points" in summary
        assert "outage_events" in summary


class TestReliabilityIndices:
    """Tests for ReliabilityIndices class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        indices = ReliabilityIndices(
            saidi=60.0,
            saifi=1.2,
            caidi=50.0,
            asai=0.9999,
            total_customers=10000
        )
        
        data = indices.to_dict()
        assert data["saidi"] == 60.0
        assert data["saifi"] == 1.2
        assert data["total_customers"] == 10000

    def test_get_summary_text(self):
        """Test getting summary text."""
        indices = ReliabilityIndices(
            saidi=60.0,
            saifi=1.2,
            caidi=50.0,
            asai=0.9999,
            total_customers=10000
        )
        
        text = indices.get_summary_text()
        assert "SAIDI" in text
        assert "SAIFI" in text
        assert "60.00" in text


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_calculate_reliability_indices(self, simple_network):
        """Test convenience function."""
        indices = calculate_reliability_indices(simple_network)
        
        assert isinstance(indices, ReliabilityIndices)
        assert indices.total_customers > 0

    def test_calculate_with_outage_events(self, simple_network):
        """Test convenience function with outage events."""
        start = datetime.now()
        outage_events = [
            {
                "event_id": "test_1",
                "outage_type": "unplanned",
                "start_time": start.isoformat(),
                "end_time": (start + timedelta(hours=1)).isoformat(),
                "affected_buses": [5, 6, 7],
                "affected_customers": 100,
                "cause": "Test"
            }
        ]
        
        indices = calculate_reliability_indices(simple_network, outage_events)
        
        assert indices.total_cmi > 0

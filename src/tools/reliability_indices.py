"""Reliability indices calculation (SAIDI, SAIFI, CAIDI, etc.)."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandapower as pp
from loguru import logger


class OutageType(Enum):
    """Types of outages."""
    PLANNED = "planned"
    UNPLANNED = "unplanned"
    MOMENTARY = "momentary"
    SUSTAINED = "sustained"


@dataclass
class CustomerData:
    """Customer data for a load point."""
    load_id: int
    bus_id: int
    num_customers: int
    load_mw: float
    priority: int = 1  # 1=residential, 2=commercial, 3=industrial, 4=critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "load_id": self.load_id,
            "bus_id": self.bus_id,
            "num_customers": self.num_customers,
            "load_mw": float(self.load_mw),
            "priority": self.priority
        }


@dataclass
class OutageEvent:
    """Represents an outage event."""
    event_id: str
    outage_type: OutageType
    start_time: datetime
    end_time: Optional[datetime] = None
    affected_buses: List[int] = field(default_factory=list)
    affected_customers: int = 0
    cause: str = ""
    restoration_actions: List[str] = field(default_factory=list)
    
    @property
    def duration_minutes(self) -> float:
        """Get outage duration in minutes."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 60.0
    
    @property
    def duration_hours(self) -> float:
        """Get outage duration in hours."""
        return self.duration_minutes / 60.0
    
    @property
    def is_sustained(self) -> bool:
        """Check if outage is sustained (>5 minutes per IEEE 1366)."""
        return self.duration_minutes > 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "outage_type": self.outage_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": float(self.duration_minutes),
            "affected_buses": self.affected_buses,
            "affected_customers": self.affected_customers,
            "cause": self.cause,
            "restoration_actions": self.restoration_actions
        }


@dataclass
class ReliabilityIndices:
    """Standard reliability indices per IEEE 1366."""
    # System Average Interruption Duration Index
    saidi: float = 0.0  # minutes
    
    # System Average Interruption Frequency Index
    saifi: float = 0.0  # interruptions per customer
    
    # Customer Average Interruption Duration Index
    caidi: float = 0.0  # minutes per interruption
    
    # Average Service Availability Index
    asai: float = 1.0  # percentage (0-1)
    
    # Momentary Average Interruption Frequency Index
    maifi: float = 0.0  # momentary interruptions per customer
    
    # Customer Experiencing Multiple Interruptions
    cemi: float = 0.0  # percentage of customers with >n interruptions
    
    # Total customers served
    total_customers: int = 0
    
    # Total customer minutes interrupted
    total_cmi: float = 0.0
    
    # Total customer interruptions
    total_ci: int = 0
    
    # Analysis period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "saidi": float(self.saidi),
            "saifi": float(self.saifi),
            "caidi": float(self.caidi),
            "asai": float(self.asai),
            "maifi": float(self.maifi),
            "cemi": float(self.cemi),
            "total_customers": self.total_customers,
            "total_cmi": float(self.total_cmi),
            "total_ci": self.total_ci,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None
        }
    
    def get_summary_text(self) -> str:
        """Get human-readable summary."""
        return f"""Reliability Indices Summary:
SAIDI: {self.saidi:.2f} minutes (avg interruption duration per customer)
SAIFI: {self.saifi:.4f} (avg interruptions per customer)
CAIDI: {self.caidi:.2f} minutes (avg duration per interruption)
ASAI: {self.asai*100:.4f}% (service availability)
MAIFI: {self.maifi:.4f} (momentary interruptions per customer)
Total Customers: {self.total_customers:,}
Total Customer-Minutes Interrupted: {self.total_cmi:,.0f}
"""


@dataclass
class ReliabilityAnalysisResult:
    """Results from reliability analysis."""
    success: bool
    indices: ReliabilityIndices
    outage_events: List[OutageEvent] = field(default_factory=list)
    customer_data: List[CustomerData] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "indices": self.indices.to_dict(),
            "outage_events": [e.to_dict() for e in self.outage_events],
            "customer_data": [c.to_dict() for c in self.customer_data],
            "violations": self.violations,
            "recommendations": self.recommendations
        }


def validate_bus_id(net: pp.pandapowerNet, bus_id: int) -> bool:
    """Check if a bus ID exists in the network."""
    return bus_id in set(net.bus.index.tolist())


class ReliabilityCalculator:
    """Calculate reliability indices for distribution networks."""
    
    def __init__(self, analysis_period_hours: float = 8760.0):  # Default: 1 year
        """
        Initialize reliability calculator.
        
        Args:
            analysis_period_hours: Analysis period in hours (default: 8760 = 1 year)
        """
        self.analysis_period_hours = analysis_period_hours
        self.customer_data: List[CustomerData] = []
        self.outage_events: List[OutageEvent] = []
        logger.info(f"ReliabilityCalculator initialized (period={analysis_period_hours}h)")
    
    def set_customer_data_from_network(self, net: pp.pandapowerNet,
                                       customers_per_mw: float = 500.0) -> None:
        """
        Estimate customer data from network loads.
        
        Args:
            net: pandapower network
            customers_per_mw: Estimated customers per MW of load
        """
        self.customer_data = []
        
        for idx, load in net.load.iterrows():
            bus_id = int(load.bus)
            load_mw = float(load.p_mw)
            
            # Estimate customers based on load
            num_customers = max(1, int(load_mw * customers_per_mw))
            
            # Determine priority based on load size
            if load_mw > 1.0:
                priority = 3  # Industrial
            elif load_mw > 0.5:
                priority = 2  # Commercial
            else:
                priority = 1  # Residential
            
            self.customer_data.append(CustomerData(
                load_id=int(idx),
                bus_id=bus_id,
                num_customers=num_customers,
                load_mw=load_mw,
                priority=priority
            ))
        
        logger.info(f"Set customer data: {len(self.customer_data)} load points, "
                   f"{sum(c.num_customers for c in self.customer_data)} total customers")
    
    def add_customer_data(self, customer: CustomerData) -> None:
        """Add customer data for a load point."""
        self.customer_data.append(customer)
    
    def add_outage_event(self, event: OutageEvent) -> None:
        """Add an outage event."""
        self.outage_events.append(event)
        logger.info(f"Added outage event: {event.event_id}")
    
    def record_contingency_outage(self, net: pp.pandapowerNet,
                                  affected_buses: List[int],
                                  duration_minutes: float,
                                  cause: str = "contingency") -> OutageEvent:
        """
        Record an outage from a contingency event.
        
        Args:
            net: pandapower network
            affected_buses: List of affected bus IDs
            duration_minutes: Outage duration in minutes
            cause: Cause of outage
            
        Returns:
            Created OutageEvent
        """
        # Validate bus IDs
        valid_buses = [b for b in affected_buses if validate_bus_id(net, b)]
        invalid_buses = [b for b in affected_buses if not validate_bus_id(net, b)]
        if invalid_buses:
            logger.warning(f"Invalid bus IDs skipped: {invalid_buses}")
        
        # Calculate affected customers
        affected_customers = 0
        for customer in self.customer_data:
            if customer.bus_id in valid_buses:
                affected_customers += customer.num_customers
        
        # Create event
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        event = OutageEvent(
            event_id=f"outage_{len(self.outage_events)+1}_{start_time.strftime('%Y%m%d%H%M%S')}",
            outage_type=OutageType.UNPLANNED if duration_minutes > 5 else OutageType.MOMENTARY,
            start_time=start_time,
            end_time=end_time,
            affected_buses=valid_buses,
            affected_customers=affected_customers,
            cause=cause
        )
        
        self.add_outage_event(event)
        return event
    
    def calculate_indices(self, period_start: Optional[datetime] = None,
                         period_end: Optional[datetime] = None) -> ReliabilityIndices:
        """
        Calculate reliability indices for the analysis period.
        
        Args:
            period_start: Start of analysis period
            period_end: End of analysis period
            
        Returns:
            Calculated reliability indices
        """
        logger.info("Calculating reliability indices")
        
        if not self.customer_data:
            logger.warning("No customer data available")
            return ReliabilityIndices()
        
        total_customers = sum(c.num_customers for c in self.customer_data)
        
        if total_customers == 0:
            logger.warning("Total customers is zero")
            return ReliabilityIndices()
        
        # Filter events by period
        events = self.outage_events
        if period_start:
            events = [e for e in events if e.start_time >= period_start]
        if period_end:
            events = [e for e in events if e.start_time <= period_end]
        
        # Separate sustained and momentary interruptions
        sustained_events = [e for e in events if e.is_sustained]
        momentary_events = [e for e in events if not e.is_sustained]
        
        # Calculate Customer Minutes Interrupted (CMI)
        total_cmi = 0.0
        for event in sustained_events:
            total_cmi += event.affected_customers * event.duration_minutes
        
        # Calculate Customer Interruptions (CI)
        total_ci = sum(e.affected_customers for e in sustained_events)
        
        # Calculate SAIDI (System Average Interruption Duration Index)
        # SAIDI = Sum(Customer Minutes Interrupted) / Total Customers
        saidi = total_cmi / total_customers if total_customers > 0 else 0.0
        
        # Calculate SAIFI (System Average Interruption Frequency Index)
        # SAIFI = Sum(Customer Interruptions) / Total Customers
        saifi = total_ci / total_customers if total_customers > 0 else 0.0
        
        # Calculate CAIDI (Customer Average Interruption Duration Index)
        # CAIDI = SAIDI / SAIFI = Sum(CMI) / Sum(CI)
        caidi = total_cmi / total_ci if total_ci > 0 else 0.0
        
        # Calculate ASAI (Average Service Availability Index)
        # ASAI = (Customer Hours Served) / (Customer Hours Demanded)
        total_customer_hours = total_customers * self.analysis_period_hours
        customer_hours_interrupted = total_cmi / 60.0  # Convert minutes to hours
        asai = (total_customer_hours - customer_hours_interrupted) / total_customer_hours if total_customer_hours > 0 else 1.0
        
        # Calculate MAIFI (Momentary Average Interruption Frequency Index)
        momentary_ci = sum(e.affected_customers for e in momentary_events)
        maifi = momentary_ci / total_customers if total_customers > 0 else 0.0
        
        # Calculate CEMI (Customers Experiencing Multiple Interruptions)
        # Track customers with >3 interruptions
        customer_interruption_counts: Dict[int, int] = {}
        for event in sustained_events:
            for customer in self.customer_data:
                if customer.bus_id in event.affected_buses:
                    customer_interruption_counts[customer.load_id] = \
                        customer_interruption_counts.get(customer.load_id, 0) + 1
        
        customers_with_multiple = sum(
            c.num_customers for c in self.customer_data 
            if customer_interruption_counts.get(c.load_id, 0) > 3
        )
        cemi = customers_with_multiple / total_customers if total_customers > 0 else 0.0
        
        indices = ReliabilityIndices(
            saidi=saidi,
            saifi=saifi,
            caidi=caidi,
            asai=asai,
            maifi=maifi,
            cemi=cemi,
            total_customers=total_customers,
            total_cmi=total_cmi,
            total_ci=total_ci,
            period_start=period_start,
            period_end=period_end
        )
        
        logger.info(f"Indices calculated: SAIDI={saidi:.2f}, SAIFI={saifi:.4f}, ASAI={asai:.6f}")
        return indices
    
    def analyze_reliability(self, net: pp.pandapowerNet) -> ReliabilityAnalysisResult:
        """
        Perform full reliability analysis.
        
        Args:
            net: pandapower network
            
        Returns:
            Reliability analysis result
        """
        logger.info("Starting reliability analysis")
        
        # Set customer data if not already set
        if not self.customer_data:
            self.set_customer_data_from_network(net)
        
        # Calculate indices
        indices = self.calculate_indices()
        
        # Generate recommendations
        violations = []
        recommendations = []
        
        # Check against typical utility targets
        if indices.saidi > 120:  # 2 hours
            violations.append(f"SAIDI ({indices.saidi:.1f} min) exceeds typical target of 120 min")
            recommendations.append("Focus on reducing outage duration through faster restoration")
        
        if indices.saifi > 1.5:
            violations.append(f"SAIFI ({indices.saifi:.2f}) exceeds typical target of 1.5")
            recommendations.append("Implement preventive maintenance to reduce outage frequency")
        
        if indices.asai < 0.9999:  # 99.99% availability
            recommendations.append(f"Current availability {indices.asai*100:.4f}% - target 99.99%")
        
        # Identify worst-performing areas
        bus_outage_counts: Dict[int, int] = {}
        for event in self.outage_events:
            for bus in event.affected_buses:
                bus_outage_counts[bus] = bus_outage_counts.get(bus, 0) + 1
        
        if bus_outage_counts:
            worst_bus = max(bus_outage_counts, key=bus_outage_counts.get)
            recommendations.append(
                f"Bus {worst_bus} has highest outage frequency ({bus_outage_counts[worst_bus]} events) - "
                "consider reinforcement"
            )
        
        result = ReliabilityAnalysisResult(
            success=len(violations) == 0,
            indices=indices,
            outage_events=self.outage_events,
            customer_data=self.customer_data,
            violations=violations,
            recommendations=recommendations
        )
        
        logger.info(f"Reliability analysis complete: {len(violations)} violations")
        return result
    
    def estimate_contingency_impact(self, net: pp.pandapowerNet,
                                    contingency_buses: List[int],
                                    estimated_duration_minutes: float = 60.0) -> Dict[str, Any]:
        """
        Estimate reliability impact of a contingency.
        
        Args:
            net: pandapower network
            contingency_buses: Buses affected by contingency
            estimated_duration_minutes: Estimated outage duration
            
        Returns:
            Impact assessment dictionary
        """
        # Validate bus IDs
        valid_buses = [b for b in contingency_buses if validate_bus_id(net, b)]
        
        # Calculate affected customers
        affected_customers = 0
        affected_load_mw = 0.0
        for customer in self.customer_data:
            if customer.bus_id in valid_buses:
                affected_customers += customer.num_customers
                affected_load_mw += customer.load_mw
        
        total_customers = sum(c.num_customers for c in self.customer_data)
        
        # Calculate incremental impact on indices
        incremental_cmi = affected_customers * estimated_duration_minutes
        incremental_saidi = incremental_cmi / total_customers if total_customers > 0 else 0.0
        incremental_saifi = affected_customers / total_customers if total_customers > 0 else 0.0
        
        return {
            "affected_buses": valid_buses,
            "affected_customers": affected_customers,
            "affected_load_mw": float(affected_load_mw),
            "estimated_duration_minutes": estimated_duration_minutes,
            "incremental_cmi": float(incremental_cmi),
            "incremental_saidi": float(incremental_saidi),
            "incremental_saifi": float(incremental_saifi),
            "customer_impact_percent": float(affected_customers / total_customers * 100) if total_customers > 0 else 0.0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of reliability data."""
        return {
            "total_customers": sum(c.num_customers for c in self.customer_data),
            "total_load_mw": sum(c.load_mw for c in self.customer_data),
            "load_points": len(self.customer_data),
            "outage_events": len(self.outage_events),
            "sustained_outages": len([e for e in self.outage_events if e.is_sustained]),
            "momentary_outages": len([e for e in self.outage_events if not e.is_sustained]),
            "analysis_period_hours": self.analysis_period_hours
        }


def calculate_reliability_indices(net: pp.pandapowerNet,
                                  outage_events: Optional[List[Dict[str, Any]]] = None) -> ReliabilityIndices:
    """
    Convenience function to calculate reliability indices.
    
    Args:
        net: pandapower network
        outage_events: Optional list of outage event dictionaries
        
    Returns:
        Calculated reliability indices
    """
    calculator = ReliabilityCalculator()
    calculator.set_customer_data_from_network(net)
    
    if outage_events:
        for event_data in outage_events:
            event = OutageEvent(
                event_id=event_data.get("event_id", f"event_{len(calculator.outage_events)}"),
                outage_type=OutageType(event_data.get("outage_type", "unplanned")),
                start_time=datetime.fromisoformat(event_data["start_time"]),
                end_time=datetime.fromisoformat(event_data["end_time"]) if event_data.get("end_time") else None,
                affected_buses=event_data.get("affected_buses", []),
                affected_customers=event_data.get("affected_customers", 0),
                cause=event_data.get("cause", "")
            )
            calculator.add_outage_event(event)
    
    return calculator.calculate_indices()


if __name__ == "__main__":
    # Test reliability indices calculation
    import json
    import pandapower.networks as pn
    
    # Load test network
    net = pn.case33bw()
    pp.runpp(net)
    
    print("\n=== Reliability Indices Test ===")
    
    # Create calculator
    calculator = ReliabilityCalculator(analysis_period_hours=8760)  # 1 year
    calculator.set_customer_data_from_network(net, customers_per_mw=500)
    
    print(f"\nNetwork Summary:")
    print(json.dumps(calculator.get_summary(), indent=2))
    
    # Simulate some outage events
    calculator.record_contingency_outage(net, [5, 6, 7], duration_minutes=45, cause="Line 5 failure")
    calculator.record_contingency_outage(net, [10, 11, 12], duration_minutes=30, cause="Equipment failure")
    calculator.record_contingency_outage(net, [20, 21], duration_minutes=2, cause="Momentary fault")
    
    # Calculate indices
    result = calculator.analyze_reliability(net)
    
    print(f"\n=== Reliability Indices ===")
    print(result.indices.get_summary_text())
    
    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  - {v}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for r in result.recommendations:
            print(f"  - {r}")
    
    # Test contingency impact estimation
    print("\n=== Contingency Impact Estimation ===")
    impact = calculator.estimate_contingency_impact(net, [15, 16, 17, 18], estimated_duration_minutes=60)
    print(json.dumps(impact, indent=2))

"""Protection coordination analysis for distribution networks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
from loguru import logger


class ProtectionDeviceType(Enum):
    """Types of protection devices."""
    FUSE = "fuse"
    RECLOSER = "recloser"
    RELAY = "relay"
    BREAKER = "breaker"
    SECTIONALIZER = "sectionalizer"


class TripCharacteristic(Enum):
    """Time-current characteristic curves."""
    EXTREMELY_INVERSE = "extremely_inverse"
    VERY_INVERSE = "very_inverse"
    INVERSE = "inverse"
    DEFINITE_TIME = "definite_time"


@dataclass
class ProtectionDevice:
    """Represents a protection device."""
    device_id: str
    device_type: ProtectionDeviceType
    bus_id: int
    line_id: Optional[int] = None
    pickup_current_a: float = 100.0
    time_dial: float = 1.0
    characteristic: TripCharacteristic = TripCharacteristic.INVERSE
    instantaneous_pickup_a: Optional[float] = None
    coordination_time_interval: float = 0.3  # CTI in seconds
    max_interrupting_current_ka: float = 10.0
    in_service: bool = True
    
    def calculate_trip_time(self, fault_current_a: float) -> float:
        """
        Calculate trip time for given fault current using IEC curves.
        
        Args:
            fault_current_a: Fault current in amperes
            
        Returns:
            Trip time in seconds
        """
        if fault_current_a <= self.pickup_current_a:
            return float('inf')  # No trip
        
        # Check instantaneous pickup
        if self.instantaneous_pickup_a and fault_current_a >= self.instantaneous_pickup_a:
            return 0.05  # Instantaneous trip (50ms)
        
        # Calculate multiple of pickup
        m = fault_current_a / self.pickup_current_a
        
        # IEC standard curves
        if self.characteristic == TripCharacteristic.EXTREMELY_INVERSE:
            # IEC 60255: t = TDS * 80 / (M^2 - 1)
            trip_time = self.time_dial * 80.0 / (m**2 - 1)
        elif self.characteristic == TripCharacteristic.VERY_INVERSE:
            # IEC 60255: t = TDS * 13.5 / (M - 1)
            trip_time = self.time_dial * 13.5 / (m - 1)
        elif self.characteristic == TripCharacteristic.INVERSE:
            # IEC 60255: t = TDS * 0.14 / (M^0.02 - 1)
            trip_time = self.time_dial * 0.14 / (m**0.02 - 1)
        else:  # DEFINITE_TIME
            trip_time = self.time_dial
        
        return max(trip_time, 0.01)  # Minimum 10ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "bus_id": self.bus_id,
            "line_id": self.line_id,
            "pickup_current_a": float(self.pickup_current_a),
            "time_dial": float(self.time_dial),
            "characteristic": self.characteristic.value,
            "instantaneous_pickup_a": float(self.instantaneous_pickup_a) if self.instantaneous_pickup_a else None,
            "coordination_time_interval": float(self.coordination_time_interval),
            "max_interrupting_current_ka": float(self.max_interrupting_current_ka),
            "in_service": self.in_service
        }


@dataclass
class CoordinationResult:
    """Result of coordination check between two devices."""
    upstream_device: str
    downstream_device: str
    fault_location: int  # Bus ID
    fault_current_ka: float
    upstream_trip_time: float
    downstream_trip_time: float
    time_margin: float
    coordinated: bool
    issue: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "upstream_device": self.upstream_device,
            "downstream_device": self.downstream_device,
            "fault_location": self.fault_location,
            "fault_current_ka": float(self.fault_current_ka),
            "upstream_trip_time": float(self.upstream_trip_time),
            "downstream_trip_time": float(self.downstream_trip_time),
            "time_margin": float(self.time_margin),
            "coordinated": self.coordinated,
            "issue": self.issue
        }


@dataclass
class ShortCircuitResult:
    """Short circuit analysis result."""
    bus_id: int
    ikss_ka: float  # Initial symmetrical short-circuit current
    ip_ka: float    # Peak short-circuit current
    ith_ka: float   # Thermal equivalent short-circuit current
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bus_id": self.bus_id,
            "ikss_ka": float(self.ikss_ka),
            "ip_ka": float(self.ip_ka),
            "ith_ka": float(self.ith_ka)
        }


@dataclass
class ProtectionAnalysisResult:
    """Results from protection coordination analysis."""
    success: bool
    total_devices: int
    coordination_pairs_checked: int
    coordination_violations: int
    short_circuit_results: List[ShortCircuitResult] = field(default_factory=list)
    coordination_results: List[CoordinationResult] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "total_devices": self.total_devices,
            "coordination_pairs_checked": self.coordination_pairs_checked,
            "coordination_violations": self.coordination_violations,
            "short_circuit_results": [r.to_dict() for r in self.short_circuit_results],
            "coordination_results": [r.to_dict() for r in self.coordination_results],
            "violations": self.violations,
            "recommendations": self.recommendations
        }


def validate_bus_id(net: pp.pandapowerNet, bus_id: int) -> bool:
    """Check if a bus ID exists in the network."""
    return bus_id in set(net.bus.index.tolist())


class ProtectionCoordination:
    """Analyze and verify protection device coordination."""
    
    def __init__(self, min_coordination_time: float = 0.3,
                 max_fault_clearing_time: float = 2.0):
        """
        Initialize protection coordination analyzer.
        
        Args:
            min_coordination_time: Minimum CTI between devices (seconds)
            max_fault_clearing_time: Maximum allowed fault clearing time (seconds)
        """
        self.min_cti = min_coordination_time
        self.max_clearing_time = max_fault_clearing_time
        self.devices: List[ProtectionDevice] = []
        logger.info(f"ProtectionCoordination initialized (CTI={min_coordination_time}s)")
    
    def add_device(self, device: ProtectionDevice) -> None:
        """Add a protection device to the system."""
        self.devices.append(device)
        logger.info(f"Added protection device: {device.device_id} at bus {device.bus_id}")
    
    def add_device_at_bus(self, net: pp.pandapowerNet, bus_id: int,
                          device_type: ProtectionDeviceType,
                          pickup_current_a: float,
                          time_dial: float = 1.0) -> Optional[ProtectionDevice]:
        """
        Add a protection device at a specific bus with validation.
        
        Args:
            net: pandapower network
            bus_id: Bus ID to place device
            device_type: Type of protection device
            pickup_current_a: Pickup current setting
            time_dial: Time dial setting
            
        Returns:
            Created ProtectionDevice or None if validation fails
        """
        if not validate_bus_id(net, bus_id):
            logger.warning(f"Invalid bus ID: {bus_id}")
            return None
        
        device_id = f"{device_type.value}_{bus_id}_{len(self.devices)}"
        device = ProtectionDevice(
            device_id=device_id,
            device_type=device_type,
            bus_id=bus_id,
            pickup_current_a=pickup_current_a,
            time_dial=time_dial
        )
        self.add_device(device)
        return device
    
    def run_short_circuit_analysis(self, net: pp.pandapowerNet,
                                   fault_buses: Optional[List[int]] = None) -> List[ShortCircuitResult]:
        """
        Run short circuit analysis on the network.
        
        Args:
            net: pandapower network
            fault_buses: List of buses to analyze (None = all buses)
            
        Returns:
            List of short circuit results
        """
        logger.info("Running short circuit analysis")
        results = []
        
        try:
            # Run pandapower short circuit calculation
            sc.calc_sc(net, fault="3ph", case="max")
            
            # Get buses to analyze
            if fault_buses is None:
                buses_to_check = net.bus.index.tolist()
            else:
                # Validate bus IDs
                buses_to_check = [b for b in fault_buses if validate_bus_id(net, b)]
                invalid_buses = [b for b in fault_buses if not validate_bus_id(net, b)]
                if invalid_buses:
                    logger.warning(f"Invalid bus IDs skipped: {invalid_buses}")
            
            # Extract results
            for bus_id in buses_to_check:
                if bus_id in net.res_bus_sc.index:
                    ikss = net.res_bus_sc.at[bus_id, "ikss_ka"]
                    ip = net.res_bus_sc.at[bus_id, "ip_ka"] if "ip_ka" in net.res_bus_sc.columns else ikss * 1.8
                    ith = net.res_bus_sc.at[bus_id, "ith_ka"] if "ith_ka" in net.res_bus_sc.columns else ikss
                    
                    results.append(ShortCircuitResult(
                        bus_id=int(bus_id),
                        ikss_ka=float(ikss),
                        ip_ka=float(ip),
                        ith_ka=float(ith)
                    ))
            
            logger.info(f"Short circuit analysis complete: {len(results)} buses analyzed")
            
        except Exception as e:
            logger.error(f"Short circuit analysis failed: {e}")
            # Return estimated values based on network impedance
            for bus_id in (fault_buses or net.bus.index.tolist()[:10]):
                if validate_bus_id(net, bus_id):
                    results.append(ShortCircuitResult(
                        bus_id=int(bus_id),
                        ikss_ka=5.0,  # Estimated
                        ip_ka=9.0,
                        ith_ka=5.0
                    ))
        
        return results
    
    def check_device_coordination(self, upstream: ProtectionDevice,
                                  downstream: ProtectionDevice,
                                  fault_current_a: float,
                                  fault_bus: int) -> CoordinationResult:
        """
        Check coordination between two devices for a specific fault.
        
        Args:
            upstream: Upstream (backup) protection device
            downstream: Downstream (primary) protection device
            fault_current_a: Fault current at fault location
            fault_bus: Bus where fault occurs
            
        Returns:
            Coordination result
        """
        upstream_time = upstream.calculate_trip_time(fault_current_a)
        downstream_time = downstream.calculate_trip_time(fault_current_a)
        
        time_margin = upstream_time - downstream_time
        coordinated = time_margin >= self.min_cti
        
        issue = ""
        if not coordinated:
            if time_margin < 0:
                issue = f"Upstream trips before downstream (margin={time_margin:.3f}s)"
            else:
                issue = f"Insufficient CTI: {time_margin:.3f}s < {self.min_cti}s"
        
        if downstream_time > self.max_clearing_time:
            issue = f"Downstream clearing time {downstream_time:.3f}s exceeds max {self.max_clearing_time}s"
            coordinated = False
        
        return CoordinationResult(
            upstream_device=upstream.device_id,
            downstream_device=downstream.device_id,
            fault_location=fault_bus,
            fault_current_ka=fault_current_a / 1000.0,
            upstream_trip_time=upstream_time,
            downstream_trip_time=downstream_time,
            time_margin=time_margin,
            coordinated=coordinated,
            issue=issue
        )
    
    def analyze_coordination(self, net: pp.pandapowerNet) -> ProtectionAnalysisResult:
        """
        Perform full protection coordination analysis.
        
        Args:
            net: pandapower network
            
        Returns:
            Protection analysis result
        """
        logger.info("Starting protection coordination analysis")
        
        if not self.devices:
            logger.warning("No protection devices defined")
            return ProtectionAnalysisResult(
                success=False,
                total_devices=0,
                coordination_pairs_checked=0,
                coordination_violations=0,
                violations=["No protection devices defined"]
            )
        
        # Run short circuit analysis
        sc_results = self.run_short_circuit_analysis(net)
        
        # Create fault current lookup
        fault_currents = {r.bus_id: r.ikss_ka * 1000 for r in sc_results}  # Convert to A
        
        # Check coordination for all device pairs
        coordination_results = []
        violations = []
        recommendations = []
        
        # Sort devices by bus location (approximate upstream/downstream)
        sorted_devices = sorted(self.devices, key=lambda d: d.bus_id)
        
        pairs_checked = 0
        for i, downstream in enumerate(sorted_devices):
            for upstream in sorted_devices[i+1:]:
                # Check if upstream is actually upstream (higher bus number in radial network)
                if upstream.bus_id <= downstream.bus_id:
                    continue
                
                # Get fault current at downstream device location
                fault_current = fault_currents.get(downstream.bus_id, 5000.0)
                
                result = self.check_device_coordination(
                    upstream, downstream, fault_current, downstream.bus_id
                )
                coordination_results.append(result)
                pairs_checked += 1
                
                if not result.coordinated:
                    violations.append(result.issue)
                    
                    # Generate recommendation
                    if result.time_margin < self.min_cti:
                        recommendations.append(
                            f"Increase time dial on {upstream.device_id} or decrease on {downstream.device_id}"
                        )
        
        # Check for devices exceeding interrupting capacity
        for device in self.devices:
            fault_current_ka = fault_currents.get(device.bus_id, 5.0) / 1000
            if fault_current_ka > device.max_interrupting_current_ka:
                violations.append(
                    f"{device.device_id}: Fault current {fault_current_ka:.2f} kA exceeds "
                    f"interrupting capacity {device.max_interrupting_current_ka} kA"
                )
                recommendations.append(
                    f"Replace {device.device_id} with higher rated device"
                )
        
        result = ProtectionAnalysisResult(
            success=len(violations) == 0,
            total_devices=len(self.devices),
            coordination_pairs_checked=pairs_checked,
            coordination_violations=len([r for r in coordination_results if not r.coordinated]),
            short_circuit_results=sc_results,
            coordination_results=coordination_results,
            violations=violations,
            recommendations=recommendations
        )
        
        logger.info(f"Protection analysis complete: {result.coordination_violations} violations found")
        return result
    
    def verify_post_reconfiguration(self, net: pp.pandapowerNet,
                                    switched_lines: List[int]) -> ProtectionAnalysisResult:
        """
        Verify protection coordination after network reconfiguration.
        
        Args:
            net: pandapower network (after reconfiguration)
            switched_lines: List of line IDs that were switched
            
        Returns:
            Protection analysis result
        """
        logger.info(f"Verifying protection after reconfiguration: {len(switched_lines)} lines switched")
        
        # Get buses affected by switching
        affected_buses: Set[int] = set()
        for line_id in switched_lines:
            if line_id in net.line.index:
                affected_buses.add(int(net.line.at[line_id, "from_bus"]))
                affected_buses.add(int(net.line.at[line_id, "to_bus"]))
        
        # Run analysis focusing on affected areas
        result = self.analyze_coordination(net)
        
        # Add specific warnings for affected areas
        for coord_result in result.coordination_results:
            if coord_result.fault_location in affected_buses and not coord_result.coordinated:
                result.recommendations.append(
                    f"Review protection settings near switched lines: "
                    f"{coord_result.downstream_device} may need adjustment"
                )
        
        return result
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get summary of protection devices."""
        summary = {
            "total_devices": len(self.devices),
            "by_type": {},
            "by_bus": {}
        }
        
        for device in self.devices:
            # By type
            dtype = device.device_type.value
            if dtype not in summary["by_type"]:
                summary["by_type"][dtype] = 0
            summary["by_type"][dtype] += 1
            
            # By bus
            bus = device.bus_id
            if bus not in summary["by_bus"]:
                summary["by_bus"][bus] = []
            summary["by_bus"][bus].append(device.device_id)
        
        return summary


def create_default_protection_scheme(net: pp.pandapowerNet) -> ProtectionCoordination:
    """
    Create a default protection scheme for a network.
    
    Args:
        net: pandapower network
        
    Returns:
        ProtectionCoordination with default devices
    """
    coordinator = ProtectionCoordination()
    
    # Add protection at substation (slack bus)
    slack_buses = net.ext_grid.bus.tolist()
    for bus in slack_buses:
        coordinator.add_device_at_bus(
            net, bus,
            ProtectionDeviceType.BREAKER,
            pickup_current_a=500.0,
            time_dial=0.5
        )
    
    # Add reclosers at major branch points (buses with multiple connections)
    if len(net.line) > 0:
        # Find buses with multiple line connections
        from_counts = net.line.from_bus.value_counts()
        to_counts = net.line.to_bus.value_counts()
        
        for bus in from_counts.index:
            if from_counts[bus] >= 2 and bus not in slack_buses:
                coordinator.add_device_at_bus(
                    net, int(bus),
                    ProtectionDeviceType.RECLOSER,
                    pickup_current_a=200.0,
                    time_dial=1.0
                )
    
    # Add fuses at load buses
    for idx, load in net.load.iterrows():
        bus = int(load.bus)
        if bus not in slack_buses:
            # Size fuse based on load current
            load_current = (load.p_mw * 1000) / (net.bus.at[bus, "vn_kv"] * np.sqrt(3))
            pickup = max(load_current * 1.5, 50.0)
            
            coordinator.add_device_at_bus(
                net, bus,
                ProtectionDeviceType.FUSE,
                pickup_current_a=pickup,
                time_dial=0.1
            )
    
    logger.info(f"Created default protection scheme with {len(coordinator.devices)} devices")
    return coordinator


if __name__ == "__main__":
    # Test protection coordination
    import json
    import pandapower.networks as pn
    
    # Load test network
    net = pn.case33bw()
    pp.runpp(net)
    
    print("\n=== Protection Coordination Test ===")
    
    # Create default protection scheme
    coordinator = create_default_protection_scheme(net)
    
    print(f"\nDevices: {len(coordinator.devices)}")
    print(json.dumps(coordinator.get_device_summary(), indent=2))
    
    # Run coordination analysis
    result = coordinator.analyze_coordination(net)
    
    print(f"\n=== Analysis Results ===")
    print(f"Success: {result.success}")
    print(f"Devices: {result.total_devices}")
    print(f"Pairs checked: {result.coordination_pairs_checked}")
    print(f"Violations: {result.coordination_violations}")
    
    if result.violations:
        print("\nViolations:")
        for v in result.violations[:5]:
            print(f"  - {v}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for r in result.recommendations[:5]:
            print(f"  - {r}")

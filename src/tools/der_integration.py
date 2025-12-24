"""Distributed Energy Resources (DER) integration module."""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
from loguru import logger


class DERType(Enum):
    """Types of distributed energy resources."""
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    BATTERY_STORAGE = "battery_storage"
    DIESEL_GENERATOR = "diesel_generator"
    FUEL_CELL = "fuel_cell"
    MICROTURBINE = "microturbine"


@dataclass
class DERUnit:
    """Represents a single DER unit."""
    der_id: str
    der_type: DERType
    bus_id: int
    rated_power_mw: float
    min_power_mw: float = 0.0
    max_power_mw: Optional[float] = None
    power_factor: float = 0.95
    controllable: bool = True
    in_service: bool = True
    cost_per_mwh: float = 0.0  # Operating cost

    def __post_init__(self):
        if self.max_power_mw is None:
            self.max_power_mw = self.rated_power_mw

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "der_id": self.der_id,
            "der_type": self.der_type.value,
            "bus_id": self.bus_id,
            "rated_power_mw": self.rated_power_mw,
            "min_power_mw": self.min_power_mw,
            "max_power_mw": self.max_power_mw,
            "power_factor": self.power_factor,
            "controllable": self.controllable,
            "in_service": self.in_service,
            "cost_per_mwh": self.cost_per_mwh
        }


@dataclass
class DERDispatch:
    """Dispatch result for a DER unit."""
    der_id: str
    active_power_mw: float
    reactive_power_mvar: float
    curtailed_mw: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "der_id": self.der_id,
            "active_power_mw": float(self.active_power_mw),
            "reactive_power_mvar": float(self.reactive_power_mvar),
            "curtailed_mw": float(self.curtailed_mw)
        }


@dataclass
class BatteryState:
    """State of a battery storage system."""
    battery_id: str
    capacity_mwh: float
    current_soc: float  # State of charge (0-1)
    max_charge_rate_mw: float
    max_discharge_rate_mw: float
    efficiency: float = 0.92  # Round-trip efficiency
    min_soc: float = 0.1  # Minimum state of charge
    max_soc: float = 0.9  # Maximum state of charge
    
    @property
    def available_energy_mwh(self) -> float:
        """Energy available for discharge."""
        return (self.current_soc - self.min_soc) * self.capacity_mwh
    
    @property
    def available_capacity_mwh(self) -> float:
        """Capacity available for charging."""
        return (self.max_soc - self.current_soc) * self.capacity_mwh
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "battery_id": self.battery_id,
            "capacity_mwh": float(self.capacity_mwh),
            "current_soc": float(self.current_soc),
            "max_charge_rate_mw": float(self.max_charge_rate_mw),
            "max_discharge_rate_mw": float(self.max_discharge_rate_mw),
            "efficiency": float(self.efficiency),
            "available_energy_mwh": float(self.available_energy_mwh),
            "available_capacity_mwh": float(self.available_capacity_mwh)
        }


@dataclass
class BatteryDispatchResult:
    """Result of battery dispatch optimization."""
    success: bool
    battery_id: str
    dispatch_mw: float  # Positive = discharge, negative = charge
    new_soc: float
    energy_throughput_mwh: float
    revenue_or_cost: float = 0.0
    mode: str = "idle"  # "charging", "discharging", "idle"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "battery_id": self.battery_id,
            "dispatch_mw": float(self.dispatch_mw),
            "new_soc": float(self.new_soc),
            "energy_throughput_mwh": float(self.energy_throughput_mwh),
            "revenue_or_cost": float(self.revenue_or_cost),
            "mode": self.mode
        }


@dataclass
class DERAnalysisResult:
    """Results from DER integration analysis."""
    success: bool
    total_der_generation_mw: float
    total_curtailment_mw: float
    hosting_capacity_mw: float
    voltage_rise_max_pu: float
    reverse_power_flow: bool
    dispatches: List[DERDispatch] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "total_der_generation_mw": float(self.total_der_generation_mw),
            "total_curtailment_mw": float(self.total_curtailment_mw),
            "hosting_capacity_mw": float(self.hosting_capacity_mw),
            "voltage_rise_max_pu": float(self.voltage_rise_max_pu),
            "reverse_power_flow": self.reverse_power_flow,
            "dispatches": [d.to_dict() for d in self.dispatches],
            "violations": self.violations,
            "recommendations": self.recommendations
        }


class DERIntegration:
    """Manage DER integration and analysis."""

    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
        self.der_units: List[DERUnit] = []
        logger.info("DERIntegration initialized")

    def add_der(self, net: pp.pandapowerNet, der: DERUnit) -> int:
        """
        Add a DER unit to the network.

        Returns:
            Index of the created sgen element
        """
        # Calculate reactive power from power factor
        q_mvar = der.rated_power_mw * np.tan(np.arccos(der.power_factor))

        # Add as static generator
        sgen_idx = pp.create_sgen(
            net,
            bus=der.bus_id,
            p_mw=der.rated_power_mw,
            q_mvar=q_mvar,
            name=der.der_id,
            type=der.der_type.value,
            in_service=der.in_service,
            controllable=der.controllable
        )

        # Set min/max for OPF
        net.sgen.at[sgen_idx, 'min_p_mw'] = der.min_power_mw
        net.sgen.at[sgen_idx, 'max_p_mw'] = der.max_power_mw
        net.sgen.at[sgen_idx, 'min_q_mvar'] = -q_mvar
        net.sgen.at[sgen_idx, 'max_q_mvar'] = q_mvar

        self.der_units.append(der)
        logger.info(f"Added DER: {der.der_id} ({der.der_type.value}) at bus {der.bus_id}")

        return sgen_idx

    def add_solar_pv(self, net: pp.pandapowerNet, bus_id: int,
                     rated_power_mw: float, name: str = None) -> int:
        """Add a solar PV system."""
        if name is None:
            name = f"PV_{bus_id}"

        der = DERUnit(
            der_id=name,
            der_type=DERType.SOLAR_PV,
            bus_id=bus_id,
            rated_power_mw=rated_power_mw,
            min_power_mw=0.0,
            power_factor=1.0,  # Unity PF for PV
            cost_per_mwh=0.0   # Zero marginal cost
        )

        return self.add_der(net, der)

    def add_battery_storage(self, net: pp.pandapowerNet, bus_id: int,
                           rated_power_mw: float, name: str = None) -> int:
        """Add a battery storage system."""
        if name is None:
            name = f"BESS_{bus_id}"

        # Battery can both charge and discharge
        der = DERUnit(
            der_id=name,
            der_type=DERType.BATTERY_STORAGE,
            bus_id=bus_id,
            rated_power_mw=rated_power_mw,
            min_power_mw=-rated_power_mw,  # Can charge (negative)
            max_power_mw=rated_power_mw,   # Can discharge (positive)
            power_factor=0.95,
            cost_per_mwh=10.0  # Small cycling cost
        )

        return self.add_der(net, der)

    def add_wind_turbine(self, net: pp.pandapowerNet, bus_id: int,
                        rated_power_mw: float, name: str = None) -> int:
        """Add a wind turbine."""
        if name is None:
            name = f"Wind_{bus_id}"

        der = DERUnit(
            der_id=name,
            der_type=DERType.WIND,
            bus_id=bus_id,
            rated_power_mw=rated_power_mw,
            min_power_mw=0.0,
            power_factor=0.95,
            cost_per_mwh=0.0
        )

        return self.add_der(net, der)

    def set_der_output(self, net: pp.pandapowerNet, der_id: str,
                       power_mw: float) -> bool:
        """Set the output power of a DER unit."""
        # Find sgen by name
        matches = net.sgen[net.sgen.name == der_id]

        if len(matches) == 0:
            logger.warning(f"DER not found: {der_id}")
            return False

        idx = matches.index[0]
        net.sgen.at[idx, 'p_mw'] = power_mw

        # Update reactive power based on power factor
        der = next((d for d in self.der_units if d.der_id == der_id), None)
        if der:
            q_mvar = power_mw * np.tan(np.arccos(der.power_factor))
            net.sgen.at[idx, 'q_mvar'] = q_mvar

        return True

    def analyze_der_impact(self, net: pp.pandapowerNet) -> DERAnalysisResult:
        """Analyze the impact of DERs on the network."""
        logger.info("Analyzing DER impact on network")

        net_work = copy.deepcopy(net)

        try:
            pp.runpp(net_work)

            if not net_work.converged:
                return DERAnalysisResult(
                    success=False,
                    total_der_generation_mw=0.0,
                    total_curtailment_mw=0.0,
                    hosting_capacity_mw=0.0,
                    voltage_rise_max_pu=0.0,
                    reverse_power_flow=False,
                    violations=["Power flow did not converge"]
                )

            # Calculate DER generation
            total_der_gen = 0.0
            if len(net_work.res_sgen) > 0:
                total_der_gen = net_work.res_sgen.p_mw.sum()

            # Check for voltage rise
            max_voltage = net_work.res_bus.vm_pu.max()
            voltage_rise = max_voltage - 1.0

            # Check for reverse power flow at substation
            reverse_flow = False
            if len(net_work.res_ext_grid) > 0:
                ext_grid_p = net_work.res_ext_grid.p_mw.sum()
                reverse_flow = ext_grid_p < 0

            # Identify violations
            violations = []
            recommendations = []

            # Voltage violations
            overvoltage_buses = net_work.res_bus[net_work.res_bus.vm_pu > self.v_max]
            for idx in overvoltage_buses.index:
                v = overvoltage_buses.at[idx, 'vm_pu']
                violations.append(f"Bus {idx}: {v:.4f} pu (overvoltage)")

            if len(overvoltage_buses) > 0:
                recommendations.append("Consider DER curtailment or reactive power control")
                recommendations.append("Evaluate voltage regulator settings")

            # Thermal violations
            if len(net_work.res_line) > 0:
                overloaded = net_work.res_line[net_work.res_line.loading_percent > self.thermal_limit]
                for idx in overloaded.index:
                    loading = overloaded.at[idx, 'loading_percent']
                    violations.append(f"Line {idx}: {loading:.1f}% (overloaded)")

            if reverse_flow:
                recommendations.append("Reverse power flow detected - verify protection settings")

            # Create dispatch results
            dispatches = []
            for idx, row in net_work.sgen.iterrows():
                if row.in_service:
                    dispatches.append(DERDispatch(
                        der_id=str(row.name) if row.name else f"sgen_{idx}",
                        active_power_mw=float(net_work.res_sgen.at[idx, 'p_mw']),
                        reactive_power_mvar=float(net_work.res_sgen.at[idx, 'q_mvar'])
                    ))

            return DERAnalysisResult(
                success=True,
                total_der_generation_mw=float(total_der_gen),
                total_curtailment_mw=0.0,
                hosting_capacity_mw=0.0,  # Calculated separately
                voltage_rise_max_pu=float(voltage_rise),
                reverse_power_flow=reverse_flow,
                dispatches=dispatches,
                violations=violations,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"DER impact analysis failed: {e}")
            return DERAnalysisResult(
                success=False,
                total_der_generation_mw=0.0,
                total_curtailment_mw=0.0,
                hosting_capacity_mw=0.0,
                voltage_rise_max_pu=0.0,
                reverse_power_flow=False,
                violations=[str(e)]
            )

    def calculate_hosting_capacity(self, net: pp.pandapowerNet, bus_id: int,
                                   max_der_mw: float = 10.0,
                                   step_mw: float = 0.1) -> float:
        """
        Calculate DER hosting capacity at a specific bus.

        Uses iterative approach to find maximum DER that can be connected
        without causing voltage or thermal violations.
        """
        logger.info(f"Calculating hosting capacity at bus {bus_id}")

        net_work = copy.deepcopy(net)

        # Add test DER
        test_sgen = pp.create_sgen(
            net_work,
            bus=bus_id,
            p_mw=0.0,
            q_mvar=0.0,
            name="test_der"
        )

        hosting_capacity = 0.0
        current_power = step_mw

        while current_power <= max_der_mw:
            net_work.sgen.at[test_sgen, 'p_mw'] = current_power

            try:
                pp.runpp(net_work)

                if not net_work.converged:
                    break

                # Check constraints
                max_v = net_work.res_bus.vm_pu.max()
                max_loading = 0.0
                if len(net_work.res_line) > 0:
                    max_loading = net_work.res_line.loading_percent.max()

                if max_v > self.v_max or max_loading > self.thermal_limit:
                    break

                hosting_capacity = current_power
                current_power += step_mw

            except Exception:
                break

        logger.info(f"Hosting capacity at bus {bus_id}: {hosting_capacity:.2f} MW")
        return hosting_capacity

    def optimize_der_dispatch(self, net: pp.pandapowerNet,
                             available_power: Dict[str, float]) -> DERAnalysisResult:
        """
        Optimize DER dispatch considering constraints.

        Args:
            net: pandapower network
            available_power: Dict mapping DER ID to available power (e.g., from forecast)

        Returns:
            Analysis result with optimal dispatch
        """
        logger.info("Optimizing DER dispatch")

        net_work = copy.deepcopy(net)

        # Set available power for each DER
        for der_id, power in available_power.items():
            self.set_der_output(net_work, der_id, power)

        # Run power flow
        try:
            pp.runpp(net_work)

            if not net_work.converged:
                # Try curtailment
                return self._curtail_for_feasibility(net_work, available_power)

            # Check for violations
            result = self.analyze_der_impact(net_work)

            if result.violations:
                # Need curtailment
                return self._curtail_for_feasibility(net_work, available_power)

            return result

        except Exception as e:
            logger.error(f"DER dispatch optimization failed: {e}")
            return DERAnalysisResult(
                success=False,
                total_der_generation_mw=0.0,
                total_curtailment_mw=0.0,
                hosting_capacity_mw=0.0,
                voltage_rise_max_pu=0.0,
                reverse_power_flow=False,
                violations=[str(e)]
            )

    def _curtail_for_feasibility(self, net: pp.pandapowerNet,
                                 available_power: Dict[str, float]) -> DERAnalysisResult:
        """Curtail DERs to achieve feasible operation."""
        logger.info("Applying DER curtailment for feasibility")

        total_curtailment = 0.0
        curtailment_step = 0.1  # 10% steps

        for step in range(10):  # Max 10 iterations
            curtailment_factor = 1.0 - (step * curtailment_step)

            for der_id, power in available_power.items():
                curtailed_power = power * curtailment_factor
                self.set_der_output(net, der_id, curtailed_power)

            try:
                pp.runpp(net)

                if not net.converged:
                    continue

                # Check constraints
                max_v = net.res_bus.vm_pu.max()
                max_loading = 0.0
                if len(net.res_line) > 0:
                    max_loading = net.res_line.loading_percent.max()

                if max_v <= self.v_max and max_loading <= self.thermal_limit:
                    # Feasible solution found
                    total_available = sum(available_power.values())
                    total_dispatched = sum(p * curtailment_factor for p in available_power.values())
                    total_curtailment = total_available - total_dispatched

                    result = self.analyze_der_impact(net)
                    result.total_curtailment_mw = total_curtailment
                    result.recommendations.append(
                        f"Curtailed {total_curtailment:.2f} MW ({(1-curtailment_factor)*100:.0f}%)"
                    )
                    return result

            except Exception:
                continue

        # Could not find feasible solution
        return DERAnalysisResult(
            success=False,
            total_der_generation_mw=0.0,
            total_curtailment_mw=sum(available_power.values()),
            hosting_capacity_mw=0.0,
            voltage_rise_max_pu=0.0,
            reverse_power_flow=False,
            violations=["Could not find feasible dispatch even with full curtailment"]
        )

    def optimize_battery_dispatch(self, net: pp.pandapowerNet,
                                   battery_state: BatteryState,
                                   target_power_mw: Optional[float] = None,
                                   price_signal: Optional[float] = None,
                                   duration_hours: float = 1.0,
                                   mode: str = "auto") -> BatteryDispatchResult:
        """
        Optimize battery dispatch based on grid conditions and price signals.
        
        Args:
            net: pandapower network
            battery_state: Current state of the battery
            target_power_mw: Target power output (positive=discharge, negative=charge)
            price_signal: Current electricity price ($/MWh) for economic dispatch
            duration_hours: Duration of dispatch period
            mode: Dispatch mode ('auto', 'peak_shaving', 'arbitrage', 'voltage_support')
            
        Returns:
            BatteryDispatchResult with optimized dispatch
        """
        logger.info(f"Optimizing battery dispatch: mode={mode}, SOC={battery_state.current_soc:.2%}")
        
        try:
            # Run power flow to get current state
            pp.runpp(net)
            if not net.converged:
                return BatteryDispatchResult(
                    success=False,
                    battery_id=battery_state.battery_id,
                    dispatch_mw=0.0,
                    new_soc=battery_state.current_soc,
                    energy_throughput_mwh=0.0,
                    mode="idle"
                )
            
            # Determine optimal dispatch based on mode
            if target_power_mw is not None:
                dispatch_mw = self._constrain_battery_dispatch(
                    target_power_mw, battery_state, duration_hours
                )
            elif mode == "peak_shaving":
                dispatch_mw = self._peak_shaving_dispatch(net, battery_state, duration_hours)
            elif mode == "arbitrage" and price_signal is not None:
                dispatch_mw = self._arbitrage_dispatch(battery_state, price_signal, duration_hours)
            elif mode == "voltage_support":
                dispatch_mw = self._voltage_support_dispatch(net, battery_state, duration_hours)
            else:
                # Auto mode - use peak shaving as default
                dispatch_mw = self._peak_shaving_dispatch(net, battery_state, duration_hours)
            
            # Calculate new SOC
            energy_mwh = dispatch_mw * duration_hours
            if dispatch_mw > 0:  # Discharging
                new_soc = battery_state.current_soc - (energy_mwh / battery_state.capacity_mwh)
                dispatch_mode = "discharging"
            elif dispatch_mw < 0:  # Charging
                new_soc = battery_state.current_soc - (energy_mwh * battery_state.efficiency / battery_state.capacity_mwh)
                dispatch_mode = "charging"
            else:
                new_soc = battery_state.current_soc
                dispatch_mode = "idle"
            
            # Calculate revenue/cost if price signal provided
            revenue = 0.0
            if price_signal is not None:
                revenue = dispatch_mw * duration_hours * price_signal
            
            return BatteryDispatchResult(
                success=True,
                battery_id=battery_state.battery_id,
                dispatch_mw=dispatch_mw,
                new_soc=new_soc,
                energy_throughput_mwh=abs(energy_mwh),
                revenue_or_cost=revenue,
                mode=dispatch_mode
            )
            
        except Exception as e:
            logger.error(f"Battery dispatch optimization failed: {e}")
            return BatteryDispatchResult(
                success=False,
                battery_id=battery_state.battery_id,
                dispatch_mw=0.0,
                new_soc=battery_state.current_soc,
                energy_throughput_mwh=0.0,
                mode="idle"
            )
    
    def _constrain_battery_dispatch(self, target_mw: float, 
                                    battery_state: BatteryState,
                                    duration_hours: float) -> float:
        """Constrain dispatch to battery limits."""
        if target_mw > 0:  # Discharge
            # Check rate limit
            max_discharge = min(
                battery_state.max_discharge_rate_mw,
                battery_state.available_energy_mwh / duration_hours
            )
            return min(target_mw, max_discharge)
        else:  # Charge
            # Check rate limit
            max_charge = min(
                battery_state.max_charge_rate_mw,
                battery_state.available_capacity_mwh / (duration_hours * battery_state.efficiency)
            )
            return max(target_mw, -max_charge)
    
    def _peak_shaving_dispatch(self, net: pp.pandapowerNet,
                               battery_state: BatteryState,
                               duration_hours: float) -> float:
        """Calculate dispatch for peak shaving."""
        # Get current load
        total_load = net.res_load.p_mw.sum() if len(net.res_load) > 0 else 0.0
        
        # Get current generation
        total_gen = net.res_ext_grid.p_mw.sum() if len(net.res_ext_grid) > 0 else 0.0
        
        # Target: reduce peak by discharging during high load
        # Simple heuristic: discharge if load > 80% of typical peak
        typical_peak = total_load * 1.2  # Assume current is 80% of peak
        
        if total_load > typical_peak * 0.8:
            # High load - discharge
            target_discharge = min(
                total_load * 0.1,  # Shave 10% of load
                battery_state.max_discharge_rate_mw
            )
            return self._constrain_battery_dispatch(target_discharge, battery_state, duration_hours)
        elif total_load < typical_peak * 0.4:
            # Low load - charge
            target_charge = min(
                battery_state.max_charge_rate_mw * 0.5,  # Charge at 50% rate
                battery_state.max_charge_rate_mw
            )
            return self._constrain_battery_dispatch(-target_charge, battery_state, duration_hours)
        else:
            return 0.0  # Idle
    
    def _arbitrage_dispatch(self, battery_state: BatteryState,
                           price_signal: float,
                           duration_hours: float) -> float:
        """Calculate dispatch for price arbitrage."""
        # Simple thresholds for arbitrage
        high_price_threshold = 100.0  # $/MWh
        low_price_threshold = 30.0    # $/MWh
        
        if price_signal > high_price_threshold:
            # High price - discharge
            return self._constrain_battery_dispatch(
                battery_state.max_discharge_rate_mw,
                battery_state,
                duration_hours
            )
        elif price_signal < low_price_threshold:
            # Low price - charge
            return self._constrain_battery_dispatch(
                -battery_state.max_charge_rate_mw,
                battery_state,
                duration_hours
            )
        else:
            return 0.0  # Idle
    
    def _voltage_support_dispatch(self, net: pp.pandapowerNet,
                                  battery_state: BatteryState,
                                  duration_hours: float) -> float:
        """Calculate dispatch for voltage support."""
        min_v = net.res_bus.vm_pu.min()
        max_v = net.res_bus.vm_pu.max()
        
        if min_v < self.v_min:
            # Low voltage - discharge to support
            voltage_deficit = self.v_min - min_v
            target_discharge = min(
                voltage_deficit * 10.0,  # Heuristic: 10 MW per 0.1 pu
                battery_state.max_discharge_rate_mw
            )
            return self._constrain_battery_dispatch(target_discharge, battery_state, duration_hours)
        elif max_v > self.v_max:
            # High voltage - charge to absorb
            voltage_excess = max_v - self.v_max
            target_charge = min(
                voltage_excess * 10.0,
                battery_state.max_charge_rate_mw
            )
            return self._constrain_battery_dispatch(-target_charge, battery_state, duration_hours)
        else:
            return 0.0  # Idle

    def get_der_summary(self, net: pp.pandapowerNet) -> Dict[str, Any]:
        """Get summary of DERs in the network."""
        summary = {
            "total_count": len(net.sgen),
            "total_rated_mw": float(net.sgen.p_mw.sum()) if len(net.sgen) > 0 else 0.0,
            "by_type": {},
            "by_bus": {}
        }

        for _idx, row in net.sgen.iterrows():
            # By type
            der_type = row.type if hasattr(row, 'type') and row.type else "unknown"
            if der_type not in summary["by_type"]:
                summary["by_type"][der_type] = {"count": 0, "total_mw": 0.0}
            summary["by_type"][der_type]["count"] += 1
            summary["by_type"][der_type]["total_mw"] += float(row.p_mw)

            # By bus
            bus = int(row.bus)
            if bus not in summary["by_bus"]:
                summary["by_bus"][bus] = {"count": 0, "total_mw": 0.0}
            summary["by_bus"][bus]["count"] += 1
            summary["by_bus"][bus]["total_mw"] += float(row.p_mw)

        return summary


def add_der_to_network(net: pp.pandapowerNet, bus_id: int,
                       der_type: str, rated_power_mw: float) -> int:
    """Convenience function to add a DER to the network."""
    integration = DERIntegration()

    type_map = {
        "solar": DERType.SOLAR_PV,
        "pv": DERType.SOLAR_PV,
        "wind": DERType.WIND,
        "battery": DERType.BATTERY_STORAGE,
        "bess": DERType.BATTERY_STORAGE
    }

    der_type_enum = type_map.get(der_type.lower(), DERType.SOLAR_PV)

    der = DERUnit(
        der_id=f"{der_type}_{bus_id}",
        der_type=der_type_enum,
        bus_id=bus_id,
        rated_power_mw=rated_power_mw
    )

    return integration.add_der(net, der)

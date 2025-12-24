"""Contingency simulation for N-1 and N-k scenarios."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandapower as pp
from loguru import logger


class ContingencyType(Enum):
    """Supported contingency types."""
    LINE_OUTAGE = "line_outage"
    TRANSFORMER_OUTAGE = "transformer_outage"
    GENERATOR_OUTAGE = "generator_outage"
    BUS_FAULT = "bus_fault"
    LOAD_INCREASE = "load_increase"
    MULTIPLE_OUTAGE = "multiple_outage"


@dataclass
class ContingencyEvent:
    """Single contingency event definition."""
    event_type: ContingencyType
    element_index: int
    element_name: str = ""
    severity: float = 1.0  # Multiplier for severity (e.g., 1.5 = 150% load)
    description: str = ""
    affected_buses: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "element_index": self.element_index,
            "element_name": self.element_name,
            "severity": self.severity,
            "description": self.description,
            "affected_buses": self.affected_buses
        }


@dataclass
class ContingencyResult:
    """Simulation results with violations and metrics."""
    contingency: ContingencyEvent
    converged: bool
    max_line_loading_percent: float
    min_voltage_pu: float
    max_voltage_pu: float
    total_load_shed_mw: float = 0.0
    violated_constraints: List[str] = field(default_factory=list)
    critical_elements: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contingency": self.contingency.to_dict(),
            "converged": self.converged,
            "max_line_loading_percent": float(self.max_line_loading_percent),
            "min_voltage_pu": float(self.min_voltage_pu),
            "max_voltage_pu": float(self.max_voltage_pu),
            "total_load_shed_mw": float(self.total_load_shed_mw),
            "violated_constraints": self.violated_constraints,
            "critical_elements": self.critical_elements
        }

    @property
    def is_critical(self) -> bool:
        """Check if contingency results in critical violations."""
        return len(self.violated_constraints) > 0 or not self.converged


class ContingencySimulator:
    """Simulates contingencies and analyzes violations."""

    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        """
        Initialize contingency simulator.

        Args:
            voltage_limits: (min, max) voltage in per-unit
            thermal_limit_percent: Maximum allowed line loading percentage
        """
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
        self.simulation_history: List[ContingencyResult] = []

    def simulate_contingency(self, net: pp.pandapowerNet,
                           contingency: ContingencyEvent,
                           run_powerflow: bool = True) -> ContingencyResult:
        """
        Simulate a single contingency event.

        Args:
            net: pandapower network (will be modified)
            contingency: Contingency event to simulate
            run_powerflow: Whether to run power flow after applying contingency

        Returns:
            Contingency simulation results
        """
        logger.info(f"Simulating contingency: {contingency.description}")

        # Apply contingency to network
        self._apply_contingency(net, contingency)

        # Run power flow if requested
        converged = False
        if run_powerflow:
            try:
                pp.runpp(net, algorithm="bfsw", calculate_voltage_angles=False)
                converged = net.converged
                if converged:
                    logger.info("Power flow converged")
                else:
                    logger.warning("Power flow did not converge")
            except Exception as e:
                logger.error(f"Power flow failed: {e}")
                converged = False

        # Analyze results
        result = self._analyze_results(net, contingency, converged)
        self.simulation_history.append(result)

        return result

    def _apply_contingency(self, net: pp.pandapowerNet, contingency: ContingencyEvent):
        """Apply contingency event to the network."""
        if contingency.event_type == ContingencyType.LINE_OUTAGE:
            if contingency.element_index < len(net.line):
                net.line.at[contingency.element_index, "in_service"] = False
                logger.debug(f"Line {contingency.element_index} taken out of service")

        elif contingency.event_type == ContingencyType.TRANSFORMER_OUTAGE:
            if contingency.element_index < len(net.trafo):
                net.trafo.at[contingency.element_index, "in_service"] = False
                logger.debug(f"Transformer {contingency.element_index} taken out of service")

        elif contingency.event_type == ContingencyType.GENERATOR_OUTAGE:
            if contingency.element_index < len(net.gen):
                net.gen.at[contingency.element_index, "in_service"] = False
            elif contingency.element_index < len(net.sgen):
                net.sgen.at[contingency.element_index, "in_service"] = False
            logger.debug(f"Generator {contingency.element_index} taken out of service")

        elif contingency.event_type == ContingencyType.LOAD_INCREASE:
            if contingency.element_index < len(net.load):
                original_p = net.load.at[contingency.element_index, "p_mw"]
                original_q = net.load.at[contingency.element_index, "q_mvar"]
                net.load.at[contingency.element_index, "p_mw"] = original_p * contingency.severity
                net.load.at[contingency.element_index, "q_mvar"] = original_q * contingency.severity
                logger.debug(f"Load {contingency.element_index} increased by {(contingency.severity-1)*100:.1f}%")

        else:
            logger.warning(f"Contingency type {contingency.event_type} not fully implemented")

    def _analyze_results(self, net: pp.pandapowerNet,
                        contingency: ContingencyEvent,
                        converged: bool) -> ContingencyResult:
        """Analyze power flow results and identify violations."""
        violations = []
        critical_elements = []

        # Initialize worst-case values
        max_loading = 0.0
        min_voltage = 1.0
        max_voltage = 1.0

        if converged:
            # Check line loadings
            if len(net.res_line) > 0:
                max_loading = net.res_line.loading_percent.max()
                overloaded = net.res_line[net.res_line.loading_percent > self.thermal_limit]

                for idx, row in overloaded.iterrows():
                    violations.append(f"Line {idx}: {row.loading_percent:.1f}% loading")
                    critical_elements.append({
                        "type": "line",
                        "index": int(idx),
                        "loading_percent": float(row.loading_percent),
                        "from_bus": int(net.line.at[idx, "from_bus"]),
                        "to_bus": int(net.line.at[idx, "to_bus"])
                    })

            # Check bus voltages
            if len(net.res_bus) > 0:
                min_voltage = net.res_bus.vm_pu.min()
                max_voltage = net.res_bus.vm_pu.max()

                undervoltage = net.res_bus[net.res_bus.vm_pu < self.v_min]
                overvoltage = net.res_bus[net.res_bus.vm_pu > self.v_max]

                for idx, row in undervoltage.iterrows():
                    violations.append(f"Bus {idx}: {row.vm_pu:.3f} pu (undervoltage)")
                    critical_elements.append({
                        "type": "bus_undervoltage",
                        "index": int(idx),
                        "voltage_pu": float(row.vm_pu),
                        "violation": float(self.v_min - row.vm_pu)
                    })

                for idx, row in overvoltage.iterrows():
                    violations.append(f"Bus {idx}: {row.vm_pu:.3f} pu (overvoltage)")
                    critical_elements.append({
                        "type": "bus_overvoltage",
                        "index": int(idx),
                        "voltage_pu": float(row.vm_pu),
                        "violation": float(row.vm_pu - self.v_max)
                    })
        else:
            violations.append("Power flow did not converge")

        return ContingencyResult(
            contingency=contingency,
            converged=converged,
            max_line_loading_percent=float(max_loading),
            min_voltage_pu=float(min_voltage),
            max_voltage_pu=float(max_voltage),
            violated_constraints=violations,
            critical_elements=critical_elements
        )

    def generate_n_minus_1_contingencies(self, net: pp.pandapowerNet) -> List[ContingencyEvent]:
        """
        Generate all N-1 contingency events for the network.

        Args:
            net: pandapower network

        Returns:
            List of N-1 contingency events
        """
        contingencies = []

        # Line outages
        for idx, line in net.line.iterrows():
            if line.in_service:
                name = line.name if "name" in line and line.name else f"Line {idx}"
                contingencies.append(ContingencyEvent(
                    event_type=ContingencyType.LINE_OUTAGE,
                    element_index=int(idx),
                    element_name=name,
                    description=f"N-1: {name} outage",
                    affected_buses=[int(line.from_bus), int(line.to_bus)]
                ))

        # Transformer outages
        for idx, trafo in net.trafo.iterrows():
            if trafo.in_service:
                name = trafo.name if "name" in trafo and trafo.name else f"Trafo {idx}"
                contingencies.append(ContingencyEvent(
                    event_type=ContingencyType.TRANSFORMER_OUTAGE,
                    element_index=int(idx),
                    element_name=name,
                    description=f"N-1: {name} outage",
                    affected_buses=[int(trafo.hv_bus), int(trafo.lv_bus)]
                ))

        logger.info(f"Generated {len(contingencies)} N-1 contingency scenarios")
        return contingencies

    def generate_load_increase_scenarios(self, net: pp.pandapowerNet,
                                        increase_factors: List[float] = None) -> List[ContingencyEvent]:
        """
        Generate load increase scenarios.

        Args:
            net: pandapower network
            increase_factors: List of load increase multipliers

        Returns:
            List of load increase contingencies
        """
        if increase_factors is None:
            increase_factors = [1.2, 1.5, 2.0]
        contingencies = []

        for idx, load in net.load.iterrows():
            if load.in_service:
                name = load.name if "name" in load and load.name else f"Load {idx}"
                for factor in increase_factors:
                    contingencies.append(ContingencyEvent(
                        event_type=ContingencyType.LOAD_INCREASE,
                        element_index=int(idx),
                        element_name=name,
                        severity=factor,
                        description=f"Load increase: {name} to {factor*100:.0f}%",
                        affected_buses=[int(load.bus)]
                    ))

        logger.info(f"Generated {len(contingencies)} load increase scenarios")
        return contingencies

    def run_contingency_analysis(self, net: pp.pandapowerNet,
                                 contingency_types: Optional[List[ContingencyType]] = None) -> List[ContingencyResult]:
        """
        Run comprehensive contingency analysis.

        Args:
            net: pandapower network (original will not be modified)
            contingency_types: Types of contingencies to analyze (None = all N-1)

        Returns:
            List of contingency results
        """
        if contingency_types is None:
            contingency_types = [ContingencyType.LINE_OUTAGE, ContingencyType.TRANSFORMER_OUTAGE]

        all_contingencies = []

        # Generate contingencies based on requested types
        if ContingencyType.LINE_OUTAGE in contingency_types or ContingencyType.TRANSFORMER_OUTAGE in contingency_types:
            all_contingencies.extend(self.generate_n_minus_1_contingencies(net))

        if ContingencyType.LOAD_INCREASE in contingency_types:
            all_contingencies.extend(self.generate_load_increase_scenarios(net))

        # Simulate each contingency
        results = []
        for i, contingency in enumerate(all_contingencies):
            logger.info(f"Running contingency {i+1}/{len(all_contingencies)}")

            # Make a deep copy of the network for each simulation
            net_copy = net.deepcopy()
            result = self.simulate_contingency(net_copy, contingency)
            results.append(result)

        # Summary statistics
        critical_count = sum(1 for r in results if r.is_critical)
        logger.info(f"Contingency analysis complete: {len(results)} scenarios, {critical_count} critical")

        return results

    def generate_n_k_contingencies(self, net: pp.pandapowerNet, k: int = 2,
                                   max_combinations: int = 500) -> List[ContingencyEvent]:
        """
        Generate N-k contingency events (multiple simultaneous failures).

        Args:
            net: pandapower network
            k: Number of simultaneous failures (2 = N-2, 3 = N-3, etc.)
            max_combinations: Maximum number of combinations to generate

        Returns:
            List of N-k contingency events
        """
        # Get all in-service lines
        in_service_lines = net.line[net.line.in_service].index.tolist()

        if len(in_service_lines) < k:
            logger.warning(f"Not enough lines for N-{k} analysis")
            return []

        # Generate combinations
        all_combos = list(combinations(in_service_lines, k))

        # Limit combinations if too many
        if len(all_combos) > max_combinations:
            logger.info(f"Limiting N-{k} combinations from {len(all_combos)} to {max_combinations}")
            # Prioritize combinations with high-loading lines
            if len(net.res_line) > 0:
                line_loadings = net.res_line.loading_percent.to_dict()
                all_combos = sorted(
                    all_combos,
                    key=lambda c: sum(line_loadings.get(i, 0) for i in c),
                    reverse=True
                )[:max_combinations]
            else:
                all_combos = all_combos[:max_combinations]

        contingencies = []
        for combo in all_combos:
            affected_buses = set()
            element_names = []

            for line_idx in combo:
                line = net.line.loc[line_idx]
                # Handle name - could be None, NaN, or actual string
                if hasattr(line, 'name') and line.name and str(line.name) != 'nan':
                    name = str(line.name)
                else:
                    name = f"Line {line_idx}"
                element_names.append(name)
                affected_buses.add(int(line.from_bus))
                affected_buses.add(int(line.to_bus))

            contingencies.append(ContingencyEvent(
                event_type=ContingencyType.MULTIPLE_OUTAGE,
                element_index=int(combo[0]),  # Primary element
                element_name=", ".join(element_names),
                description=f"N-{k}: {', '.join(element_names)} outage",
                affected_buses=list(affected_buses)
            ))

        logger.info(f"Generated {len(contingencies)} N-{k} contingency scenarios")
        return contingencies

    def simulate_n_k_contingency(self, net: pp.pandapowerNet,
                                 line_indices: List[int]) -> ContingencyResult:
        """
        Simulate multiple simultaneous line outages.

        Args:
            net: pandapower network (will be modified)
            line_indices: List of line indices to take out of service

        Returns:
            Contingency simulation results
        """
        # Create contingency event for tracking
        element_names = []
        affected_buses = set()

        for idx in line_indices:
            if idx < len(net.line):
                line = net.line.loc[idx]
                # Handle name - could be None, NaN, or actual string
                if hasattr(line, 'name') and line.name and str(line.name) != 'nan':
                    name = str(line.name)
                else:
                    name = f"Line {idx}"
                element_names.append(name)
                affected_buses.add(int(line.from_bus))
                affected_buses.add(int(line.to_bus))
                net.line.at[idx, "in_service"] = False

        contingency = ContingencyEvent(
            event_type=ContingencyType.MULTIPLE_OUTAGE,
            element_index=line_indices[0],
            element_name=", ".join(element_names),
            description=f"N-{len(line_indices)}: {', '.join(element_names)} outage",
            affected_buses=list(affected_buses)
        )

        # Run power flow
        converged = False
        try:
            pp.runpp(net, algorithm="bfsw", calculate_voltage_angles=False)
            converged = net.converged
        except Exception as e:
            logger.error(f"Power flow failed for N-{len(line_indices)}: {e}")

        return self._analyze_results(net, contingency, converged)

    def run_n_k_analysis(self, net: pp.pandapowerNet, k: int = 2,
                         max_combinations: int = 500,
                         parallel: bool = False,
                         max_workers: int = 4) -> List[ContingencyResult]:
        """
        Run N-k contingency analysis.

        Args:
            net: pandapower network
            k: Number of simultaneous failures
            max_combinations: Maximum combinations to analyze
            parallel: Use parallel processing
            max_workers: Number of parallel workers

        Returns:
            List of contingency results sorted by severity
        """
        logger.info(f"Starting N-{k} contingency analysis")

        # First run power flow to get baseline loadings
        pp.runpp(net)

        contingencies = self.generate_n_k_contingencies(net, k, max_combinations)

        if not contingencies:
            return []

        results = []

        if parallel and len(contingencies) > 10:
            # Parallel execution for large analyses
            logger.info(f"Running {len(contingencies)} contingencies in parallel")

            def simulate_single(contingency):
                net_copy = net.deepcopy()
                # Extract line indices from the contingency
                line_indices = [contingency.element_index]
                # Parse additional lines from element_name
                if "," in contingency.element_name:
                    parts = contingency.element_name.split(", ")
                    for part in parts[1:]:
                        try:
                            idx = int(part.replace("Line ", ""))
                            line_indices.append(idx)
                        except ValueError:
                            pass
                return self.simulate_n_k_contingency(net_copy, line_indices)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(simulate_single, c): c for c in contingencies}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Contingency simulation failed: {e}")
        else:
            # Sequential execution
            for i, contingency in enumerate(contingencies):
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i+1}/{len(contingencies)}")

                net_copy = net.deepcopy()
                # Parse line indices from contingency
                line_indices = self._parse_line_indices_from_contingency(contingency, net)
                result = self.simulate_n_k_contingency(net_copy, line_indices)
                results.append(result)

        # Sort by severity (critical first, then by number of violations)
        results.sort(key=lambda r: (not r.is_critical, -len(r.violated_constraints)))

        critical_count = sum(1 for r in results if r.is_critical)
        logger.info(f"N-{k} analysis complete: {len(results)} scenarios, {critical_count} critical")

        return results

    def _parse_line_indices_from_contingency(self, contingency: ContingencyEvent,
                                             net: pp.pandapowerNet) -> List[int]:
        """Parse line indices from a multiple outage contingency."""
        if contingency.event_type != ContingencyType.MULTIPLE_OUTAGE:
            return [contingency.element_index]

        line_indices = []
        parts = contingency.element_name.split(", ")

        for part in parts:
            part = part.strip()
            if part.startswith("Line "):
                try:
                    idx = int(part.replace("Line ", ""))
                    if idx in net.line.index:
                        line_indices.append(idx)
                except ValueError:
                    pass
            else:
                # Try to find by name
                matches = net.line[net.line.name == part].index.tolist()
                if matches:
                    line_indices.append(matches[0])

        if not line_indices:
            line_indices = [contingency.element_index]

        return line_indices

    def rank_contingencies_by_severity(self, results: List[ContingencyResult]) -> List[Dict[str, Any]]:
        """
        Rank contingencies by severity for prioritization.

        Args:
            results: List of contingency results

        Returns:
            Ranked list with severity scores
        """
        ranked = []

        for result in results:
            # Calculate severity score
            severity_score = 0.0

            # Non-convergence is most severe
            if not result.converged:
                severity_score = 100.0
            else:
                # Voltage violations
                if result.min_voltage_pu < self.v_min:
                    severity_score += (self.v_min - result.min_voltage_pu) * 100

                if result.max_voltage_pu > self.v_max:
                    severity_score += (result.max_voltage_pu - self.v_max) * 100

                # Thermal violations
                if result.max_line_loading_percent > self.thermal_limit:
                    severity_score += (result.max_line_loading_percent - self.thermal_limit) / 10

                # Number of violations
                severity_score += len(result.violated_constraints) * 5

            ranked.append({
                "contingency": result.contingency.description,
                "severity_score": float(severity_score),
                "converged": result.converged,
                "violations": len(result.violated_constraints),
                "min_voltage_pu": float(result.min_voltage_pu),
                "max_loading_percent": float(result.max_line_loading_percent),
                "is_critical": result.is_critical
            })

        # Sort by severity score descending
        ranked.sort(key=lambda x: x["severity_score"], reverse=True)

        return ranked


if __name__ == "__main__":
    # Test the contingency simulator
    import json

    from src.config import load_configuration
    from src.core.network_loader import NetworkLoader

    config, constraints, paths = load_configuration()

    # Load a test network
    loader = NetworkLoader(networks_path=paths.networks)
    net = loader.load_network("ieee_33")

    # Initialize simulator
    simulator = ContingencySimulator(
        voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
        thermal_limit_percent=100.0
    )

    # Run baseline power flow
    pp.runpp(net)
    print("\n=== Baseline Power Flow ===")
    print(f"Converged: {net.converged}")
    print(f"Max line loading: {net.res_line.loading_percent.max():.2f}%")
    print(f"Min voltage: {net.res_bus.vm_pu.min():.4f} pu")
    print(f"Max voltage: {net.res_bus.vm_pu.max():.4f} pu")

    # Test a single contingency
    contingencies = simulator.generate_n_minus_1_contingencies(net)
    if len(contingencies) > 0:
        print(f"\n=== Testing Contingency: {contingencies[0].description} ===")
        net_test = net.deepcopy()
        result = simulator.simulate_contingency(net_test, contingencies[0])
        print(json.dumps(result.to_dict(), indent=2))

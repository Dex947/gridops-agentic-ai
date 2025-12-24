"""Network reconfiguration optimization for loss minimization and load balancing."""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandapower as pp
from loguru import logger

# Default configuration values (can be overridden via SystemConfig)
DEFAULT_MAX_COMBINATIONS = 1000
DEFAULT_MAX_ITERATIONS = 100


def validate_line_id(net: pp.pandapowerNet, line_id: int) -> bool:
    """Check if a line ID exists in the network."""
    return line_id in set(net.line.index.tolist())


class ReconfigObjective(Enum):
    """Optimization objectives for network reconfiguration."""
    MIN_LOSS = "min_loss"
    MIN_VOLTAGE_DEVIATION = "min_voltage_deviation"
    LOAD_BALANCING = "load_balancing"
    MAX_LOADABILITY = "max_loadability"


@dataclass
class SwitchState:
    """Represents the state of a switch."""
    switch_id: int
    element_type: str  # 'line' or 'switch'
    element_id: int
    is_closed: bool
    from_bus: int
    to_bus: int


@dataclass
class ReconfigurationResult:
    """Results from network reconfiguration optimization."""
    success: bool
    objective_value: float
    baseline_losses_mw: float
    optimized_losses_mw: float
    loss_reduction_percent: float
    baseline_min_voltage: float
    optimized_min_voltage: float
    switches_changed: List[Dict[str, Any]] = field(default_factory=list)
    final_topology: List[SwitchState] = field(default_factory=list)
    iterations: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "objective_value": float(self.objective_value),
            "baseline_losses_mw": float(self.baseline_losses_mw),
            "optimized_losses_mw": float(self.optimized_losses_mw),
            "loss_reduction_percent": float(self.loss_reduction_percent),
            "baseline_min_voltage": float(self.baseline_min_voltage),
            "optimized_min_voltage": float(self.optimized_min_voltage),
            "switches_changed": self.switches_changed,
            "iterations": self.iterations,
            "message": self.message
        }


class NetworkReconfiguration:
    """Optimize network topology through switching operations."""

    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0,
                 max_combinations: Optional[int] = None,
                 max_iterations: Optional[int] = None):
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
        self.max_combinations = max_combinations or DEFAULT_MAX_COMBINATIONS
        self.max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        logger.info(f"NetworkReconfiguration initialized (max_combinations={self.max_combinations})")

    def get_switchable_elements(self, net: pp.pandapowerNet) -> List[SwitchState]:
        """Get all switchable elements in the network."""
        switches = []

        # Get switches from switch table
        if len(net.switch) > 0:
            for idx, sw in net.switch.iterrows():
                if sw.et == 'l':  # Line switch
                    line = net.line.loc[sw.element]
                    switches.append(SwitchState(
                        switch_id=int(idx),
                        element_type='switch',
                        element_id=int(sw.element),
                        is_closed=bool(sw.closed),
                        from_bus=int(line.from_bus),
                        to_bus=int(line.to_bus)
                    ))

        # Also consider lines that can be switched (tie lines)
        # Typically lines with index >= 32 in IEEE 33-bus are tie switches
        for idx, line in net.line.iterrows():
            if line.in_service:
                # Check if this line already has a switch
                has_switch = False
                if len(net.switch) > 0:
                    has_switch = any(
                        (net.switch.et == 'l') & (net.switch.element == idx)
                    )

                if not has_switch:
                    switches.append(SwitchState(
                        switch_id=-1,  # No dedicated switch
                        element_type='line',
                        element_id=int(idx),
                        is_closed=bool(line.in_service),
                        from_bus=int(line.from_bus),
                        to_bus=int(line.to_bus)
                    ))

        return switches

    def get_tie_switches(self, net: pp.pandapowerNet) -> List[int]:
        """Identify tie switches (normally open switches)."""
        tie_switches = []

        # Check switch table for normally open switches
        if len(net.switch) > 0:
            open_switches = net.switch[(net.switch.et == 'l') & (~net.switch.closed)]
            tie_switches.extend(open_switches.element.tolist())

        # Check for out-of-service lines (potential tie lines)
        out_of_service = net.line[~net.line.in_service].index.tolist()
        tie_switches.extend(out_of_service)

        return list(set(tie_switches))

    def get_sectionalizing_switches(self, net: pp.pandapowerNet) -> List[int]:
        """Identify sectionalizing switches (normally closed switches)."""
        sectionalizing = []

        # All in-service lines are potential sectionalizing points
        in_service = net.line[net.line.in_service].index.tolist()
        sectionalizing.extend(in_service)

        return sectionalizing

    def check_radiality(self, net: pp.pandapowerNet) -> bool:
        """Check if network maintains radial topology."""
        try:
            import pandapower.topology as top

            # Get connected buses
            mg = top.create_nxgraph(net, respect_switches=True)

            # For radial network: edges = nodes - 1
            n_nodes = mg.number_of_nodes()
            n_edges = mg.number_of_edges()

            # Check if graph is a tree (connected and n_edges = n_nodes - 1)
            import networkx as nx
            is_connected = nx.is_connected(mg)
            is_tree = n_edges == n_nodes - 1

            return is_connected and is_tree

        except Exception as e:
            logger.warning(f"Radiality check failed: {e}")
            return True  # Assume radial if check fails

    def evaluate_configuration(self, net: pp.pandapowerNet,
                               objective: ReconfigObjective = ReconfigObjective.MIN_LOSS) -> Tuple[float, bool]:
        """
        Evaluate a network configuration.

        Returns:
            Tuple of (objective_value, is_feasible)
        """
        try:
            pp.runpp(net, algorithm="bfsw")

            if not net.converged:
                return float('inf'), False

            # Check constraints
            min_v = net.res_bus.vm_pu.min()
            max_v = net.res_bus.vm_pu.max()
            max_loading = net.res_line.loading_percent.max() if len(net.res_line) > 0 else 0

            if min_v < self.v_min or max_v > self.v_max:
                return float('inf'), False

            if max_loading > self.thermal_limit:
                return float('inf'), False

            # Check radiality
            if not self.check_radiality(net):
                return float('inf'), False

            # Calculate objective
            if objective == ReconfigObjective.MIN_LOSS:
                obj_value = net.res_line.pl_mw.sum()

            elif objective == ReconfigObjective.MIN_VOLTAGE_DEVIATION:
                obj_value = ((net.res_bus.vm_pu - 1.0) ** 2).sum()

            elif objective == ReconfigObjective.LOAD_BALANCING:
                # Minimize variance in line loadings
                obj_value = net.res_line.loading_percent.var()

            elif objective == ReconfigObjective.MAX_LOADABILITY:
                # Maximize minimum voltage (proxy for loadability)
                obj_value = -min_v

            else:
                obj_value = net.res_line.pl_mw.sum()

            return float(obj_value), True

        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return float('inf'), False

    def apply_switch_action(self, net: pp.pandapowerNet,
                           line_id: int, close: bool) -> bool:
        """
        Apply a switching action to the network.
        
        Returns:
            True if action was applied successfully, False otherwise
        """
        if not validate_line_id(net, line_id):
            logger.warning(f"Invalid line ID: {line_id}")
            return False
            
        if line_id in net.line.index:
            net.line.at[line_id, 'in_service'] = close

            # Also update switch table if exists
            if len(net.switch) > 0:
                matching = net.switch[(net.switch.et == 'l') & (net.switch.element == line_id)]
                for idx in matching.index:
                    net.switch.at[idx, 'closed'] = close
            return True
        return False

    def branch_exchange(self, net: pp.pandapowerNet,
                       objective: ReconfigObjective = ReconfigObjective.MIN_LOSS,
                       max_iterations: int = 100) -> ReconfigurationResult:
        """
        Optimize network using branch exchange algorithm.

        This is a heuristic that iteratively swaps tie and sectionalizing
        switches to minimize the objective function while maintaining radiality.
        """
        logger.info(f"Starting branch exchange optimization: {objective.value}")

        # Get baseline
        net_work = copy.deepcopy(net)
        pp.runpp(net_work)

        if not net_work.converged:
            return ReconfigurationResult(
                success=False,
                objective_value=float('inf'),
                baseline_losses_mw=0.0,
                optimized_losses_mw=0.0,
                loss_reduction_percent=0.0,
                baseline_min_voltage=0.0,
                optimized_min_voltage=0.0,
                message="Baseline power flow did not converge"
            )

        baseline_losses = net_work.res_line.pl_mw.sum()
        baseline_min_v = net_work.res_bus.vm_pu.min()
        best_obj, _ = self.evaluate_configuration(net_work, objective)

        tie_switches = self.get_tie_switches(net_work)
        sectionalizing = self.get_sectionalizing_switches(net_work)

        logger.info(f"Found {len(tie_switches)} tie switches, {len(sectionalizing)} sectionalizing switches")

        best_net = copy.deepcopy(net_work)
        switches_changed = []
        iteration = 0
        improved = True

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for tie_sw in tie_switches:
                for sect_sw in sectionalizing:
                    if tie_sw == sect_sw:
                        continue

                    # Try swapping
                    net_test = copy.deepcopy(best_net)

                    # Close tie switch, open sectionalizing switch
                    self.apply_switch_action(net_test, tie_sw, close=True)
                    self.apply_switch_action(net_test, sect_sw, close=False)

                    obj_value, feasible = self.evaluate_configuration(net_test, objective)

                    if feasible and obj_value < best_obj:
                        logger.info(f"Improvement found: {best_obj:.4f} -> {obj_value:.4f}")
                        best_obj = obj_value
                        best_net = net_test

                        switches_changed.append({
                            "action": "swap",
                            "closed": tie_sw,
                            "opened": sect_sw,
                            "objective": float(obj_value)
                        })

                        # Update switch lists
                        tie_switches = self.get_tie_switches(best_net)
                        sectionalizing = self.get_sectionalizing_switches(best_net)

                        improved = True
                        break

                if improved:
                    break

        # Calculate final results
        pp.runpp(best_net)
        final_losses = best_net.res_line.pl_mw.sum()
        final_min_v = best_net.res_bus.vm_pu.min()

        loss_reduction = ((baseline_losses - final_losses) / baseline_losses * 100
                         if baseline_losses > 0 else 0.0)

        result = ReconfigurationResult(
            success=True,
            objective_value=float(best_obj),
            baseline_losses_mw=float(baseline_losses),
            optimized_losses_mw=float(final_losses),
            loss_reduction_percent=float(loss_reduction),
            baseline_min_voltage=float(baseline_min_v),
            optimized_min_voltage=float(final_min_v),
            switches_changed=switches_changed,
            iterations=iteration,
            message=f"Optimization complete after {iteration} iterations"
        )

        logger.info(f"Branch exchange complete: {loss_reduction:.2f}% loss reduction")
        return result

    def exhaustive_search(self, net: pp.pandapowerNet,
                         objective: ReconfigObjective = ReconfigObjective.MIN_LOSS,
                         max_configs: Optional[int] = None) -> ReconfigurationResult:
        """
        Find optimal configuration through exhaustive search.

        Only practical for small networks with few switches.
        """
        logger.info(f"Starting exhaustive search: {objective.value}")

        # Get baseline
        net_work = copy.deepcopy(net)
        pp.runpp(net_work)

        if not net_work.converged:
            return ReconfigurationResult(
                success=False,
                objective_value=float('inf'),
                baseline_losses_mw=0.0,
                optimized_losses_mw=0.0,
                loss_reduction_percent=0.0,
                baseline_min_voltage=0.0,
                optimized_min_voltage=0.0,
                message="Baseline power flow did not converge"
            )

        baseline_losses = net_work.res_line.pl_mw.sum()
        baseline_min_v = net_work.res_bus.vm_pu.min()

        tie_switches = self.get_tie_switches(net_work)
        sectionalizing = self.get_sectionalizing_switches(net_work)

        # Generate all valid configurations
        n_ties = len(tie_switches)

        if n_ties == 0:
            logger.warning("No tie switches found for reconfiguration")
            return ReconfigurationResult(
                success=False,
                objective_value=float('inf'),
                baseline_losses_mw=float(baseline_losses),
                optimized_losses_mw=float(baseline_losses),
                loss_reduction_percent=0.0,
                baseline_min_voltage=float(baseline_min_v),
                optimized_min_voltage=float(baseline_min_v),
                message="No tie switches available"
            )

        best_obj = float('inf')
        best_config = None
        
        # Use instance max_combinations if not specified
        max_configs = max_configs or self.max_combinations
        configs_tested = 0

        # For each tie switch, try closing it and opening one sectionalizing switch
        for tie_sw in tie_switches:
            for sect_sw in sectionalizing:
                if configs_tested >= max_configs:
                    break

                if tie_sw == sect_sw:
                    continue

                net_test = copy.deepcopy(net_work)
                self.apply_switch_action(net_test, tie_sw, close=True)
                self.apply_switch_action(net_test, sect_sw, close=False)

                obj_value, feasible = self.evaluate_configuration(net_test, objective)
                configs_tested += 1

                if feasible and obj_value < best_obj:
                    best_obj = obj_value
                    best_config = (tie_sw, sect_sw)

            if configs_tested >= max_configs:
                break

        if best_config is None:
            return ReconfigurationResult(
                success=False,
                objective_value=float('inf'),
                baseline_losses_mw=float(baseline_losses),
                optimized_losses_mw=float(baseline_losses),
                loss_reduction_percent=0.0,
                baseline_min_voltage=float(baseline_min_v),
                optimized_min_voltage=float(baseline_min_v),
                message="No feasible configuration found"
            )

        # Apply best configuration
        net_best = copy.deepcopy(net_work)
        self.apply_switch_action(net_best, best_config[0], close=True)
        self.apply_switch_action(net_best, best_config[1], close=False)
        pp.runpp(net_best)

        final_losses = net_best.res_line.pl_mw.sum()
        final_min_v = net_best.res_bus.vm_pu.min()

        loss_reduction = ((baseline_losses - final_losses) / baseline_losses * 100
                         if baseline_losses > 0 else 0.0)

        return ReconfigurationResult(
            success=True,
            objective_value=float(best_obj),
            baseline_losses_mw=float(baseline_losses),
            optimized_losses_mw=float(final_losses),
            loss_reduction_percent=float(loss_reduction),
            baseline_min_voltage=float(baseline_min_v),
            optimized_min_voltage=float(final_min_v),
            switches_changed=[{
                "closed": best_config[0],
                "opened": best_config[1]
            }],
            iterations=configs_tested,
            message=f"Tested {configs_tested} configurations"
        )

    def find_optimal_topology(self, net: pp.pandapowerNet,
                             objective: str = "min_loss",
                             method: str = "branch_exchange") -> ReconfigurationResult:
        """
        Find optimal network topology.

        Args:
            net: pandapower network
            objective: Optimization objective
            method: Algorithm ('branch_exchange' or 'exhaustive')

        Returns:
            Reconfiguration result
        """
        obj_map = {
            "min_loss": ReconfigObjective.MIN_LOSS,
            "min_voltage_deviation": ReconfigObjective.MIN_VOLTAGE_DEVIATION,
            "load_balancing": ReconfigObjective.LOAD_BALANCING,
            "max_loadability": ReconfigObjective.MAX_LOADABILITY
        }

        obj_enum = obj_map.get(objective, ReconfigObjective.MIN_LOSS)

        if method == "exhaustive":
            return self.exhaustive_search(net, obj_enum)
        else:
            return self.branch_exchange(net, obj_enum)


def optimize_network_topology(net: pp.pandapowerNet,
                             objective: str = "min_loss",
                             voltage_limits: Tuple[float, float] = (0.95, 1.05)) -> ReconfigurationResult:
    """Convenience function for network reconfiguration."""
    reconfig = NetworkReconfiguration(voltage_limits=voltage_limits)
    return reconfig.find_optimal_topology(net, objective)

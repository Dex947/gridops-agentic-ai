"""
Power flow tools for agent use.
Provides callable functions for power system analysis and modification.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandapower as pp
import numpy as np
from loguru import logger
from dataclasses import dataclass


@dataclass
class PowerFlowResults:
    """Structured power flow results."""
    converged: bool
    max_line_loading_percent: float
    min_voltage_pu: float
    max_voltage_pu: float
    total_losses_mw: float
    violations: List[str]
    bus_voltages: Dict[int, float]
    line_loadings: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "converged": self.converged,
            "max_line_loading_percent": float(self.max_line_loading_percent),
            "min_voltage_pu": float(self.min_voltage_pu),
            "max_voltage_pu": float(self.max_voltage_pu),
            "total_losses_mw": float(self.total_losses_mw),
            "violations": self.violations,
            "bus_voltages": {int(k): float(v) for k, v in self.bus_voltages.items()},
            "line_loadings": {int(k): float(v) for k, v in self.line_loadings.items()}
        }
    
    def get_summary_text(self) -> str:
        """Get human-readable summary."""
        status = "✓ Converged" if self.converged else "✗ Did not converge"
        text = f"""Power Flow Results:
Status: {status}
Voltage Range: {self.min_voltage_pu:.4f} to {self.max_voltage_pu:.4f} pu
Max Line Loading: {self.max_line_loading_percent:.2f}%
Total Losses: {self.total_losses_mw:.3f} MW
Violations: {len(self.violations)}
"""
        if self.violations:
            text += "\nConstraint Violations:\n"
            for v in self.violations[:10]:  # Limit to 10
                text += f"  - {v}\n"
        
        return text


class PowerFlowTool:
    """Tool for running power flow analysis."""
    
    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        """
        Initialize power flow tool.
        
        Args:
            voltage_limits: (min, max) voltage in per-unit
            thermal_limit_percent: Maximum allowed line loading
        """
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
    
    def run(self, net: pp.pandapowerNet, 
            algorithm: str = "bfsw") -> PowerFlowResults:
        """
        Run power flow analysis.
        
        Args:
            net: pandapower network
            algorithm: Power flow algorithm ("bfsw", "nr", "gs")
        
        Returns:
            Power flow results
        """
        try:
            pp.runpp(net, algorithm=algorithm, calculate_voltage_angles=False)
            converged = net.converged
            
            if not converged:
                logger.warning("Power flow did not converge")
        except Exception as e:
            logger.error(f"Power flow execution failed: {e}")
            converged = False
        
        return self._extract_results(net, converged)
    
    def _extract_results(self, net: pp.pandapowerNet, 
                        converged: bool) -> PowerFlowResults:
        """Extract and analyze power flow results."""
        violations = []
        
        # Initialize default values
        max_loading = 0.0
        min_voltage = 1.0
        max_voltage = 1.0
        total_losses = 0.0
        bus_voltages = {}
        line_loadings = {}
        
        if converged:
            # Extract bus voltages
            if len(net.res_bus) > 0:
                min_voltage = net.res_bus.vm_pu.min()
                max_voltage = net.res_bus.vm_pu.max()
                bus_voltages = net.res_bus.vm_pu.to_dict()
                
                # Check voltage violations
                for idx, row in net.res_bus.iterrows():
                    if row.vm_pu < self.v_min:
                        violations.append(f"Bus {idx}: {row.vm_pu:.4f} pu (< {self.v_min} pu)")
                    elif row.vm_pu > self.v_max:
                        violations.append(f"Bus {idx}: {row.vm_pu:.4f} pu (> {self.v_max} pu)")
            
            # Extract line loadings
            if len(net.res_line) > 0:
                max_loading = net.res_line.loading_percent.max()
                line_loadings = net.res_line.loading_percent.to_dict()
                
                # Check thermal violations
                for idx, row in net.res_line.iterrows():
                    if row.loading_percent > self.thermal_limit:
                        violations.append(f"Line {idx}: {row.loading_percent:.2f}% loading (> {self.thermal_limit}%)")
                
                # Calculate losses
                total_losses = net.res_line.pl_mw.sum()
        else:
            violations.append("Power flow did not converge")
        
        return PowerFlowResults(
            converged=converged,
            max_line_loading_percent=max_loading,
            min_voltage_pu=min_voltage,
            max_voltage_pu=max_voltage,
            total_losses_mw=total_losses,
            violations=violations,
            bus_voltages=bus_voltages,
            line_loadings=line_loadings
        )


def run_powerflow_analysis(net: pp.pandapowerNet,
                          voltage_limits: Tuple[float, float] = (0.95, 1.05),
                          thermal_limit: float = 100.0) -> Dict[str, Any]:
    """
    Standalone function to run power flow analysis.
    Designed to be called by LLM agents.
    
    Args:
        net: pandapower network
        voltage_limits: (min, max) voltage in per-unit
        thermal_limit: Maximum allowed line loading percentage
    
    Returns:
        Dictionary with power flow results
    """
    tool = PowerFlowTool(voltage_limits=voltage_limits, thermal_limit_percent=thermal_limit)
    results = tool.run(net)
    return results.to_dict()


def apply_switching_action(net: pp.pandapowerNet,
                          line_switches: Optional[List[Dict[str, Any]]] = None,
                          explicit_switches: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Apply switching actions to the network.
    
    Args:
        net: pandapower network (will be modified)
        line_switches: List of line switching actions
            Format: [{"line_id": int, "close": bool}, ...]
        explicit_switches: List of explicit switch actions
            Format: [{"switch_id": int, "close": bool}, ...]
    
    Returns:
        Dictionary with applied actions and results
    """
    applied_actions = []
    
    try:
        # Apply line switches
        if line_switches:
            for action in line_switches:
                line_id = action["line_id"]
                close = action["close"]
                
                if line_id < len(net.line):
                    net.line.at[line_id, "in_service"] = close
                    applied_actions.append({
                        "type": "line_switch",
                        "line_id": line_id,
                        "action": "closed" if close else "opened",
                        "success": True
                    })
                    logger.info(f"Line {line_id} {'closed' if close else 'opened'}")
                else:
                    logger.warning(f"Invalid line ID: {line_id}")
                    applied_actions.append({
                        "type": "line_switch",
                        "line_id": line_id,
                        "success": False,
                        "error": "Invalid line ID"
                    })
        
        # Apply explicit switches
        if explicit_switches:
            for action in explicit_switches:
                switch_id = action["switch_id"]
                close = action["close"]
                
                if switch_id < len(net.switch):
                    net.switch.at[switch_id, "closed"] = close
                    applied_actions.append({
                        "type": "explicit_switch",
                        "switch_id": switch_id,
                        "action": "closed" if close else "opened",
                        "success": True
                    })
                    logger.info(f"Switch {switch_id} {'closed' if close else 'opened'}")
                else:
                    logger.warning(f"Invalid switch ID: {switch_id}")
                    applied_actions.append({
                        "type": "explicit_switch",
                        "switch_id": switch_id,
                        "success": False,
                        "error": "Invalid switch ID"
                    })
        
        return {
            "success": True,
            "applied_actions": applied_actions,
            "total_actions": len(applied_actions)
        }
    
    except Exception as e:
        logger.error(f"Error applying switching actions: {e}")
        return {
            "success": False,
            "error": str(e),
            "applied_actions": applied_actions
        }


def apply_load_shedding(net: pp.pandapowerNet,
                       load_reductions: List[Dict[str, Any]],
                       max_shed_percent: float = 30.0) -> Dict[str, Any]:
    """
    Apply load shedding to the network.
    
    Args:
        net: pandapower network (will be modified)
        load_reductions: List of load reduction actions
            Format: [{"load_id": int, "reduction_percent": float}, ...]
        max_shed_percent: Maximum allowed load shedding per load
    
    Returns:
        Dictionary with applied load shedding and totals
    """
    applied_reductions = []
    total_p_shed = 0.0
    total_q_shed = 0.0
    
    try:
        for action in load_reductions:
            load_id = action["load_id"]
            reduction = min(action["reduction_percent"], max_shed_percent)
            
            if load_id < len(net.load):
                # Get original load
                original_p = net.load.at[load_id, "p_mw"]
                original_q = net.load.at[load_id, "q_mvar"]
                
                # Calculate reduction
                reduction_factor = 1.0 - (reduction / 100.0)
                new_p = original_p * reduction_factor
                new_q = original_q * reduction_factor
                
                # Apply reduction
                net.load.at[load_id, "p_mw"] = new_p
                net.load.at[load_id, "q_mvar"] = new_q
                
                p_shed = original_p - new_p
                q_shed = original_q - new_q
                
                total_p_shed += p_shed
                total_q_shed += q_shed
                
                applied_reductions.append({
                    "load_id": load_id,
                    "reduction_percent": reduction,
                    "p_shed_mw": float(p_shed),
                    "q_shed_mvar": float(q_shed),
                    "success": True
                })
                logger.info(f"Load {load_id} reduced by {reduction:.1f}% ({p_shed:.3f} MW)")
            else:
                logger.warning(f"Invalid load ID: {load_id}")
                applied_reductions.append({
                    "load_id": load_id,
                    "success": False,
                    "error": "Invalid load ID"
                })
        
        return {
            "success": True,
            "applied_reductions": applied_reductions,
            "total_p_shed_mw": float(total_p_shed),
            "total_q_shed_mvar": float(total_q_shed),
            "num_loads_affected": len(applied_reductions)
        }
    
    except Exception as e:
        logger.error(f"Error applying load shedding: {e}")
        return {
            "success": False,
            "error": str(e),
            "applied_reductions": applied_reductions
        }


def validate_action_feasibility(net: pp.pandapowerNet,
                               action_plan: Dict[str, Any],
                               voltage_limits: Tuple[float, float] = (0.95, 1.05),
                               thermal_limit: float = 100.0) -> Dict[str, Any]:
    """
    Validate if a proposed action plan is feasible.
    
    Args:
        net: pandapower network (will be copied, not modified)
        action_plan: Dictionary containing proposed actions
        voltage_limits: Voltage limits for validation
        thermal_limit: Thermal limit for validation
    
    Returns:
        Validation results with feasibility determination
    """
    # Create a copy for testing
    net_test = net.deepcopy()
    
    try:
        # Apply proposed actions
        if "line_switches" in action_plan:
            result = apply_switching_action(net_test, line_switches=action_plan["line_switches"])
            if not result["success"]:
                return {
                    "feasible": False,
                    "reason": "Failed to apply switching actions",
                    "details": result
                }
        
        if "load_reductions" in action_plan:
            result = apply_load_shedding(net_test, load_reductions=action_plan["load_reductions"])
            if not result["success"]:
                return {
                    "feasible": False,
                    "reason": "Failed to apply load shedding",
                    "details": result
                }
        
        # Run power flow
        pf_results = run_powerflow_analysis(net_test, voltage_limits, thermal_limit)
        
        # Determine feasibility
        feasible = pf_results["converged"] and len(pf_results["violations"]) == 0
        
        return {
            "feasible": feasible,
            "powerflow_results": pf_results,
            "reason": "Action plan is feasible" if feasible else "Constraints violated or non-convergence"
        }
    
    except Exception as e:
        logger.error(f"Error validating action feasibility: {e}")
        return {
            "feasible": False,
            "reason": f"Validation error: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test power flow tools
    from src.config import load_configuration
    from src.core.network_loader import NetworkLoader
    import json
    
    config, constraints, paths = load_configuration()
    
    # Load network
    loader = NetworkLoader(networks_path=paths.networks)
    net = loader.load_network("ieee_33")
    
    # Test power flow analysis
    print("\n=== Power Flow Analysis ===")
    pf_tool = PowerFlowTool(
        voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
        thermal_limit_percent=100.0
    )
    results = pf_tool.run(net)
    print(results.get_summary_text())
    
    # Test switching action
    print("\n=== Test Switching Action ===")
    net_copy = net.deepcopy()
    switch_result = apply_switching_action(
        net_copy,
        line_switches=[{"line_id": 5, "close": False}]
    )
    print(json.dumps(switch_result, indent=2))
    
    # Run power flow after switching
    results_after = pf_tool.run(net_copy)
    print("\nPower Flow After Switching:")
    print(results_after.get_summary_text())

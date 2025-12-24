"""PowerFlow User Agent for simulation and network modifications."""

from typing import Any, Dict

import pandapower as pp
from loguru import logger

from src.core.state_manager import ActionProposal
from src.tools.powerflow_tools import PowerFlowTool


class PowerFlowUserAgent:
    """Runs power flow and applies switching/load-shedding actions."""

    def __init__(self, voltage_limits: tuple = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        self.voltage_limits = voltage_limits
        self.thermal_limit = thermal_limit_percent
        self.pf_tool = PowerFlowTool(voltage_limits, thermal_limit_percent)

        logger.info("PowerFlowUserAgent initialized")

    def run_baseline_analysis(self, net: pp.pandapowerNet) -> Dict[str, Any]:
        """Run power flow on healthy network."""
        logger.info("Running baseline power flow analysis...")

        try:
            results = self.pf_tool.run(net)

            baseline = {
                "status": "success",
                "converged": results.converged,
                "max_line_loading_percent": results.max_line_loading_percent,
                "min_voltage_pu": results.min_voltage_pu,
                "max_voltage_pu": results.max_voltage_pu,
                "total_losses_mw": results.total_losses_mw,
                "violations": results.violations,
                "summary": results.get_summary_text()
            }

            logger.info("Baseline analysis complete")
            return baseline

        except Exception as e:
            logger.error(f"Baseline analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "converged": False
            }

    def simulate_contingency(self, net: pp.pandapowerNet,
                           contingency_desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a contingency event and analyze results.

        Args:
            net: pandapower network (will be modified)
            contingency_desc: Contingency description with element info

        Returns:
            Contingency simulation results
        """
        logger.info(f"Simulating contingency: {contingency_desc.get('description', 'N/A')}")

        try:
            # Apply contingency
            self._apply_contingency(net, contingency_desc)

            # Run power flow
            results = self.pf_tool.run(net)

            contingency_results = {
                "status": "success",
                "converged": results.converged,
                "max_line_loading_percent": results.max_line_loading_percent,
                "min_voltage_pu": results.min_voltage_pu,
                "max_voltage_pu": results.max_voltage_pu,
                "total_losses_mw": results.total_losses_mw,
                "violations": results.violations,
                "num_violations": len(results.violations),
                "summary": results.get_summary_text()
            }

            logger.info(f"Contingency simulation complete: {len(results.violations)} violations")
            return contingency_results

        except Exception as e:
            logger.error(f"Contingency simulation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "converged": False,
                "violations": ["Simulation failed"]
            }

    def _apply_contingency(self, net: pp.pandapowerNet, contingency: Dict[str, Any]):
        """Apply contingency to network."""
        cont_type = contingency.get('type', '')
        elements = contingency.get('elements', [])

        if cont_type == 'line_outage':
            for line_id in elements:
                if line_id < len(net.line):
                    net.line.at[line_id, 'in_service'] = False
                    logger.debug(f"Line {line_id} taken out of service")

        elif cont_type == 'transformer_outage':
            for trafo_id in elements:
                if trafo_id < len(net.trafo):
                    net.trafo.at[trafo_id, 'in_service'] = False
                    logger.debug(f"Transformer {trafo_id} taken out of service")

        elif cont_type == 'load_increase':
            factor = contingency.get('factor', 1.5)
            for load_id in elements:
                if load_id < len(net.load):
                    net.load.at[load_id, 'p_mw'] *= factor
                    net.load.at[load_id, 'q_mvar'] *= factor
                    logger.debug(f"Load {load_id} increased by factor {factor}")

    def evaluate_action_plan(self, net: pp.pandapowerNet,
                            action: ActionProposal) -> Dict[str, Any]:
        """
        Evaluate an action plan by simulating its effects.

        Args:
            net: pandapower network (will be copied, not modified)
            action: Action proposal to evaluate

        Returns:
            Evaluation results including power flow outcomes
        """
        logger.info(f"Evaluating action plan: {action.action_id}")

        # Create a copy for testing
        net_test = net.deepcopy()

        try:
            # Apply the proposed action
            applied = self._apply_action(net_test, action)

            if not applied['success']:
                return {
                    "action_id": action.action_id,
                    "feasible": False,
                    "reason": "Failed to apply action",
                    "details": applied
                }

            # Run power flow after action
            results = self.pf_tool.run(net_test)

            evaluation = {
                "action_id": action.action_id,
                "feasible": results.converged,
                "converged": results.converged,
                "max_line_loading_percent": results.max_line_loading_percent,
                "min_voltage_pu": results.min_voltage_pu,
                "max_voltage_pu": results.max_voltage_pu,
                "violations": results.violations,
                "num_violations": len(results.violations),
                "violations_resolved": results.converged and len(results.violations) == 0,
                "applied_actions": applied,
                "summary": results.get_summary_text()
            }

            logger.info(f"Action evaluation complete: feasible={evaluation['feasible']}, violations={evaluation['num_violations']}")
            return evaluation

        except Exception as e:
            logger.error(f"Action evaluation failed: {e}")
            return {
                "action_id": action.action_id,
                "feasible": False,
                "reason": f"Evaluation error: {str(e)}",
                "error": str(e)
            }

    def _apply_action(self, net: pp.pandapowerNet,
                     action: ActionProposal) -> Dict[str, Any]:
        """Apply an action proposal to the network."""
        results = {"success": True, "applied": []}

        try:
            for element in action.target_elements:
                elem_type = element.get('type', '')

                if elem_type == 'line_switch':
                    line_id = element.get('line_id')
                    close = element.get('close', True)

                    if line_id is not None and line_id < len(net.line):
                        net.line.at[line_id, 'in_service'] = close
                        results['applied'].append({
                            'type': 'line_switch',
                            'line_id': line_id,
                            'action': 'closed' if close else 'opened'
                        })
                        logger.debug(f"Line {line_id} {'closed' if close else 'opened'}")

                elif elem_type == 'load_shed':
                    load_id = element.get('load_id')
                    reduction_percent = element.get('reduction_percent', 0)

                    if load_id is not None and load_id < len(net.load):
                        reduction_factor = 1.0 - (reduction_percent / 100.0)
                        original_p = net.load.at[load_id, 'p_mw']
                        original_q = net.load.at[load_id, 'q_mvar']

                        net.load.at[load_id, 'p_mw'] = original_p * reduction_factor
                        net.load.at[load_id, 'q_mvar'] = original_q * reduction_factor

                        results['applied'].append({
                            'type': 'load_shed',
                            'load_id': load_id,
                            'reduction_percent': reduction_percent,
                            'p_shed_mw': original_p * (1 - reduction_factor)
                        })
                        logger.debug(f"Load {load_id} reduced by {reduction_percent}%")

                elif elem_type == 'switch':
                    switch_id = element.get('switch_id')
                    close = element.get('close', True)

                    if switch_id is not None and switch_id < len(net.switch):
                        net.switch.at[switch_id, 'closed'] = close
                        results['applied'].append({
                            'type': 'switch',
                            'switch_id': switch_id,
                            'action': 'closed' if close else 'opened'
                        })
                        logger.debug(f"Switch {switch_id} {'closed' if close else 'opened'}")

            return results

        except Exception as e:
            logger.error(f"Error applying action: {e}")
            return {"success": False, "error": str(e)}

    def compare_scenarios(self, baseline: Dict[str, Any],
                         post_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare baseline and post-action scenarios.

        Args:
            baseline: Baseline power flow results
            post_action: Post-action power flow results

        Returns:
            Comparison metrics
        """
        comparison = {
            "improvement": False,
            "metrics": {}
        }

        if baseline.get('converged') and post_action.get('converged'):
            # Calculate improvements
            voltage_improvement = (post_action.get('min_voltage_pu', 0) -
                                 baseline.get('min_voltage_pu', 0))

            loading_improvement = (baseline.get('max_line_loading_percent', 100) -
                                 post_action.get('max_line_loading_percent', 100))

            violations_improvement = (len(baseline.get('violations', [])) -
                                    len(post_action.get('violations', [])))

            comparison['metrics'] = {
                "voltage_improvement_pu": float(voltage_improvement),
                "loading_improvement_percent": float(loading_improvement),
                "violations_resolved": int(violations_improvement),
                "losses_change_mw": float(post_action.get('total_losses_mw', 0) -
                                        baseline.get('total_losses_mw', 0))
            }

            # Determine if overall improvement
            comparison['improvement'] = (violations_improvement > 0 or
                                       (voltage_improvement > 0 and violations_improvement >= 0))

        logger.info(f"Scenario comparison: improvement={comparison['improvement']}")
        return comparison


if __name__ == "__main__":
    # Test PowerFlow User Agent
    import json

    from src.config import load_configuration
    from src.core.network_loader import NetworkLoader
    from src.core.state_manager import ActionProposal

    config, constraints, paths = load_configuration()

    # Load network
    loader = NetworkLoader(networks_path=paths.networks)
    net = loader.load_network("ieee_33")

    # Initialize agent
    agent = PowerFlowUserAgent(
        voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
        thermal_limit_percent=100.0
    )

    # Test baseline analysis
    print("\n=== Baseline Analysis ===")
    baseline = agent.run_baseline_analysis(net)
    print(baseline['summary'])

    # Test contingency simulation
    print("\n=== Contingency Simulation ===")
    net_cont = net.deepcopy()
    contingency = {
        'type': 'line_outage',
        'elements': [5],
        'description': 'Line 5 outage'
    }
    cont_results = agent.simulate_contingency(net_cont, contingency)
    print(cont_results['summary'])

    # Test action evaluation
    print("\n=== Action Evaluation ===")
    test_action = ActionProposal(
        action_id="test_action",
        action_type="switch_line",
        target_elements=[
            {"type": "line_switch", "line_id": 20, "close": True}
        ],
        expected_impact="Restore connectivity",
        priority=8,
        proposed_by="Test"
    )

    evaluation = agent.evaluate_action_plan(net_cont, test_action)
    print(json.dumps(evaluation, indent=2))

"""Constraint Checker Agent for validating power system limits."""

from typing import Any, Dict, List, Tuple

import pandapower as pp
from loguru import logger

from src.config import NetworkConstraints
from src.core.state_manager import ActionEvaluation


class ConstraintCheckerAgent:
    """Validates voltage, thermal, and protection constraints."""

    def __init__(self, constraints: NetworkConstraints):
        self.constraints = constraints
        self.v_min = constraints.v_min_pu
        self.v_max = constraints.v_max_pu
        self.thermal_margin = constraints.thermal_margin

        logger.info("ConstraintCheckerAgent initialized")

    def validate_action(self, evaluation_results: Dict[str, Any],
                       baseline_violations: List[str]) -> ActionEvaluation:
        """
        Validate an evaluated action against all constraints.

        Args:
            evaluation_results: Results from PowerFlowUserAgent evaluation
            baseline_violations: Original violations before action

        Returns:
            Action evaluation with recommendation
        """
        action_id = evaluation_results.get('action_id', 'unknown')
        logger.info(f"Validating action: {action_id}")

        # Check if action is feasible (power flow converged)
        feasible = evaluation_results.get('converged', False)

        if not feasible:
            return ActionEvaluation(
                action_id=action_id,
                feasible=False,
                powerflow_converged=False,
                violations_resolved=[],
                violations_remaining=baseline_violations,
                new_violations=[],
                performance_metrics={},
                safety_score=0.0,
                recommendation="reject",
                rationale="Power flow did not converge after applying action"
            )

        # Analyze constraint compliance
        current_violations = evaluation_results.get('violations', [])

        # Determine which violations were resolved
        violations_resolved = [v for v in baseline_violations if v not in current_violations]
        violations_remaining = [v for v in baseline_violations if v in current_violations]
        new_violations = [v for v in current_violations if v not in baseline_violations]

        # Calculate performance metrics
        performance = self._calculate_performance_metrics(evaluation_results)

        # Calculate safety score
        safety_score = self._calculate_safety_score(
            evaluation_results,
            violations_resolved,
            violations_remaining,
            new_violations
        )

        # Make recommendation
        recommendation, rationale = self._make_recommendation(
            safety_score,
            violations_resolved,
            violations_remaining,
            new_violations,
            performance
        )

        evaluation = ActionEvaluation(
            action_id=action_id,
            feasible=feasible,
            powerflow_converged=feasible,
            violations_resolved=violations_resolved,
            violations_remaining=violations_remaining,
            new_violations=new_violations,
            performance_metrics=performance,
            safety_score=safety_score,
            recommendation=recommendation,
            rationale=rationale
        )

        logger.info(f"Validation complete: {recommendation} (safety={safety_score:.2f})")
        return evaluation

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from power flow results."""
        metrics = {
            "min_voltage_pu": results.get('min_voltage_pu', 0.0),
            "max_voltage_pu": results.get('max_voltage_pu', 1.0),
            "max_line_loading_percent": results.get('max_line_loading_percent', 0.0),
            "voltage_margin_pu": min(
                results.get('min_voltage_pu', 1.0) - self.v_min,
                self.v_max - results.get('max_voltage_pu', 1.0)
            ),
            "thermal_margin_percent": 100.0 - results.get('max_line_loading_percent', 0.0)
        }

        return metrics

    def _calculate_safety_score(self, results: Dict[str, Any],
                                violations_resolved: List[str],
                                violations_remaining: List[str],
                                new_violations: List[str]) -> float:
        """
        Calculate safety score (0-1, higher = safer).

        Scoring criteria:
        - Constraint compliance (voltage, thermal)
        - Violations resolved vs. remaining
        - No new violations introduced
        - Operating margins
        """
        score = 0.0

        # Base score for convergence
        if results.get('converged', False):
            score += 0.2

        # Score for constraint compliance
        min_v = results.get('min_voltage_pu', 0.0)
        max_v = results.get('max_voltage_pu', 1.0)
        max_loading = results.get('max_line_loading_percent', 0.0)

        # Voltage compliance (0-0.3 points)
        if min_v >= self.v_min and max_v <= self.v_max:
            voltage_score = 0.3
            # Bonus for good margins
            margin = min(min_v - self.v_min, self.v_max - max_v)
            voltage_score += min(0.1, margin * 2)  # Up to 0.1 bonus
            score += voltage_score

        # Thermal compliance (0-0.2 points)
        thermal_limit = 100.0 + (self.thermal_margin * 100.0)
        if max_loading <= thermal_limit:
            thermal_score = 0.2
            # Bonus for good margin
            margin_percent = thermal_limit - max_loading
            thermal_score += min(0.1, margin_percent / 100.0)
            score += thermal_score

        # Violations resolution (0-0.3 points)
        if len(baseline_violations := violations_resolved + violations_remaining) > 0:
            resolution_ratio = len(violations_resolved) / len(baseline_violations)
            score += 0.3 * resolution_ratio
        else:
            score += 0.3  # No violations to begin with

        # Penalty for new violations
        if new_violations:
            score -= 0.2 * len(new_violations)

        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))

    def _make_recommendation(self, safety_score: float,
                            violations_resolved: List[str],
                            violations_remaining: List[str],
                            new_violations: List[str],
                            performance: Dict[str, float]) -> Tuple[str, str]:
        """
        Make recommendation based on validation results.

        Returns:
            (recommendation, rationale) tuple
        """
        # Reject if safety score is too low
        if safety_score < 0.4:
            return ("reject", f"Safety score {safety_score:.2f} below threshold (0.4). "
                             f"Action introduces unacceptable risk.")

        # Reject if new violations introduced
        if new_violations:
            return ("reject", f"Action introduces {len(new_violations)} new violations: "
                             f"{', '.join(new_violations[:3])}")

        # Approve if all violations resolved and good safety score
        if not violations_remaining and safety_score >= 0.7:
            return ("approve", f"All violations resolved. Safety score: {safety_score:.2f}. "
                              f"Adequate operating margins maintained.")

        # Approve if significant improvement and acceptable safety
        if len(violations_resolved) > len(violations_remaining) and safety_score >= 0.6:
            return ("approve", f"{len(violations_resolved)} violations resolved, "
                              f"{len(violations_remaining)} remaining. "
                              f"Safety score: {safety_score:.2f}. Acceptable improvement.")

        # Suggest modification if some violations remain but action is promising
        if violations_resolved and safety_score >= 0.5:
            return ("modify", f"Partial success: {len(violations_resolved)} violations resolved, "
                             f"{len(violations_remaining)} remaining. "
                             f"Consider additional actions or refinement.")

        # Default to reject
        return ("reject", f"Insufficient improvement. Safety score: {safety_score:.2f}. "
                         f"Only {len(violations_resolved)} violations resolved.")

    def check_n_minus_1_security(self, net: pp.pandapowerNet) -> Dict[str, Any]:
        """
        Check N-1 security after an action is applied.

        Args:
            net: pandapower network in post-action state

        Returns:
            N-1 security assessment
        """
        logger.info("Checking N-1 security...")

        from src.core.contingency_simulator import ContingencySimulator

        simulator = ContingencySimulator(
            voltage_limits=(self.v_min, self.v_max),
            thermal_limit_percent=100.0
        )

        # Generate N-1 contingencies
        contingencies = simulator.generate_n_minus_1_contingencies(net)

        # Test a sample of contingencies (testing all may be time-consuming)
        sample_size = min(10, len(contingencies))
        critical_count = 0

        for _i, contingency in enumerate(contingencies[:sample_size]):
            net_test = net.deepcopy()
            result = simulator.simulate_contingency(net_test, contingency, run_powerflow=True)

            if result.is_critical:
                critical_count += 1

        security_level = 1.0 - (critical_count / sample_size) if sample_size > 0 else 1.0

        return {
            "security_level": security_level,
            "tested_contingencies": sample_size,
            "critical_contingencies": critical_count,
            "secure": critical_count == 0,
            "summary": f"N-1 security: {security_level*100:.1f}% "
                      f"({sample_size - critical_count}/{sample_size} contingencies passed)"
        }


if __name__ == "__main__":
    # Test Constraint Checker Agent
    from src.config import load_configuration

    config, constraints, paths = load_configuration()

    # Initialize agent
    agent = ConstraintCheckerAgent(constraints)

    # Mock evaluation results
    baseline_violations = [
        "Bus 10: 0.92 pu (undervoltage)",
        "Bus 11: 0.91 pu (undervoltage)",
        "Line 8: 105% loading"
    ]

    evaluation_results = {
        "action_id": "test_action",
        "converged": True,
        "min_voltage_pu": 0.96,
        "max_voltage_pu": 1.03,
        "max_line_loading_percent": 95.0,
        "violations": ["Bus 11: 0.94 pu (undervoltage)"]
    }

    print("\n=== Constraint Validation ===")
    validation = agent.validate_action(evaluation_results, baseline_violations)

    print(f"Action ID: {validation.action_id}")
    print(f"Feasible: {validation.feasible}")
    print(f"Safety Score: {validation.safety_score:.2f}")
    print(f"Recommendation: {validation.recommendation}")
    print(f"Rationale: {validation.rationale}")
    print(f"\nViolations Resolved: {len(validation.violations_resolved)}")
    print(f"Violations Remaining: {len(validation.violations_remaining)}")
    print(f"New Violations: {len(validation.new_violations)}")

    import json
    print("\nPerformance Metrics:")
    print(json.dumps(validation.performance_metrics, indent=2))

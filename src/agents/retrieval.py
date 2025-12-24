"""Retrieval Agent for IEEE standards and technical references."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class RetrievalAgent:
    """Retrieves IEEE standards and best practices."""

    # Standard reference database
    STANDARD_REFERENCES = {
        "voltage_limits": {
            "title": "ANSI C84.1-2020: Electric Power Systems and Equipment - Voltage Ratings",
            "summary": "Defines voltage ranges for power systems. Range A: ±5% of nominal (0.95-1.05 pu).",
            "application": "Voltage constraint validation",
            "citation": "ANSI C84.1-2020, Section 4.1"
        },
        "distributed_generation": {
            "title": "IEEE Std 1547-2018: Standard for Interconnection and Interoperability of DER",
            "summary": "Requirements for distributed energy resource interconnection with power systems.",
            "application": "DER integration and voltage regulation",
            "citation": "IEEE Std 1547-2018"
        },
        "protection_coordination": {
            "title": "IEEE Std 242-2001: IEEE Recommended Practice for Protection and Coordination",
            "summary": "Guidelines for protection device coordination in industrial and commercial power systems.",
            "application": "Ensuring protective device coordination after reconfiguration",
            "citation": "IEEE Std 242-2001 (Buff Book)"
        },
        "distribution_reliability": {
            "title": "IEEE Std 1366-2012: Guide for Electric Power Distribution Reliability Indices",
            "summary": "Defines reliability indices including SAIDI, SAIFI, CAIDI for distribution systems.",
            "application": "Reliability impact assessment",
            "citation": "IEEE Std 1366-2012"
        },
        "n_minus_1": {
            "title": "NERC Standard TPL-001-4: Transmission System Planning Performance Requirements",
            "summary": "Requires systems to withstand N-1 contingencies without cascading failures.",
            "application": "Contingency planning and N-1 security analysis",
            "citation": "NERC TPL-001-4, Category B events"
        },
        "load_shedding": {
            "title": "IEEE Std C37.117-2007: Guide for the Application of Protective Relays Used for UFLS",
            "summary": "Guidelines for under-frequency and under-voltage load shedding schemes.",
            "application": "Load shedding strategy development",
            "citation": "IEEE Std C37.117-2007"
        },
        "switching_operations": {
            "title": "IEEE Std 1434-2014: Guide for the Measurement of Partial Discharges in AC Electric Machinery",
            "summary": "Best practices for switching operations to minimize equipment stress and transients.",
            "application": "Safe switching procedure design",
            "citation": "IEEE Std 1434-2014"
        }
    }

    def __init__(self, knowledge_base_path: Optional[Path] = None):
        self.knowledge_base_path = knowledge_base_path
        self.retrieved_cache: Dict[str, Any] = {}

        logger.info("RetrievalAgent initialized")

    def retrieve_relevant_standards(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Retrieve relevant standards based on context.

        Args:
            context: Context including contingency type, violations, etc.

        Returns:
            List of relevant standard references
        """
        logger.info("Retrieving relevant standards...")

        relevant_standards = []

        # Always include voltage limits if voltage violations present
        violations = context.get('violations', [])
        if any('voltage' in v.lower() or 'pu' in v for v in violations):
            relevant_standards.append(self.STANDARD_REFERENCES['voltage_limits'])

        # Include N-1 standard for contingency analysis
        if context.get('contingency_type'):
            relevant_standards.append(self.STANDARD_REFERENCES['n_minus_1'])

        # Include thermal/loading standards if thermal violations
        if any('loading' in v.lower() or '%' in v for v in violations):
            relevant_standards.append(self.STANDARD_REFERENCES['protection_coordination'])

        # Include load shedding standard if load shedding proposed
        action_type = context.get('action_type', '')
        if 'load' in action_type.lower() or 'shed' in action_type.lower():
            relevant_standards.append(self.STANDARD_REFERENCES['load_shedding'])

        # Include switching standard if switching operations proposed
        if 'switch' in action_type.lower():
            relevant_standards.append(self.STANDARD_REFERENCES['switching_operations'])

        # Include reliability standard
        relevant_standards.append(self.STANDARD_REFERENCES['distribution_reliability'])

        logger.info(f"Retrieved {len(relevant_standards)} relevant standards")
        return relevant_standards

    def get_best_practices(self, action_type: str) -> Dict[str, Any]:
        """
        Get best practices for a specific action type.

        Args:
            action_type: Type of action being performed

        Returns:
            Best practices information
        """
        best_practices = {
            "switch_line": {
                "title": "Line Switching Best Practices",
                "practices": [
                    "Verify no-load or low-load conditions before switching",
                    "Ensure alternative path exists before opening lines",
                    "Check protection coordination after topology changes",
                    "Monitor transient voltages during switching operations",
                    "Document all switching actions with timestamps"
                ],
                "safety_considerations": [
                    "Maintain minimum conductor clearances",
                    "Ensure proper grounding",
                    "Coordinate with field personnel",
                    "Verify switch ratings adequate for expected currents"
                ]
            },
            "shed_load": {
                "title": "Load Shedding Best Practices",
                "practices": [
                    "Prioritize non-critical loads for shedding",
                    "Implement staged load shedding to avoid over-correction",
                    "Maintain minimum load for voltage stability",
                    "Document customer notification procedures",
                    "Prepare load restoration sequence"
                ],
                "safety_considerations": [
                    "Avoid shedding critical infrastructure loads",
                    "Maintain hospital and emergency service power",
                    "Consider community impact and duration",
                    "Ensure equitable distribution of load shedding"
                ]
            },
            "multiple": {
                "title": "Coordinated Action Best Practices",
                "practices": [
                    "Sequence actions to minimize system disturbance",
                    "Validate each step before proceeding to next",
                    "Maintain system observability throughout process",
                    "Prepare rollback procedures for each action",
                    "Monitor key performance indicators continuously"
                ],
                "safety_considerations": [
                    "Avoid simultaneous major changes",
                    "Maintain N-1 security at all intermediate states",
                    "Ensure communication with all stakeholders",
                    "Document decision rationale for each action"
                ]
            }
        }

        return best_practices.get(action_type, best_practices["multiple"])

    def retrieve_historical_cases(self, contingency_type: str,
                                  network_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical contingency cases.

        Args:
            contingency_type: Type of contingency
            network_characteristics: Network properties

        Returns:
            List of historical case summaries
        """
        # In a real system, this would query a database of historical events
        # For now, return example cases

        historical_cases = [
            {
                "case_id": "HIST_001",
                "contingency": f"{contingency_type} on similar network",
                "network_type": "distribution_radial",
                "resolution": "Alternative path activation via tie switches",
                "outcome": "Successful restoration within 15 minutes",
                "lessons_learned": "Pre-identified alternative paths reduced response time"
            },
            {
                "case_id": "HIST_002",
                "contingency": f"{contingency_type} with multiple violations",
                "network_type": "distribution_radial",
                "resolution": "Combination of switching and 10% load reduction",
                "outcome": "All constraints satisfied, minimal customer impact",
                "lessons_learned": "Staged approach prevented over-correction"
            }
        ]

        logger.info(f"Retrieved {len(historical_cases)} historical cases")
        return historical_cases

    def get_constraint_references(self, constraint_type: str) -> Dict[str, Any]:
        """
        Get detailed reference information for a constraint type.

        Args:
            constraint_type: Type of constraint (voltage, thermal, etc.)

        Returns:
            Constraint reference information
        """
        constraint_refs = {
            "voltage": {
                "standard": "ANSI C84.1-2020",
                "normal_range": "0.95 - 1.05 pu (±5%)",
                "emergency_range": "0.90 - 1.05 pu",
                "rationale": "Ensures equipment operates within design specifications and maintains power quality",
                "monitoring": "Continuous voltage monitoring at critical buses"
            },
            "thermal": {
                "standard": "IEEE Std 738-2012 (Conductor Temperature)",
                "normal_limit": "100% of rated capacity",
                "emergency_limit": "120% for short duration (<15 min)",
                "rationale": "Prevents conductor overheating and accelerated aging",
                "monitoring": "Real-time thermal monitoring or conservative ratings"
            },
            "frequency": {
                "standard": "NERC Standard BAL-003-1",
                "normal_range": "59.95 - 60.05 Hz",
                "emergency_range": "59.5 - 60.5 Hz",
                "rationale": "Maintains generation-load balance and equipment synchronization",
                "monitoring": "Frequency monitoring at generation sites"
            }
        }

        return constraint_refs.get(constraint_type, {})

    def format_references_for_report(self, context: Dict[str, Any]) -> str:
        """
        Format references for inclusion in report.

        Args:
            context: Analysis context

        Returns:
            Formatted reference section
        """
        standards = self.retrieve_relevant_standards(context)

        references_text = "## Technical References\n\n"

        for i, std in enumerate(standards, 1):
            references_text += f"{i}. **{std['title']}**\n"
            references_text += f"   - Application: {std['application']}\n"
            references_text += f"   - Citation: {std['citation']}\n"
            references_text += f"   - Summary: {std['summary']}\n\n"

        return references_text

    def get_citation_list(self, context: Dict[str, Any]) -> List[str]:
        """
        Get simple citation list for state management.

        Args:
            context: Analysis context

        Returns:
            List of citation strings
        """
        standards = self.retrieve_relevant_standards(context)
        return [std['citation'] for std in standards]


if __name__ == "__main__":
    # Test Retrieval Agent
    from src.config import load_configuration

    config, constraints, paths = load_configuration()

    # Initialize agent
    agent = RetrievalAgent(knowledge_base_path=paths.knowledge_base)

    # Test context
    test_context = {
        'contingency_type': 'line_outage',
        'violations': [
            'Bus 10: 0.92 pu (undervoltage)',
            'Line 8: 105% loading'
        ],
        'action_type': 'switch_line'
    }

    print("\n=== Relevant Standards ===")
    standards = agent.retrieve_relevant_standards(test_context)
    for std in standards:
        print(f"\n{std['title']}")
        print(f"  Application: {std['application']}")
        print(f"  Citation: {std['citation']}")

    print("\n=== Best Practices ===")
    practices = agent.get_best_practices('switch_line')
    print(f"\n{practices['title']}")
    print("\nPractices:")
    for p in practices['practices']:
        print(f"  - {p}")

    print("\n=== Formatted References ===")
    formatted = agent.format_references_for_report(test_context)
    print(formatted)

    print("\n=== Citation List ===")
    citations = agent.get_citation_list(test_context)
    for cite in citations:
        print(f"  - {cite}")

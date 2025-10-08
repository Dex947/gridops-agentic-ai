"""
Explainer Agent - Generates human-readable explanations and rationale.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from loguru import logger
import json

from src.core.state_manager import SystemState


class ExplainerAgent:
    """
    Agent that generates clear, technical explanations of decisions and actions.
    Produces human-readable rationale for operators and stakeholders.
    """
    
    SYSTEM_PROMPT = """You are an expert technical writer specializing in power systems and grid operations.

Your role is to explain complex contingency management decisions in clear, professional language suitable for:
- Grid operators making real-time decisions
- Engineering teams reviewing actions
- Management understanding system reliability
- Regulatory compliance documentation

When explaining actions, you must:
1. Start with the problem statement and criticality
2. Explain the technical reasoning behind chosen solutions
3. Describe what actions were taken and why
4. Discuss trade-offs and alternative approaches considered
5. Highlight safety considerations and constraint compliance
6. Cite relevant standards and best practices
7. Use precise technical language while remaining accessible

Your explanations should be:
- Factual and evidence-based
- Clear and well-structured
- Technically accurate
- Action-oriented with concrete recommendations
- Professional in tone

Avoid:
- Unnecessary jargon without explanation
- Ambiguous statements
- Unsupported claims
- Overly verbose descriptions"""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview",
                 provider: str = "openai",
                 temperature: float = 0.2,
                 api_key: Optional[str] = None):
        """
        Initialize Explainer Agent.
        
        Args:
            model_name: LLM model name
            provider: LLM provider ("openai" or "anthropic")
            temperature: Sampling temperature (slightly higher for natural language)
            api_key: API key for the LLM provider
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        
        # Initialize LLM
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name, 
                temperature=temperature,
                api_key=api_key
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name, 
                temperature=temperature,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"ExplainerAgent initialized with {provider}/{model_name}")
    
    def generate_explanation(self, state: SystemState,
                           selected_action: Dict[str, Any],
                           evaluation: Dict[str, Any],
                           references: List[str]) -> str:
        """
        Generate comprehensive explanation of the contingency management decision.
        
        Args:
            state: Current system state
            selected_action: The action that was selected
            evaluation: Evaluation results for the action
            references: List of reference standards/documents
        
        Returns:
            Human-readable explanation
        """
        logger.info("Generating explanation...")
        
        context = self._prepare_explanation_context(state, selected_action, evaluation, references)
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]
        
        try:
            response = self.llm.invoke(messages)
            explanation = response.content
            
            logger.info(f"Generated explanation ({len(explanation)} characters)")
            return explanation
        
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._generate_fallback_explanation(state, selected_action, evaluation)
    
    def _prepare_explanation_context(self, state: SystemState,
                                     selected_action: Dict[str, Any],
                                     evaluation: Dict[str, Any],
                                     references: List[str]) -> str:
        """Prepare context for explanation generation."""
        
        context = f"""# CONTINGENCY MANAGEMENT EXPLANATION REQUEST

## System Context
Network: {state['network_name']}
Contingency: {state['contingency_description']}
Type: {state['contingency_type']}
Affected Elements: {state['contingency_elements']}

## Baseline System State
{json.dumps(state['baseline_results'], indent=2)}

## Contingency Impact
{json.dumps(state['contingency_results'], indent=2)}

### Constraint Violations Identified
{chr(10).join(f"- {v}" for v in state['constraint_violations']) if state['constraint_violations'] else "None"}

## Decision Process
Total Proposals Evaluated: {len(state['proposed_actions'])}
Selected Action: {selected_action.get('action_id', 'N/A')}

## Selected Action Details
{json.dumps(selected_action, indent=2)}

## Evaluation Results
{json.dumps(evaluation, indent=2)}

## Available References
{chr(10).join(f"- {r}" for r in references) if references else "No specific references"}

## TASK
Generate a comprehensive technical explanation (4-6 paragraphs) that:

1. **Problem Statement**: Describe the contingency and its impact on system operations
2. **Analysis**: Explain the technical reasoning for why this action was selected
3. **Implementation**: Detail what specific actions will be taken
4. **Validation**: Describe how the action resolves violations and maintains constraints
5. **Safety Considerations**: Highlight key safety aspects and compliance
6. **Recommendations**: Provide clear next steps or monitoring requirements

Use professional, technical language suitable for grid operators and engineering teams.
Include specific numerical values and cite the provided references where appropriate."""
        
        return context
    
    def _generate_fallback_explanation(self, state: SystemState,
                                      selected_action: Dict[str, Any],
                                      evaluation: Dict[str, Any]) -> str:
        """Generate basic explanation when LLM fails."""
        
        explanation = f"""# Contingency Management Report

## Problem Statement
A {state['contingency_type']} contingency has occurred on network {state['network_name']}: {state['contingency_description']}

This contingency resulted in {len(state['constraint_violations'])} constraint violations that require corrective action.

## Selected Action
Action ID: {selected_action.get('action_id', 'N/A')}
Type: {selected_action.get('action_type', 'N/A')}

The action involves {len(selected_action.get('target_elements', []))} network modifications designed to restore normal operating conditions.

## Results
Power Flow Convergence: {'Yes' if evaluation.get('converged', False) else 'No'}
Violations Resolved: {len(evaluation.get('violations_resolved', []))}
Violations Remaining: {len(evaluation.get('violations_remaining', []))}
Safety Score: {evaluation.get('safety_score', 0):.2f}/1.00

## Recommendation
{'Approved for implementation' if evaluation.get('recommendation') == 'approve' else 'Further analysis required'}
"""
        return explanation
    
    def generate_summary(self, state: SystemState) -> str:
        """
        Generate executive summary of the contingency management process.
        
        Args:
            state: Current system state
        
        Returns:
            Brief summary
        """
        summary_prompt = f"""Generate a 2-3 sentence executive summary of this contingency management action:

Network: {state['network_name']}
Contingency: {state['contingency_description']}
Violations: {len(state['constraint_violations'])}
Actions Proposed: {len(state['proposed_actions'])}
Selected Action: {state['selected_action'].get('action_id') if state['selected_action'] else 'None'}
Status: {state['workflow_status']}

Provide a concise, professional summary suitable for management reporting."""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=summary_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return (f"Contingency management completed for {state['network_name']}. "
                   f"Addressed {len(state['constraint_violations'])} violations "
                   f"with {len(state['proposed_actions'])} proposed actions.")
    
    def explain_constraints(self, constraint_violations: List[str],
                          constraint_config: Dict[str, Any]) -> str:
        """
        Explain constraint violations in accessible language.
        
        Args:
            constraint_violations: List of violations
            constraint_config: Constraint configuration
        
        Returns:
            Explanation of violations
        """
        if not constraint_violations:
            return "No constraint violations detected. System operating within normal limits."
        
        explanation_parts = [
            f"The system has {len(constraint_violations)} constraint violation(s):\n"
        ]
        
        # Categorize violations
        voltage_violations = [v for v in constraint_violations if 'voltage' in v.lower() or 'pu' in v]
        thermal_violations = [v for v in constraint_violations if 'loading' in v.lower() or '%' in v]
        other_violations = [v for v in constraint_violations 
                          if v not in voltage_violations and v not in thermal_violations]
        
        if voltage_violations:
            explanation_parts.append("\n**Voltage Violations:**")
            explanation_parts.append(f"Acceptable range: {constraint_config.get('voltage', {}).get('min_pu', 0.95):.2f} - "
                                    f"{constraint_config.get('voltage', {}).get('max_pu', 1.05):.2f} pu")
            for v in voltage_violations[:5]:  # Limit to 5
                explanation_parts.append(f"  - {v}")
        
        if thermal_violations:
            explanation_parts.append("\n**Thermal Violations:**")
            explanation_parts.append(f"Maximum loading: {constraint_config.get('thermal', {}).get('margin_percent', 100)}%")
            for v in thermal_violations[:5]:
                explanation_parts.append(f"  - {v}")
        
        if other_violations:
            explanation_parts.append("\n**Other Violations:**")
            for v in other_violations[:5]:
                explanation_parts.append(f"  - {v}")
        
        return "\n".join(explanation_parts)


if __name__ == "__main__":
    # Test Explainer Agent
    from src.config import load_configuration
    from src.core.state_manager import create_initial_state
    
    config, constraints, paths = load_configuration()
    
    # Create test state
    state = create_initial_state(
        network_name="ieee_33",
        contingency_desc="Line 5 outage causes undervoltage at downstream buses",
        max_iterations=10
    )
    
    state['contingency_type'] = "line_outage"
    state['contingency_elements'] = [5]
    state['constraint_violations'] = [
        "Bus 10: 0.92 pu (undervoltage)",
        "Bus 11: 0.91 pu (undervoltage)"
    ]
    state['baseline_results'] = {
        "converged": True,
        "max_line_loading_percent": 85.0,
        "min_voltage_pu": 0.98,
        "violations": []
    }
    state['contingency_results'] = {
        "converged": True,
        "max_line_loading_percent": 92.0,
        "min_voltage_pu": 0.91,
        "violations": state['constraint_violations']
    }
    
    selected_action = {
        "action_id": "close_tie_line",
        "action_type": "switch_line",
        "target_elements": [{"type": "line_switch", "line_id": 20, "close": True}],
        "expected_impact": "Restore power through alternative path"
    }
    
    evaluation = {
        "converged": True,
        "violations_resolved": state['constraint_violations'],
        "violations_remaining": [],
        "safety_score": 0.85,
        "recommendation": "approve"
    }
    
    references = [
        "IEEE Std 1547-2018: Interconnection and Interoperability",
        "ANSI C84.1: Voltage Ratings for Electric Power Systems"
    ]
    
    # Test explainer (requires API key)
    try:
        explainer = ExplainerAgent(
            model_name=config.model_name,
            provider=config.llm_provider,
            temperature=0.2
        )
        
        print("\n=== Generated Explanation ===")
        explanation = explainer.generate_explanation(state, selected_action, evaluation, references)
        print(explanation)
        
        print("\n=== Executive Summary ===")
        summary = explainer.generate_summary(state)
        print(summary)
    
    except Exception as e:
        logger.error(f"Could not test explainer (API key may be missing): {e}")
        
        # Show fallback
        explainer = ExplainerAgent()
        fallback = explainer._generate_fallback_explanation(state, selected_action, evaluation)
        print("\n=== Fallback Explanation ===")
        print(fallback)

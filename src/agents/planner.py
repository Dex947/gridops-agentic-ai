"""
Planner Agent - Generates candidate switching and load-shedding strategies.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from loguru import logger
import json

from src.core.state_manager import SystemState, ActionProposal
from src.tools.network_analysis import NetworkAnalyzer, identify_reconfiguration_options


class PlannerAgent:
    """
    Agent responsible for generating reconfiguration strategies.
    Uses LLM to reason about contingencies and propose solutions.
    """
    
    SYSTEM_PROMPT = """You are an expert power systems engineer specializing in distribution network operations and contingency management.

Your role is to analyze network contingencies and propose safe reconfiguration strategies including:
- Line switching actions (opening/closing switches)
- Load shedding plans (reducing load at specific buses)
- Voltage regulation adjustments
- Generator dispatch modifications

When proposing solutions, you must:
1. Prioritize safety and constraint compliance
2. Minimize load shedding and customer impact
3. Consider N-1 security and protection coordination
4. Provide clear reasoning for each proposed action
5. Rank proposals by feasibility and effectiveness

You will receive:
- Network topology and current state
- Contingency description and affected elements
- Baseline power flow results
- Available reconfiguration options

Generate 2-5 candidate action plans with detailed justification."""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview",
                 provider: str = "openai",
                 temperature: float = 0.1,
                 api_key: Optional[str] = None):
        """
        Initialize Planner Agent.
        
        Args:
            model_name: LLM model name
            provider: LLM provider ("openai" or "anthropic")
            temperature: Sampling temperature
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
        
        logger.info(f"PlannerAgent initialized with {provider}/{model_name}")
    
    def generate_action_plans(self, state: SystemState, 
                             network_data: Dict[str, Any]) -> List[ActionProposal]:
        """
        Generate candidate action plans for the contingency.
        
        Args:
            state: Current system state
            network_data: Network topology and analysis data
        
        Returns:
            List of action proposals
        """
        logger.info("Generating action plans...")
        
        # Prepare context for LLM
        context = self._prepare_context(state, network_data)
        
        # Create messages
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]
        
        try:
            # Get LLM response
            response = self.llm.invoke(messages)
            
            # Parse response into action proposals
            proposals = self._parse_llm_response(response.content, state)
            
            logger.info(f"Generated {len(proposals)} action proposals")
            return proposals
        
        except Exception as e:
            logger.error(f"Error generating action plans: {e}")
            return []
    
    def _prepare_context(self, state: SystemState, network_data: Dict[str, Any]) -> str:
        """Prepare context string for LLM."""
        
        context = f"""# NETWORK CONTINGENCY ANALYSIS REQUEST

## Network Information
- Network: {state['network_name']}
- Total Buses: {network_data.get('num_buses', 'N/A')}
- Total Lines: {network_data.get('num_lines', 'N/A')}
- Total Load: {network_data.get('total_load_mw', 0):.2f} MW

## Contingency Description
{state['contingency_description']}

Type: {state['contingency_type']}
Affected Elements: {state['contingency_elements']}

## Current System Status
Baseline Results:
{json.dumps(state['baseline_results'], indent=2)}

Contingency Results:
{json.dumps(state['contingency_results'], indent=2)}

## Constraint Violations
{chr(10).join(f"- {v}" for v in state['constraint_violations']) if state['constraint_violations'] else "None"}

## Available Reconfiguration Options
{json.dumps(network_data.get('reconfiguration_options', []), indent=2)}

## TASK
Generate 2-5 candidate action plans to resolve this contingency. For each plan, provide:
1. Action ID (unique identifier)
2. Action Type (switch_line, shed_load, or multiple)
3. Target Elements (specific lines/loads to act on)
4. Expected Impact (what this achieves)
5. Priority (1-10, higher = more urgent)
6. Reasoning (why this approach makes sense)

Return your response as a valid JSON array of action plans."""
        
        return context
    
    def _parse_llm_response(self, response: str, state: SystemState) -> List[ActionProposal]:
        """Parse LLM response into ActionProposal objects."""
        proposals = []
        
        try:
            # Try to extract JSON from response
            # LLMs often wrap JSON in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            # Parse JSON
            plans = json.loads(json_str)
            
            # Convert to ActionProposal objects
            for i, plan in enumerate(plans):
                proposal = ActionProposal(
                    action_id=plan.get('action_id', f"plan_{i+1}"),
                    action_type=plan.get('action_type', 'multiple'),
                    target_elements=plan.get('target_elements', []),
                    expected_impact=plan.get('expected_impact', ''),
                    priority=int(plan.get('priority', 5)),
                    reasoning=plan.get('reasoning', ''),
                    proposed_by="PlannerAgent"
                )
                proposals.append(proposal)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response: {response}")
            
            # Fallback: Create a generic proposal
            proposals.append(ActionProposal(
                action_id="fallback_plan",
                action_type="shed_load",
                target_elements=[],
                expected_impact="Generic load shedding to resolve violations",
                priority=5,
                reasoning="Fallback plan due to parsing error",
                proposed_by="PlannerAgent"
            ))
        
        except Exception as e:
            logger.error(f"Error parsing proposals: {e}")
        
        return proposals
    
    def refine_action_plan(self, state: SystemState, 
                          evaluation_feedback: Dict[str, Any]) -> Optional[ActionProposal]:
        """
        Refine an action plan based on evaluation feedback.
        
        Args:
            state: Current system state
            evaluation_feedback: Feedback from constraint checker
        
        Returns:
            Refined action proposal or None
        """
        logger.info("Refining action plan based on feedback...")
        
        refinement_prompt = f"""# ACTION PLAN REFINEMENT

The following action plan was evaluated and needs refinement:

## Original Plan
{json.dumps(state.get('selected_action', {}), indent=2)}

## Evaluation Feedback
{json.dumps(evaluation_feedback, indent=2)}

## TASK
Modify the action plan to address the feedback while maintaining safety and effectiveness.
Return the refined plan as a valid JSON object with the same structure as the original."""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=refinement_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            proposals = self._parse_llm_response(response.content, state)
            
            if proposals:
                logger.info("Action plan refined successfully")
                return proposals[0]
            
        except Exception as e:
            logger.error(f"Error refining action plan: {e}")
        
        return None


if __name__ == "__main__":
    # Test planner agent
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
        "Bus 11: 0.91 pu (undervoltage)",
        "Line 8: 105% loading"
    ]
    
    # Mock network data
    network_data = {
        'num_buses': 33,
        'num_lines': 32,
        'total_load_mw': 3.715,
        'reconfiguration_options': [
            {
                'target_bus': 10,
                'lines_to_close': [{'line_id': 20, 'from_bus': 9, 'to_bus': 10}],
                'priority': 8
            }
        ]
    }
    
    # Test planner (requires API key)
    try:
        planner = PlannerAgent(
            model_name=config.model_name,
            provider=config.llm_provider,
            temperature=config.temperature
        )
        
        proposals = planner.generate_action_plans(state, network_data)
        
        print("\n=== Generated Action Plans ===")
        for proposal in proposals:
            print(json.dumps(proposal.to_dict(), indent=2))
            print()
    
    except Exception as e:
        logger.error(f"Could not test planner (API key may be missing): {e}")

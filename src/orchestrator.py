"""
Multi-Agent Orchestrator using LangGraph.
Coordinates all agents in the contingency management workflow.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from loguru import logger
import pandapower as pp

from src.core.state_manager import (
    SystemState, 
    create_initial_state,
    StateManager,
    ActionProposal
)
from src.core.network_loader import NetworkLoader
from src.core.contingency_simulator import ContingencySimulator, ContingencyEvent, ContingencyType
from src.agents.planner import PlannerAgent
from src.agents.powerflow_user import PowerFlowUserAgent
from src.agents.constraint_checker import ConstraintCheckerAgent
from src.agents.explainer import ExplainerAgent
from src.agents.retrieval import RetrievalAgent
from src.tools.network_analysis import NetworkAnalyzer, calculate_network_metrics
from src.config import SystemConfig, NetworkConstraints, PathConfig


class GridOpsOrchestrator:
    """
    Orchestrates multi-agent workflow for contingency management.
    Uses LangGraph for agent coordination and state management.
    """
    
    def __init__(self, config: SystemConfig, constraints: NetworkConstraints, paths: PathConfig):
        """
        Initialize orchestrator with all agents and tools.
        
        Args:
            config: System configuration
            constraints: Network constraints
            paths: Path configuration
        """
        self.config = config
        self.constraints = constraints
        self.paths = paths
        
        # Initialize state manager
        self.state_manager = StateManager(state_dir=paths.base / "state")
        
        # Initialize network loader
        self.network_loader = NetworkLoader(networks_path=paths.networks)
        
        # Get API key based on provider
        api_key = None
        if config.llm_provider == "openai":
            api_key = config.openai_api_key
        elif config.llm_provider == "anthropic":
            api_key = config.anthropic_api_key
        
        # Initialize agents
        self.planner = PlannerAgent(
            model_name=config.model_name,
            provider=config.llm_provider,
            temperature=config.temperature,
            api_key=api_key
        )
        
        self.powerflow_agent = PowerFlowUserAgent(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
            thermal_limit_percent=100.0
        )
        
        self.constraint_checker = ConstraintCheckerAgent(constraints=constraints)
        
        self.explainer = ExplainerAgent(
            model_name=config.model_name,
            provider=config.llm_provider,
            temperature=0.2,
            api_key=api_key
        )
        
        self.retrieval = RetrievalAgent(knowledge_base_path=paths.knowledge_base)
        
        # Initialize contingency simulator
        self.contingency_simulator = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
            thermal_limit_percent=100.0
        )
        
        # Network data
        self.network: Optional[pp.pandapowerNet] = None
        self.network_data: Dict[str, Any] = {}
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("GridOpsOrchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for agent coordination."""
        
        # Create state graph
        workflow = StateGraph(SystemState)
        
        # Add nodes for each workflow step
        workflow.add_node("load_network", self._load_network_node)
        workflow.add_node("analyze_baseline", self._analyze_baseline_node)
        workflow.add_node("simulate_contingency", self._simulate_contingency_node)
        workflow.add_node("retrieve_references", self._retrieve_references_node)
        workflow.add_node("generate_plans", self._generate_plans_node)
        workflow.add_node("evaluate_plans", self._evaluate_plans_node)
        workflow.add_node("select_action", self._select_action_node)
        workflow.add_node("generate_explanation", self._generate_explanation_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges (workflow sequence)
        workflow.set_entry_point("load_network")
        workflow.add_edge("load_network", "analyze_baseline")
        workflow.add_edge("analyze_baseline", "simulate_contingency")
        workflow.add_edge("simulate_contingency", "retrieve_references")
        workflow.add_edge("retrieve_references", "generate_plans")
        workflow.add_edge("generate_plans", "evaluate_plans")
        workflow.add_edge("evaluate_plans", "select_action")
        workflow.add_edge("select_action", "generate_explanation")
        workflow.add_edge("generate_explanation", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _load_network_node(self, state: SystemState) -> SystemState:
        """Node: Load and prepare network."""
        logger.info("=== Node: Load Network ===")
        
        try:
            # Load network
            self.network = self.network_loader.load_network(state['network_name'])
            
            # Calculate network metrics
            self.network_data = calculate_network_metrics(self.network)
            
            # Add network summary to state
            state['network_loaded'] = True
            state['network_summary'] = self.network_loader.get_network_summary(self.network)
            
            state = self.state_manager.update_workflow_status(state, "analyzing")
            
            logger.info(f"Network loaded: {state['network_name']}")
            return state
        
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            return self.state_manager.update_workflow_status(state, "failed", str(e))
    
    def _analyze_baseline_node(self, state: SystemState) -> SystemState:
        """Node: Analyze baseline (healthy) network."""
        logger.info("=== Node: Analyze Baseline ===")
        
        try:
            # Run baseline power flow
            baseline_results = self.powerflow_agent.run_baseline_analysis(self.network)
            state['baseline_results'] = baseline_results
            
            logger.info("Baseline analysis complete")
            return state
        
        except Exception as e:
            logger.error(f"Baseline analysis failed: {e}")
            return self.state_manager.update_workflow_status(state, "failed", str(e))
    
    def _simulate_contingency_node(self, state: SystemState) -> SystemState:
        """Node: Simulate contingency scenario."""
        logger.info("=== Node: Simulate Contingency ===")
        
        try:
            # Create contingency event
            contingency = self._parse_contingency(state)
            
            # Make a copy of network for contingency simulation
            net_contingency = self.network.deepcopy()
            
            # Simulate contingency
            result = self.contingency_simulator.simulate_contingency(
                net_contingency, 
                contingency,
                run_powerflow=True
            )
            
            # Store results
            state['contingency_results'] = result.to_dict()
            state['constraint_violations'] = result.violated_constraints
            
            # Store contingency network for later use
            self.network_contingency = net_contingency
            
            logger.info(f"Contingency simulation complete: {len(result.violated_constraints)} violations")
            return state
        
        except Exception as e:
            logger.error(f"Contingency simulation failed: {e}")
            return self.state_manager.update_workflow_status(state, "failed", str(e))
    
    def _retrieve_references_node(self, state: SystemState) -> SystemState:
        """Node: Retrieve relevant standards and references."""
        logger.info("=== Node: Retrieve References ===")
        
        try:
            context = {
                'contingency_type': state['contingency_type'],
                'violations': state['constraint_violations'],
                'action_type': 'switch_line'  # Default, will be refined
            }
            
            # Get relevant standards
            citations = self.retrieval.get_citation_list(context)
            state['references'] = citations
            
            logger.info(f"Retrieved {len(citations)} references")
            return state
        
        except Exception as e:
            logger.error(f"Reference retrieval failed: {e}")
            state['references'] = []
            return state
    
    def _generate_plans_node(self, state: SystemState) -> SystemState:
        """Node: Generate action plans using Planner Agent."""
        logger.info("=== Node: Generate Plans ===")
        
        state = self.state_manager.update_workflow_status(state, "planning")
        
        try:
            # Prepare network analysis data
            analyzer = NetworkAnalyzer(self.network_contingency)
            
            # Find isolated buses
            isolated_buses = analyzer.find_isolated_buses()
            
            # Get critical lines
            critical_lines = analyzer.identify_critical_lines()
            
            network_analysis = {
                **self.network_data,
                'isolated_buses': isolated_buses,
                'critical_lines': critical_lines,
                'reconfiguration_options': []
            }
            
            # Generate action proposals
            proposals = self.planner.generate_action_plans(state, network_analysis)
            
            # Add proposals to state
            for proposal in proposals:
                state = self.state_manager.add_proposed_action(state, proposal)
            
            logger.info(f"Generated {len(proposals)} action plans")
            return state
        
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            # Continue with empty proposals
            return state
    
    def _evaluate_plans_node(self, state: SystemState) -> SystemState:
        """Node: Evaluate all proposed plans."""
        logger.info("=== Node: Evaluate Plans ===")
        
        state = self.state_manager.update_workflow_status(state, "validating")
        
        try:
            baseline_violations = state['constraint_violations']
            
            # Evaluate each proposed action
            for action_dict in state['proposed_actions']:
                action = ActionProposal.from_dict(action_dict)
                
                logger.info(f"Evaluating action: {action.action_id}")
                
                # Evaluate with PowerFlow agent
                evaluation_results = self.powerflow_agent.evaluate_action_plan(
                    self.network_contingency,
                    action
                )
                
                # Validate with Constraint Checker
                validation = self.constraint_checker.validate_action(
                    evaluation_results,
                    baseline_violations
                )
                
                # Add evaluation to state
                state = self.state_manager.add_action_evaluation(state, validation)
            
            logger.info(f"Evaluated {len(state['evaluated_actions'])} actions")
            return state
        
        except Exception as e:
            logger.error(f"Plan evaluation failed: {e}")
            return state
    
    def _select_action_node(self, state: SystemState) -> SystemState:
        """Node: Select best action from evaluations."""
        logger.info("=== Node: Select Action ===")
        
        try:
            state = self.state_manager.select_best_action(state)
            
            if state['selected_action']:
                logger.info(f"Selected action: {state['selected_action']['action_id']}")
            else:
                logger.warning("No suitable action found")
            
            return state
        
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return state
    
    def _generate_explanation_node(self, state: SystemState) -> SystemState:
        """Node: Generate explanation and rationale."""
        logger.info("=== Node: Generate Explanation ===")
        
        try:
            if not state['selected_action']:
                state['explanation'] = "No suitable action was identified that meets all constraints."
                return state
            
            # Find the evaluation for selected action
            selected_eval = None
            for eval_dict in state['evaluated_actions']:
                if eval_dict['action_id'] == state['selected_action']['action_id']:
                    selected_eval = eval_dict
                    break
            
            if selected_eval:
                # Generate comprehensive explanation
                explanation = self.explainer.generate_explanation(
                    state,
                    state['selected_action'],
                    selected_eval,
                    state['references']
                )
                state['explanation'] = explanation
            else:
                state['explanation'] = "Selected action evaluation not found."
            
            logger.info("Explanation generated")
            return state
        
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            state['explanation'] = self.explainer._generate_fallback_explanation(
                state,
                state['selected_action'] or {},
                {}
            )
            return state
    
    def _finalize_node(self, state: SystemState) -> SystemState:
        """Node: Finalize workflow and update status."""
        logger.info("=== Node: Finalize ===")
        
        state = self.state_manager.update_workflow_status(state, "complete")
        logger.info("Workflow complete")
        
        return state
    
    def _parse_contingency(self, state: SystemState) -> ContingencyEvent:
        """Parse contingency description into ContingencyEvent."""
        
        # Map contingency type string to enum
        type_map = {
            'line_outage': ContingencyType.LINE_OUTAGE,
            'transformer_outage': ContingencyType.TRANSFORMER_OUTAGE,
            'load_increase': ContingencyType.LOAD_INCREASE,
            'generator_outage': ContingencyType.GENERATOR_OUTAGE
        }
        
        cont_type = type_map.get(
            state['contingency_type'], 
            ContingencyType.LINE_OUTAGE
        )
        
        elements = state['contingency_elements']
        element_idx = elements[0] if elements else 0
        
        return ContingencyEvent(
            event_type=cont_type,
            element_index=element_idx,
            description=state['contingency_description']
        )
    
    def run(self, network_name: str, contingency_description: str,
            contingency_type: str = "line_outage",
            contingency_elements: Optional[List[int]] = None) -> SystemState:
        """
        Run complete contingency management workflow.
        
        Args:
            network_name: Name of network to analyze
            contingency_description: Human-readable contingency description
            contingency_type: Type of contingency
            contingency_elements: List of affected element indices
        
        Returns:
            Final system state
        """
        logger.info("=" * 60)
        logger.info("STARTING CONTINGENCY MANAGEMENT WORKFLOW")
        logger.info("=" * 60)
        
        # Create initial state
        initial_state = create_initial_state(
            network_name=network_name,
            contingency_desc=contingency_description,
            max_iterations=self.config.max_iterations
        )
        
        initial_state['contingency_type'] = contingency_type
        initial_state['contingency_elements'] = contingency_elements or []
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("=" * 60)
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return self.state_manager.update_workflow_status(
                initial_state,
                "failed",
                str(e)
            )


if __name__ == "__main__":
    # Test orchestrator
    from src.config import load_configuration
    import json
    
    config, constraints, paths = load_configuration()
    
    # Initialize orchestrator
    orchestrator = GridOpsOrchestrator(config, constraints, paths)
    
    # Run test scenario
    print("\n" + "=" * 60)
    print("TESTING GRIDOPS ORCHESTRATOR")
    print("=" * 60)
    
    final_state = orchestrator.run(
        network_name="ieee_33",
        contingency_description="Line 5 outage causing downstream undervoltage",
        contingency_type="line_outage",
        contingency_elements=[5]
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)
    
    summary = orchestrator.state_manager.get_summary(final_state)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    
    if final_state['selected_action']:
        print("\nSelected Action:")
        print(json.dumps(final_state['selected_action'], indent=2))
    
    if final_state['explanation']:
        print("\n" + "=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        print(final_state['explanation'])

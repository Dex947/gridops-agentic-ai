"""
State management module for GridOps Agentic AI System.
Manages system state across multi-agent workflow using LangGraph.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import operator
from loguru import logger


class SystemState(TypedDict):
    """
    System state for LangGraph workflow.
    Uses TypedDict for LangGraph state management.
    """
    # Network information
    network_name: str
    network_loaded: bool
    network_summary: Dict[str, Any]
    
    # Contingency information
    contingency_description: str
    contingency_type: str
    contingency_elements: List[int]
    
    # Analysis results
    baseline_results: Dict[str, Any]
    contingency_results: Dict[str, Any]
    constraint_violations: List[str]
    
    # Agent proposals and decisions
    proposed_actions: Annotated[List[Dict[str, Any]], operator.add]
    evaluated_actions: List[Dict[str, Any]]
    selected_action: Optional[Dict[str, Any]]
    
    # Explanation and references
    explanation: str
    references: List[str]
    
    # Workflow control
    iteration: int
    max_iterations: int
    workflow_status: str  # "initializing", "analyzing", "planning", "validating", "complete", "failed"
    error_message: str
    
    # Timestamps
    started_at: str
    completed_at: str


def create_initial_state(network_name: str, 
                         contingency_desc: str,
                         max_iterations: int = 10) -> SystemState:
    """
    Create initial system state.
    
    Args:
        network_name: Name of the network to analyze
        contingency_desc: Description of contingency scenario
        max_iterations: Maximum workflow iterations
    
    Returns:
        Initial system state
    """
    now = datetime.utcnow().isoformat()
    
    return SystemState(
        network_name=network_name,
        network_loaded=False,
        network_summary={},
        contingency_description=contingency_desc,
        contingency_type="",
        contingency_elements=[],
        baseline_results={},
        contingency_results={},
        constraint_violations=[],
        proposed_actions=[],
        evaluated_actions=[],
        selected_action=None,
        explanation="",
        references=[],
        iteration=0,
        max_iterations=max_iterations,
        workflow_status="initializing",
        error_message="",
        started_at=now,
        completed_at=""
    )


@dataclass
class ActionProposal:
    """Represents a proposed reconfiguration action."""
    action_id: str
    action_type: str  # "switch_line", "shed_load", "adjust_setpoint", "multiple"
    target_elements: List[Dict[str, Any]]
    expected_impact: str
    priority: int = 5  # 1-10, higher = more critical
    estimated_cost: float = 0.0  # Operational cost estimate
    reasoning: str = ""
    proposed_by: str = ""  # Agent that proposed this action
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionProposal":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ActionEvaluation:
    """Evaluation results for a proposed action."""
    action_id: str
    feasible: bool
    powerflow_converged: bool
    violations_resolved: List[str]
    violations_remaining: List[str]
    new_violations: List[str]
    performance_metrics: Dict[str, float]
    safety_score: float  # 0-1, higher = safer
    recommendation: str  # "approve", "modify", "reject"
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionEvaluation":
        """Create from dictionary."""
        return cls(**data)


class StateManager:
    """Manages system state persistence and transitions."""
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for state persistence
        """
        self.state_dir = state_dir
        if state_dir:
            state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state: SystemState, session_id: str):
        """
        Save current state to disk.
        
        Args:
            state: Current system state
            session_id: Unique session identifier
        """
        if not self.state_dir:
            logger.warning("State directory not set, skipping save")
            return
        
        file_path = self.state_dir / f"state_{session_id}.json"
        
        # Convert state to serializable format
        state_dict = dict(state)
        
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"State saved to {file_path}")
    
    def load_state(self, session_id: str) -> Optional[SystemState]:
        """
        Load state from disk.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            System state or None if not found
        """
        if not self.state_dir:
            logger.warning("State directory not set")
            return None
        
        file_path = self.state_dir / f"state_{session_id}.json"
        
        if not file_path.exists():
            logger.warning(f"State file not found: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            state_dict = json.load(f)
        
        logger.info(f"State loaded from {file_path}")
        return SystemState(**state_dict)
    
    def update_workflow_status(self, state: SystemState, 
                              new_status: str,
                              error_message: str = "") -> SystemState:
        """
        Update workflow status.
        
        Args:
            state: Current state
            new_status: New workflow status
            error_message: Error message if status is "failed"
        
        Returns:
            Updated state
        """
        state["workflow_status"] = new_status
        state["error_message"] = error_message
        
        if new_status in ["complete", "failed"]:
            state["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Workflow status updated: {new_status}")
        return state
    
    def increment_iteration(self, state: SystemState) -> SystemState:
        """
        Increment iteration counter and check limits.
        
        Args:
            state: Current state
        
        Returns:
            Updated state
        """
        state["iteration"] += 1
        
        if state["iteration"] >= state["max_iterations"]:
            logger.warning(f"Maximum iterations reached: {state['max_iterations']}")
            state = self.update_workflow_status(
                state, 
                "failed", 
                "Maximum iterations exceeded"
            )
        
        return state
    
    def add_proposed_action(self, state: SystemState, 
                           action: ActionProposal) -> SystemState:
        """
        Add a proposed action to state.
        
        Args:
            state: Current state
            action: Action proposal
        
        Returns:
            Updated state
        """
        state["proposed_actions"].append(action.to_dict())
        logger.info(f"Action proposed: {action.action_id} by {action.proposed_by}")
        return state
    
    def add_action_evaluation(self, state: SystemState,
                             evaluation: ActionEvaluation) -> SystemState:
        """
        Add action evaluation to state.
        
        Args:
            state: Current state
            evaluation: Action evaluation
        
        Returns:
            Updated state
        """
        state["evaluated_actions"].append(evaluation.to_dict())
        logger.info(f"Action evaluated: {evaluation.action_id} -> {evaluation.recommendation}")
        return state
    
    def select_best_action(self, state: SystemState) -> SystemState:
        """
        Select the best action from evaluated actions.
        
        Args:
            state: Current state
        
        Returns:
            Updated state with selected action
        """
        if not state["evaluated_actions"]:
            logger.warning("No actions to select from")
            return state
        
        # Filter approved actions
        approved = [a for a in state["evaluated_actions"] 
                   if a["recommendation"] == "approve"]
        
        if not approved:
            logger.warning("No approved actions found")
            return state
        
        # Select based on safety score and feasibility
        best = max(approved, key=lambda a: (a["safety_score"], -len(a["violations_remaining"])))
        state["selected_action"] = best
        
        logger.info(f"Selected action: {best['action_id']}")
        return state
    
    def get_summary(self, state: SystemState) -> Dict[str, Any]:
        """
        Get summary of current state.
        
        Args:
            state: Current state
        
        Returns:
            Summary dictionary
        """
        return {
            "network": state["network_name"],
            "contingency": state["contingency_description"],
            "status": state["workflow_status"],
            "iteration": state["iteration"],
            "proposed_actions": len(state["proposed_actions"]),
            "evaluated_actions": len(state["evaluated_actions"]),
            "selected_action": state["selected_action"]["action_id"] if state["selected_action"] else None,
            "violations": len(state["constraint_violations"]),
            "has_explanation": bool(state["explanation"]),
            "references_count": len(state["references"])
        }


if __name__ == "__main__":
    # Test state management
    from src.config import load_configuration
    
    config, _, paths = load_configuration()
    
    # Initialize state manager
    state_dir = paths.base / "state"
    manager = StateManager(state_dir=state_dir)
    
    # Create initial state
    state = create_initial_state(
        network_name="ieee_33",
        contingency_desc="Line 5 outage",
        max_iterations=10
    )
    
    print("\n=== Initial State ===")
    print(json.dumps(manager.get_summary(state), indent=2))
    
    # Simulate workflow
    state["network_loaded"] = True
    state = manager.update_workflow_status(state, "analyzing")
    
    # Add a proposed action
    action = ActionProposal(
        action_id="action_001",
        action_type="switch_line",
        target_elements=[{"line_id": 10, "close": True}],
        expected_impact="Restore connectivity to isolated buses",
        priority=8,
        proposed_by="PlannerAgent"
    )
    state = manager.add_proposed_action(state, action)
    
    # Save state
    session_id = "test_session_001"
    manager.save_state(state, session_id)
    
    print("\n=== Updated State ===")
    print(json.dumps(manager.get_summary(state), indent=2))
    
    # Load state
    loaded_state = manager.load_state(session_id)
    print("\n=== Loaded State ===")
    print(json.dumps(manager.get_summary(loaded_state), indent=2))

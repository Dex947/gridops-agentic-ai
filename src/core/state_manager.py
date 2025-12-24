"""State management for LangGraph workflow."""

import json
import operator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Tuple

from loguru import logger


class ActionPlanValidationError(Exception):
    """Raised when action plan validation fails."""
    pass


def validate_action_plan(action_plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an action plan schema.
    
    Args:
        action_plan: Dictionary containing action plan data
        
    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []
    
    # Required fields
    required_fields = ["action_id", "action_type", "target_elements"]
    for field in required_fields:
        if field not in action_plan:
            errors.append(f"Missing required field: {field}")
    
    # Validate action_type
    valid_action_types = ["switch_line", "shed_load", "adjust_setpoint", "multiple", "close_tie", "open_sectionalizer"]
    if "action_type" in action_plan:
        if action_plan["action_type"] not in valid_action_types:
            errors.append(f"Invalid action_type: {action_plan['action_type']}. Must be one of {valid_action_types}")
    
    # Validate target_elements
    if "target_elements" in action_plan:
        if not isinstance(action_plan["target_elements"], list):
            errors.append("target_elements must be a list")
        else:
            for i, elem in enumerate(action_plan["target_elements"]):
                if not isinstance(elem, dict):
                    errors.append(f"target_elements[{i}] must be a dictionary")
                else:
                    # Validate based on action type
                    action_type = action_plan.get("action_type", "")
                    elem_errors = _validate_target_element(elem, action_type, i)
                    errors.extend(elem_errors)
    
    # Validate priority if present
    if "priority" in action_plan:
        if not isinstance(action_plan["priority"], (int, float)):
            errors.append("priority must be a number")
        elif not 1 <= action_plan["priority"] <= 10:
            errors.append("priority must be between 1 and 10")
    
    # Validate estimated_cost if present
    if "estimated_cost" in action_plan:
        if not isinstance(action_plan["estimated_cost"], (int, float)):
            errors.append("estimated_cost must be a number")
        elif action_plan["estimated_cost"] < 0:
            errors.append("estimated_cost cannot be negative")
    
    return len(errors) == 0, errors


def _validate_target_element(element: Dict[str, Any], action_type: str, index: int) -> List[str]:
    """Validate a single target element based on action type."""
    errors = []
    prefix = f"target_elements[{index}]"
    
    if action_type in ["switch_line", "close_tie", "open_sectionalizer"]:
        if "line_id" not in element:
            errors.append(f"{prefix}: missing 'line_id' for {action_type} action")
        elif not isinstance(element["line_id"], int):
            errors.append(f"{prefix}: 'line_id' must be an integer")
        
        if "close" in element and not isinstance(element["close"], bool):
            errors.append(f"{prefix}: 'close' must be a boolean")
            
    elif action_type == "shed_load":
        if "load_id" not in element:
            errors.append(f"{prefix}: missing 'load_id' for shed_load action")
        elif not isinstance(element["load_id"], int):
            errors.append(f"{prefix}: 'load_id' must be an integer")
        
        if "reduction_percent" in element:
            if not isinstance(element["reduction_percent"], (int, float)):
                errors.append(f"{prefix}: 'reduction_percent' must be a number")
            elif not 0 <= element["reduction_percent"] <= 100:
                errors.append(f"{prefix}: 'reduction_percent' must be between 0 and 100")
                
    elif action_type == "adjust_setpoint":
        if "element_type" not in element:
            errors.append(f"{prefix}: missing 'element_type' for adjust_setpoint action")
        if "element_id" not in element:
            errors.append(f"{prefix}: missing 'element_id' for adjust_setpoint action")
        if "new_value" not in element:
            errors.append(f"{prefix}: missing 'new_value' for adjust_setpoint action")
    
    return errors


def validate_action_proposal(proposal: 'ActionProposal') -> Tuple[bool, List[str]]:
    """Validate an ActionProposal object."""
    return validate_action_plan(proposal.to_dict())


def validate_and_sanitize_action_plan(action_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize an action plan, raising an error if invalid.
    
    Args:
        action_plan: Dictionary containing action plan data
        
    Returns:
        Sanitized action plan dictionary
        
    Raises:
        ActionPlanValidationError: If validation fails
    """
    is_valid, errors = validate_action_plan(action_plan)
    
    if not is_valid:
        error_msg = "Action plan validation failed: " + "; ".join(errors)
        logger.error(error_msg)
        raise ActionPlanValidationError(error_msg)
    
    # Sanitize: ensure default values
    sanitized = action_plan.copy()
    sanitized.setdefault("priority", 5)
    sanitized.setdefault("estimated_cost", 0.0)
    sanitized.setdefault("reasoning", "")
    sanitized.setdefault("proposed_by", "unknown")
    sanitized.setdefault("expected_impact", "")
    
    return sanitized


class SystemState(TypedDict):
    """LangGraph workflow state container."""
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
    """Create initial workflow state."""
    now = datetime.now(timezone.utc).isoformat()

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
            state["completed_at"] = datetime.now(timezone.utc).isoformat()

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
                           action: ActionProposal,
                           validate: bool = True) -> SystemState:
        """
        Add a proposed action to state.

        Args:
            state: Current state
            action: Action proposal
            validate: Whether to validate the action before adding

        Returns:
            Updated state
            
        Raises:
            ActionPlanValidationError: If validation is enabled and fails
        """
        action_dict = action.to_dict()
        
        if validate:
            is_valid, errors = validate_action_plan(action_dict)
            if not is_valid:
                error_msg = f"Action {action.action_id} validation failed: {'; '.join(errors)}"
                logger.warning(error_msg)
                # Log but don't raise - allow workflow to continue
        
        state["proposed_actions"].append(action_dict)
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

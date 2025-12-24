"""Historical case database for learning from past contingencies."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class HistoricalCase:
    """Represents a historical contingency case."""
    case_id: str
    timestamp: datetime
    network_name: str
    contingency_type: str
    contingency_description: str
    affected_elements: List[int]
    
    # Pre-contingency state
    baseline_min_voltage: float = 1.0
    baseline_max_loading: float = 0.0
    baseline_losses_mw: float = 0.0
    
    # Post-contingency state
    post_contingency_min_voltage: float = 1.0
    post_contingency_max_loading: float = 0.0
    post_contingency_losses_mw: float = 0.0
    violations: List[str] = field(default_factory=list)
    
    # Action taken
    action_type: str = ""
    action_elements: List[Dict[str, Any]] = field(default_factory=list)
    action_description: str = ""
    
    # Outcome
    action_successful: bool = False
    post_action_min_voltage: float = 1.0
    post_action_max_loading: float = 0.0
    post_action_losses_mw: float = 0.0
    violations_resolved: List[str] = field(default_factory=list)
    
    # Metadata
    operator_feedback: str = ""
    operator_rating: int = 0  # 1-5 stars
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalCase":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def get_similarity_features(self) -> Dict[str, Any]:
        """Get features for similarity matching."""
        return {
            "network_name": self.network_name,
            "contingency_type": self.contingency_type,
            "num_affected_elements": len(self.affected_elements),
            "severity": len(self.violations),
            "voltage_drop": self.baseline_min_voltage - self.post_contingency_min_voltage,
            "loading_increase": self.post_contingency_max_loading - self.baseline_max_loading,
            "action_type": self.action_type,
            "success": self.action_successful
        }


@dataclass
class CaseSearchResult:
    """Result from case database search."""
    case: HistoricalCase
    similarity_score: float
    matching_features: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case": self.case.to_dict(),
            "similarity_score": float(self.similarity_score),
            "matching_features": self.matching_features
        }


class CaseDatabase:
    """Database for storing and retrieving historical contingency cases."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize case database.
        
        Args:
            db_path: Path to database file (JSON)
        """
        self.db_path = db_path
        self.cases: List[HistoricalCase] = []
        self._action_success_rates: Dict[str, Dict[str, float]] = {}
        
        if db_path and db_path.exists():
            self.load()
        
        logger.info(f"CaseDatabase initialized with {len(self.cases)} cases")
    
    def add_case(self, case: HistoricalCase) -> None:
        """Add a case to the database."""
        self.cases.append(case)
        self._update_success_rates(case)
        logger.info(f"Added case: {case.case_id}")
        
        if self.db_path:
            self.save()
    
    def _update_success_rates(self, case: HistoricalCase) -> None:
        """Update action success rate tracking."""
        key = f"{case.contingency_type}:{case.action_type}"
        
        if key not in self._action_success_rates:
            self._action_success_rates[key] = {"successes": 0, "total": 0}
        
        self._action_success_rates[key]["total"] += 1
        if case.action_successful:
            self._action_success_rates[key]["successes"] += 1
    
    def get_action_success_rate(self, contingency_type: str, action_type: str) -> float:
        """
        Get historical success rate for an action type.
        
        Args:
            contingency_type: Type of contingency
            action_type: Type of action
            
        Returns:
            Success rate (0-1)
        """
        key = f"{contingency_type}:{action_type}"
        stats = self._action_success_rates.get(key, {"successes": 0, "total": 0})
        
        if stats["total"] == 0:
            return 0.5  # No data, assume 50%
        
        return stats["successes"] / stats["total"]
    
    def find_similar_cases(self, contingency_type: str,
                          affected_elements: List[int],
                          network_name: Optional[str] = None,
                          max_results: int = 5) -> List[CaseSearchResult]:
        """
        Find similar historical cases.
        
        Args:
            contingency_type: Type of contingency
            affected_elements: Elements affected by contingency
            network_name: Optional network name filter
            max_results: Maximum number of results
            
        Returns:
            List of similar cases with similarity scores
        """
        results = []
        
        for case in self.cases:
            score = 0.0
            matching_features = []
            
            # Match contingency type (high weight)
            if case.contingency_type == contingency_type:
                score += 0.4
                matching_features.append("contingency_type")
            
            # Match network (medium weight)
            if network_name and case.network_name == network_name:
                score += 0.2
                matching_features.append("network_name")
            
            # Match affected elements (medium weight)
            common_elements = set(case.affected_elements) & set(affected_elements)
            if common_elements:
                element_similarity = len(common_elements) / max(len(case.affected_elements), len(affected_elements))
                score += 0.2 * element_similarity
                matching_features.append(f"elements:{len(common_elements)}")
            
            # Match number of affected elements (low weight)
            size_diff = abs(len(case.affected_elements) - len(affected_elements))
            if size_diff <= 2:
                score += 0.1 * (1 - size_diff / 3)
                matching_features.append("similar_size")
            
            # Bonus for successful cases
            if case.action_successful:
                score += 0.1
                matching_features.append("successful")
            
            if score > 0.2:  # Minimum threshold
                results.append(CaseSearchResult(
                    case=case,
                    similarity_score=score,
                    matching_features=matching_features
                ))
        
        # Sort by similarity score
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return results[:max_results]
    
    def get_recommended_actions(self, contingency_type: str,
                               affected_elements: List[int],
                               network_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recommended actions based on historical cases.
        
        Args:
            contingency_type: Type of contingency
            affected_elements: Elements affected by contingency
            network_name: Optional network name filter
            
        Returns:
            List of recommended actions with confidence scores
        """
        similar_cases = self.find_similar_cases(
            contingency_type, affected_elements, network_name, max_results=10
        )
        
        if not similar_cases:
            return []
        
        # Aggregate action recommendations
        action_scores: Dict[str, Dict[str, Any]] = {}
        
        for result in similar_cases:
            case = result.case
            action_key = case.action_type
            
            if action_key not in action_scores:
                action_scores[action_key] = {
                    "action_type": case.action_type,
                    "total_score": 0.0,
                    "success_count": 0,
                    "total_count": 0,
                    "example_elements": [],
                    "example_descriptions": []
                }
            
            action_scores[action_key]["total_score"] += result.similarity_score
            action_scores[action_key]["total_count"] += 1
            
            if case.action_successful:
                action_scores[action_key]["success_count"] += 1
            
            if len(action_scores[action_key]["example_elements"]) < 3:
                action_scores[action_key]["example_elements"].append(case.action_elements)
                action_scores[action_key]["example_descriptions"].append(case.action_description)
        
        # Calculate confidence and sort
        recommendations = []
        for action_key, data in action_scores.items():
            success_rate = data["success_count"] / data["total_count"] if data["total_count"] > 0 else 0
            avg_similarity = data["total_score"] / data["total_count"] if data["total_count"] > 0 else 0
            confidence = (success_rate * 0.6 + avg_similarity * 0.4)
            
            recommendations.append({
                "action_type": data["action_type"],
                "confidence": float(confidence),
                "success_rate": float(success_rate),
                "based_on_cases": data["total_count"],
                "example_elements": data["example_elements"][0] if data["example_elements"] else [],
                "example_description": data["example_descriptions"][0] if data["example_descriptions"] else ""
            })
        
        recommendations.sort(key=lambda r: r["confidence"], reverse=True)
        return recommendations
    
    def record_case_from_workflow(self, state: Dict[str, Any],
                                  action_successful: bool) -> HistoricalCase:
        """
        Record a case from workflow state.
        
        Args:
            state: Workflow state dictionary
            action_successful: Whether the action was successful
            
        Returns:
            Created HistoricalCase
        """
        case_id = f"case_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{len(self.cases)}"
        
        # Extract baseline results
        baseline = state.get("baseline_results", {})
        contingency = state.get("contingency_results", {})
        selected = state.get("selected_action", {})
        
        case = HistoricalCase(
            case_id=case_id,
            timestamp=datetime.now(timezone.utc),
            network_name=state.get("network_name", "unknown"),
            contingency_type=state.get("contingency_type", "unknown"),
            contingency_description=state.get("contingency_description", ""),
            affected_elements=state.get("contingency_elements", []),
            baseline_min_voltage=baseline.get("min_voltage_pu", 1.0),
            baseline_max_loading=baseline.get("max_line_loading_percent", 0.0),
            baseline_losses_mw=baseline.get("total_losses_mw", 0.0),
            post_contingency_min_voltage=contingency.get("min_voltage_pu", 1.0),
            post_contingency_max_loading=contingency.get("max_line_loading_percent", 0.0),
            post_contingency_losses_mw=contingency.get("total_losses_mw", 0.0),
            violations=state.get("constraint_violations", []),
            action_type=selected.get("action_type", "") if selected else "",
            action_elements=selected.get("target_elements", []) if selected else [],
            action_description=selected.get("expected_impact", "") if selected else "",
            action_successful=action_successful,
            violations_resolved=selected.get("violations_resolved", []) if selected else []
        )
        
        self.add_case(case)
        return case
    
    def add_operator_feedback(self, case_id: str, feedback: str, rating: int) -> bool:
        """
        Add operator feedback to a case.
        
        Args:
            case_id: Case ID
            feedback: Operator feedback text
            rating: Rating (1-5)
            
        Returns:
            True if case was found and updated
        """
        for case in self.cases:
            if case.case_id == case_id:
                case.operator_feedback = feedback
                case.operator_rating = max(1, min(5, rating))
                
                if self.db_path:
                    self.save()
                
                logger.info(f"Added feedback to case {case_id}: rating={rating}")
                return True
        
        logger.warning(f"Case not found: {case_id}")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.cases:
            return {"total_cases": 0}
        
        successful = sum(1 for c in self.cases if c.action_successful)
        
        # Count by contingency type
        by_type: Dict[str, int] = {}
        for case in self.cases:
            by_type[case.contingency_type] = by_type.get(case.contingency_type, 0) + 1
        
        # Count by action type
        by_action: Dict[str, int] = {}
        for case in self.cases:
            by_action[case.action_type] = by_action.get(case.action_type, 0) + 1
        
        # Average rating
        rated_cases = [c for c in self.cases if c.operator_rating > 0]
        avg_rating = sum(c.operator_rating for c in rated_cases) / len(rated_cases) if rated_cases else 0
        
        return {
            "total_cases": len(self.cases),
            "successful_cases": successful,
            "success_rate": successful / len(self.cases) if self.cases else 0,
            "by_contingency_type": by_type,
            "by_action_type": by_action,
            "rated_cases": len(rated_cases),
            "average_rating": avg_rating
        }
    
    def save(self) -> None:
        """Save database to file."""
        if not self.db_path:
            logger.warning("No database path set")
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "cases": [c.to_dict() for c in self.cases],
            "success_rates": self._action_success_rates
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.cases)} cases to {self.db_path}")
    
    def load(self) -> None:
        """Load database from file."""
        if not self.db_path or not self.db_path.exists():
            logger.warning(f"Database file not found: {self.db_path}")
            return
        
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        self.cases = [HistoricalCase.from_dict(c) for c in data.get("cases", [])]
        self._action_success_rates = data.get("success_rates", {})
        
        logger.info(f"Loaded {len(self.cases)} cases from {self.db_path}")
    
    def export_for_training(self) -> List[Dict[str, Any]]:
        """Export cases in format suitable for ML training."""
        training_data = []
        
        for case in self.cases:
            features = case.get_similarity_features()
            features["label"] = 1 if case.action_successful else 0
            features["case_id"] = case.case_id
            training_data.append(features)
        
        return training_data


def create_case_database(db_path: Optional[Path] = None) -> CaseDatabase:
    """Create a case database instance."""
    return CaseDatabase(db_path)


if __name__ == "__main__":
    # Test case database
    from pathlib import Path
    import tempfile
    
    print("\n=== Case Database Test ===")
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cases.json"
        db = CaseDatabase(db_path)
        
        # Add some test cases
        case1 = HistoricalCase(
            case_id="case_001",
            timestamp=datetime.now(timezone.utc),
            network_name="ieee_33",
            contingency_type="line_outage",
            contingency_description="Line 5 outage",
            affected_elements=[5],
            baseline_min_voltage=0.98,
            post_contingency_min_voltage=0.92,
            violations=["Bus 18: 0.92 pu (< 0.95 pu)"],
            action_type="switch_line",
            action_elements=[{"line_id": 32, "close": True}],
            action_description="Close tie switch 32",
            action_successful=True,
            violations_resolved=["Bus 18: 0.92 pu (< 0.95 pu)"]
        )
        db.add_case(case1)
        
        case2 = HistoricalCase(
            case_id="case_002",
            timestamp=datetime.now(timezone.utc),
            network_name="ieee_33",
            contingency_type="line_outage",
            contingency_description="Line 10 outage",
            affected_elements=[10],
            baseline_min_voltage=0.98,
            post_contingency_min_voltage=0.94,
            violations=["Bus 25: 0.94 pu (< 0.95 pu)"],
            action_type="shed_load",
            action_elements=[{"load_id": 25, "reduction_percent": 20}],
            action_description="Shed 20% load at bus 25",
            action_successful=True
        )
        db.add_case(case2)
        
        print(f"\nDatabase Statistics:")
        print(json.dumps(db.get_statistics(), indent=2))
        
        # Find similar cases
        print("\n=== Finding Similar Cases ===")
        similar = db.find_similar_cases("line_outage", [5, 6], "ieee_33")
        for result in similar:
            print(f"Case: {result.case.case_id}, Score: {result.similarity_score:.2f}")
            print(f"  Matching: {result.matching_features}")
        
        # Get recommendations
        print("\n=== Recommended Actions ===")
        recommendations = db.get_recommended_actions("line_outage", [5], "ieee_33")
        for rec in recommendations:
            print(f"Action: {rec['action_type']}, Confidence: {rec['confidence']:.2f}")
        
        # Test success rate
        print(f"\nSuccess rate for line_outage:switch_line: "
              f"{db.get_action_success_rate('line_outage', 'switch_line'):.2f}")

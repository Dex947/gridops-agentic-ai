"""Tests for case database module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.case_database import (
    CaseDatabase,
    CaseSearchResult,
    HistoricalCase,
    create_case_database,
)


@pytest.fixture
def sample_case():
    """Create a sample historical case."""
    return HistoricalCase(
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


@pytest.fixture
def case_database():
    """Create a case database without persistence."""
    return CaseDatabase()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_cases.json"


class TestHistoricalCase:
    """Tests for HistoricalCase class."""

    def test_create_case(self, sample_case):
        """Test creating a historical case."""
        assert sample_case.case_id == "case_001"
        assert sample_case.network_name == "ieee_33"
        assert sample_case.contingency_type == "line_outage"
        assert sample_case.action_successful is True

    def test_to_dict(self, sample_case):
        """Test conversion to dictionary."""
        data = sample_case.to_dict()
        
        assert data["case_id"] == "case_001"
        assert data["network_name"] == "ieee_33"
        assert data["action_successful"] is True
        assert isinstance(data["timestamp"], str)

    def test_from_dict(self, sample_case):
        """Test creation from dictionary."""
        data = sample_case.to_dict()
        restored = HistoricalCase.from_dict(data)
        
        assert restored.case_id == sample_case.case_id
        assert restored.network_name == sample_case.network_name
        assert restored.action_successful == sample_case.action_successful

    def test_get_similarity_features(self, sample_case):
        """Test getting similarity features."""
        features = sample_case.get_similarity_features()
        
        assert "network_name" in features
        assert "contingency_type" in features
        assert "num_affected_elements" in features
        assert "success" in features
        assert features["success"] is True


class TestCaseDatabase:
    """Tests for CaseDatabase class."""

    def test_init_empty(self, case_database):
        """Test initialization with empty database."""
        assert len(case_database.cases) == 0

    def test_add_case(self, case_database, sample_case):
        """Test adding a case."""
        case_database.add_case(sample_case)
        assert len(case_database.cases) == 1
        assert case_database.cases[0].case_id == "case_001"

    def test_get_action_success_rate(self, case_database, sample_case):
        """Test getting action success rate."""
        case_database.add_case(sample_case)
        
        rate = case_database.get_action_success_rate("line_outage", "switch_line")
        assert rate == 1.0  # 100% success (1 successful case)

    def test_get_action_success_rate_no_data(self, case_database):
        """Test success rate with no data."""
        rate = case_database.get_action_success_rate("unknown", "unknown")
        assert rate == 0.5  # Default 50% when no data

    def test_find_similar_cases(self, case_database, sample_case):
        """Test finding similar cases."""
        case_database.add_case(sample_case)
        
        # Add another case
        case2 = HistoricalCase(
            case_id="case_002",
            timestamp=datetime.now(timezone.utc),
            network_name="ieee_33",
            contingency_type="line_outage",
            contingency_description="Line 10 outage",
            affected_elements=[10],
            action_type="shed_load",
            action_successful=True
        )
        case_database.add_case(case2)
        
        # Search for similar cases
        results = case_database.find_similar_cases(
            "line_outage", [5], "ieee_33"
        )
        
        assert len(results) > 0
        assert all(isinstance(r, CaseSearchResult) for r in results)
        # First result should be most similar
        assert results[0].similarity_score >= results[-1].similarity_score

    def test_find_similar_cases_no_match(self, case_database, sample_case):
        """Test finding similar cases with no match."""
        case_database.add_case(sample_case)
        
        results = case_database.find_similar_cases(
            "transformer_outage", [100], "different_network"
        )
        
        # Should return empty or low-score results
        assert all(r.similarity_score < 0.5 for r in results)

    def test_get_recommended_actions(self, case_database, sample_case):
        """Test getting recommended actions."""
        case_database.add_case(sample_case)
        
        recommendations = case_database.get_recommended_actions(
            "line_outage", [5], "ieee_33"
        )
        
        assert len(recommendations) > 0
        assert "action_type" in recommendations[0]
        assert "confidence" in recommendations[0]

    def test_add_operator_feedback(self, case_database, sample_case):
        """Test adding operator feedback."""
        case_database.add_case(sample_case)
        
        success = case_database.add_operator_feedback(
            "case_001", "Good recommendation", 5
        )
        
        assert success is True
        assert case_database.cases[0].operator_feedback == "Good recommendation"
        assert case_database.cases[0].operator_rating == 5

    def test_add_operator_feedback_invalid_case(self, case_database):
        """Test adding feedback to non-existent case."""
        success = case_database.add_operator_feedback(
            "invalid_case", "Feedback", 3
        )
        assert success is False

    def test_get_statistics(self, case_database, sample_case):
        """Test getting database statistics."""
        case_database.add_case(sample_case)
        
        stats = case_database.get_statistics()
        
        assert stats["total_cases"] == 1
        assert stats["successful_cases"] == 1
        assert stats["success_rate"] == 1.0
        assert "line_outage" in stats["by_contingency_type"]

    def test_get_statistics_empty(self, case_database):
        """Test statistics with empty database."""
        stats = case_database.get_statistics()
        assert stats["total_cases"] == 0


class TestCaseDatabasePersistence:
    """Tests for database persistence."""

    def test_save_and_load(self, temp_db_path, sample_case):
        """Test saving and loading database."""
        # Create and save
        db1 = CaseDatabase(temp_db_path)
        db1.add_case(sample_case)
        db1.save()
        
        # Load in new instance
        db2 = CaseDatabase(temp_db_path)
        
        assert len(db2.cases) == 1
        assert db2.cases[0].case_id == "case_001"

    def test_auto_save_on_add(self, temp_db_path, sample_case):
        """Test auto-save when adding case."""
        db = CaseDatabase(temp_db_path)
        db.add_case(sample_case)
        
        # Check file exists
        assert temp_db_path.exists()
        
        # Verify content
        with open(temp_db_path) as f:
            data = json.load(f)
        assert len(data["cases"]) == 1

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        db = CaseDatabase(Path("/nonexistent/path/db.json"))
        assert len(db.cases) == 0


class TestRecordFromWorkflow:
    """Tests for recording cases from workflow state."""

    def test_record_case_from_workflow(self, case_database):
        """Test recording case from workflow state."""
        state = {
            "network_name": "ieee_33",
            "contingency_type": "line_outage",
            "contingency_description": "Line 5 failure",
            "contingency_elements": [5],
            "baseline_results": {
                "min_voltage_pu": 0.98,
                "max_line_loading_percent": 80.0,
                "total_losses_mw": 0.5
            },
            "contingency_results": {
                "min_voltage_pu": 0.92,
                "max_line_loading_percent": 95.0,
                "total_losses_mw": 0.8
            },
            "constraint_violations": ["Bus 18: 0.92 pu"],
            "selected_action": {
                "action_type": "switch_line",
                "target_elements": [{"line_id": 32}],
                "expected_impact": "Restore voltage"
            }
        }
        
        case = case_database.record_case_from_workflow(state, action_successful=True)
        
        assert case is not None
        assert case.network_name == "ieee_33"
        assert case.contingency_type == "line_outage"
        assert case.action_successful is True
        assert len(case_database.cases) == 1


class TestExportForTraining:
    """Tests for ML training export."""

    def test_export_for_training(self, case_database, sample_case):
        """Test exporting cases for ML training."""
        case_database.add_case(sample_case)
        
        training_data = case_database.export_for_training()
        
        assert len(training_data) == 1
        assert "label" in training_data[0]
        assert training_data[0]["label"] == 1  # Successful case


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_create_case_database(self):
        """Test creating case database via convenience function."""
        db = create_case_database()
        assert isinstance(db, CaseDatabase)
        assert len(db.cases) == 0

    def test_create_case_database_with_path(self, temp_db_path):
        """Test creating case database with path."""
        db = create_case_database(temp_db_path)
        assert isinstance(db, CaseDatabase)
        assert db.db_path == temp_db_path

"""Tests for report_generator module."""

import pytest
from pathlib import Path

from src.report_generator import ReportGenerator
from src.core.state_manager import create_initial_state


@pytest.fixture
def report_gen(tmp_path):
    """Create report generator with temp output directory."""
    return ReportGenerator(output_dir=tmp_path)


@pytest.fixture
def sample_state():
    """Create sample state for report generation."""
    state = create_initial_state("ieee_33", "Line 5 outage", 5)
    state["network_loaded"] = True
    state["network_summary"] = {
        "buses": 33,
        "lines": 37,
        "loads": 32,
        "total_load_p_mw": 3.715,
        "total_load_q_mvar": 2.3,
        "voltage_levels_kv": [12.66]
    }
    state["contingency_type"] = "line_outage"
    state["contingency_elements"] = [5]
    state["baseline_results"] = {
        "converged": True,
        "min_voltage_pu": 0.913,
        "max_voltage_pu": 1.0,
        "max_line_loading_percent": 66.67,
        "total_losses_mw": 0.203,
        "violations": ["Bus 28: undervoltage"]
    }
    state["contingency_results"] = {
        "converged": True,
        "min_voltage_pu": 0.938,
        "max_voltage_pu": 1.0,
        "max_line_loading_percent": 58.4,
        "total_losses_mw": 0.0
    }
    state["constraint_violations"] = [
        "Bus 28: 0.947 pu (undervoltage)",
        "Bus 29: 0.943 pu (undervoltage)"
    ]
    state["proposed_actions"] = []
    state["evaluated_actions"] = []
    state["selected_action"] = None
    state["explanation"] = "No suitable action identified."
    state["references"] = ["ANSI C84.1-2020", "NERC TPL-001-4"]
    state["workflow_status"] = "complete"
    return state


class TestReportGeneratorInit:
    """Tests for initialization."""

    def test_init_creates_output_dir(self, tmp_path):
        """Should create output directory."""
        output_dir = tmp_path / "reports"
        gen = ReportGenerator(output_dir=output_dir)
        
        assert output_dir.exists()


class TestMarkdownReport:
    """Tests for Markdown report generation."""

    def test_generate_markdown_report(self, report_gen, sample_state):
        """Should generate Markdown report."""
        plots = {"voltage_profile": None, "line_loading": None}
        
        output_path = report_gen.generate_markdown_report(
            state=sample_state,
            plots=plots,
            session_id="test_session"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".md"
        
        content = output_path.read_text()
        assert "ieee_33" in content
        assert "Line 5 outage" in content

    def test_markdown_includes_network_info(self, report_gen, sample_state):
        """Should include network information."""
        plots = {}
        output_path = report_gen.generate_markdown_report(
            state=sample_state,
            plots=plots,
            session_id="test_net_info"
        )
        
        content = output_path.read_text()
        assert "33" in content  # buses
        assert "37" in content  # lines

    def test_markdown_includes_violations(self, report_gen, sample_state):
        """Should include constraint violations."""
        plots = {}
        output_path = report_gen.generate_markdown_report(
            state=sample_state,
            plots=plots,
            session_id="test_violations"
        )
        
        content = output_path.read_text()
        assert "undervoltage" in content.lower()

    def test_markdown_includes_references(self, report_gen, sample_state):
        """Should include references."""
        plots = {}
        output_path = report_gen.generate_markdown_report(
            state=sample_state,
            plots=plots,
            session_id="test_refs"
        )
        
        content = output_path.read_text()
        assert "ANSI" in content or "NERC" in content


class TestLatexReport:
    """Tests for LaTeX report generation."""

    def test_generate_latex_report(self, report_gen, sample_state):
        """Should generate LaTeX report."""
        plots = {}
        output_path = report_gen.generate_latex_report(
            state=sample_state,
            plots=plots,
            session_id="test_latex"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".tex"
        
        content = output_path.read_text()
        assert "documentclass" in content or "ieee_33" in content

    def test_latex_escapes_special_chars(self, report_gen, sample_state):
        """Should handle special LaTeX characters."""
        sample_state["contingency_description"] = "Test with 50% load & special_chars"
        plots = {}
        
        output_path = report_gen.generate_latex_report(
            state=sample_state,
            plots=plots,
            session_id="test_escape"
        )
        
        assert output_path.exists()


class TestReportWithActions:
    """Tests for reports with action proposals."""

    def test_report_with_selected_action(self, report_gen, sample_state):
        """Should include selected action details."""
        sample_state["proposed_actions"] = [
            {"action_id": "switch_1", "action_type": "switch_line"}
        ]
        sample_state["evaluated_actions"] = [
            {
                "action_id": "switch_1",
                "feasible": True,
                "safety_score": 0.85,
                "recommendation": "approve"
            }
        ]
        sample_state["selected_action"] = {
            "action_id": "switch_1",
            "action_type": "switch_line"
        }
        plots = {}
        
        output_path = report_gen.generate_markdown_report(
            state=sample_state,
            plots=plots,
            session_id="test_action"
        )
        
        content = output_path.read_text()
        assert "switch_1" in content or "switch_line" in content

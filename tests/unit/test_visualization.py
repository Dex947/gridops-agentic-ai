"""Tests for visualization module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from src.visualization import NetworkVisualizer


@pytest.fixture
def visualizer(tmp_path):
    """Create visualizer with temp output directory."""
    return NetworkVisualizer(output_dir=tmp_path, dpi=100)


class TestNetworkVisualizerInit:
    """Tests for visualizer initialization."""

    def test_init_creates_output_dir(self, tmp_path):
        """Should create output directory."""
        output_dir = tmp_path / "plots"
        viz = NetworkVisualizer(output_dir=output_dir)
        
        assert output_dir.exists()

    def test_init_sets_dpi(self, tmp_path):
        """Should set DPI value."""
        viz = NetworkVisualizer(output_dir=tmp_path, dpi=150)
        
        assert viz.dpi == 150


class TestPlotVoltageProfile:
    """Tests for voltage profile plotting."""

    def test_plot_voltage_profile(self, visualizer, converged_network):
        """Should generate voltage profile plot."""
        output_path = visualizer.plot_voltage_profile(
            converged_network,
            filename="test_voltage.png"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_plot_voltage_with_baseline(self, visualizer, converged_network):
        """Should plot with baseline comparison."""
        import copy
        baseline = copy.deepcopy(converged_network)
        
        output_path = visualizer.plot_voltage_profile(
            converged_network,
            baseline_net=baseline,
            filename="test_voltage_compare.png"
        )
        
        assert output_path.exists()

    def test_plot_voltage_custom_limits(self, visualizer, converged_network):
        """Should respect custom voltage limits."""
        output_path = visualizer.plot_voltage_profile(
            converged_network,
            v_limits=(0.90, 1.10),
            filename="test_voltage_limits.png"
        )
        
        assert output_path.exists()


class TestPlotLineLoading:
    """Tests for line loading plotting."""

    def test_plot_line_loading(self, visualizer, converged_network):
        """Should generate line loading plot."""
        output_path = visualizer.plot_line_loading(
            converged_network,
            filename="test_loading.png"
        )
        
        assert output_path.exists()

    def test_plot_line_loading_custom_limit(self, visualizer, converged_network):
        """Should plot with custom thermal limit."""
        output_path = visualizer.plot_line_loading(
            converged_network,
            thermal_limit=80.0,
            filename="test_loading_limit.png"
        )
        
        assert output_path.exists()


class TestPlotNetworkTopology:
    """Tests for network topology plotting."""

    def test_plot_topology(self, visualizer, converged_network):
        """Should generate topology plot."""
        output_path = visualizer.plot_network_topology(
            converged_network,
            filename="test_topology.png"
        )
        
        assert output_path.exists()

    def test_plot_topology_with_switched_elements(self, visualizer, converged_network):
        """Should highlight switched elements."""
        output_path = visualizer.plot_network_topology(
            converged_network,
            switched_elements=[0, 1],
            filename="test_topology_switched.png"
        )
        
        assert output_path.exists()


class TestGenerateAllPlots:
    """Tests for batch plot generation."""

    def test_generate_all_plots(self, visualizer, converged_network):
        """Should generate all plot types."""
        import copy
        baseline_net = copy.deepcopy(converged_network)
        
        baseline_results = {
            "min_voltage_pu": 0.95,
            "max_voltage_pu": 1.0,
            "max_line_loading_percent": 60.0,
            "violations": [],
            "total_losses_mw": 0.1
        }
        post_results = {
            "min_voltage_pu": 0.97,
            "max_voltage_pu": 1.0,
            "max_line_loading_percent": 55.0,
            "violations": [],
            "total_losses_mw": 0.08
        }
        
        plots = visualizer.generate_all_plots(
            net_baseline=baseline_net,
            net_post_action=converged_network,
            baseline_results=baseline_results,
            post_action_results=post_results
        )
        
        assert len(plots) >= 3
        for plot_type, plot_path in plots.items():
            if plot_path:
                assert plot_path.exists()


class TestPlotComparisonMetrics:
    """Tests for comparison metrics plotting."""

    def test_plot_comparison(self, visualizer, converged_network):
        """Should generate comparison plot."""
        # API takes dicts, not networks
        baseline = {
            "min_voltage_pu": 0.95,
            "max_voltage_pu": 1.0,
            "max_line_loading_percent": 60.0,
            "violations": ["v1"],
            "total_losses_mw": 0.1
        }
        post_action = {
            "min_voltage_pu": 0.97,
            "max_voltage_pu": 1.0,
            "max_line_loading_percent": 55.0,
            "violations": [],
            "total_losses_mw": 0.08
        }
        
        output_path = visualizer.plot_comparison_metrics(
            baseline=baseline,
            post_action=post_action,
            filename="test_comparison.png"
        )
        
        assert output_path.exists()

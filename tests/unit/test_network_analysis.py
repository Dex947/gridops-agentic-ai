"""Tests for network_analysis module."""

import pytest
import pandapower as pp

from src.tools.network_analysis import (
    NetworkAnalyzer,
    find_alternative_paths,
    calculate_network_metrics,
    identify_reconfiguration_options
)


class TestNetworkAnalyzer:
    """Tests for NetworkAnalyzer class."""

    def test_init_builds_graph(self, converged_network):
        """Should build NetworkX graph on init."""
        analyzer = NetworkAnalyzer(converged_network)
        
        assert analyzer.graph is not None
        assert analyzer.graph.number_of_nodes() > 0

    def test_find_alternative_paths(self, converged_network):
        """Should find paths between buses."""
        analyzer = NetworkAnalyzer(converged_network)
        
        paths = analyzer.find_alternative_paths(0, 2, max_paths=3)
        
        assert isinstance(paths, list)
        if len(paths) > 0:
            assert paths[0][0] == 0  # Starts at source
            assert paths[0][-1] == 2  # Ends at destination

    def test_find_isolated_buses_none(self, converged_network):
        """Connected network should have no isolated buses."""
        analyzer = NetworkAnalyzer(converged_network)
        
        isolated = analyzer.find_isolated_buses()
        
        assert len(isolated) == 0

    def test_get_buses_downstream(self, ieee33_network):
        """Should find downstream buses in radial network."""
        analyzer = NetworkAnalyzer(ieee33_network)
        
        # Bus 1 should have downstream buses
        downstream = analyzer.get_buses_downstream(1)
        
        assert isinstance(downstream, list)

    def test_identify_critical_lines(self, converged_network):
        """Should identify critical lines."""
        analyzer = NetworkAnalyzer(converged_network)
        
        critical = analyzer.identify_critical_lines(threshold_loading=0.0)
        
        assert isinstance(critical, list)
        # All bridge elements should be identified
        for line in critical:
            assert "line_id" in line
            assert "reason" in line

    def test_calculate_centrality(self, converged_network):
        """Should calculate bus centrality."""
        analyzer = NetworkAnalyzer(converged_network)
        
        centrality = analyzer.calculate_centrality()
        
        assert isinstance(centrality, dict)
        assert len(centrality) == len(converged_network.bus)

    def test_get_network_metrics(self, converged_network):
        """Should return network metrics."""
        analyzer = NetworkAnalyzer(converged_network)
        
        metrics = analyzer.get_network_metrics()
        
        assert "num_buses" in metrics
        assert "num_edges" in metrics
        assert "is_connected" in metrics
        assert metrics["is_connected"] == True


class TestStandaloneFunctions:
    """Tests for standalone analysis functions."""

    def test_find_alternative_paths_func(self, converged_network):
        """Standalone function should work."""
        paths = find_alternative_paths(converged_network, 0, 2, max_paths=2)
        
        assert isinstance(paths, list)

    def test_calculate_network_metrics_func(self, converged_network):
        """Should include power system metrics."""
        metrics = calculate_network_metrics(converged_network)
        
        assert "total_load_mw" in metrics
        assert "num_loads" in metrics
        assert metrics["total_load_mw"] > 0

    def test_identify_reconfiguration_options(self, ieee33_network):
        """Should find reconfiguration options for affected buses."""
        options = identify_reconfiguration_options(
            ieee33_network,
            affected_buses=[10, 15]
        )
        
        assert isinstance(options, list)

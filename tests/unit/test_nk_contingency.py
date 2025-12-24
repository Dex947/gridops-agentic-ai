"""Tests for N-k contingency analysis."""

import pytest
import pandapower as pp
import pandapower.networks as pn

from src.core.contingency_simulator import (
    ContingencySimulator,
    ContingencyType,
    ContingencyEvent,
    ContingencyResult,
)


@pytest.fixture
def ieee33_network():
    """Load IEEE 33-bus network."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


@pytest.fixture
def simulator():
    """Create contingency simulator."""
    return ContingencySimulator(
        voltage_limits=(0.95, 1.05),
        thermal_limit_percent=100.0
    )


class TestGenerateNkContingencies:
    """Tests for N-k contingency generation."""

    def test_generate_n2_contingencies(self, simulator, ieee33_network):
        """Should generate N-2 contingencies."""
        contingencies = simulator.generate_n_k_contingencies(
            ieee33_network, k=2, max_combinations=100
        )
        
        assert len(contingencies) > 0
        assert len(contingencies) <= 100
        
        # All should be MULTIPLE_OUTAGE type
        for c in contingencies:
            assert c.event_type == ContingencyType.MULTIPLE_OUTAGE

    def test_generate_n3_contingencies(self, simulator, ieee33_network):
        """Should generate N-3 contingencies."""
        contingencies = simulator.generate_n_k_contingencies(
            ieee33_network, k=3, max_combinations=50
        )
        
        assert len(contingencies) > 0
        assert len(contingencies) <= 50

    def test_max_combinations_limit(self, simulator, ieee33_network):
        """Should respect max_combinations limit."""
        contingencies = simulator.generate_n_k_contingencies(
            ieee33_network, k=2, max_combinations=20
        )
        
        assert len(contingencies) <= 20

    def test_contingency_has_affected_buses(self, simulator, ieee33_network):
        """Contingencies should list affected buses."""
        contingencies = simulator.generate_n_k_contingencies(
            ieee33_network, k=2, max_combinations=10
        )
        
        for c in contingencies:
            assert len(c.affected_buses) >= 2  # At least 2 buses per outage

    def test_contingency_description(self, simulator, ieee33_network):
        """Contingencies should have descriptive names."""
        contingencies = simulator.generate_n_k_contingencies(
            ieee33_network, k=2, max_combinations=5
        )
        
        for c in contingencies:
            assert "N-2" in c.description
            assert "outage" in c.description.lower()


class TestSimulateNkContingency:
    """Tests for N-k contingency simulation."""

    def test_simulate_n2_contingency(self, simulator, ieee33_network):
        """Should simulate N-2 contingency."""
        import copy
        net = copy.deepcopy(ieee33_network)
        
        result = simulator.simulate_n_k_contingency(net, line_indices=[0, 1])
        
        assert isinstance(result, ContingencyResult)
        assert result.contingency.event_type == ContingencyType.MULTIPLE_OUTAGE

    def test_lines_taken_out_of_service(self, simulator, ieee33_network):
        """Lines should be taken out of service."""
        import copy
        net = copy.deepcopy(ieee33_network)
        
        simulator.simulate_n_k_contingency(net, line_indices=[5, 10])
        
        assert net.line.at[5, 'in_service'] == False
        assert net.line.at[10, 'in_service'] == False

    def test_result_has_violations(self, simulator, ieee33_network):
        """Result should track violations."""
        import copy
        net = copy.deepcopy(ieee33_network)
        
        result = simulator.simulate_n_k_contingency(net, line_indices=[0, 1])
        
        assert hasattr(result, 'violated_constraints')
        assert isinstance(result.violated_constraints, list)


class TestRunNkAnalysis:
    """Tests for full N-k analysis."""

    def test_run_n2_analysis(self, simulator, ieee33_network):
        """Should run complete N-2 analysis."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=2, max_combinations=20
        )
        
        assert len(results) > 0
        assert len(results) <= 20
        
        for r in results:
            assert isinstance(r, ContingencyResult)

    def test_results_sorted_by_severity(self, simulator, ieee33_network):
        """Results should be sorted by severity."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=2, max_combinations=30
        )
        
        if len(results) >= 2:
            # Critical results should come first
            critical_indices = [i for i, r in enumerate(results) if r.is_critical]
            non_critical_indices = [i for i, r in enumerate(results) if not r.is_critical]
            
            if critical_indices and non_critical_indices:
                assert max(critical_indices) < min(non_critical_indices)

    def test_analysis_with_small_k(self, simulator, ieee33_network):
        """Should handle k=1 (same as N-1)."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=1, max_combinations=10
        )
        
        assert len(results) > 0


class TestRankContingencies:
    """Tests for contingency ranking."""

    def test_rank_contingencies_by_severity(self, simulator, ieee33_network):
        """Should rank contingencies by severity score."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=2, max_combinations=20
        )
        
        ranked = simulator.rank_contingencies_by_severity(results)
        
        assert len(ranked) == len(results)
        
        # Should be sorted by severity_score descending
        for i in range(len(ranked) - 1):
            assert ranked[i]['severity_score'] >= ranked[i+1]['severity_score']

    def test_ranking_includes_required_fields(self, simulator, ieee33_network):
        """Ranking should include all required fields."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=2, max_combinations=10
        )
        
        ranked = simulator.rank_contingencies_by_severity(results)
        
        for item in ranked:
            assert 'contingency' in item
            assert 'severity_score' in item
            assert 'converged' in item
            assert 'violations' in item
            assert 'is_critical' in item

    def test_non_converged_has_high_severity(self, simulator, ieee33_network):
        """Non-converged contingencies should have high severity."""
        results = simulator.run_n_k_analysis(
            ieee33_network, k=2, max_combinations=50
        )
        
        ranked = simulator.rank_contingencies_by_severity(results)
        
        # Find non-converged results
        non_converged = [r for r in ranked if not r['converged']]
        converged = [r for r in ranked if r['converged']]
        
        if non_converged and converged:
            # Non-converged should have severity >= 100
            for nc in non_converged:
                assert nc['severity_score'] >= 100.0


class TestParseLineIndices:
    """Tests for line index parsing."""

    def test_parse_single_line(self, simulator, ieee33_network):
        """Should parse single line contingency."""
        contingency = ContingencyEvent(
            event_type=ContingencyType.LINE_OUTAGE,
            element_index=5,
            element_name="Line 5",
            description="N-1: Line 5 outage"
        )
        
        indices = simulator._parse_line_indices_from_contingency(contingency, ieee33_network)
        
        assert indices == [5]

    def test_parse_multiple_lines(self, simulator, ieee33_network):
        """Should parse multiple line contingency."""
        contingency = ContingencyEvent(
            event_type=ContingencyType.MULTIPLE_OUTAGE,
            element_index=5,
            element_name="Line 5, Line 10, Line 15",
            description="N-3 outage"
        )
        
        indices = simulator._parse_line_indices_from_contingency(contingency, ieee33_network)
        
        assert 5 in indices
        assert 10 in indices
        assert 15 in indices

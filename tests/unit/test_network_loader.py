"""Tests for network_loader module."""

import pytest
import pandapower as pp

from src.core.network_loader import NetworkLoader, create_test_network


class TestNetworkLoader:
    """Tests for NetworkLoader class."""

    def test_list_available_networks(self, paths):
        """Should return list of built-in networks."""
        loader = NetworkLoader(networks_path=paths.networks)
        networks = loader.list_available_networks()
        
        assert isinstance(networks, list)
        assert len(networks) >= 5
        assert "ieee_33" in networks

    def test_load_ieee33(self, paths):
        """Should load IEEE 33-bus network."""
        loader = NetworkLoader(networks_path=paths.networks)
        net = loader.load_network("ieee_33")
        
        assert isinstance(net, pp.pandapowerNet)
        assert len(net.bus) == 33
        assert len(net.line) == 37
        assert len(net.load) == 32

    def test_load_network_caching(self, paths):
        """Should cache loaded networks."""
        loader = NetworkLoader(networks_path=paths.networks)
        
        net1 = loader.load_network("ieee_33")
        net2 = loader.load_network("ieee_33")
        
        # Should be different objects (deepcopy)
        assert net1 is not net2
        # But loader should have cached version
        assert "ieee_33" in loader.loaded_networks

    def test_load_unknown_network_raises(self, paths):
        """Should raise ValueError for unknown network."""
        loader = NetworkLoader(networks_path=paths.networks)
        
        with pytest.raises(ValueError, match="Unknown network"):
            loader.load_network("nonexistent_network")

    def test_get_network_summary(self, paths, ieee33_network):
        """Should return correct network statistics."""
        loader = NetworkLoader(networks_path=paths.networks)
        summary = loader.get_network_summary(ieee33_network)
        
        assert summary["buses"] == 33
        assert summary["lines"] == 37
        assert summary["loads"] == 32
        assert summary["total_load_p_mw"] > 0
        assert "voltage_levels_kv" in summary

    def test_get_switchable_elements(self, paths, ieee33_network):
        """Should identify switchable lines."""
        loader = NetworkLoader(networks_path=paths.networks)
        switchable = loader.get_switchable_elements(ieee33_network)
        
        assert isinstance(switchable, list)
        assert len(switchable) > 0
        # All lines should be switchable
        line_switches = [s for s in switchable if s["type"] == "line"]
        assert len(line_switches) == 37

    def test_line_ratings_fixed(self, paths):
        """Should fix unrealistic line ratings on load."""
        loader = NetworkLoader(networks_path=paths.networks)
        net = loader.load_network("ieee_33")
        
        # Ratings should be realistic (< 1000 kA)
        assert net.line.max_i_ka.max() < 1000
        # Should have meaningful values
        assert net.line.max_i_ka.min() >= 0.1


class TestCreateTestNetwork:
    """Tests for create_test_network function."""

    def test_create_simple_network(self):
        """Should create valid 4-bus network."""
        net = create_test_network("simple")
        
        assert len(net.bus) == 4
        assert len(net.line) == 3
        assert len(net.load) == 3
        assert len(net.ext_grid) == 1

    def test_simple_network_converges(self):
        """Simple network should converge."""
        net = create_test_network("simple")
        pp.runpp(net)
        
        assert net.converged

    def test_unknown_type_raises(self):
        """Should raise for unknown network type."""
        with pytest.raises(ValueError):
            create_test_network("unknown_type")

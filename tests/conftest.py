"""Pytest fixtures for GridOps tests."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandapower as pp
import pandapower.networks as pn

from src.config import SystemConfig, NetworkConstraints, PathConfig


@pytest.fixture(scope="session")
def config():
    """System configuration fixture."""
    return SystemConfig()


@pytest.fixture(scope="session")
def constraints(config):
    """Network constraints fixture."""
    return NetworkConstraints(config)


@pytest.fixture(scope="session")
def paths():
    """Path configuration fixture."""
    return PathConfig()


@pytest.fixture
def ieee33_network():
    """Load IEEE 33-bus network for testing."""
    net = pn.case33bw()
    # Fix unrealistic line ratings
    pp.runpp(net, algorithm="bfsw")
    if net.converged:
        import numpy as np
        max_currents = np.maximum(
            net.res_line.i_from_ka.values,
            net.res_line.i_to_ka.values
        )
        net.line['max_i_ka'] = np.maximum(max_currents * 1.5, 0.1)
    return net


@pytest.fixture
def simple_network():
    """Create minimal 4-bus test network."""
    net = pp.create_empty_network(name="Test Network")
    
    # Buses
    b0 = pp.create_bus(net, vn_kv=11.0, name="Slack")
    b1 = pp.create_bus(net, vn_kv=11.0, name="Bus1")
    b2 = pp.create_bus(net, vn_kv=11.0, name="Bus2")
    b3 = pp.create_bus(net, vn_kv=11.0, name="Bus3")
    
    # External grid
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02)
    
    # Lines
    pp.create_line(net, b0, b1, length_km=1.0, std_type="NAYY 4x150 SE")
    pp.create_line(net, b1, b2, length_km=1.0, std_type="NAYY 4x150 SE")
    pp.create_line(net, b1, b3, length_km=0.5, std_type="NAYY 4x150 SE")
    
    # Loads
    pp.create_load(net, bus=b1, p_mw=0.3, q_mvar=0.1)
    pp.create_load(net, bus=b2, p_mw=0.5, q_mvar=0.2)
    pp.create_load(net, bus=b3, p_mw=0.4, q_mvar=0.15)
    
    return net


@pytest.fixture
def converged_network(simple_network):
    """Simple network with power flow already run."""
    pp.runpp(simple_network)
    return simple_network

"""
Network loader module for GridOps Agentic AI System.
Handles loading IEEE test feeders and custom distribution networks.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import pandapower as pp
import pandapower.networks as pn
from loguru import logger
import json


class NetworkLoader:
    """Load and manage distribution network models."""
    
    # Available built-in pandapower distribution networks
    AVAILABLE_NETWORKS = {
        "ieee_33": "case33bw",  # IEEE 33-bus distribution feeder
        "cigre_mv": "create_cigre_network_mv",  # CIGRE MV benchmark
        "kerber_landnetz": "kb_extrem_landnetz",  # German rural network
        "kerber_dorfnetz": "create_kerber_dorfnetz_kabel_1",  # German village network
        "simple_dist": "simple_four_bus_system",  # Simple 4-bus test
    }
    
    def __init__(self, networks_path: Optional[Path] = None):
        """
        Initialize network loader.
        
        Args:
            networks_path: Path to custom network files
        """
        self.networks_path = networks_path
        self.loaded_networks: Dict[str, pp.pandapowerNet] = {}
    
    def load_network(self, network_name: str) -> pp.pandapowerNet:
        """
        Load a network by name.
        
        Args:
            network_name: Name of network (from AVAILABLE_NETWORKS or custom file)
        
        Returns:
            pandapower network object
        """
        # Check if already loaded
        if network_name in self.loaded_networks:
            logger.info(f"Network '{network_name}' already loaded from cache")
            return self.loaded_networks[network_name].deepcopy()
        
        # Try to load from built-in networks
        if network_name in self.AVAILABLE_NETWORKS:
            net = self._load_builtin_network(network_name)
        elif self.networks_path and (self.networks_path / f"{network_name}.json").exists():
            net = self._load_custom_network(network_name)
        else:
            logger.error(f"Network '{network_name}' not found")
            raise ValueError(f"Unknown network: {network_name}")
        
        # Cache the network
        self.loaded_networks[network_name] = net.deepcopy()
        
        logger.info(f"Network '{network_name}' loaded successfully")
        logger.info(f"  Buses: {len(net.bus)}")
        logger.info(f"  Lines: {len(net.line)}")
        logger.info(f"  Loads: {len(net.load)}")
        logger.info(f"  Generators: {len(net.gen) if 'gen' in net else len(net.sgen)}")
        
        return net
    
    def _load_builtin_network(self, network_name: str) -> pp.pandapowerNet:
        """Load a built-in pandapower network."""
        function_name = self.AVAILABLE_NETWORKS[network_name]
        
        try:
            if hasattr(pn, function_name):
                # Direct function call
                net = getattr(pn, function_name)()
            else:
                # Try as power system test case
                from pandapower.networks import power_system_test_cases as pstc
                net = getattr(pstc, function_name)()
            
            return net
        except Exception as e:
            logger.error(f"Failed to load built-in network '{network_name}': {e}")
            raise
    
    def _load_custom_network(self, network_name: str) -> pp.pandapowerNet:
        """Load a custom network from file."""
        file_path = self.networks_path / f"{network_name}.json"
        
        try:
            net = pp.from_json(str(file_path))
            return net
        except Exception as e:
            logger.error(f"Failed to load custom network from '{file_path}': {e}")
            raise
    
    def save_network(self, net: pp.pandapowerNet, network_name: str):
        """
        Save a network to file.
        
        Args:
            net: pandapower network
            network_name: Name for the saved network
        """
        if self.networks_path is None:
            raise ValueError("networks_path not set")
        
        file_path = self.networks_path / f"{network_name}.json"
        pp.to_json(net, str(file_path))
        logger.info(f"Network saved to '{file_path}'")
    
    def get_network_summary(self, net: pp.pandapowerNet) -> Dict[str, Any]:
        """
        Get summary statistics of a network.
        
        Args:
            net: pandapower network
        
        Returns:
            Dictionary with network statistics
        """
        summary = {
            "buses": len(net.bus),
            "lines": len(net.line),
            "transformers": len(net.trafo) if len(net.trafo) > 0 else 0,
            "loads": len(net.load),
            "generators": len(net.gen) if len(net.gen) > 0 else 0,
            "static_generators": len(net.sgen) if len(net.sgen) > 0 else 0,
            "switches": len(net.switch) if len(net.switch) > 0 else 0,
            "shunts": len(net.shunt) if len(net.shunt) > 0 else 0,
        }
        
        # Calculate total load
        total_p_mw = net.load.p_mw.sum()
        total_q_mvar = net.load.q_mvar.sum()
        summary["total_load_p_mw"] = float(total_p_mw)
        summary["total_load_q_mvar"] = float(total_q_mvar)
        
        # Get voltage levels
        voltage_levels = net.bus.vn_kv.unique().tolist()
        summary["voltage_levels_kv"] = sorted(voltage_levels, reverse=True)
        
        return summary
    
    def get_switchable_elements(self, net: pp.pandapowerNet) -> List[Dict[str, Any]]:
        """
        Identify switchable elements in the network.
        
        Args:
            net: pandapower network
        
        Returns:
            List of switchable elements with metadata
        """
        switchable = []
        
        # Check explicit switches
        if len(net.switch) > 0:
            for idx, switch in net.switch.iterrows():
                switchable.append({
                    "type": "switch",
                    "index": int(idx),
                    "element_type": switch.et,
                    "element_index": int(switch.element),
                    "bus": int(switch.bus),
                    "closed": bool(switch.closed),
                    "name": switch.name if "name" in switch and switch.name else f"sw_{idx}"
                })
        
        # Lines can also be switched
        for idx, line in net.line.iterrows():
            switchable.append({
                "type": "line",
                "index": int(idx),
                "from_bus": int(line.from_bus),
                "to_bus": int(line.to_bus),
                "in_service": bool(line.in_service),
                "name": line.name if "name" in line and line.name else f"line_{idx}"
            })
        
        logger.info(f"Found {len(switchable)} switchable elements")
        return switchable
    
    def list_available_networks(self) -> List[str]:
        """List all available networks."""
        networks = list(self.AVAILABLE_NETWORKS.keys())
        
        # Add custom networks if path exists
        if self.networks_path and self.networks_path.exists():
            custom = [f.stem for f in self.networks_path.glob("*.json")]
            networks.extend(custom)
        
        return sorted(networks)


def create_test_network(network_type: str = "simple") -> pp.pandapowerNet:
    """
    Create a simple test distribution network for debugging.
    
    Args:
        network_type: Type of test network ("simple", "medium", "complex")
    
    Returns:
        pandapower network
    """
    if network_type == "simple":
        # Create simple 4-bus radial distribution network
        net = pp.create_empty_network(name="Simple 4-Bus Test Network")
        
        # Create buses
        bus0 = pp.create_bus(net, vn_kv=11.0, name="Substation", type="b")
        bus1 = pp.create_bus(net, vn_kv=11.0, name="Bus 1", type="b")
        bus2 = pp.create_bus(net, vn_kv=11.0, name="Bus 2", type="b")
        bus3 = pp.create_bus(net, vn_kv=11.0, name="Bus 3", type="b")
        
        # Create external grid (slack bus)
        pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid Connection")
        
        # Create lines
        pp.create_line(net, from_bus=bus0, to_bus=bus1, length_km=1.5, 
                      std_type="NAYY 4x150 SE", name="Line 0-1")
        pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=1.0, 
                      std_type="NAYY 4x150 SE", name="Line 1-2")
        pp.create_line(net, from_bus=bus1, to_bus=bus3, length_km=0.8, 
                      std_type="NAYY 4x150 SE", name="Line 1-3")
        
        # Create loads
        pp.create_load(net, bus=bus1, p_mw=0.5, q_mvar=0.2, name="Load 1")
        pp.create_load(net, bus=bus2, p_mw=0.8, q_mvar=0.3, name="Load 2")
        pp.create_load(net, bus=bus3, p_mw=0.6, q_mvar=0.25, name="Load 3")
        
        # Create switches for reconfiguration
        pp.create_switch(net, bus=bus1, element=1, et="l", closed=True, 
                        type="LBS", name="Switch Line 1-2")
        
        logger.info("Created simple 4-bus test network")
        
    elif network_type == "medium":
        # Use IEEE 33-bus as medium complexity
        net = pn.case33bw()
        logger.info("Created IEEE 33-bus test network")
        
    else:
        raise ValueError(f"Unknown test network type: {network_type}")
    
    return net


if __name__ == "__main__":
    # Test the network loader
    from src.config import load_configuration
    
    config, _, paths = load_configuration()
    loader = NetworkLoader(networks_path=paths.networks)
    
    print("\n=== Available Networks ===")
    for net_name in loader.list_available_networks():
        print(f"  - {net_name}")
    
    print("\n=== Loading IEEE 33-bus Network ===")
    net = loader.load_network("ieee_33")
    
    summary = loader.get_network_summary(net)
    print("\nNetwork Summary:")
    print(json.dumps(summary, indent=2))
    
    print("\n=== Switchable Elements ===")
    switchable = loader.get_switchable_elements(net)
    print(f"Total switchable elements: {len(switchable)}")
    for elem in switchable[:5]:  # Show first 5
        print(f"  {elem}")

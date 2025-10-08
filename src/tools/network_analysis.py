"""
Network analysis tools for GridOps Agentic AI System.
Provides graph-based analysis and path finding capabilities.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import pandapower as pp
import pandapower.topology as top
import networkx as nx
import numpy as np
from loguru import logger


class NetworkAnalyzer:
    """Analyze distribution network topology and find alternative paths."""
    
    def __init__(self, net: pp.pandapowerNet):
        """
        Initialize network analyzer.
        
        Args:
            net: pandapower network
        """
        self.net = net
        self.graph: Optional[nx.Graph] = None
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from pandapower network."""
        try:
            # Create graph from network topology
            self.graph = top.create_nxgraph(self.net, respect_switches=True)
            logger.info(f"Network graph created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to create network graph: {e}")
            self.graph = nx.Graph()
    
    def find_alternative_paths(self, from_bus: int, to_bus: int, 
                              max_paths: int = 5) -> List[List[int]]:
        """
        Find alternative paths between two buses.
        
        Args:
            from_bus: Starting bus ID
            to_bus: Destination bus ID
            max_paths: Maximum number of paths to return
        
        Returns:
            List of paths (each path is a list of bus IDs)
        """
        if self.graph is None or not self.graph.has_node(from_bus) or not self.graph.has_node(to_bus):
            logger.warning(f"Cannot find path: buses {from_bus} or {to_bus} not in graph")
            return []
        
        try:
            # Convert MultiGraph to simple Graph if needed
            if isinstance(self.graph, nx.MultiGraph):
                simple_graph = nx.Graph(self.graph)
            else:
                simple_graph = self.graph
            
            # Use k-shortest paths algorithm on simple graph
            paths = list(nx.shortest_simple_paths(simple_graph, from_bus, to_bus))
            paths = paths[:max_paths]
            
            logger.info(f"Found {len(paths)} alternative paths from bus {from_bus} to {to_bus}")
            return paths
        except nx.NetworkXNoPath:
            logger.warning(f"No path exists between bus {from_bus} and {to_bus}")
            return []
        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []
    
    def find_isolated_buses(self) -> List[int]:
        """
        Find buses that are isolated (not connected to the main network).
        
        Returns:
            List of isolated bus IDs
        """
        if self.graph is None:
            return []
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        if len(components) <= 1:
            logger.info("No isolated buses found")
            return []
        
        # The largest component is the main network
        main_component = max(components, key=len)
        
        # All other buses are isolated
        isolated = []
        for component in components:
            if component != main_component:
                isolated.extend(list(component))
        
        logger.warning(f"Found {len(isolated)} isolated buses: {isolated}")
        return sorted(isolated)
    
    def get_buses_downstream(self, bus_id: int) -> List[int]:
        """
        Get all buses downstream of a given bus (radial networks).
        
        Args:
            bus_id: Starting bus ID
        
        Returns:
            List of downstream bus IDs
        """
        if self.graph is None or not self.graph.has_node(bus_id):
            return []
        
        # For radial networks, use BFS from slack bus to determine direction
        slack_buses = self.net.ext_grid.bus.tolist()
        
        if not slack_buses:
            logger.warning("No slack bus found")
            return []
        
        slack_bus = slack_buses[0]
        
        try:
            # Find path from slack to current bus
            path_from_slack = nx.shortest_path(self.graph, slack_bus, bus_id)
            
            # Get all nodes reachable from bus_id without going back through path_from_slack
            visited = set(path_from_slack[:-1])  # Don't include current bus
            downstream = []
            
            # BFS from bus_id
            queue = [bus_id]
            visited.add(bus_id)
            
            while queue:
                current = queue.pop(0)
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        downstream.append(neighbor)
                        queue.append(neighbor)
            
            logger.info(f"Found {len(downstream)} downstream buses from bus {bus_id}")
            return sorted(downstream)
        
        except nx.NetworkXNoPath:
            logger.warning(f"No path from slack bus to bus {bus_id}")
            return []
    
    def identify_critical_lines(self, threshold_loading: float = 80.0) -> List[Dict[str, Any]]:
        """
        Identify critical lines based on loading or connectivity.
        
        Args:
            threshold_loading: Loading percentage threshold for criticality
        
        Returns:
            List of critical line information
        """
        critical_lines = []
        
        # Check if power flow has been run
        if len(self.net.res_line) == 0:
            logger.warning("No power flow results available")
            return critical_lines
        
        # Lines with high loading
        for idx, row in self.net.res_line.iterrows():
            if row.loading_percent > threshold_loading:
                line_info = self.net.line.loc[idx]
                critical_lines.append({
                    "line_id": int(idx),
                    "from_bus": int(line_info.from_bus),
                    "to_bus": int(line_info.to_bus),
                    "loading_percent": float(row.loading_percent),
                    "reason": "high_loading",
                    "severity": "critical" if row.loading_percent > 100 else "warning"
                })
        
        # Check for articulation points (lines whose removal disconnects the network)
        if self.graph is not None:
            bridges = list(nx.bridges(self.graph))
            
            for from_bus, to_bus in bridges:
                # Find line index
                line_idx = self._find_line_index(from_bus, to_bus)
                
                if line_idx is not None:
                    # Check if already in list
                    if not any(l["line_id"] == line_idx for l in critical_lines):
                        loading = self.net.res_line.at[line_idx, "loading_percent"] if line_idx in self.net.res_line.index else 0.0
                        critical_lines.append({
                            "line_id": line_idx,
                            "from_bus": from_bus,
                            "to_bus": to_bus,
                            "loading_percent": float(loading),
                            "reason": "bridge_element",
                            "severity": "critical"
                        })
        
        logger.info(f"Identified {len(critical_lines)} critical lines")
        return critical_lines
    
    def _find_line_index(self, from_bus: int, to_bus: int) -> Optional[int]:
        """Find line index connecting two buses."""
        for idx, row in self.net.line.iterrows():
            if (row.from_bus == from_bus and row.to_bus == to_bus) or \
               (row.from_bus == to_bus and row.to_bus == from_bus):
                return int(idx)
        return None
    
    def calculate_centrality(self) -> Dict[int, float]:
        """
        Calculate betweenness centrality for all buses.
        Higher centrality = more critical for network connectivity.
        
        Returns:
            Dictionary mapping bus ID to centrality score
        """
        if self.graph is None:
            return {}
        
        try:
            centrality = nx.betweenness_centrality(self.graph)
            logger.info("Calculated betweenness centrality for all buses")
            return {int(k): float(v) for k, v in centrality.items()}
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """
        Calculate various network topology metrics.
        
        Returns:
            Dictionary of network metrics
        """
        if self.graph is None:
            return {}
        
        metrics = {
            "num_buses": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_components": nx.number_connected_components(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
        }
        
        # Calculate diameter if connected
        if metrics["is_connected"]:
            metrics["diameter"] = nx.diameter(self.graph)
            metrics["average_shortest_path"] = nx.average_shortest_path_length(self.graph)
        
        # Identify articulation points
        articulation_points = list(nx.articulation_points(self.graph))
        metrics["num_articulation_points"] = len(articulation_points)
        metrics["articulation_points"] = [int(x) for x in articulation_points]
        
        logger.info("Calculated network metrics")
        return metrics


def find_alternative_paths(net: pp.pandapowerNet, from_bus: int, to_bus: int,
                          max_paths: int = 5) -> List[List[int]]:
    """
    Standalone function to find alternative paths between buses.
    
    Args:
        net: pandapower network
        from_bus: Starting bus ID
        to_bus: Destination bus ID
        max_paths: Maximum number of paths
    
    Returns:
        List of paths
    """
    analyzer = NetworkAnalyzer(net)
    return analyzer.find_alternative_paths(from_bus, to_bus, max_paths)


def calculate_network_metrics(net: pp.pandapowerNet) -> Dict[str, Any]:
    """
    Standalone function to calculate network metrics.
    
    Args:
        net: pandapower network
    
    Returns:
        Dictionary of metrics
    """
    analyzer = NetworkAnalyzer(net)
    metrics = analyzer.get_network_metrics()
    
    # Add power system specific metrics
    metrics["total_load_mw"] = float(net.load.p_mw.sum())
    metrics["total_load_mvar"] = float(net.load.q_mvar.sum())
    metrics["num_loads"] = len(net.load)
    metrics["num_generators"] = len(net.gen) + len(net.sgen)
    
    return metrics


def identify_reconfiguration_options(net: pp.pandapowerNet,
                                     affected_buses: List[int]) -> List[Dict[str, Any]]:
    """
    Identify possible reconfiguration options for affected buses.
    
    Args:
        net: pandapower network
        affected_buses: List of buses affected by contingency
    
    Returns:
        List of reconfiguration options
    """
    analyzer = NetworkAnalyzer(net)
    options = []
    
    # Find slack bus
    slack_buses = net.ext_grid.bus.tolist()
    if not slack_buses:
        return options
    
    slack_bus = slack_buses[0]
    
    # For each affected bus, find alternative connections
    for bus in affected_buses:
        paths = analyzer.find_alternative_paths(slack_bus, bus, max_paths=3)
        
        for path_idx, path in enumerate(paths):
            # Check which lines in this path are currently open
            open_lines = []
            
            for i in range(len(path) - 1):
                from_b = path[i]
                to_b = path[i + 1]
                
                # Find line
                for line_idx, line in net.line.iterrows():
                    if ((line.from_bus == from_b and line.to_bus == to_b) or
                        (line.from_bus == to_b and line.to_bus == from_b)):
                        
                        if not line.in_service:
                            open_lines.append({
                                "line_id": int(line_idx),
                                "from_bus": int(from_b),
                                "to_bus": int(to_b)
                            })
            
            if open_lines:
                options.append({
                    "target_bus": bus,
                    "path": path,
                    "lines_to_close": open_lines,
                    "path_length": len(path),
                    "priority": 10 - path_idx  # Shorter paths have higher priority
                })
    
    logger.info(f"Identified {len(options)} reconfiguration options")
    return options


if __name__ == "__main__":
    # Test network analysis tools
    from src.config import load_configuration
    from src.core.network_loader import NetworkLoader
    import json
    
    config, _, paths = load_configuration()
    
    # Load network
    loader = NetworkLoader(networks_path=paths.networks)
    net = loader.load_network("ieee_33")
    
    # Run power flow first
    pp.runpp(net)
    
    print("\n=== Network Analysis ===")
    analyzer = NetworkAnalyzer(net)
    
    # Get network metrics
    metrics = analyzer.get_network_metrics()
    print("\nNetwork Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Find alternative paths
    print("\n=== Alternative Paths (Bus 0 to Bus 10) ===")
    paths = analyzer.find_alternative_paths(0, 10, max_paths=3)
    for i, path in enumerate(paths):
        print(f"Path {i+1}: {' -> '.join(map(str, path))}")
    
    # Identify critical lines
    print("\n=== Critical Lines ===")
    critical = analyzer.identify_critical_lines(threshold_loading=50.0)
    for line in critical[:5]:
        print(json.dumps(line, indent=2))
    
    # Calculate centrality
    print("\n=== Bus Centrality (Top 5) ===")
    centrality = analyzer.calculate_centrality()
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for bus, score in sorted_centrality:
        print(f"Bus {bus}: {score:.4f}")

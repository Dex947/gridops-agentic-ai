"""
Visualization module for GridOps Agentic AI System.
Generates plots for network analysis and contingency reports.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import networkx as nx
from loguru import logger


class NetworkVisualizer:
    """Create visualizations for network analysis and reports."""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved figures
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        logger.info(f"NetworkVisualizer initialized: output={output_dir}")
    
    def plot_voltage_profile(self, net: pp.pandapowerNet,
                            baseline_net: Optional[pp.pandapowerNet] = None,
                            v_limits: Tuple[float, float] = (0.95, 1.05),
                            title: str = "Voltage Profile",
                            filename: str = "voltage_profile.png") -> Path:
        """
        Plot voltage profile across all buses.
        
        Args:
            net: pandapower network (post-action)
            baseline_net: Baseline network for comparison (optional)
            v_limits: Voltage limits (min, max) in pu
            title: Plot title
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        logger.info("Generating voltage profile plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get bus indices and voltages
        bus_indices = net.res_bus.index.tolist()
        voltages = net.res_bus.vm_pu.values
        
        # Plot post-action voltages
        ax.plot(bus_indices, voltages, 'o-', label='Post-Action', linewidth=2, markersize=6)
        
        # Plot baseline if provided
        if baseline_net is not None and len(baseline_net.res_bus) > 0:
            baseline_voltages = baseline_net.res_bus.vm_pu.values
            ax.plot(bus_indices, baseline_voltages, 's--', 
                   label='Baseline', alpha=0.7, linewidth=1.5, markersize=5)
        
        # Add voltage limit lines
        ax.axhline(y=v_limits[0], color='r', linestyle='--', linewidth=1.5, 
                  label=f'Min Limit ({v_limits[0]} pu)')
        ax.axhline(y=v_limits[1], color='r', linestyle='--', linewidth=1.5,
                  label=f'Max Limit ({v_limits[1]} pu)')
        
        # Highlight violations
        violations = (voltages < v_limits[0]) | (voltages > v_limits[1])
        if np.any(violations):
            violated_buses = np.array(bus_indices)[violations]
            violated_voltages = voltages[violations]
            ax.scatter(violated_buses, violated_voltages, color='red', 
                      s=100, marker='x', linewidths=3, label='Violations', zorder=5)
        
        ax.set_xlabel('Bus Index', fontsize=12)
        ax.set_ylabel('Voltage (pu)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Voltage profile saved: {output_path}")
        return output_path
    
    def plot_line_loading(self, net: pp.pandapowerNet,
                         thermal_limit: float = 100.0,
                         title: str = "Line Loading Distribution",
                         filename: str = "line_loading.png") -> Path:
        """
        Plot line loading histogram.
        
        Args:
            net: pandapower network
            thermal_limit: Thermal limit percentage
            title: Plot title
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        logger.info("Generating line loading plot...")
        
        if len(net.res_line) == 0:
            logger.warning("No line results available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        loadings = net.res_line.loading_percent.values
        
        # Histogram
        ax1.hist(loadings, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=thermal_limit, color='r', linestyle='--', linewidth=2,
                   label=f'Thermal Limit ({thermal_limit}%)')
        ax1.set_xlabel('Loading (%)', fontsize=12)
        ax1.set_ylabel('Number of Lines', fontsize=12)
        ax1.set_title('Loading Distribution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of individual lines
        line_indices = net.res_line.index.tolist()
        colors = ['red' if l > thermal_limit else 'steelblue' for l in loadings]
        
        ax2.bar(line_indices, loadings, color=colors, edgecolor='black', alpha=0.7)
        ax2.axhline(y=thermal_limit, color='r', linestyle='--', linewidth=2,
                   label=f'Thermal Limit ({thermal_limit}%)')
        ax2.set_xlabel('Line Index', fontsize=12)
        ax2.set_ylabel('Loading (%)', fontsize=12)
        ax2.set_title('Line-by-Line Loading', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        # Save plot
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Line loading plot saved: {output_path}")
        return output_path
    
    def plot_network_topology(self, net: pp.pandapowerNet,
                             switched_elements: Optional[List[int]] = None,
                             title: str = "Network Topology",
                             filename: str = "network_topology.png") -> Path:
        """
        Plot network topology with highlighted switched elements.
        
        Args:
            net: pandapower network
            switched_elements: List of line indices that were switched
            title: Plot title
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        logger.info("Generating network topology plot...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Try to use pandapower's plotting if coordinates available
            if hasattr(net, 'bus_geodata') and len(net.bus_geodata) > 0:
                # Use pandapower plotting
                plot.simple_plot(net, ax=ax, plot_loads=True, plot_sgens=True,
                               line_width=2.0, bus_size=1.0, show_plot=False)
            else:
                # Create layout using networkx
                from src.tools.network_analysis import NetworkAnalyzer
                analyzer = NetworkAnalyzer(net)
                
                if analyzer.graph:
                    pos = nx.spring_layout(analyzer.graph, k=2, iterations=50)
                    
                    # Draw network
                    nx.draw_networkx_nodes(analyzer.graph, pos, node_size=300, 
                                         node_color='lightblue', ax=ax)
                    nx.draw_networkx_labels(analyzer.graph, pos, font_size=8, ax=ax)
                    
                    # Draw lines
                    edge_colors = []
                    edge_widths = []
                    
                    for edge in analyzer.graph.edges():
                        # Check if this edge corresponds to a switched line
                        is_switched = False
                        if switched_elements:
                            for line_idx in switched_elements:
                                if line_idx < len(net.line):
                                    line = net.line.iloc[line_idx]
                                    if (line.from_bus, line.to_bus) == edge or \
                                       (line.to_bus, line.from_bus) == edge:
                                        is_switched = True
                                        break
                        
                        if is_switched:
                            edge_colors.append('red')
                            edge_widths.append(3)
                        else:
                            edge_colors.append('black')
                            edge_widths.append(1)
                    
                    nx.draw_networkx_edges(analyzer.graph, pos, 
                                         edge_color=edge_colors,
                                         width=edge_widths, ax=ax)
                    
                    # Add legend
                    if switched_elements:
                        red_patch = mpatches.Patch(color='red', label='Switched Lines')
                        ax.legend(handles=[red_patch], loc='best')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Save plot
            output_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Network topology saved: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to generate network topology plot: {e}")
            return None
    
    def plot_comparison_metrics(self, baseline: Dict[str, Any],
                               post_action: Dict[str, Any],
                               title: str = "Performance Comparison",
                               filename: str = "comparison.png") -> Path:
        """
        Plot comparison between baseline and post-action metrics.
        
        Args:
            baseline: Baseline metrics
            post_action: Post-action metrics
            title: Plot title
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        logger.info("Generating comparison metrics plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Voltage comparison
        ax = axes[0, 0]
        metrics = ['Min Voltage', 'Max Voltage']
        baseline_v = [baseline.get('min_voltage_pu', 1.0), baseline.get('max_voltage_pu', 1.0)]
        post_v = [post_action.get('min_voltage_pu', 1.0), post_action.get('max_voltage_pu', 1.0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, baseline_v, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, post_v, width, label='Post-Action', alpha=0.8)
        ax.set_ylabel('Voltage (pu)')
        ax.set_title('Voltage Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loading comparison
        ax = axes[0, 1]
        baseline_load = baseline.get('max_line_loading_percent', 0)
        post_load = post_action.get('max_line_loading_percent', 0)
        
        ax.bar(['Baseline', 'Post-Action'], [baseline_load, post_load], 
              color=['steelblue', 'coral'], alpha=0.8)
        ax.axhline(y=100, color='r', linestyle='--', label='100% Limit')
        ax.set_ylabel('Loading (%)')
        ax.set_title('Maximum Line Loading')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Violations comparison
        ax = axes[1, 0]
        baseline_viol = len(baseline.get('violations', []))
        post_viol = len(post_action.get('violations', []))
        
        colors = ['red' if v > 0 else 'green' for v in [baseline_viol, post_viol]]
        ax.bar(['Baseline', 'Post-Action'], [baseline_viol, post_viol],
              color=colors, alpha=0.8)
        ax.set_ylabel('Number of Violations')
        ax.set_title('Constraint Violations')
        ax.grid(True, alpha=0.3)
        
        # Losses comparison
        ax = axes[1, 1]
        baseline_loss = baseline.get('total_losses_mw', 0)
        post_loss = post_action.get('total_losses_mw', 0)
        
        ax.bar(['Baseline', 'Post-Action'], [baseline_loss, post_loss],
              color=['steelblue', 'coral'], alpha=0.8)
        ax.set_ylabel('Losses (MW)')
        ax.set_title('System Losses')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved: {output_path}")
        return output_path
    
    def generate_all_plots(self, net_baseline: pp.pandapowerNet,
                          net_post_action: pp.pandapowerNet,
                          baseline_results: Dict[str, Any],
                          post_action_results: Dict[str, Any],
                          switched_elements: Optional[List[int]] = None,
                          v_limits: Tuple[float, float] = (0.95, 1.05)) -> Dict[str, Path]:
        """
        Generate all plots for a complete analysis.
        
        Args:
            net_baseline: Baseline network
            net_post_action: Post-action network
            baseline_results: Baseline power flow results
            post_action_results: Post-action power flow results
            switched_elements: List of switched line indices
            v_limits: Voltage limits
        
        Returns:
            Dictionary mapping plot type to file path
        """
        logger.info("Generating all visualization plots...")
        
        plots = {}
        
        # Voltage profile
        plots['voltage_profile'] = self.plot_voltage_profile(
            net_post_action, net_baseline, v_limits,
            title="Voltage Profile: Baseline vs Post-Action"
        )
        
        # Line loading
        plots['line_loading'] = self.plot_line_loading(
            net_post_action,
            title="Line Loading After Corrective Action"
        )
        
        # Network topology
        plots['network_topology'] = self.plot_network_topology(
            net_post_action, switched_elements,
            title="Network Topology with Switched Elements"
        )
        
        # Comparison metrics
        plots['comparison'] = self.plot_comparison_metrics(
            baseline_results, post_action_results,
            title="Performance Metrics: Before and After"
        )
        
        logger.info(f"Generated {len(plots)} plots")
        return plots


if __name__ == "__main__":
    # Test visualization
    from src.config import load_configuration
    from src.core.network_loader import NetworkLoader
    from src.tools.powerflow_tools import PowerFlowTool
    
    config, constraints, paths = load_configuration()
    
    # Load network
    loader = NetworkLoader(networks_path=paths.networks)
    net = loader.load_network("ieee_33")
    
    # Run power flow
    pp.runpp(net)
    
    # Initialize visualizer
    viz = NetworkVisualizer(output_dir=paths.plots, dpi=150)
    
    # Create power flow tool
    pf_tool = PowerFlowTool(
        voltage_limits=(constraints.v_min_pu, constraints.v_max_pu)
    )
    
    # Get baseline results
    baseline_results = pf_tool.run(net)
    
    # Simulate a contingency and action
    net_cont = net.deepcopy()
    net_cont.line.at[5, 'in_service'] = False  # Create contingency
    pp.runpp(net_cont)
    
    post_results = pf_tool.run(net_cont)
    
    print("\n=== Generating Test Visualizations ===")
    
    # Generate plots
    plots = viz.generate_all_plots(
        net, net_cont,
        baseline_results.to_dict(),
        post_results.to_dict(),
        switched_elements=[5],
        v_limits=(constraints.v_min_pu, constraints.v_max_pu)
    )
    
    print("\nGenerated plots:")
    for plot_type, path in plots.items():
        if path:
            print(f"  {plot_type}: {path}")

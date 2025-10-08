"""
Quick system validation script.
Tests core functionality without requiring LLM API keys.
"""

import sys
from pathlib import Path
import pandapower as pp
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_configuration
from src.core.network_loader import NetworkLoader, create_test_network
from src.core.contingency_simulator import ContingencySimulator, ContingencyEvent, ContingencyType
from src.tools.powerflow_tools import PowerFlowTool, apply_switching_action
from src.tools.network_analysis import NetworkAnalyzer, calculate_network_metrics
from src.visualization import NetworkVisualizer
from src.report_generator import ReportGenerator
from src.core.state_manager import create_initial_state, StateManager, ActionProposal


def test_configuration():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TEST 1: Configuration System")
    print("=" * 60)
    
    try:
        config, constraints, paths = load_configuration()
        print("✓ Configuration loaded successfully")
        print(f"  - LLM Provider: {config.llm_provider}")
        print(f"  - Model: {config.model_name}")
        print(f"  - Voltage Limits: {constraints.v_min_pu} - {constraints.v_max_pu} pu")
        print(f"  - Thermal Margin: {constraints.thermal_margin * 100}%")
        return True, (config, constraints, paths)
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False, None


def test_network_loader(paths):
    """Test network loading."""
    print("\n" + "=" * 60)
    print("TEST 2: Network Loading")
    print("=" * 60)
    
    try:
        loader = NetworkLoader(networks_path=paths.networks)
        
        # List available networks
        networks = loader.list_available_networks()
        print(f"✓ Found {len(networks)} available networks")
        
        # Load IEEE 33-bus network
        net = loader.load_network("ieee_33")
        summary = loader.get_network_summary(net)
        
        print(f"✓ Loaded ieee_33 network")
        print(f"  - Buses: {summary['buses']}")
        print(f"  - Lines: {summary['lines']}")
        print(f"  - Loads: {summary['loads']}")
        print(f"  - Total Load: {summary['total_load_p_mw']:.3f} MW")
        
        return True, net
    except Exception as e:
        print(f"✗ Network loading failed: {e}")
        return False, None


def test_power_flow(net, constraints):
    """Test power flow analysis."""
    print("\n" + "=" * 60)
    print("TEST 3: Power Flow Analysis")
    print("=" * 60)
    
    try:
        pf_tool = PowerFlowTool(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
            thermal_limit_percent=100.0
        )
        
        results = pf_tool.run(net)
        
        print(f"✓ Power flow executed")
        print(f"  - Converged: {results.converged}")
        print(f"  - Min Voltage: {results.min_voltage_pu:.4f} pu")
        print(f"  - Max Voltage: {results.max_voltage_pu:.4f} pu")
        print(f"  - Max Loading: {results.max_line_loading_percent:.2f}%")
        print(f"  - Violations: {len(results.violations)}")
        
        return True, results
    except Exception as e:
        print(f"✗ Power flow failed: {e}")
        return False, None


def test_contingency_simulation(net, constraints):
    """Test contingency simulation."""
    print("\n" + "=" * 60)
    print("TEST 4: Contingency Simulation")
    print("=" * 60)
    
    try:
        simulator = ContingencySimulator(
            voltage_limits=(constraints.v_min_pu, constraints.v_max_pu),
            thermal_limit_percent=100.0
        )
        
        # Generate N-1 contingencies
        contingencies = simulator.generate_n_minus_1_contingencies(net)
        print(f"✓ Generated {len(contingencies)} N-1 contingencies")
        
        # Test first contingency
        if contingencies:
            net_test = net.deepcopy()
            result = simulator.simulate_contingency(net_test, contingencies[0])
            
            print(f"✓ Simulated contingency: {contingencies[0].description}")
            print(f"  - Converged: {result.converged}")
            print(f"  - Violations: {len(result.violated_constraints)}")
            print(f"  - Critical: {result.is_critical}")
        
        return True, simulator
    except Exception as e:
        print(f"✗ Contingency simulation failed: {e}")
        return False, None


def test_network_analysis(net):
    """Test network analysis tools."""
    print("\n" + "=" * 60)
    print("TEST 5: Network Analysis")
    print("=" * 60)
    
    try:
        analyzer = NetworkAnalyzer(net)
        
        # Get network metrics
        metrics = calculate_network_metrics(net)
        print(f"✓ Network metrics calculated")
        print(f"  - Buses: {metrics['num_buses']}")
        print(f"  - Edges: {metrics['num_edges']}")
        print(f"  - Connected: {metrics['is_connected']}")
        
        # Find alternative paths
        if metrics['num_buses'] > 2:
            paths = analyzer.find_alternative_paths(0, min(10, metrics['num_buses'] - 1), max_paths=3)
            print(f"✓ Found {len(paths)} alternative paths")
        
        # Identify critical lines
        critical = analyzer.identify_critical_lines(threshold_loading=50.0)
        print(f"✓ Identified {len(critical)} critical elements")
        
        return True, analyzer
    except Exception as e:
        print(f"✗ Network analysis failed: {e}")
        return False, None


def test_state_management(paths):
    """Test state management."""
    print("\n" + "=" * 60)
    print("TEST 6: State Management")
    print("=" * 60)
    
    try:
        state = create_initial_state(
            network_name="ieee_33",
            contingency_desc="Line 5 outage test",
            max_iterations=10
        )
        
        print("✓ Initial state created")
        print(f"  - Network: {state['network_name']}")
        print(f"  - Status: {state['workflow_status']}")
        
        # Test state manager
        manager = StateManager(state_dir=paths.base / "state")
        
        # Add test action
        action = ActionProposal(
            action_id="test_action",
            action_type="switch_line",
            target_elements=[{"line_id": 5, "close": False}],
            expected_impact="Test impact",
            priority=5,
            proposed_by="TestScript"
        )
        
        state = manager.add_proposed_action(state, action)
        print(f"✓ Added action proposal: {action.action_id}")
        
        # Save and load state
        manager.save_state(state, "test_session")
        loaded = manager.load_state("test_session")
        print(f"✓ State persistence verified")
        
        return True, manager
    except Exception as e:
        print(f"✗ State management failed: {e}")
        return False, None


def test_visualization(net, paths, constraints):
    """Test visualization generation."""
    print("\n" + "=" * 60)
    print("TEST 7: Visualization")
    print("=" * 60)
    
    try:
        viz = NetworkVisualizer(output_dir=paths.plots, dpi=150)
        
        # Run power flow
        pp.runpp(net)
        
        # Generate voltage profile
        plot_path = viz.plot_voltage_profile(
            net,
            v_limits=(constraints.v_min_pu, constraints.v_max_pu),
            title="Test Voltage Profile",
            filename="test_voltage_profile.png"
        )
        
        if plot_path and plot_path.exists():
            print(f"✓ Voltage profile generated: {plot_path.name}")
        
        # Generate line loading
        plot_path = viz.plot_line_loading(
            net,
            title="Test Line Loading",
            filename="test_line_loading.png"
        )
        
        if plot_path and plot_path.exists():
            print(f"✓ Line loading plot generated: {plot_path.name}")
        
        print("✓ Visualization system operational")
        return True, viz
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False, None


def test_report_generation(paths):
    """Test report generation."""
    print("\n" + "=" * 60)
    print("TEST 8: Report Generation")
    print("=" * 60)
    
    try:
        generator = ReportGenerator(output_dir=paths.reports)
        
        # Create test state
        state = create_initial_state("ieee_33", "Test contingency", 10)
        state['workflow_status'] = "complete"
        state['baseline_results'] = {
            "converged": True,
            "min_voltage_pu": 0.98,
            "max_voltage_pu": 1.02,
            "max_line_loading_percent": 85.0,
            "violations": []
        }
        
        # Generate markdown report
        md_path = generator.generate_markdown_report(state, {}, "test_report")
        
        if md_path and md_path.exists():
            print(f"✓ Markdown report generated: {md_path.name}")
        
        # Generate LaTeX report
        tex_path = generator.generate_latex_report(state, {}, "test_report")
        
        if tex_path and tex_path.exists():
            print(f"✓ LaTeX report generated: {tex_path.name}")
        
        print("✓ Report generation system operational")
        return True, generator
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        return False, None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GRIDOPS SYSTEM VALIDATION")
    print("=" * 60)
    print("\nTesting core functionality (no LLM required)...\n")
    
    results = []
    config_data = None
    net = None
    
    # Test 1: Configuration
    success, config_data = test_configuration()
    results.append(("Configuration", success))
    
    if not success or not config_data:
        print("\n✗ Configuration failed - cannot continue")
        return 1
    
    config, constraints, paths = config_data
    
    # Test 2: Network Loading
    success, net = test_network_loader(paths)
    results.append(("Network Loading", success))
    
    if success and net is not None:
        # Test 3: Power Flow
        success, _ = test_power_flow(net, constraints)
        results.append(("Power Flow", success))
        
        # Test 4: Contingency Simulation
        success, _ = test_contingency_simulation(net, constraints)
        results.append(("Contingency Simulation", success))
        
        # Test 5: Network Analysis
        success, _ = test_network_analysis(net)
        results.append(("Network Analysis", success))
    
    # Test 6: State Management
    success, _ = test_state_management(paths)
    results.append(("State Management", success))
    
    # Test 7: Visualization
    if net is not None:
        success, _ = test_visualization(net, paths, constraints)
        results.append(("Visualization", success))
    
    # Test 8: Report Generation
    success, _ = test_report_generation(paths)
    results.append(("Report Generation", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is operational.")
        print("\nNote: Full agent orchestration requires LLM API keys.")
        print("Configure .env file with OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("to test the complete multi-agent workflow.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

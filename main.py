"""
GridOps Agentic AI System - Main Execution Script
Command-line interface for contingency management system.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

from src.config import load_configuration
from src.orchestrator import GridOpsOrchestrator
from src.visualization import NetworkVisualizer
from src.report_generator import ReportGenerator
from src.core.network_loader import NetworkLoader


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GridOps Agentic AI System - Distribution Network Contingency Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze line outage on IEEE 33-bus network
  python main.py --network ieee_33 --contingency "Line 5 outage" --type line_outage --elements 5

  # Analyze with custom settings
  python main.py --network ieee_33 --contingency "Line 10 outage" --type line_outage --elements 10 --no-plots

  # List available networks
  python main.py --list-networks
        """
    )
    
    parser.add_argument(
        '--network',
        type=str,
        default='ieee_33',
        help='Network name to analyze (default: ieee_33)'
    )
    
    parser.add_argument(
        '--contingency',
        type=str,
        required=False,
        help='Contingency description (e.g., "Line 5 outage")'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['line_outage', 'transformer_outage', 'load_increase', 'generator_outage'],
        default='line_outage',
        help='Contingency type (default: line_outage)'
    )
    
    parser.add_argument(
        '--elements',
        type=int,
        nargs='+',
        help='Affected element indices (e.g., 5 or 5 6 7)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--no-latex',
        action='store_true',
        help='Skip LaTeX report generation'
    )
    
    parser.add_argument(
        '--list-networks',
        action='store_true',
        help='List available networks and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory for reports'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level from config'
    )
    
    return parser


def list_available_networks(config, paths):
    """List all available networks."""
    loader = NetworkLoader(networks_path=paths.networks)
    networks = loader.list_available_networks()
    
    print("\n" + "=" * 60)
    print("AVAILABLE NETWORKS")
    print("=" * 60)
    
    for net_name in networks:
        try:
            net = loader.load_network(net_name)
            summary = loader.get_network_summary(net)
            
            print(f"\n{net_name}:")
            print(f"  Buses: {summary['buses']}")
            print(f"  Lines: {summary['lines']}")
            print(f"  Loads: {summary['loads']}")
            print(f"  Total Load: {summary['total_load_p_mw']:.3f} MW")
        except Exception as e:
            print(f"\n{net_name}: (Error loading: {e})")
    
    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Load configuration
    config, constraints, paths = load_configuration()
    
    # Override log level if specified
    if args.log_level:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)
    
    # List networks and exit if requested
    if args.list_networks:
        list_available_networks(config, paths)
        return 0
    
    # Validate required arguments
    if not args.contingency or not args.elements:
        logger.error("Both --contingency and --elements are required (unless using --list-networks)")
        parser.print_help()
        return 1
    
    # Generate session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("GRIDOPS AGENTIC AI SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Network: {args.network}")
    logger.info(f"Contingency: {args.contingency}")
    logger.info(f"Type: {args.type}")
    logger.info(f"Elements: {args.elements}")
    logger.info("=" * 80)
    
    try:
        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = GridOpsOrchestrator(config, constraints, paths)
        
        # Run contingency management workflow
        logger.info("Starting workflow execution...")
        final_state = orchestrator.run(
            network_name=args.network,
            contingency_description=args.contingency,
            contingency_type=args.type,
            contingency_elements=args.elements
        )
        
        # Check workflow status
        if final_state['workflow_status'] != 'complete':
            logger.error(f"Workflow failed: {final_state.get('error_message', 'Unknown error')}")
            return 1
        
        logger.info("Workflow completed successfully")
        
        # Generate visualizations
        plots = {}
        if not args.no_plots and config.generate_plots:
            logger.info("Generating visualizations...")
            viz = NetworkVisualizer(output_dir=paths.plots, dpi=config.plot_dpi)
            
            try:
                # Get networks for visualization
                baseline_net = orchestrator.network
                contingency_net = orchestrator.network_contingency
                
                # Extract switched elements from selected action
                switched_elements = []
                if final_state.get('selected_action'):
                    for elem in final_state['selected_action'].get('target_elements', []):
                        if elem.get('type') == 'line_switch':
                            switched_elements.append(elem.get('line_id'))
                
                # Generate all plots
                plots = viz.generate_all_plots(
                    baseline_net,
                    contingency_net,
                    final_state['baseline_results'],
                    final_state['contingency_results'],
                    switched_elements=switched_elements,
                    v_limits=(constraints.v_min_pu, constraints.v_max_pu)
                )
                
                logger.info(f"Generated {len(plots)} visualization plots")
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
        
        # Generate reports
        logger.info("Generating reports...")
        output_dir = Path(args.output_dir) if args.output_dir else paths.reports
        generator = ReportGenerator(output_dir=output_dir)
        
        # Markdown report
        md_path = generator.generate_markdown_report(final_state, plots, session_id)
        logger.info(f"Markdown report: {md_path}")
        
        # LaTeX report
        if not args.no_latex:
            tex_path = generator.generate_latex_report(final_state, plots, session_id)
            logger.info(f"LaTeX report: {tex_path}")
        
        # Save final state
        orchestrator.state_manager.save_state(final_state, session_id)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        
        summary = orchestrator.state_manager.get_summary(final_state)
        print(json.dumps(summary, indent=2))
        
        if final_state.get('selected_action'):
            print("\n" + "-" * 80)
            print("SELECTED ACTION")
            print("-" * 80)
            selected = final_state['selected_action']
            print(f"Action ID: {selected.get('action_id')}")
            print(f"Type: {selected.get('action_type')}")
            print(f"Safety Score: {selected.get('safety_score', 0):.2f}/1.00")
            print(f"Recommendation: {selected.get('recommendation')}")
            print(f"Violations Resolved: {len(selected.get('violations_resolved', []))}")
            print(f"Violations Remaining: {len(selected.get('violations_remaining', []))}")
        
        print("\n" + "=" * 80)
        print("OUTPUTS")
        print("=" * 80)
        print(f"Markdown Report: {md_path}")
        if not args.no_latex:
            print(f"LaTeX Report: {tex_path}")
        if plots:
            print(f"Plots Directory: {paths.plots}")
        print(f"State File: {paths.base / 'state' / f'state_{session_id}.json'}")
        print("=" * 80)
        
        logger.info("Execution completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return 130
    
    except Exception as e:
        logger.exception(f"Execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

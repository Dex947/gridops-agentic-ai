"""
Report generation module for GridOps Agentic AI System.
Generates comprehensive reports in Markdown and LaTeX formats.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

from src.core.state_manager import SystemState


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReportGenerator initialized: output={output_dir}")
    
    def generate_markdown_report(self, state: SystemState,
                                 plots: Dict[str, Path],
                                 session_id: str) -> Path:
        """
        Generate comprehensive Markdown report.
        
        Args:
            state: Final system state
            plots: Dictionary of generated plots
            session_id: Unique session identifier
        
        Returns:
            Path to generated report
        """
        logger.info("Generating Markdown report...")
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        filename = f"run_{session_id}.md"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# GridOps Contingency Management Report\n\n")
            f.write(f"**Generated:** {timestamp}  \n")
            f.write(f"**Session ID:** {session_id}  \n")
            f.write(f"**Status:** {state['workflow_status']}  \n\n")
            
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"**Network:** {state['network_name']}  \n")
            f.write(f"**Contingency:** {state['contingency_description']}  \n")
            f.write(f"**Type:** {state['contingency_type']}  \n")
            f.write(f"**Affected Elements:** {', '.join(map(str, state['contingency_elements']))}  \n\n")
            
            # Network Information
            f.write("## Network Information\n\n")
            summary = state.get('network_summary', {})
            f.write(f"- **Total Buses:** {summary.get('buses', 'N/A')}\n")
            f.write(f"- **Total Lines:** {summary.get('lines', 'N/A')}\n")
            f.write(f"- **Total Loads:** {summary.get('loads', 'N/A')}\n")
            f.write(f"- **Total Load (P):** {summary.get('total_load_p_mw', 0):.3f} MW\n")
            f.write(f"- **Total Load (Q):** {summary.get('total_load_q_mvar', 0):.3f} MVAr\n")
            f.write(f"- **Voltage Levels:** {', '.join(map(str, summary.get('voltage_levels_kv', [])))} kV\n\n")
            
            # Baseline Analysis
            f.write("## Baseline Analysis\n\n")
            baseline = state.get('baseline_results', {})
            f.write("### Pre-Contingency System State\n\n")
            f.write(f"- **Power Flow Converged:** {'Yes' if baseline.get('converged') else 'No'}\n")
            f.write(f"- **Min Voltage:** {baseline.get('min_voltage_pu', 0):.4f} pu\n")
            f.write(f"- **Max Voltage:** {baseline.get('max_voltage_pu', 0):.4f} pu\n")
            f.write(f"- **Max Line Loading:** {baseline.get('max_line_loading_percent', 0):.2f}%\n")
            f.write(f"- **Total Losses:** {baseline.get('total_losses_mw', 0):.3f} MW\n")
            f.write(f"- **Violations:** {len(baseline.get('violations', []))}\n\n")
            
            # Contingency Impact
            f.write("## Contingency Impact\n\n")
            contingency = state.get('contingency_results', {})
            f.write(f"- **Power Flow Converged:** {'Yes' if contingency.get('converged') else 'No'}\n")
            f.write(f"- **Min Voltage:** {contingency.get('min_voltage_pu', 0):.4f} pu\n")
            f.write(f"- **Max Voltage:** {contingency.get('max_voltage_pu', 0):.4f} pu\n")
            f.write(f"- **Max Line Loading:** {contingency.get('max_line_loading_percent', 0):.2f}%\n")
            f.write(f"- **Total Losses:** {contingency.get('total_losses_mw', 0):.3f} MW\n\n")
            
            # Constraint Violations
            f.write("### Constraint Violations\n\n")
            violations = state.get('constraint_violations', [])
            if violations:
                f.write(f"**Total Violations:** {len(violations)}\n\n")
                for i, violation in enumerate(violations, 1):
                    f.write(f"{i}. {violation}\n")
                f.write("\n")
            else:
                f.write("No constraint violations detected.\n\n")
            
            # Proposed Actions
            f.write("## Proposed Actions\n\n")
            f.write(f"**Total Proposals Generated:** {len(state.get('proposed_actions', []))}\n\n")
            
            for i, action in enumerate(state.get('proposed_actions', []), 1):
                f.write(f"### Action {i}: {action.get('action_id', 'N/A')}\n\n")
                f.write(f"- **Type:** {action.get('action_type', 'N/A')}\n")
                f.write(f"- **Priority:** {action.get('priority', 'N/A')}/10\n")
                f.write(f"- **Expected Impact:** {action.get('expected_impact', 'N/A')}\n")
                f.write(f"- **Target Elements:** {len(action.get('target_elements', []))}\n")
                f.write(f"- **Reasoning:** {action.get('reasoning', 'N/A')}\n\n")
            
            # Evaluation Results
            f.write("## Action Evaluation Results\n\n")
            f.write("| Action ID | Converged | Violations | Safety Score | Recommendation |\n")
            f.write("|-----------|-----------|------------|--------------|----------------|\n")
            
            for eval_result in state.get('evaluated_actions', []):
                f.write(f"| {eval_result.get('action_id', 'N/A')} | ")
                f.write(f"{'✓' if eval_result.get('powerflow_converged') else '✗'} | ")
                f.write(f"{len(eval_result.get('violations_remaining', []))} | ")
                f.write(f"{eval_result.get('safety_score', 0):.2f} | ")
                f.write(f"{eval_result.get('recommendation', 'N/A')} |\n")
            
            f.write("\n")
            
            # Selected Action
            f.write("## Selected Action\n\n")
            selected = state.get('selected_action')
            if selected:
                f.write(f"**Action ID:** {selected.get('action_id', 'N/A')}  \n")
                f.write(f"**Type:** {selected.get('action_type', 'N/A')}  \n")
                f.write(f"**Recommendation:** {selected.get('recommendation', 'N/A')}  \n")
                f.write(f"**Safety Score:** {selected.get('safety_score', 0):.2f}/1.00  \n\n")
                
                # Performance metrics
                f.write("### Performance Metrics\n\n")
                metrics = selected.get('performance_metrics', {})
                f.write(f"- **Min Voltage:** {metrics.get('min_voltage_pu', 0):.4f} pu\n")
                f.write(f"- **Max Voltage:** {metrics.get('max_voltage_pu', 0):.4f} pu\n")
                f.write(f"- **Max Line Loading:** {metrics.get('max_line_loading_percent', 0):.2f}%\n")
                f.write(f"- **Voltage Margin:** {metrics.get('voltage_margin_pu', 0):.4f} pu\n")
                f.write(f"- **Thermal Margin:** {metrics.get('thermal_margin_percent', 0):.2f}%\n\n")
                
                # Violations resolved
                f.write("### Violations Resolved\n\n")
                resolved = selected.get('violations_resolved', [])
                if resolved:
                    for v in resolved:
                        f.write(f"- ✓ {v}\n")
                else:
                    f.write("None\n")
                f.write("\n")
                
                # Remaining violations
                remaining = selected.get('violations_remaining', [])
                if remaining:
                    f.write("### Violations Remaining\n\n")
                    for v in remaining:
                        f.write(f"- ✗ {v}\n")
                    f.write("\n")
            else:
                f.write("*No suitable action was identified.*\n\n")
            
            # Technical Explanation
            f.write("## Technical Explanation\n\n")
            explanation = state.get('explanation', 'No explanation available.')
            f.write(f"{explanation}\n\n")
            
            # References
            f.write("## References and Standards\n\n")
            references = state.get('references', [])
            if references:
                for i, ref in enumerate(references, 1):
                    f.write(f"{i}. {ref}\n")
            else:
                f.write("No specific references cited.\n")
            f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            for plot_name, plot_path in plots.items():
                if plot_path and plot_path.exists():
                    # Use relative path for markdown
                    rel_path = plot_path.name
                    f.write(f"### {plot_name.replace('_', ' ').title()}\n\n")
                    f.write(f"![{plot_name}](../plots/{rel_path})\n\n")
            
            # Timestamp footer
            f.write("---\n\n")
            f.write(f"*Report generated by GridOps Agentic AI System on {timestamp}*\n")
        
        logger.info(f"Markdown report generated: {output_path}")
        return output_path
    
    def generate_latex_report(self, state: SystemState,
                             plots: Dict[str, Path],
                             session_id: str) -> Path:
        """
        Generate LaTeX technical report.
        
        Args:
            state: Final system state
            plots: Dictionary of generated plots
            session_id: Unique session identifier
        
        Returns:
            Path to generated LaTeX file
        """
        logger.info("Generating LaTeX report...")
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        filename = f"run_{session_id}.tex"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # LaTeX preamble
            f.write(r"\documentclass[11pt,a4paper]{article}" + "\n")
            f.write(r"\usepackage[utf8]{inputenc}" + "\n")
            f.write(r"\usepackage{graphicx}" + "\n")
            f.write(r"\usepackage{booktabs}" + "\n")
            f.write(r"\usepackage{geometry}" + "\n")
            f.write(r"\usepackage{hyperref}" + "\n")
            f.write(r"\usepackage{float}" + "\n")
            f.write(r"\geometry{margin=1in}" + "\n\n")
            
            # Title and metadata
            f.write(r"\title{GridOps Contingency Management Report}" + "\n")
            f.write(r"\author{GridOps Agentic AI System}" + "\n")
            f.write(r"\date{" + timestamp.replace("UTC", "").strip() + "}\n\n")
            
            f.write(r"\begin{document}" + "\n\n")
            f.write(r"\maketitle" + "\n\n")
            
            # Abstract
            f.write(r"\begin{abstract}" + "\n")
            f.write(f"This report presents the analysis and resolution of a {self._escape_latex(state['contingency_type'])} ")
            f.write(f"contingency on the {self._escape_latex(state['network_name'])} distribution network. ")
            f.write(f"The contingency resulted in {len(state.get('constraint_violations', []))} ")
            f.write(f"constraint violations, which were addressed through automated multi-agent analysis ")
            f.write(f"and action planning. ")
            
            if state.get('selected_action'):
                f.write(f"A corrective action was identified with a safety score of ")
                f.write(f"{state['selected_action'].get('safety_score', 0):.2f}/1.00.")
            
            f.write(r"\end{abstract}" + "\n\n")
            
            f.write(r"\tableofcontents" + "\n")
            f.write(r"\newpage" + "\n\n")
            
            # Section 1: Introduction
            f.write(r"\section{Introduction}" + "\n\n")
            f.write(f"Network: \\textbf{{{self._escape_latex(state['network_name'])}}} \\\\\n")
            f.write(f"Contingency: {self._escape_latex(state['contingency_description'])} \\\\\n")
            f.write(f"Analysis Date: {timestamp} \\\\\n")
            f.write(f"Session ID: \\texttt{{{self._escape_latex(session_id)}}}\n\n")
            
            # Section 2: Network Characteristics
            f.write(r"\section{Network Characteristics}" + "\n\n")
            summary = state.get('network_summary', {})
            
            f.write(r"\begin{tabular}{ll}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Parameter & Value \\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(f"Total Buses & {summary.get('buses', 'N/A')} \\\\\n")
            f.write(f"Total Lines & {summary.get('lines', 'N/A')} \\\\\n")
            f.write(f"Total Loads & {summary.get('loads', 'N/A')} \\\\\n")
            f.write(f"Total Active Power & {summary.get('total_load_p_mw', 0):.3f} MW \\\\\n")
            f.write(f"Total Reactive Power & {summary.get('total_load_q_mvar', 0):.3f} MVAr \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n\n")
            
            # Section 3: Contingency Analysis
            f.write(r"\section{Contingency Analysis}" + "\n\n")
            f.write(r"\subsection{Baseline Conditions}" + "\n\n")
            
            baseline = state.get('baseline_results', {})
            f.write(f"Power Flow Converged: {'Yes' if baseline.get('converged') else 'No'} \\\\\n")
            f.write(f"Minimum Voltage: {baseline.get('min_voltage_pu', 0):.4f} pu \\\\\n")
            f.write(f"Maximum Voltage: {baseline.get('max_voltage_pu', 0):.4f} pu \\\\\n")
            f.write(f"Maximum Line Loading: {baseline.get('max_line_loading_percent', 0):.2f}\\% \\\\\n\n")
            
            f.write(r"\subsection{Post-Contingency State}" + "\n\n")
            contingency = state.get('contingency_results', {})
            
            violations = state.get('constraint_violations', [])
            f.write(f"The contingency resulted in {len(violations)} constraint violation(s):\n\n")
            
            if violations:
                f.write(r"\begin{itemize}" + "\n")
                for violation in violations:
                    f.write(f"\\item {self._escape_latex(violation)}\n")
                f.write(r"\end{itemize}" + "\n\n")
            
            # Section 4: Corrective Actions
            f.write(r"\section{Corrective Actions}" + "\n\n")
            f.write(f"Total action proposals generated: {len(state.get('proposed_actions', []))}\n\n")
            
            if state.get('selected_action'):
                selected = state['selected_action']
                f.write(r"\subsection{Selected Action}" + "\n\n")
                f.write(f"Action ID: \\texttt{{{selected.get('action_id', 'N/A')}}} \\\\\n")
                f.write(f"Type: {selected.get('action_type', 'N/A')} \\\\\n")
                f.write(f"Safety Score: {selected.get('safety_score', 0):.2f}/1.00 \\\\\n")
                f.write(f"Recommendation: {selected.get('recommendation', 'N/A')}\n\n")
            
            # Section 5: Results and Validation
            f.write(r"\section{Results and Validation}" + "\n\n")
            
            if state.get('selected_action'):
                metrics = state['selected_action'].get('performance_metrics', {})
                
                f.write(r"\begin{tabular}{ll}" + "\n")
                f.write(r"\toprule" + "\n")
                f.write(r"Metric & Value \\" + "\n")
                f.write(r"\midrule" + "\n")
                f.write(f"Min Voltage & {metrics.get('min_voltage_pu', 0):.4f} pu \\\\\n")
                f.write(f"Max Voltage & {metrics.get('max_voltage_pu', 0):.4f} pu \\\\\n")
                f.write(f"Max Line Loading & {metrics.get('max_line_loading_percent', 0):.2f}\\% \\\\\n")
                f.write(f"Voltage Margin & {metrics.get('voltage_margin_pu', 0):.4f} pu \\\\\n")
                f.write(f"Thermal Margin & {metrics.get('thermal_margin_percent', 0):.2f}\\% \\\\\n")
                f.write(r"\bottomrule" + "\n")
                f.write(r"\end{tabular}" + "\n\n")
            
            # Section 6: Visualizations
            f.write(r"\section{Visualizations}" + "\n\n")
            
            for plot_name, plot_path in plots.items():
                if plot_path and plot_path.exists():
                    f.write(r"\subsection{" + plot_name.replace('_', ' ').title() + "}\n\n")
                    f.write(r"\begin{figure}[H]" + "\n")
                    f.write(r"\centering" + "\n")
                    f.write(r"\includegraphics[width=0.9\textwidth]{../plots/" + plot_path.name + "}\n")
                    f.write(r"\caption{" + plot_name.replace('_', ' ').title() + "}\n")
                    f.write(r"\end{figure}" + "\n\n")
            
            # Section 7: References
            f.write(r"\section{References}" + "\n\n")
            references = state.get('references', [])
            
            if references:
                f.write(r"\begin{enumerate}" + "\n")
                for ref in references:
                    f.write(f"\\item {self._escape_latex(ref)}\n")
                f.write(r"\end{enumerate}" + "\n\n")
            
            f.write(r"\end{document}" + "\n")
        
        logger.info(f"LaTeX report generated: {output_path}")
        return output_path
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


if __name__ == "__main__":
    # Test report generation
    from src.config import load_configuration
    from src.core.state_manager import create_initial_state
    
    config, constraints, paths = load_configuration()
    
    # Create test state
    state = create_initial_state(
        network_name="ieee_33",
        contingency_desc="Line 5 outage",
        max_iterations=10
    )
    
    state['contingency_type'] = "line_outage"
    state['contingency_elements'] = [5]
    state['workflow_status'] = "complete"
    state['constraint_violations'] = ["Bus 10: 0.92 pu (undervoltage)"]
    state['baseline_results'] = {
        "converged": True,
        "min_voltage_pu": 0.98,
        "max_voltage_pu": 1.02,
        "max_line_loading_percent": 85.0,
        "violations": []
    }
    state['selected_action'] = {
        "action_id": "test_action",
        "action_type": "switch_line",
        "safety_score": 0.85,
        "recommendation": "approve",
        "performance_metrics": {
            "min_voltage_pu": 0.96,
            "max_voltage_pu": 1.03,
            "max_line_loading_percent": 90.0
        },
        "violations_resolved": ["Bus 10: 0.92 pu (undervoltage)"],
        "violations_remaining": []
    }
    state['explanation'] = "Test explanation of the corrective action."
    state['references'] = ["IEEE Std 1547-2018", "ANSI C84.1-2020"]
    
    # Initialize report generator
    generator = ReportGenerator(output_dir=paths.reports)
    
    # Generate reports
    print("\n=== Generating Test Reports ===")
    
    md_path = generator.generate_markdown_report(state, {}, "test_session")
    print(f"Markdown report: {md_path}")
    
    tex_path = generator.generate_latex_report(state, {}, "test_session")
    print(f"LaTeX report: {tex_path}")

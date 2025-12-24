"""Optimal Power Flow (OPF) tools for network optimization."""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import pandapower as pp
from loguru import logger


class OPFObjective(Enum):
    """Optimization objectives for OPF."""
    MIN_LOSS = "min_loss"
    MIN_GENERATION_COST = "min_generation_cost"
    MIN_LOAD_SHEDDING = "min_load_shedding"
    MAX_LOADABILITY = "max_loadability"


@dataclass
class OPFResult:
    """Results from optimal power flow calculation."""
    converged: bool
    objective_value: float
    total_generation_mw: float
    total_load_mw: float
    total_losses_mw: float
    min_voltage_pu: float
    max_voltage_pu: float
    max_line_loading_percent: float
    generator_dispatch: Dict[int, float] = field(default_factory=dict)
    voltage_setpoints: Dict[int, float] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "converged": self.converged,
            "objective_value": float(self.objective_value),
            "total_generation_mw": float(self.total_generation_mw),
            "total_load_mw": float(self.total_load_mw),
            "total_losses_mw": float(self.total_losses_mw),
            "min_voltage_pu": float(self.min_voltage_pu),
            "max_voltage_pu": float(self.max_voltage_pu),
            "max_line_loading_percent": float(self.max_line_loading_percent),
            "generator_dispatch": self.generator_dispatch,
            "voltage_setpoints": self.voltage_setpoints,
            "constraint_violations": self.constraint_violations
        }


class OPFTool:
    """Tool for running optimal power flow analysis."""

    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
        logger.info("OPFTool initialized")

    def setup_opf_constraints(self, net: pp.pandapowerNet) -> pp.pandapowerNet:
        """Set up OPF constraints on the network."""
        net = copy.deepcopy(net)

        # Set voltage constraints on all buses
        net.bus['min_vm_pu'] = self.v_min
        net.bus['max_vm_pu'] = self.v_max

        # Set thermal constraints on lines
        net.line['max_loading_percent'] = self.thermal_limit

        # Set thermal constraints on transformers if present
        if len(net.trafo) > 0:
            net.trafo['max_loading_percent'] = self.thermal_limit

        return net

    def setup_cost_functions(self, net: pp.pandapowerNet,
                            objective: OPFObjective = OPFObjective.MIN_LOSS) -> pp.pandapowerNet:
        """Set up cost functions for OPF optimization."""
        # Clear existing cost functions
        if 'poly_cost' in net:
            net.poly_cost = net.poly_cost.iloc[0:0]
        if 'pwl_cost' in net:
            net.pwl_cost = net.pwl_cost.iloc[0:0]

        if objective == OPFObjective.MIN_LOSS:
            # Minimize losses by minimizing total generation
            # Add small cost to external grid generation
            for idx in net.ext_grid.index:
                if net.ext_grid.at[idx, 'in_service']:
                    pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1.0)

            # Add cost to generators if present
            for idx in net.gen.index:
                if net.gen.at[idx, 'in_service']:
                    pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=1.0)

        elif objective == OPFObjective.MIN_GENERATION_COST:
            # Minimize generation cost with different costs per generator
            for idx in net.ext_grid.index:
                if net.ext_grid.at[idx, 'in_service']:
                    # External grid has higher cost (represents purchased power)
                    pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=50.0)

            for idx in net.gen.index:
                if net.gen.at[idx, 'in_service']:
                    # Local generation has lower cost
                    pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=30.0)

        elif objective == OPFObjective.MIN_LOAD_SHEDDING:
            # Minimize load shedding (maximize served load)
            for idx in net.load.index:
                if net.load.at[idx, 'in_service']:
                    # High cost for not serving load
                    net.load.at[idx, 'controllable'] = True
                    net.load.at[idx, 'min_p_mw'] = 0.0
                    net.load.at[idx, 'max_p_mw'] = net.load.at[idx, 'p_mw']
                    pp.create_poly_cost(net, idx, 'load', cp1_eur_per_mw=-1000.0)

        return net

    def setup_controllable_elements(self, net: pp.pandapowerNet,
                                    controllable_gens: bool = True,
                                    controllable_ext_grid: bool = True) -> pp.pandapowerNet:
        """Mark elements as controllable for OPF."""
        # External grid controllability
        if controllable_ext_grid and len(net.ext_grid) > 0:
            net.ext_grid['controllable'] = True
            # Set power limits for external grid
            net.ext_grid['min_p_mw'] = -1000.0  # Can absorb power
            net.ext_grid['max_p_mw'] = 1000.0   # Can supply power
            net.ext_grid['min_q_mvar'] = -1000.0
            net.ext_grid['max_q_mvar'] = 1000.0

        # Generator controllability
        if controllable_gens and len(net.gen) > 0:
            net.gen['controllable'] = True
            # Ensure min/max are set
            if 'min_p_mw' not in net.gen.columns:
                net.gen['min_p_mw'] = 0.0
            if 'max_p_mw' not in net.gen.columns:
                net.gen['max_p_mw'] = net.gen['p_mw'] * 2

        # Static generators (DERs)
        if len(net.sgen) > 0:
            net.sgen['controllable'] = True
            if 'min_p_mw' not in net.sgen.columns:
                net.sgen['min_p_mw'] = 0.0
            if 'max_p_mw' not in net.sgen.columns:
                net.sgen['max_p_mw'] = net.sgen['p_mw']

        return net

    def run_ac_opf(self, net: pp.pandapowerNet,
                   objective: OPFObjective = OPFObjective.MIN_LOSS,
                   init: str = "flat") -> OPFResult:
        """Run AC Optimal Power Flow."""
        logger.info(f"Running AC OPF with objective: {objective.value}")

        try:
            # Prepare network
            net_opf = self.setup_opf_constraints(net)
            net_opf = self.setup_controllable_elements(net_opf)
            net_opf = self.setup_cost_functions(net_opf, objective)

            # Run OPF
            pp.runopp(net_opf, init=init, verbose=False)

            if not net_opf.OPF_converged:
                logger.warning("AC OPF did not converge")
                return OPFResult(
                    converged=False,
                    objective_value=float('inf'),
                    total_generation_mw=0.0,
                    total_load_mw=0.0,
                    total_losses_mw=0.0,
                    min_voltage_pu=0.0,
                    max_voltage_pu=0.0,
                    max_line_loading_percent=0.0,
                    constraint_violations=["OPF did not converge"]
                )

            # Extract results
            result = self._extract_opf_results(net_opf)
            logger.info(f"AC OPF converged: losses={result.total_losses_mw:.3f} MW")
            return result

        except Exception as e:
            logger.error(f"AC OPF failed: {e}")
            return OPFResult(
                converged=False,
                objective_value=float('inf'),
                total_generation_mw=0.0,
                total_load_mw=0.0,
                total_losses_mw=0.0,
                min_voltage_pu=0.0,
                max_voltage_pu=0.0,
                max_line_loading_percent=0.0,
                constraint_violations=[str(e)]
            )

    def run_dc_opf(self, net: pp.pandapowerNet,
                   objective: OPFObjective = OPFObjective.MIN_LOSS) -> OPFResult:
        """Run DC Optimal Power Flow (faster, linear approximation)."""
        logger.info(f"Running DC OPF with objective: {objective.value}")

        try:
            # Prepare network
            net_opf = self.setup_opf_constraints(net)
            net_opf = self.setup_controllable_elements(net_opf)
            net_opf = self.setup_cost_functions(net_opf, objective)

            # Run DC OPF
            pp.rundcopp(net_opf, verbose=False)

            if not net_opf.OPF_converged:
                logger.warning("DC OPF did not converge")
                return OPFResult(
                    converged=False,
                    objective_value=float('inf'),
                    total_generation_mw=0.0,
                    total_load_mw=0.0,
                    total_losses_mw=0.0,
                    min_voltage_pu=1.0,
                    max_voltage_pu=1.0,
                    max_line_loading_percent=0.0,
                    constraint_violations=["DC OPF did not converge"]
                )

            # Extract results (DC OPF doesn't have voltage results)
            total_gen = net_opf.res_ext_grid.p_mw.sum()
            if len(net_opf.res_gen) > 0:
                total_gen += net_opf.res_gen.p_mw.sum()

            total_load = net_opf.load[net_opf.load.in_service].p_mw.sum()
            losses = total_gen - total_load

            max_loading = 0.0
            if len(net_opf.res_line) > 0:
                max_loading = net_opf.res_line.loading_percent.max()

            result = OPFResult(
                converged=True,
                objective_value=net_opf.res_cost if hasattr(net_opf, 'res_cost') else losses,
                total_generation_mw=float(total_gen),
                total_load_mw=float(total_load),
                total_losses_mw=float(losses),
                min_voltage_pu=1.0,  # DC OPF assumes flat voltage
                max_voltage_pu=1.0,
                max_line_loading_percent=float(max_loading)
            )

            logger.info(f"DC OPF converged: losses={result.total_losses_mw:.3f} MW")
            return result

        except Exception as e:
            logger.error(f"DC OPF failed: {e}")
            return OPFResult(
                converged=False,
                objective_value=float('inf'),
                total_generation_mw=0.0,
                total_load_mw=0.0,
                total_losses_mw=0.0,
                min_voltage_pu=1.0,
                max_voltage_pu=1.0,
                max_line_loading_percent=0.0,
                constraint_violations=[str(e)]
            )

    def _extract_opf_results(self, net: pp.pandapowerNet) -> OPFResult:
        """Extract results from completed OPF."""
        # Generation
        total_gen = net.res_ext_grid.p_mw.sum()
        if len(net.res_gen) > 0:
            total_gen += net.res_gen.p_mw.sum()
        if len(net.res_sgen) > 0:
            total_gen += net.res_sgen.p_mw.sum()

        # Load
        total_load = net.res_load.p_mw.sum()

        # Losses
        losses = total_gen - total_load

        # Voltages
        min_v = net.res_bus.vm_pu.min()
        max_v = net.res_bus.vm_pu.max()

        # Line loading
        max_loading = 0.0
        if len(net.res_line) > 0:
            max_loading = net.res_line.loading_percent.max()

        # Generator dispatch
        gen_dispatch = {}
        for idx in net.ext_grid.index:
            gen_dispatch[f"ext_grid_{idx}"] = float(net.res_ext_grid.at[idx, 'p_mw'])
        for idx in net.gen.index:
            gen_dispatch[f"gen_{idx}"] = float(net.res_gen.at[idx, 'p_mw'])

        # Voltage setpoints
        voltage_setpoints = {}
        for idx in net.bus.index:
            voltage_setpoints[int(idx)] = float(net.res_bus.at[idx, 'vm_pu'])

        # Check for violations
        violations = []

        # Voltage violations
        v_low = net.res_bus[net.res_bus.vm_pu < self.v_min]
        for idx in v_low.index:
            violations.append(f"Bus {idx}: {v_low.at[idx, 'vm_pu']:.4f} pu (undervoltage)")

        v_high = net.res_bus[net.res_bus.vm_pu > self.v_max]
        for idx in v_high.index:
            violations.append(f"Bus {idx}: {v_high.at[idx, 'vm_pu']:.4f} pu (overvoltage)")

        # Thermal violations
        if len(net.res_line) > 0:
            overloaded = net.res_line[net.res_line.loading_percent > self.thermal_limit]
            for idx in overloaded.index:
                violations.append(f"Line {idx}: {overloaded.at[idx, 'loading_percent']:.1f}% (overloaded)")

        return OPFResult(
            converged=True,
            objective_value=net.res_cost if hasattr(net, 'res_cost') else losses,
            total_generation_mw=float(total_gen),
            total_load_mw=float(total_load),
            total_losses_mw=float(losses),
            min_voltage_pu=float(min_v),
            max_voltage_pu=float(max_v),
            max_line_loading_percent=float(max_loading),
            generator_dispatch=gen_dispatch,
            voltage_setpoints=voltage_setpoints,
            constraint_violations=violations
        )

    def optimize_voltage_setpoints(self, net: pp.pandapowerNet,
                                   target_profile: str = "flat") -> Dict[int, float]:
        """Optimize voltage setpoints for minimum losses."""
        logger.info(f"Optimizing voltage setpoints with target: {target_profile}")

        # Run OPF to get optimal voltages
        result = self.run_ac_opf(net, OPFObjective.MIN_LOSS)

        if result.converged:
            return result.voltage_setpoints
        else:
            # Return default setpoints
            return {int(idx): 1.0 for idx in net.bus.index}

    def calculate_loss_sensitivity(self, net: pp.pandapowerNet) -> Dict[str, float]:
        """Calculate sensitivity of losses to various parameters."""
        logger.info("Calculating loss sensitivity")

        sensitivities = {}

        # Run baseline
        pp.runpp(net)
        if not net.converged:
            return sensitivities

        baseline_losses = net.res_line.pl_mw.sum()

        # Sensitivity to voltage setpoint
        net_test = copy.deepcopy(net)
        net_test.ext_grid.at[0, 'vm_pu'] = net.ext_grid.at[0, 'vm_pu'] + 0.01
        pp.runpp(net_test)
        if net_test.converged:
            new_losses = net_test.res_line.pl_mw.sum()
            sensitivities['voltage_setpoint'] = (new_losses - baseline_losses) / 0.01

        # Sensitivity to load (per MW)
        if len(net.load) > 0:
            net_test = copy.deepcopy(net)
            net_test.load.at[0, 'p_mw'] = net.load.at[0, 'p_mw'] + 0.1
            pp.runpp(net_test)
            if net_test.converged:
                new_losses = net_test.res_line.pl_mw.sum()
                sensitivities['load_increase'] = (new_losses - baseline_losses) / 0.1

        return sensitivities


def run_opf_analysis(net: pp.pandapowerNet,
                     objective: str = "min_loss",
                     voltage_limits: Tuple[float, float] = (0.95, 1.05)) -> OPFResult:
    """Convenience function to run OPF analysis."""
    tool = OPFTool(voltage_limits=voltage_limits)

    obj_map = {
        "min_loss": OPFObjective.MIN_LOSS,
        "min_cost": OPFObjective.MIN_GENERATION_COST,
        "min_shed": OPFObjective.MIN_LOAD_SHEDDING
    }

    objective_enum = obj_map.get(objective, OPFObjective.MIN_LOSS)
    return tool.run_ac_opf(net, objective_enum)

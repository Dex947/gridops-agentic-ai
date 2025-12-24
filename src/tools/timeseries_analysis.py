"""Time-series analysis for power system studies."""

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandapower as pp
from loguru import logger


@dataclass
class LoadProfile:
    """Represents a load profile over time."""
    name: str
    timestamps: List[datetime]
    multipliers: List[float]  # Multipliers relative to base load

    def get_multiplier(self, timestamp: datetime) -> float:
        """Get load multiplier for a specific timestamp."""
        if not self.timestamps:
            return 1.0

        # Find closest timestamp
        min_diff = float('inf')
        closest_mult = 1.0

        for ts, mult in zip(self.timestamps, self.multipliers, strict=False):
            diff = abs((ts - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_mult = mult

        return closest_mult

    @classmethod
    def create_daily_profile(cls, name: str = "residential",
                            base_date: datetime = None) -> 'LoadProfile':
        """Create a typical daily load profile."""
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Typical residential profile (24 hours)
        hourly_multipliers = [
            0.6, 0.5, 0.5, 0.5, 0.5, 0.6,  # 00:00 - 05:00
            0.7, 0.9, 1.0, 0.9, 0.8, 0.8,  # 06:00 - 11:00
            0.9, 0.9, 0.8, 0.8, 0.9, 1.0,  # 12:00 - 17:00
            1.2, 1.3, 1.2, 1.0, 0.8, 0.7   # 18:00 - 23:00
        ]

        timestamps = [base_date + timedelta(hours=h) for h in range(24)]

        return cls(name=name, timestamps=timestamps, multipliers=hourly_multipliers)

    @classmethod
    def create_industrial_profile(cls, name: str = "industrial",
                                  base_date: datetime = None) -> 'LoadProfile':
        """Create a typical industrial load profile."""
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Industrial profile (high during work hours)
        hourly_multipliers = [
            0.3, 0.3, 0.3, 0.3, 0.3, 0.4,  # 00:00 - 05:00
            0.6, 0.9, 1.0, 1.0, 1.0, 0.9,  # 06:00 - 11:00
            0.8, 1.0, 1.0, 1.0, 1.0, 0.9,  # 12:00 - 17:00
            0.6, 0.4, 0.3, 0.3, 0.3, 0.3   # 18:00 - 23:00
        ]

        timestamps = [base_date + timedelta(hours=h) for h in range(24)]

        return cls(name=name, timestamps=timestamps, multipliers=hourly_multipliers)


@dataclass
class GenerationProfile:
    """Represents a generation profile (e.g., solar, wind)."""
    name: str
    timestamps: List[datetime]
    capacity_factors: List[float]  # 0.0 to 1.0

    def get_capacity_factor(self, timestamp: datetime) -> float:
        """Get capacity factor for a specific timestamp."""
        if not self.timestamps:
            return 0.0

        min_diff = float('inf')
        closest_cf = 0.0

        for ts, cf in zip(self.timestamps, self.capacity_factors, strict=False):
            diff = abs((ts - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_cf = cf

        return closest_cf

    @classmethod
    def create_solar_profile(cls, name: str = "solar",
                            base_date: datetime = None) -> 'GenerationProfile':
        """Create a typical solar generation profile."""
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Solar profile (peaks at noon)
        hourly_cf = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 00:00 - 05:00
            0.1, 0.3, 0.5, 0.7, 0.85, 0.95, # 06:00 - 11:00
            1.0, 0.95, 0.85, 0.7, 0.5, 0.3, # 12:00 - 17:00
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0   # 18:00 - 23:00
        ]

        timestamps = [base_date + timedelta(hours=h) for h in range(24)]

        return cls(name=name, timestamps=timestamps, capacity_factors=hourly_cf)

    @classmethod
    def create_wind_profile(cls, name: str = "wind",
                           base_date: datetime = None,
                           wind_pattern: str = "moderate") -> 'GenerationProfile':
        """
        Create a typical wind generation profile.
        
        Args:
            name: Profile name
            base_date: Base date for timestamps
            wind_pattern: Wind pattern type ('low', 'moderate', 'high', 'variable')
            
        Returns:
            GenerationProfile for wind generation
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Wind profiles vary by pattern type
        if wind_pattern == "low":
            # Low wind day - peaks in afternoon
            hourly_cf = [
                0.15, 0.12, 0.10, 0.10, 0.12, 0.15,  # 00:00 - 05:00
                0.18, 0.20, 0.22, 0.25, 0.28, 0.30,  # 06:00 - 11:00
                0.32, 0.35, 0.38, 0.40, 0.38, 0.35,  # 12:00 - 17:00
                0.30, 0.25, 0.22, 0.20, 0.18, 0.15   # 18:00 - 23:00
            ]
        elif wind_pattern == "high":
            # High wind day - strong throughout
            hourly_cf = [
                0.70, 0.72, 0.75, 0.78, 0.80, 0.82,  # 00:00 - 05:00
                0.85, 0.88, 0.90, 0.88, 0.85, 0.82,  # 06:00 - 11:00
                0.80, 0.78, 0.80, 0.82, 0.85, 0.88,  # 12:00 - 17:00
                0.90, 0.88, 0.85, 0.80, 0.75, 0.72   # 18:00 - 23:00
            ]
        elif wind_pattern == "variable":
            # Variable wind day - gusty conditions
            hourly_cf = [
                0.25, 0.40, 0.30, 0.55, 0.45, 0.60,  # 00:00 - 05:00
                0.50, 0.35, 0.65, 0.55, 0.70, 0.45,  # 06:00 - 11:00
                0.60, 0.75, 0.50, 0.65, 0.80, 0.55,  # 12:00 - 17:00
                0.70, 0.45, 0.60, 0.40, 0.35, 0.30   # 18:00 - 23:00
            ]
        else:  # moderate (default)
            # Moderate wind day - typical diurnal pattern with evening peak
            hourly_cf = [
                0.35, 0.32, 0.30, 0.28, 0.30, 0.35,  # 00:00 - 05:00
                0.40, 0.45, 0.48, 0.50, 0.52, 0.55,  # 06:00 - 11:00
                0.58, 0.60, 0.62, 0.65, 0.68, 0.70,  # 12:00 - 17:00
                0.72, 0.68, 0.60, 0.50, 0.42, 0.38   # 18:00 - 23:00
            ]

        timestamps = [base_date + timedelta(hours=h) for h in range(24)]

        return cls(name=name, timestamps=timestamps, capacity_factors=hourly_cf)

    @classmethod
    def create_combined_renewable_profile(cls, name: str = "combined",
                                          base_date: datetime = None,
                                          solar_weight: float = 0.6,
                                          wind_weight: float = 0.4) -> 'GenerationProfile':
        """
        Create a combined solar+wind generation profile.
        
        Args:
            name: Profile name
            base_date: Base date for timestamps
            solar_weight: Weight for solar component (0-1)
            wind_weight: Weight for wind component (0-1)
            
        Returns:
            GenerationProfile for combined renewable generation
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        solar = cls.create_solar_profile(base_date=base_date)
        wind = cls.create_wind_profile(base_date=base_date, wind_pattern="moderate")
        
        # Normalize weights
        total_weight = solar_weight + wind_weight
        solar_w = solar_weight / total_weight
        wind_w = wind_weight / total_weight
        
        # Combine capacity factors
        combined_cf = [
            solar_w * s + wind_w * w 
            for s, w in zip(solar.capacity_factors, wind.capacity_factors)
        ]

        timestamps = [base_date + timedelta(hours=h) for h in range(24)]

        return cls(name=name, timestamps=timestamps, capacity_factors=combined_cf)


@dataclass
class TimeSeriesResult:
    """Results from a single time step."""
    timestamp: datetime
    converged: bool
    total_load_mw: float
    total_generation_mw: float
    total_losses_mw: float
    min_voltage_pu: float
    max_voltage_pu: float
    max_line_loading_percent: float
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "converged": self.converged,
            "total_load_mw": float(self.total_load_mw),
            "total_generation_mw": float(self.total_generation_mw),
            "total_losses_mw": float(self.total_losses_mw),
            "min_voltage_pu": float(self.min_voltage_pu),
            "max_voltage_pu": float(self.max_voltage_pu),
            "max_line_loading_percent": float(self.max_line_loading_percent),
            "violations": self.violations
        }


@dataclass
class TimeSeriesAnalysisResult:
    """Results from time-series analysis."""
    success: bool
    num_timesteps: int
    num_converged: int
    num_violations: int
    peak_load_mw: float
    peak_load_time: Optional[datetime]
    min_voltage_pu: float
    min_voltage_time: Optional[datetime]
    max_loading_percent: float
    max_loading_time: Optional[datetime]
    total_energy_mwh: float
    total_losses_mwh: float
    timestep_results: List[TimeSeriesResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "num_timesteps": self.num_timesteps,
            "num_converged": self.num_converged,
            "num_violations": self.num_violations,
            "peak_load_mw": float(self.peak_load_mw),
            "peak_load_time": self.peak_load_time.isoformat() if self.peak_load_time else None,
            "min_voltage_pu": float(self.min_voltage_pu),
            "min_voltage_time": self.min_voltage_time.isoformat() if self.min_voltage_time else None,
            "max_loading_percent": float(self.max_loading_percent),
            "max_loading_time": self.max_loading_time.isoformat() if self.max_loading_time else None,
            "total_energy_mwh": float(self.total_energy_mwh),
            "total_losses_mwh": float(self.total_losses_mwh),
            "timestep_results": [r.to_dict() for r in self.timestep_results]
        }


class TimeSeriesAnalysis:
    """Run time-series power flow analysis."""

    def __init__(self, voltage_limits: Tuple[float, float] = (0.95, 1.05),
                 thermal_limit_percent: float = 100.0):
        self.v_min, self.v_max = voltage_limits
        self.thermal_limit = thermal_limit_percent
        logger.info("TimeSeriesAnalysis initialized")

    def run_daily_analysis(self, net: pp.pandapowerNet,
                          load_profile: LoadProfile = None,
                          generation_profile: GenerationProfile = None,
                          timestep_minutes: int = 60) -> TimeSeriesAnalysisResult:
        """
        Run 24-hour time-series analysis.

        Args:
            net: Base pandapower network
            load_profile: Load profile (default: residential)
            generation_profile: Generation profile for DERs (default: solar)
            timestep_minutes: Time step in minutes

        Returns:
            Time-series analysis results
        """
        logger.info(f"Running daily time-series analysis (timestep={timestep_minutes}min)")

        if load_profile is None:
            load_profile = LoadProfile.create_daily_profile()

        if generation_profile is None:
            generation_profile = GenerationProfile.create_solar_profile()

        # Generate timestamps
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        num_steps = 24 * 60 // timestep_minutes
        timestamps = [base_time + timedelta(minutes=i * timestep_minutes) for i in range(num_steps)]

        return self.run_timeseries(net, timestamps, load_profile, generation_profile)

    def run_timeseries(self, net: pp.pandapowerNet,
                      timestamps: List[datetime],
                      load_profile: LoadProfile,
                      generation_profile: GenerationProfile = None) -> TimeSeriesAnalysisResult:
        """
        Run time-series power flow for given timestamps.

        Args:
            net: Base pandapower network
            timestamps: List of timestamps to analyze
            load_profile: Load profile
            generation_profile: Optional generation profile for DERs

        Returns:
            Time-series analysis results
        """
        logger.info(f"Running time-series analysis for {len(timestamps)} timesteps")

        # Store base load values
        base_loads_p = net.load.p_mw.copy()
        base_loads_q = net.load.q_mvar.copy()
        base_sgen_p = net.sgen.p_mw.copy() if len(net.sgen) > 0 else None

        results = []

        for i, ts in enumerate(timestamps):
            if (i + 1) % 10 == 0:
                logger.debug(f"Processing timestep {i+1}/{len(timestamps)}")

            net_work = copy.deepcopy(net)

            # Apply load multiplier
            load_mult = load_profile.get_multiplier(ts)
            net_work.load.p_mw = base_loads_p * load_mult
            net_work.load.q_mvar = base_loads_q * load_mult

            # Apply generation profile if provided
            if generation_profile and base_sgen_p is not None:
                cf = generation_profile.get_capacity_factor(ts)
                net_work.sgen.p_mw = base_sgen_p * cf

            # Run power flow
            result = self._run_single_timestep(net_work, ts)
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results, timestamps)

    def _run_single_timestep(self, net: pp.pandapowerNet,
                            timestamp: datetime) -> TimeSeriesResult:
        """Run power flow for a single timestep."""
        try:
            pp.runpp(net, algorithm="bfsw")

            if not net.converged:
                return TimeSeriesResult(
                    timestamp=timestamp,
                    converged=False,
                    total_load_mw=0.0,
                    total_generation_mw=0.0,
                    total_losses_mw=0.0,
                    min_voltage_pu=0.0,
                    max_voltage_pu=0.0,
                    max_line_loading_percent=0.0,
                    violations=["Power flow did not converge"]
                )

            # Extract results
            total_load = net.res_load.p_mw.sum()
            total_gen = net.res_ext_grid.p_mw.sum()
            if len(net.res_sgen) > 0:
                total_gen += net.res_sgen.p_mw.sum()

            total_losses = net.res_line.pl_mw.sum() if len(net.res_line) > 0 else 0.0

            min_v = net.res_bus.vm_pu.min()
            max_v = net.res_bus.vm_pu.max()
            max_loading = net.res_line.loading_percent.max() if len(net.res_line) > 0 else 0.0

            # Check for violations
            violations = []

            if min_v < self.v_min:
                violations.append(f"Undervoltage: {min_v:.4f} pu")
            if max_v > self.v_max:
                violations.append(f"Overvoltage: {max_v:.4f} pu")
            if max_loading > self.thermal_limit:
                violations.append(f"Overload: {max_loading:.1f}%")

            return TimeSeriesResult(
                timestamp=timestamp,
                converged=True,
                total_load_mw=float(total_load),
                total_generation_mw=float(total_gen),
                total_losses_mw=float(total_losses),
                min_voltage_pu=float(min_v),
                max_voltage_pu=float(max_v),
                max_line_loading_percent=float(max_loading),
                violations=violations
            )

        except Exception as e:
            logger.error(f"Timestep failed: {e}")
            return TimeSeriesResult(
                timestamp=timestamp,
                converged=False,
                total_load_mw=0.0,
                total_generation_mw=0.0,
                total_losses_mw=0.0,
                min_voltage_pu=0.0,
                max_voltage_pu=0.0,
                max_line_loading_percent=0.0,
                violations=[str(e)]
            )

    def _aggregate_results(self, results: List[TimeSeriesResult],
                          timestamps: List[datetime]) -> TimeSeriesAnalysisResult:
        """Aggregate timestep results into summary."""
        num_converged = sum(1 for r in results if r.converged)
        num_violations = sum(1 for r in results if r.violations)

        # Find peaks
        peak_load = 0.0
        peak_load_time = None
        min_voltage = 1.0
        min_voltage_time = None
        max_loading = 0.0
        max_loading_time = None

        total_energy = 0.0
        total_losses = 0.0

        # Calculate time step duration in hours
        if len(timestamps) >= 2:
            dt_hours = (timestamps[1] - timestamps[0]).total_seconds() / 3600
        else:
            dt_hours = 1.0

        for r in results:
            if r.converged:
                # Peak load
                if r.total_load_mw > peak_load:
                    peak_load = r.total_load_mw
                    peak_load_time = r.timestamp

                # Min voltage
                if r.min_voltage_pu < min_voltage:
                    min_voltage = r.min_voltage_pu
                    min_voltage_time = r.timestamp

                # Max loading
                if r.max_line_loading_percent > max_loading:
                    max_loading = r.max_line_loading_percent
                    max_loading_time = r.timestamp

                # Energy integration
                total_energy += r.total_load_mw * dt_hours
                total_losses += r.total_losses_mw * dt_hours

        return TimeSeriesAnalysisResult(
            success=num_converged > 0,
            num_timesteps=len(results),
            num_converged=num_converged,
            num_violations=num_violations,
            peak_load_mw=peak_load,
            peak_load_time=peak_load_time,
            min_voltage_pu=min_voltage,
            min_voltage_time=min_voltage_time,
            max_loading_percent=max_loading,
            max_loading_time=max_loading_time,
            total_energy_mwh=total_energy,
            total_losses_mwh=total_losses,
            timestep_results=results
        )

    def identify_critical_periods(self, result: TimeSeriesAnalysisResult,
                                 voltage_threshold: float = None,
                                 loading_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Identify critical time periods from analysis results.

        Args:
            result: Time-series analysis result
            voltage_threshold: Voltage threshold (default: v_min)
            loading_threshold: Loading threshold (default: thermal_limit)

        Returns:
            List of critical periods with details
        """
        if voltage_threshold is None:
            voltage_threshold = self.v_min
        if loading_threshold is None:
            loading_threshold = self.thermal_limit

        critical_periods = []

        for r in result.timestep_results:
            if not r.converged:
                critical_periods.append({
                    "timestamp": r.timestamp.isoformat(),
                    "reason": "non_convergence",
                    "severity": "critical"
                })
            elif r.min_voltage_pu < voltage_threshold:
                critical_periods.append({
                    "timestamp": r.timestamp.isoformat(),
                    "reason": "undervoltage",
                    "value": r.min_voltage_pu,
                    "threshold": voltage_threshold,
                    "severity": "warning" if r.min_voltage_pu > voltage_threshold - 0.02 else "critical"
                })
            elif r.max_line_loading_percent > loading_threshold:
                critical_periods.append({
                    "timestamp": r.timestamp.isoformat(),
                    "reason": "overload",
                    "value": r.max_line_loading_percent,
                    "threshold": loading_threshold,
                    "severity": "warning" if r.max_line_loading_percent < loading_threshold + 10 else "critical"
                })

        return critical_periods


def run_daily_timeseries(net: pp.pandapowerNet,
                        profile_type: str = "residential") -> TimeSeriesAnalysisResult:
    """Convenience function for daily time-series analysis."""
    analysis = TimeSeriesAnalysis()

    if profile_type == "industrial":
        load_profile = LoadProfile.create_industrial_profile()
    else:
        load_profile = LoadProfile.create_daily_profile()

    return analysis.run_daily_analysis(net, load_profile)

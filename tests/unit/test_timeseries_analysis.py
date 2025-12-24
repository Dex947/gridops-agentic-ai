"""Tests for time-series analysis module."""

import pytest
import pandapower as pp
import pandapower.networks as pn
from datetime import datetime, timedelta

from src.tools.timeseries_analysis import (
    TimeSeriesAnalysis,
    TimeSeriesAnalysisResult,
    TimeSeriesResult,
    LoadProfile,
    GenerationProfile,
    run_daily_timeseries,
)


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    net = pp.create_empty_network()
    
    buses = [pp.create_bus(net, vn_kv=20.0, name=f"Bus {i}") for i in range(5)]
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
    
    for i in range(4):
        pp.create_line(net, from_bus=buses[i], to_bus=buses[i+1],
                      length_km=1.0, std_type="NAYY 4x50 SE")
    
    for i in range(1, 5):
        pp.create_load(net, bus=buses[i], p_mw=0.2, q_mvar=0.05)
    
    pp.runpp(net)
    return net


@pytest.fixture
def ieee33_network():
    """Load IEEE 33-bus network."""
    net = pn.case33bw()
    pp.runpp(net)
    return net


class TestLoadProfile:
    """Tests for LoadProfile class."""

    def test_create_daily_profile(self):
        """Should create daily load profile."""
        profile = LoadProfile.create_daily_profile()
        
        assert profile.name == "residential"
        assert len(profile.timestamps) == 24
        assert len(profile.multipliers) == 24

    def test_create_industrial_profile(self):
        """Should create industrial profile."""
        profile = LoadProfile.create_industrial_profile()
        
        assert profile.name == "industrial"
        assert len(profile.timestamps) == 24

    def test_get_multiplier(self):
        """Should get multiplier for timestamp."""
        profile = LoadProfile.create_daily_profile()
        
        # Get multiplier for noon
        noon = profile.timestamps[12]
        mult = profile.get_multiplier(noon)
        
        assert mult > 0
        assert mult <= 2.0

    def test_peak_evening_load(self):
        """Residential profile should peak in evening."""
        profile = LoadProfile.create_daily_profile()
        
        # Evening hours (18-20) should have high multipliers
        evening_mults = profile.multipliers[18:21]
        morning_mults = profile.multipliers[2:5]
        
        assert max(evening_mults) > max(morning_mults)


class TestGenerationProfile:
    """Tests for GenerationProfile class."""

    def test_create_solar_profile(self):
        """Should create solar generation profile."""
        profile = GenerationProfile.create_solar_profile()
        
        assert profile.name == "solar"
        assert len(profile.timestamps) == 24
        assert len(profile.capacity_factors) == 24

    def test_solar_zero_at_night(self):
        """Solar should be zero at night."""
        profile = GenerationProfile.create_solar_profile()
        
        # Night hours (0-5) should be zero
        night_cf = profile.capacity_factors[0:6]
        
        assert all(cf == 0.0 for cf in night_cf)

    def test_solar_peak_at_noon(self):
        """Solar should peak around noon."""
        profile = GenerationProfile.create_solar_profile()
        
        # Noon (12) should have highest capacity factor
        noon_cf = profile.capacity_factors[12]
        
        assert noon_cf == max(profile.capacity_factors)

    def test_get_capacity_factor(self):
        """Should get capacity factor for timestamp."""
        profile = GenerationProfile.create_solar_profile()
        
        noon = profile.timestamps[12]
        cf = profile.get_capacity_factor(noon)
        
        assert cf == 1.0  # Peak at noon


class TestTimeSeriesAnalysisInit:
    """Tests for initialization."""

    def test_init_default_limits(self):
        """Should initialize with default limits."""
        analysis = TimeSeriesAnalysis()
        
        assert analysis.v_min == 0.95
        assert analysis.v_max == 1.05
        assert analysis.thermal_limit == 100.0

    def test_init_custom_limits(self):
        """Should accept custom limits."""
        analysis = TimeSeriesAnalysis(voltage_limits=(0.90, 1.10))
        
        assert analysis.v_min == 0.90
        assert analysis.v_max == 1.10


class TestRunDailyAnalysis:
    """Tests for daily analysis."""

    def test_run_daily_analysis(self, simple_network):
        """Should run daily analysis."""
        analysis = TimeSeriesAnalysis()
        
        result = analysis.run_daily_analysis(
            simple_network,
            timestep_minutes=60
        )
        
        assert isinstance(result, TimeSeriesAnalysisResult)
        assert result.num_timesteps == 24

    def test_daily_analysis_with_custom_profile(self, simple_network):
        """Should use custom load profile."""
        analysis = TimeSeriesAnalysis()
        profile = LoadProfile.create_industrial_profile()
        
        result = analysis.run_daily_analysis(
            simple_network,
            load_profile=profile
        )
        
        assert result.num_timesteps == 24

    def test_daily_analysis_15min_resolution(self, simple_network):
        """Should handle 15-minute resolution."""
        analysis = TimeSeriesAnalysis()
        
        result = analysis.run_daily_analysis(
            simple_network,
            timestep_minutes=15
        )
        
        assert result.num_timesteps == 96  # 24 * 4


class TestTimeSeriesResult:
    """Tests for TimeSeriesResult."""

    def test_result_structure(self, simple_network):
        """Result should have all required fields."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'num_timesteps')
        assert hasattr(result, 'num_converged')
        assert hasattr(result, 'peak_load_mw')
        assert hasattr(result, 'total_energy_mwh')
        assert hasattr(result, 'total_losses_mwh')

    def test_result_to_dict(self, simple_network):
        """Result should convert to dictionary."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'success' in result_dict
        assert 'num_timesteps' in result_dict
        assert 'timestep_results' in result_dict

    def test_timestep_results_included(self, simple_network):
        """Should include individual timestep results."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        assert len(result.timestep_results) == 24
        
        for ts_result in result.timestep_results:
            assert isinstance(ts_result, TimeSeriesResult)
            assert hasattr(ts_result, 'timestamp')
            assert hasattr(ts_result, 'converged')


class TestPeakIdentification:
    """Tests for peak identification."""

    def test_peak_load_identified(self, simple_network):
        """Should identify peak load."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        assert result.peak_load_mw > 0
        assert result.peak_load_time is not None

    def test_min_voltage_identified(self, simple_network):
        """Should identify minimum voltage."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        assert result.min_voltage_pu > 0
        assert result.min_voltage_pu <= 1.0


class TestEnergyCalculation:
    """Tests for energy calculations."""

    def test_total_energy_calculated(self, simple_network):
        """Should calculate total energy."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        # Energy should be positive
        assert result.total_energy_mwh > 0
        
        # Losses should be less than total energy
        assert result.total_losses_mwh < result.total_energy_mwh

    def test_energy_proportional_to_load(self, simple_network):
        """Energy should be roughly load * hours."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        # Base load is 0.8 MW (4 loads * 0.2 MW)
        # With varying profile, expect roughly 0.8 * 24 * avg_multiplier
        assert result.total_energy_mwh > 10  # Reasonable lower bound
        assert result.total_energy_mwh < 50  # Reasonable upper bound


class TestCriticalPeriods:
    """Tests for critical period identification."""

    def test_identify_critical_periods(self, simple_network):
        """Should identify critical periods."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        critical = analysis.identify_critical_periods(result)
        
        assert isinstance(critical, list)

    def test_critical_periods_with_custom_threshold(self, simple_network):
        """Should use custom thresholds."""
        analysis = TimeSeriesAnalysis()
        result = analysis.run_daily_analysis(simple_network, timestep_minutes=60)
        
        # Very strict threshold should find more violations
        critical_strict = analysis.identify_critical_periods(
            result,
            voltage_threshold=0.99
        )
        
        critical_normal = analysis.identify_critical_periods(
            result,
            voltage_threshold=0.95
        )
        
        assert len(critical_strict) >= len(critical_normal)


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_run_daily_timeseries(self, simple_network):
        """Convenience function should work."""
        result = run_daily_timeseries(simple_network)
        
        assert isinstance(result, TimeSeriesAnalysisResult)
        assert result.num_timesteps == 24

    def test_run_daily_timeseries_industrial(self, simple_network):
        """Should accept industrial profile."""
        result = run_daily_timeseries(simple_network, profile_type="industrial")
        
        assert isinstance(result, TimeSeriesAnalysisResult)


class TestWithDER:
    """Tests with DER integration."""

    def test_timeseries_with_solar(self, simple_network):
        """Should handle solar generation."""
        # Add solar PV
        pp.create_sgen(simple_network, bus=2, p_mw=0.3, q_mvar=0.0)
        
        analysis = TimeSeriesAnalysis()
        solar_profile = GenerationProfile.create_solar_profile()
        
        result = analysis.run_daily_analysis(
            simple_network,
            generation_profile=solar_profile
        )
        
        assert result.success == True
        assert result.num_converged > 0

"""
Unit tests for the GasTemporalEvolution class.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from src.data_generation.gas.gas_temporal_evolution import GasTemporalEvolution, ReleasePattern, FireStage


class TestGasTemporalEvolution(unittest.TestCase):
    """Test cases for the GasTemporalEvolution class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic configuration for testing
        self.config = {
            'default_release_pattern': 'gradual',
            'default_duration': 300,
            'default_sample_rate': 1.0,
            'gas_properties': {
                'methane': {
                    'production_rate': 0.01,
                    'molecular_weight': 16.04,
                    'flammability_limits': (5.0, 15.0),
                    'ignition_temp': 580
                },
                'propane': {
                    'production_rate': 0.005,
                    'molecular_weight': 44.1,
                    'flammability_limits': (2.1, 9.5),
                    'ignition_temp': 470
                }
            },
            'fire_stage_durations': {
                FireStage.INCIPIENT: 0.2,
                FireStage.GROWTH: 0.3,
                FireStage.FULLY_DEVELOPED: 0.4,
                FireStage.DECAY: 0.1
            }
        }
        
        # Create the temporal evolution model
        self.model = GasTemporalEvolution(self.config)
    
    def test_initialization(self):
        """Test initialization of the temporal evolution model."""
        self.assertEqual(self.model.default_release_pattern, ReleasePattern.GRADUAL)
        self.assertEqual(self.model.default_duration, 300)
        self.assertEqual(self.model.default_sample_rate, 1.0)
        self.assertIsNotNone(self.model.gas_properties)
        self.assertIn('methane', self.model.gas_properties)
        self.assertIn('propane', self.model.gas_properties)
        self.assertEqual(self.model.fire_stage_durations[FireStage.INCIPIENT], 0.2)
        self.assertEqual(self.model.fire_stage_durations[FireStage.GROWTH], 0.3)
        self.assertEqual(self.model.fire_stage_durations[FireStage.FULLY_DEVELOPED], 0.4)
        self.assertEqual(self.model.fire_stage_durations[FireStage.DECAY], 0.1)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Test with valid config
        self.model.validate_config()
        
        # Test with invalid release pattern
        config_invalid_pattern = self.config.copy()
        config_invalid_pattern['default_release_pattern'] = 'invalid_pattern'
        with self.assertRaises(ValueError):
            GasTemporalEvolution(config_invalid_pattern).validate_config()
        
        # Test with negative duration
        config_negative_duration = self.config.copy()
        config_negative_duration['default_duration'] = -10
        with self.assertRaises(ValueError):
            GasTemporalEvolution(config_negative_duration).validate_config()
        
        # Test with negative sample rate
        config_negative_rate = self.config.copy()
        config_negative_rate['default_sample_rate'] = -1.0
        with self.assertRaises(ValueError):
            GasTemporalEvolution(config_negative_rate).validate_config()
        
        # Test with invalid fire stage durations
        config_invalid_stages = self.config.copy()
        config_invalid_stages['fire_stage_durations'] = {
            FireStage.INCIPIENT: 0.3,
            FireStage.GROWTH: 0.3,
            FireStage.FULLY_DEVELOPED: 0.5,
            FireStage.DECAY: 0.1
        }  # Sum is 1.2, not 1.0
        with self.assertRaises(ValueError):
            GasTemporalEvolution(config_invalid_stages).validate_config()
    
    def test_generate_concentration_time_series(self):
        """Test generation of concentration time series."""
        # Test with default parameters
        start_time = datetime.now()
        gas_type = 'methane'
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        result = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time
        )
        
        # Check result structure
        self.assertIn('concentrations', result)
        self.assertIn('timestamps', result)
        self.assertIn('metadata', result)
        
        # Check concentrations
        self.assertEqual(len(result['concentrations']), self.model.default_duration * self.model.default_sample_rate)
        self.assertGreater(np.max(result['concentrations']), 0.0)
        
        # Check timestamps
        self.assertEqual(len(result['timestamps']), len(result['concentrations']))
        self.assertEqual(result['timestamps'][0], start_time)
        
        # Check metadata
        self.assertEqual(result['metadata']['gas_type'], gas_type)
        self.assertEqual(result['metadata']['release_pattern'], 'gradual')
        self.assertEqual(result['metadata']['duration'], self.model.default_duration)
        self.assertEqual(result['metadata']['sample_rate'], self.model.default_sample_rate)
        
        # Test with custom parameters
        duration = 100
        sample_rate = 2.0
        release_pattern = 'sudden'
        params = {'peak_concentration': 500.0, 'rise_time': 5.0}
        
        result = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern=release_pattern,
            params=params
        )
        
        # Check result with custom parameters
        self.assertEqual(len(result['concentrations']), duration * sample_rate)
        self.assertEqual(result['metadata']['release_pattern'], release_pattern)
        self.assertEqual(result['metadata']['duration'], duration)
        self.assertEqual(result['metadata']['sample_rate'], sample_rate)
        self.assertEqual(result['metadata']['parameters']['peak_concentration'], 500.0)
    
    def test_generate_release_patterns(self):
        """Test generation of different release patterns."""
        start_time = datetime.now()
        gas_type = 'methane'
        duration = 100
        sample_rate = 1.0
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Test sudden release
        result_sudden = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='sudden'
        )
        
        # Test gradual release
        result_gradual = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='gradual'
        )
        
        # Test intermittent release
        result_intermittent = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='intermittent'
        )
        
        # Test exponential release
        result_exponential = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='exponential'
        )
        
        # Test logarithmic release
        result_logarithmic = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='logarithmic'
        )
        
        # Test sinusoidal release
        result_sinusoidal = self.model.generate_concentration_time_series(
            gas_type=gas_type,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            release_pattern='sinusoidal'
        )
        
        # Check that different patterns produce different concentration profiles
        patterns = [
            result_sudden['concentrations'],
            result_gradual['concentrations'],
            result_intermittent['concentrations'],
            result_exponential['concentrations'],
            result_logarithmic['concentrations'],
            result_sinusoidal['concentrations']
        ]
        
        # Compare each pattern with every other pattern
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Check that patterns are different (using correlation coefficient)
                corr = np.corrcoef(patterns[i], patterns[j])[0, 1]
                self.assertLess(corr, 0.99)  # Not perfectly correlated
    
    def test_generate_fire_scenario_concentrations(self):
        """Test generation of fire scenario concentrations."""
        # Test parameters
        gas_types = ['methane', 'propane']
        start_time = datetime.now()
        duration = 300
        sample_rate = 1.0
        fire_type = 'flaming'
        fuel_load = 10.0
        room_volume = 50.0
        ventilation_rate = 0.5
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Generate scenario
        result = self.model.generate_fire_scenario_concentrations(
            gas_types=gas_types,
            start_time=start_time,
            duration=duration,
            sample_rate=sample_rate,
            fire_type=fire_type,
            fuel_load=fuel_load,
            room_volume=room_volume,
            ventilation_rate=ventilation_rate
        )
        
        # Check result structure
        self.assertIn('gas_data', result)
        self.assertIn('timestamps', result)
        self.assertIn('fire_stages', result)
        self.assertIn('metadata', result)
        
        # Check gas data
        for gas_type in gas_types:
            self.assertIn(gas_type, result['gas_data'])
            self.assertIn('concentrations', result['gas_data'][gas_type])
            self.assertIn('timestamps', result['gas_data'][gas_type])
            self.assertIn('metadata', result['gas_data'][gas_type])
            
            # Check concentrations
            self.assertEqual(len(result['gas_data'][gas_type]['concentrations']), duration * sample_rate)
            self.assertGreaterEqual(np.min(result['gas_data'][gas_type]['concentrations']), 0.0)
        
        # Check timestamps
        self.assertEqual(len(result['timestamps']), duration * sample_rate)
        self.assertEqual(result['timestamps'][0], start_time)
        
        # Check fire stages
        self.assertEqual(len(result['fire_stages']), duration * sample_rate)
        
        # Check metadata
        self.assertEqual(result['metadata']['fire_type'], fire_type)
        self.assertEqual(result['metadata']['fuel_load'], fuel_load)
        self.assertEqual(result['metadata']['room_volume'], room_volume)
        self.assertEqual(result['metadata']['ventilation_rate'], ventilation_rate)
    
    def test_determine_fire_stages(self):
        """Test determination of fire stages."""
        # Create time points for a 100-second scenario
        time_points = np.linspace(0, 100, 100)
        duration = 100.0
        
        # Determine fire stages
        stages = self.model._determine_fire_stages(time_points, duration)
        
        # Check that stages are assigned correctly based on fire_stage_durations
        self.assertEqual(stages[0], FireStage.INCIPIENT)
        self.assertEqual(stages[19], FireStage.INCIPIENT)  # Last incipient point (20%)
        self.assertEqual(stages[20], FireStage.GROWTH)     # First growth point
        self.assertEqual(stages[49], FireStage.GROWTH)     # Last growth point (50%)
        self.assertEqual(stages[50], FireStage.FULLY_DEVELOPED)  # First fully developed point
        self.assertEqual(stages[89], FireStage.FULLY_DEVELOPED)  # Last fully developed point (90%)
        self.assertEqual(stages[90], FireStage.DECAY)      # First decay point
        self.assertEqual(stages[99], FireStage.DECAY)      # Last decay point
    
    def test_coordinate_with_thermal(self):
        """Test coordination with thermal data."""
        # Create mock gas data
        gas_data = {
            'gas_data': {
                'methane': {
                    'concentrations': np.array([1.0, 2.0, 3.0]),
                    'timestamps': [
                        datetime.now(),
                        datetime.now() + timedelta(seconds=1),
                        datetime.now() + timedelta(seconds=2)
                    ]
                }
            },
            'timestamps': [
                datetime.now(),
                datetime.now() + timedelta(seconds=1),
                datetime.now() + timedelta(seconds=2)
            ],
            'metadata': {}
        }
        
        # Create mock thermal data
        thermal_data = {
            'frames': [
                np.ones((10, 10)) * 25.0,
                np.ones((10, 10)) * 50.0,
                np.ones((10, 10)) * 75.0
            ],
            'timestamps': [
                datetime.now(),
                datetime.now() + timedelta(seconds=1),
                datetime.now() + timedelta(seconds=2)
            ],
            'metadata': {'start_time': datetime.now().isoformat()}
        }
        
        # Coordinate data
        result = self.model.coordinate_with_thermal(gas_data, thermal_data)
        
        # Check that coordination was performed
        self.assertIn('coordinated_with_thermal', result['metadata'])
        self.assertTrue(result['metadata']['coordinated_with_thermal'])
        
        # Check that gas concentrations were adjusted based on thermal data
        self.assertNotEqual(
            result['gas_data']['methane']['concentrations'][1],
            gas_data['gas_data']['methane']['concentrations'][1]
        )
        
        # Test with mismatched timestamps
        thermal_data_mismatched = {
            'frames': [
                np.ones((10, 10)) * 25.0,
                np.ones((10, 10)) * 50.0
            ],
            'timestamps': [
                datetime.now(),
                datetime.now() + timedelta(seconds=2)
            ],
            'metadata': {'start_time': datetime.now().isoformat()}
        }
        
        # Should still work with mismatched timestamps
        result_mismatched = self.model.coordinate_with_thermal(gas_data, thermal_data_mismatched)
        self.assertIn('coordinated_with_thermal', result_mismatched['metadata'])
        
        # Test with empty thermal data
        thermal_data_empty = {
            'frames': [],
            'timestamps': [],
            'metadata': {}
        }
        
        # Should return original gas data if thermal data is empty
        result_empty = self.model.coordinate_with_thermal(gas_data, thermal_data_empty)
        self.assertEqual(result_empty, gas_data)


if __name__ == '__main__':
    unittest.main()
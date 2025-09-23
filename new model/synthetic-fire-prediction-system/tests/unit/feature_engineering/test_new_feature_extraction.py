"""
Unit tests for new feature extraction functions.

This module tests the enhanced feature extraction capabilities including:
- Multi-scale Blob Analysis
- Temporal Signature Pattern recognition
- Edge Sharpness Metrics
- Heat Distribution Skewness
- CO₂ Accumulation Rate calculation
- Baseline Drift Detection
- Gas-Temperature Correlation analysis
- Spatio-temporal Alignment
- Risk Convergence Index
- False Positive Discriminator
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from typing import Dict, Any

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

class TestNewFeatureExtraction(unittest.TestCase):
    """Test cases for new feature extraction functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic thermal data for testing
        self.thermal_data = self._generate_synthetic_thermal_data()
        
        # Generate synthetic gas data for testing
        self.gas_data = self._generate_synthetic_gas_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _generate_synthetic_thermal_data(self) -> Dict[str, Any]:
        """Generate synthetic thermal data for testing."""
        # Create realistic thermal frames with hotspots
        np.random.seed(42)
        n_frames = 10
        height, width = 60, 80
        
        frames = []
        for i in range(n_frames):
            # Base temperature with some variation
            base_temp = 25 + np.random.normal(0, 2)
            
            # Create frame with background temperature
            frame = np.full((height, width), base_temp)
            
            # Add some hotspots
            n_hotspots = np.random.randint(1, 4)
            for _ in range(n_hotspots):
                center_y = np.random.randint(10, height-10)
                center_x = np.random.randint(10, width-10)
                hotspot_temp = base_temp + np.random.uniform(10, 30)
                radius = np.random.randint(3, 8)
                
                # Create circular hotspot
                y, x = np.ogrid[:height, :width]
                mask = (y - center_y)**2 + (x - center_x)**2 <= radius**2
                frame[mask] = np.maximum(frame[mask], hotspot_temp)
            
            # Add noise
            frame += np.random.normal(0, 0.5, (height, width))
            frames.append(frame)
        
        return {
            'frames': frames,
            'timestamps': [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(n_frames)]
        }
    
    def _generate_synthetic_gas_data(self) -> Dict[str, Any]:
        """Generate synthetic gas data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate realistic CO2 data with baseline drift and accumulation patterns
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
        
        # Base CO2 level with seasonal variation
        base_co2 = 450 + 50 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))  # Daily cycle
        
        # Add baseline drift
        drift = np.cumsum(np.random.normal(0, 0.1, n_samples))
        
        # Add fire events (sudden increases)
        fire_events = np.zeros(n_samples)
        fire_indices = [20, 50, 80]  # Times when fires occur
        for idx in fire_indices:
            # Rapid increase followed by slower decrease
            duration = 15
            if idx + duration <= n_samples:
                t = np.arange(duration)
                fire_curve = 100 * np.exp(-0.2 * t) * (1 - np.exp(-0.5 * t))
                fire_events[idx:idx+duration] += fire_curve
        
        # Combine all components
        co2_values = base_co2 + drift + fire_events + np.random.normal(0, 5, n_samples)
        
        return {
            'timestamps': timestamps,
            'co2_ppm': co2_values,
            'temperature_c': 25 + np.random.normal(0, 2, n_samples)  # Ambient temperature
        }
    
    def test_multi_scale_blob_analysis(self):
        """Test multi-scale blob analysis functionality."""
        try:
            from feature_engineering.extractors.thermal.multi_scale_blob_analyzer import MultiScaleBlobAnalyzer
            
            # Create analyzer
            config = {
                'scales': [3, 5, 7, 9],
                'threshold_method': 'otsu',
                'min_blob_area': 5
            }
            analyzer = MultiScaleBlobAnalyzer(config)
            
            # Test feature extraction
            features = analyzer.extract_features(self.thermal_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('blob_features', features)
            
            # Check blob features structure
            blob_features = features['blob_features']
            self.assertIsInstance(blob_features, dict)
            expected_keys = ['blob_count', 'blob_areas', 'blob_centroids', 'scale_responses']
            for key in expected_keys:
                self.assertIn(key, blob_features)
            
        except ImportError:
            self.skipTest("MultiScaleBlobAnalyzer not available")
        except Exception as e:
            self.fail(f"Multi-scale blob analysis test failed: {e}")
    
    def test_temporal_signature_pattern_recognition(self):
        """Test temporal signature pattern recognition."""
        try:
            from feature_engineering.extractors.thermal.temporal_signature_extractor import TemporalSignatureExtractor
            
            # Create extractor
            config = {
                'window_size': 30,
                'feature_types': ['mean', 'std', 'trend', 'seasonality']
            }
            extractor = TemporalSignatureExtractor(config)
            
            # Test feature extraction
            features = extractor.extract_features(self.thermal_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('temporal_features', features)
            
            # Check temporal features structure
            temporal_features = features['temporal_features']
            self.assertIsInstance(temporal_features, dict)
            expected_keys = ['trend_coefficients', 'seasonal_components', 'autocorrelation', 'spectral_features']
            for key in expected_keys:
                self.assertIn(key, temporal_features)
            
        except ImportError:
            self.skipTest("TemporalSignatureExtractor not available")
        except Exception as e:
            self.fail(f"Temporal signature pattern recognition test failed: {e}")
    
    def test_edge_sharpness_metrics(self):
        """Test edge sharpness metrics extraction."""
        try:
            from feature_engineering.extractors.thermal.edge_sharpness_analyzer import EdgeSharpnessAnalyzer
            
            # Create analyzer
            config = {
                'edge_detection_method': 'sobel',
                'sharpness_metric': 'gradient_magnitude',
                'smoothing_kernel_size': 3
            }
            analyzer = EdgeSharpnessAnalyzer(config)
            
            # Test feature extraction
            features = analyzer.extract_features(self.thermal_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('edge_features', features)
            
            # Check edge features structure
            edge_features = features['edge_features']
            self.assertIsInstance(edge_features, dict)
            expected_keys = ['mean_sharpness', 'max_sharpness', 'sharpness_histogram', 'edge_orientation']
            for key in expected_keys:
                self.assertIn(key, edge_features)
            
        except ImportError:
            self.skipTest("EdgeSharpnessAnalyzer not available")
        except Exception as e:
            self.fail(f"Edge sharpness metrics test failed: {e}")
    
    def test_heat_distribution_skewness(self):
        """Test heat distribution skewness analysis."""
        try:
            from feature_engineering.extractors.thermal.heat_distribution_analyzer import HeatDistributionAnalyzer
            
            # Create analyzer
            config = {
                'statistical_moments': ['mean', 'std', 'skewness', 'kurtosis'],
                'percentiles': [5, 25, 50, 75, 95],
                'histogram_bins': 20
            }
            analyzer = HeatDistributionAnalyzer(config)
            
            # Test feature extraction
            features = analyzer.extract_features(self.thermal_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('distribution_features', features)
            
            # Check distribution features structure
            dist_features = features['distribution_features']
            self.assertIsInstance(dist_features, dict)
            expected_keys = ['mean_temp', 'std_temp', 'skewness', 'kurtosis', 'percentiles']
            for key in expected_keys:
                self.assertIn(key, dist_features)
            
        except ImportError:
            self.skipTest("HeatDistributionAnalyzer not available")
        except Exception as e:
            self.fail(f"Heat distribution skewness test failed: {e}")
    
    def test_co2_accumulation_rate(self):
        """Test CO2 accumulation rate calculation."""
        try:
            from feature_engineering.extractors.gas.co2_accumulation_calculator import CO2AccumulationCalculator
            
            # Create calculator
            config = {
                'window_size': 10,
                'smoothing_factor': 0.1,
                'noise_threshold': 2.0
            }
            calculator = CO2AccumulationCalculator(config)
            
            # Test feature extraction
            features = calculator.extract_features(self.gas_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('accumulation_features', features)
            
            # Check accumulation features structure
            accum_features = features['accumulation_features']
            self.assertIsInstance(accum_features, dict)
            expected_keys = ['accumulation_rate', 'accumulation_trend', 'baseline_drift', 'noise_level']
            for key in expected_keys:
                self.assertIn(key, accum_features)
            
        except ImportError:
            self.skipTest("CO2AccumulationCalculator not available")
        except Exception as e:
            self.fail(f"CO2 accumulation rate test failed: {e}")
    
    def test_baseline_drift_detection(self):
        """Test baseline drift detection for gas sensors."""
        try:
            from feature_engineering.extractors.gas.baseline_drift_detector import BaselineDriftDetector
            
            # Create detector
            config = {
                'drift_detection_method': 'moving_average',
                'window_size': 30,
                'drift_threshold': 5.0
            }
            detector = BaselineDriftDetector(config)
            
            # Test feature extraction
            features = detector.extract_features(self.gas_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('drift_features', features)
            
            # Check drift features structure
            drift_features = features['drift_features']
            self.assertIsInstance(drift_features, dict)
            expected_keys = ['baseline_level', 'drift_magnitude', 'drift_rate', 'drift_direction']
            for key in expected_keys:
                self.assertIn(key, drift_features)
            
        except ImportError:
            self.skipTest("BaselineDriftDetector not available")
        except Exception as e:
            self.fail(f"Baseline drift detection test failed: {e}")
    
    def test_gas_temperature_correlation(self):
        """Test gas-temperature correlation analysis."""
        try:
            from feature_engineering.extractors.cross_sensor.correlation_analyzer import CorrelationAnalyzer
            
            # Create analyzer with both thermal and gas data
            combined_data = {
                'thermal': self.thermal_data,
                'gas': self.gas_data
            }
            
            config = {
                'correlation_methods': ['pearson', 'spearman'],
                'window_size': 20,
                'lag_range': [-5, 5]
            }
            analyzer = CorrelationAnalyzer(config)
            
            # Test feature extraction
            features = analyzer.extract_features(combined_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('correlation_features', features)
            
            # Check correlation features structure
            corr_features = features['correlation_features']
            self.assertIsInstance(corr_features, dict)
            expected_keys = ['pearson_correlation', 'spearman_correlation', 'optimal_lag', 'cross_correlation']
            for key in expected_keys:
                self.assertIn(key, corr_features)
            
        except ImportError:
            self.skipTest("CorrelationAnalyzer not available")
        except Exception as e:
            self.fail(f"Gas-temperature correlation test failed: {e}")
    
    def test_spatio_temporal_alignment(self):
        """Test spatio-temporal alignment features."""
        try:
            from feature_engineering.extractors.alignment.spatiotemporal_aligner import SpatiotemporalAligner
            
            # Create aligner with both thermal and gas data
            combined_data = {
                'thermal': self.thermal_data,
                'gas': self.gas_data
            }
            
            config = {
                'alignment_method': 'dynamic_time_warping',
                'max_shift': 10,
                'interpolation_method': 'linear'
            }
            aligner = SpatiotemporalAligner(config)
            
            # Test feature extraction
            features = aligner.extract_features(combined_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('alignment_features', features)
            
            # Check alignment features structure
            align_features = features['alignment_features']
            self.assertIsInstance(align_features, dict)
            expected_keys = ['temporal_offset', 'alignment_quality', 'synchronization_score', 'warping_path']
            for key in expected_keys:
                self.assertIn(key, align_features)
            
        except ImportError:
            self.skipTest("SpatiotemporalAligner not available")
        except Exception as e:
            self.fail(f"Spatio-temporal alignment test failed: {e}")
    
    def test_risk_convergence_index(self):
        """Test risk convergence index calculation."""
        try:
            from feature_engineering.extractors.fusion.risk_convergence_index import RiskConvergenceIndex
            
            # Create index calculator with both thermal and gas data
            combined_data = {
                'thermal': self.thermal_data,
                'gas': self.gas_data
            }
            
            config = {
                'weighting_scheme': 'adaptive',
                'convergence_window': 15,
                'normalization_method': 'minmax'
            }
            index_calculator = RiskConvergenceIndex(config)
            
            # Test feature extraction
            features = index_calculator.extract_features(combined_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('risk_index', features)
            
            # Check risk index structure
            risk_index = features['risk_index']
            self.assertIsInstance(risk_index, dict)
            expected_keys = ['convergence_score', 'risk_level', 'trend_direction', 'confidence']
            for key in expected_keys:
                self.assertIn(key, risk_index)
            
        except ImportError:
            self.skipTest("RiskConvergenceIndex not available")
        except Exception as e:
            self.fail(f"Risk convergence index test failed: {e}")
    
    def test_false_positive_discriminator(self):
        """Test false positive discriminator features."""
        try:
            from feature_engineering.extractors.discrimination.false_positive_discriminator import FalsePositiveDiscriminator
            
            # Create discriminator with both thermal and gas data
            combined_data = {
                'thermal': self.thermal_data,
                'gas': self.gas_data
            }
            
            config = {
                'discrimination_methods': ['statistical', 'pattern_based'],
                'confidence_threshold': 0.7,
                'validation_window': 10
            }
            discriminator = FalsePositiveDiscriminator(config)
            
            # Test feature extraction
            features = discriminator.extract_features(combined_data)
            
            # Check that features are extracted
            self.assertIsInstance(features, dict)
            self.assertIn('extraction_time', features)
            self.assertIn('discrimination_features', features)
            
            # Check discrimination features structure
            disc_features = features['discrimination_features']
            self.assertIsInstance(disc_features, dict)
            expected_keys = ['false_positive_probability', 'discrimination_score', 'validation_result', 'confidence']
            for key in expected_keys:
                self.assertIn(key, disc_features)
            
        except ImportError:
            self.skipTest("FalsePositiveDiscriminator not available")
        except Exception as e:
            self.fail(f"False positive discriminator test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
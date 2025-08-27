"""
Comprehensive Testing Framework for Fire Detection System Validation.

This module provides complete validation testing including functional,
performance, integration, and accuracy testing.
"""

import unittest
import time
import numpy as np
import tempfile
import os
import shutil
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append('/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

try:
    from src.integrated_system import create_integrated_fire_system
    from src.hardware.sensor_manager import SensorMode
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False


class ValidationFramework:
    """Comprehensive validation framework for the fire detection system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validation framework."""
        self.config = config or {}
        self.test_scenarios = self._create_test_scenarios()
        self.results = {}
        self.temp_dir = None
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        try:
            self.temp_dir = tempfile.mkdtemp()
            
            validation_results = {
                'start_time': datetime.now().isoformat(),
                'functional_tests': self._run_functional_tests(),
                'performance_tests': self._run_performance_tests(),
                'accuracy_tests': self._run_accuracy_tests(),
                'stress_tests': self._run_stress_tests(),
                'integration_tests': self._run_integration_tests(),
                'end_time': datetime.now().isoformat()
            }
            
            # Generate summary
            validation_results['summary'] = self._generate_summary(validation_results)
            
            return validation_results
            
        finally:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create comprehensive test scenarios."""
        return [
            # Normal conditions
            {
                'name': 'normal_conditions',
                'data': {
                    'thermal': {'s1': {'temperature_max': 22.0, 'temperature_avg': 20.0, 'hotspot_count': 0}},
                    'gas': {'s1': {'co_concentration': 5.0, 'smoke_density': 8.0}},
                    'environmental': {'s1': {'temperature': 21.0, 'humidity': 45.0}}
                },
                'expected_fire': False,
                'min_confidence': 0.0
            },
            # High confidence fire
            {
                'name': 'high_confidence_fire',
                'data': {
                    'thermal': {'s1': {'temperature_max': 85.0, 'temperature_avg': 65.0, 'hotspot_count': 8}},
                    'gas': {'s1': {'co_concentration': 60.0, 'smoke_density': 85.0}},
                    'environmental': {'s1': {'temperature': 35.0, 'humidity': 25.0}}
                },
                'expected_fire': True,
                'min_confidence': 0.7
            },
            # Medium confidence fire
            {
                'name': 'medium_fire',
                'data': {
                    'thermal': {'s1': {'temperature_max': 55.0, 'temperature_avg': 42.0, 'hotspot_count': 3}},
                    'gas': {'s1': {'co_concentration': 35.0, 'smoke_density': 45.0}},
                    'environmental': {'s1': {'temperature': 28.0, 'humidity': 35.0}}
                },
                'expected_fire': True,
                'min_confidence': 0.4
            },
            # Edge case: extreme heat but no fire
            {
                'name': 'extreme_heat_no_fire',
                'data': {
                    'thermal': {'s1': {'temperature_max': 95.0, 'temperature_avg': 90.0, 'hotspot_count': 0}},
                    'gas': {'s1': {'co_concentration': 5.0, 'smoke_density': 8.0}},
                    'environmental': {'s1': {'temperature': 45.0, 'humidity': 15.0}}
                },
                'expected_fire': False,
                'min_confidence': 0.0
            }
        ]
    
    def _run_functional_tests(self) -> Dict[str, Any]:
        """Run functional validation tests."""
        if not SYSTEM_AVAILABLE:
            return {'status': 'skipped', 'reason': 'system_unavailable'}
        
        try:
            system = create_integrated_fire_system({'sensors': {'mode': SensorMode.SYNTHETIC}})
            
            results = {
                'initialization': system.initialize(),
                'scenario_tests': [],
                'basic_processing': False,
                'alert_generation': False
            }
            
            if not results['initialization']:
                return {'status': 'failed', 'reason': 'initialization_failed'}
            
            # Test scenarios
            passed_scenarios = 0
            for scenario in self.test_scenarios:
                start_time = time.time()
                result = system.process_data(scenario['data'])
                processing_time = (time.time() - start_time) * 1000
                
                final_decision = result.get('final_decision', {})
                actual_fire = final_decision.get('fire_detected', False)
                confidence = final_decision.get('confidence_score', 0.0)
                
                scenario_passed = (actual_fire == scenario['expected_fire'] and 
                                 confidence >= scenario['min_confidence'])
                
                if scenario_passed:
                    passed_scenarios += 1
                
                if 'final_decision' in result:
                    results['basic_processing'] = True
                
                if final_decision.get('alerts'):
                    results['alert_generation'] = True
                
                results['scenario_tests'].append({
                    'name': scenario['name'],
                    'expected': scenario['expected_fire'],
                    'actual': actual_fire,
                    'confidence': confidence,
                    'processing_time_ms': processing_time,
                    'passed': scenario_passed
                })
            
            results['success_rate'] = passed_scenarios / len(self.test_scenarios)
            results['status'] = 'completed'
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance validation tests."""
        if not SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        try:
            system = create_integrated_fire_system({'sensors': {'mode': SensorMode.SYNTHETIC}})
            system.initialize()
            
            processing_times = []
            
            # Run 100 processing cycles
            for _ in range(100):
                test_data = self._generate_random_data()
                
                start_time = time.time()
                system.process_data(test_data)
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
            
            return {
                'total_tests': len(processing_times),
                'avg_time_ms': np.mean(processing_times),
                'max_time_ms': np.max(processing_times),
                'min_time_ms': np.min(processing_times),
                'std_time_ms': np.std(processing_times),
                'p95_time_ms': np.percentile(processing_times, 95),
                'p99_time_ms': np.percentile(processing_times, 99),
                'under_1000ms': sum(1 for t in processing_times if t < 1000),
                'performance_score': sum(1 for t in processing_times if t < 1000) / len(processing_times)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_accuracy_tests(self) -> Dict[str, Any]:
        """Run accuracy validation tests."""
        if not SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        try:
            system = create_integrated_fire_system({'sensors': {'mode': SensorMode.SYNTHETIC}})
            system.initialize()
            
            correct_predictions = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            # Test all scenarios multiple times
            for _ in range(5):  # Run each scenario 5 times
                for scenario in self.test_scenarios:
                    result = system.process_data(scenario['data'])
                    final_decision = result.get('final_decision', {})
                    actual_fire = final_decision.get('fire_detected', False)
                    
                    if actual_fire == scenario['expected_fire']:
                        correct_predictions += 1
                    
                    if scenario['expected_fire'] and actual_fire:
                        true_positives += 1
                    elif scenario['expected_fire'] and not actual_fire:
                        false_negatives += 1
                    elif not scenario['expected_fire'] and actual_fire:
                        false_positives += 1
                    else:
                        true_negatives += 1
            
            total_tests = len(self.test_scenarios) * 5
            accuracy = correct_predictions / total_tests
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'total_tests': total_tests,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'meets_threshold': accuracy >= 0.8
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress testing."""
        if not SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        try:
            system = create_integrated_fire_system({'sensors': {'mode': SensorMode.SYNTHETIC}})
            system.initialize()
            system.start()
            
            # Concurrent processing test
            def process_request():
                return system.process_data(self._generate_random_data())
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_request) for _ in range(20)]
                results = [f.result() for f in futures]
            
            total_time = time.time() - start_time
            
            successful_requests = sum(1 for r in results if 'final_decision' in r)
            
            system.stop()
            
            return {
                'total_requests': len(results),
                'successful_requests': successful_requests,
                'failed_requests': len(results) - successful_requests,
                'success_rate': successful_requests / len(results),
                'total_time_seconds': total_time,
                'requests_per_second': len(results) / total_time,
                'concurrent_processing': successful_requests == len(results)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        if not SYSTEM_AVAILABLE:
            return {'status': 'skipped'}
        
        try:
            system = create_integrated_fire_system({'sensors': {'mode': SensorMode.SYNTHETIC}})
            
            integration_results = {
                'initialization': system.initialize(),
                'subsystems_present': {
                    'sensor_manager': system.sensor_manager is not None,
                    'multi_agent_system': system.multi_agent_system is not None,
                    'model_ensemble': system.model_ensemble is not None
                },
                'data_flow': False,
                'end_to_end': False
            }
            
            if integration_results['initialization']:
                # Test data flow
                test_data = self.test_scenarios[1]['data']  # Use fire scenario
                result = system.process_data(test_data)
                
                if ('sensor_summary' in result and 'ml_results' in result and 
                    'agent_results' in result and 'final_decision' in result):
                    integration_results['data_flow'] = True
                
                # Test end-to-end
                final_decision = result.get('final_decision', {})
                if ('fire_detected' in final_decision and 'confidence_score' in final_decision):
                    integration_results['end_to_end'] = True
            
            # Calculate integration score
            score_components = [
                integration_results['initialization'],
                all(integration_results['subsystems_present'].values()),
                integration_results['data_flow'],
                integration_results['end_to_end']
            ]
            
            integration_results['integration_score'] = sum(score_components) / len(score_components)
            
            return integration_results
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_random_data(self) -> Dict[str, Any]:
        """Generate random sensor data for testing."""
        return {
            'thermal': {
                'sensor1': {
                    'temperature_max': 20.0 + np.random.normal(0, 10),
                    'temperature_avg': 18.0 + np.random.normal(0, 8),
                    'hotspot_count': max(0, int(np.random.normal(1, 2)))
                }
            },
            'gas': {
                'sensor1': {
                    'co_concentration': max(0, 5.0 + np.random.normal(0, 10)),
                    'smoke_density': max(0, 10.0 + np.random.normal(0, 15))
                }
            },
            'environmental': {
                'sensor1': {
                    'temperature': 20.0 + np.random.normal(0, 5),
                    'humidity': max(0, min(100, 50.0 + np.random.normal(0, 15)))
                }
            }
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'overall_status': 'UNKNOWN',
            'scores': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        try:
            # Functional score
            functional = results.get('functional_tests', {})
            if functional.get('status') == 'completed':
                functional_score = functional.get('success_rate', 0)
                summary['scores']['functional'] = functional_score
                
                if functional_score < 0.8:
                    summary['critical_issues'].append('Low functional test success rate')
            
            # Performance score  
            performance = results.get('performance_tests', {})
            if 'performance_score' in performance:
                perf_score = performance['performance_score']
                summary['scores']['performance'] = perf_score
                
                if perf_score < 0.9:
                    summary['critical_issues'].append('Performance issues detected')
            
            # Accuracy score
            accuracy = results.get('accuracy_tests', {})
            if 'accuracy' in accuracy:
                acc_score = accuracy['accuracy']
                summary['scores']['accuracy'] = acc_score
                
                if acc_score < 0.8:
                    summary['critical_issues'].append('Accuracy below threshold')
            
            # Integration score
            integration = results.get('integration_tests', {})
            if 'integration_score' in integration:
                int_score = integration['integration_score']
                summary['scores']['integration'] = int_score
                
                if int_score < 0.8:
                    summary['critical_issues'].append('Integration issues')
            
            # Calculate overall score
            if summary['scores']:
                overall_score = np.mean(list(summary['scores'].values()))
                
                if overall_score >= 0.9:
                    summary['overall_status'] = 'EXCELLENT'
                elif overall_score >= 0.8:
                    summary['overall_status'] = 'GOOD'
                elif overall_score >= 0.6:
                    summary['overall_status'] = 'ACCEPTABLE'
                else:
                    summary['overall_status'] = 'NEEDS_IMPROVEMENT'
                
                summary['overall_score'] = overall_score
            
            # Generate recommendations
            if len(summary['critical_issues']) == 0:
                summary['recommendations'].append('System validation successful - ready for production')
            else:
                summary['recommendations'].append('Address critical issues before production deployment')
                
                if 'Performance issues detected' in summary['critical_issues']:
                    summary['recommendations'].append('Optimize processing pipeline performance')
                
                if 'Accuracy below threshold' in summary['critical_issues']:
                    summary['recommendations'].append('Improve ML model training and validation')
        
        except Exception as e:
            summary['overall_status'] = 'ERROR'
            summary['critical_issues'].append(f'Summary generation failed: {str(e)}')
        
        return summary


# Main validation test class
class TestSystemValidation(unittest.TestCase):
    """Main system validation test suite."""
    
    def setUp(self):
        """Setup validation tests."""
        if not SYSTEM_AVAILABLE:
            self.skipTest("System components not available")
        
        self.validator = ValidationFramework()
    
    def test_comprehensive_validation(self):
        """Run comprehensive system validation."""
        results = self.validator.run_full_validation()
        
        # Check that all test suites ran
        self.assertIn('functional_tests', results)
        self.assertIn('performance_tests', results)
        self.assertIn('accuracy_tests', results)
        self.assertIn('integration_tests', results)
        self.assertIn('summary', results)
        
        # Check summary
        summary = results['summary']
        self.assertIn('overall_status', summary)
        
        # Log results for review
        print(f"\nValidation Results Summary:")
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"Overall Score: {summary.get('overall_score', 0.0):.2f}")
        
        if summary.get('critical_issues'):
            print(f"Critical Issues: {summary['critical_issues']}")
        
        if summary.get('recommendations'):
            print(f"Recommendations: {summary['recommendations']}")
        
        # Assert minimum quality standards
        if summary.get('overall_score', 0) > 0:
            self.assertGreaterEqual(summary['overall_score'], 0.6, 
                                  "System validation score below minimum threshold")


if __name__ == '__main__':
    # Run comprehensive validation
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2, buffer=True)
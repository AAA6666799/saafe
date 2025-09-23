#!/usr/bin/env python3
"""
Validation Test Runner for FLIR+SCD41 Fire Detection System.

This script runs all validation tests and generates comprehensive performance reports.
"""

import sys
import os
import unittest
import argparse
import logging
from datetime import datetime
import json
import subprocess

# Add the project root to the path
sys.path.insert(0, '/Volumes/Ajay/saafe copy 3/new model/synthetic-fire-prediction-system')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_test_suite(suite_name: str, test_module: str) -> dict:
    """
    Run a test suite and return results.
    
    Args:
        suite_name: Name of the test suite
        test_module: Python module containing tests
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running {suite_name} tests...")
    
    try:
        # Run tests using subprocess to capture output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_module, 
            '-v', '--tb=short', '--no-header'
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Parse results
        output_lines = result.stdout.strip().split('\n')
        error_lines = result.stderr.strip().split('\n')
        
        # Extract test results
        test_results = []
        passed = 0
        failed = 0
        errors = 0
        
        for line in output_lines:
            if 'PASSED' in line:
                passed += 1
                test_results.append({'status': 'PASSED', 'test': line.strip()})
            elif 'FAILED' in line:
                failed += 1
                test_results.append({'status': 'FAILED', 'test': line.strip()})
            elif 'ERROR' in line:
                errors += 1
                test_results.append({'status': 'ERROR', 'test': line.strip()})
        
        return {
            'suite_name': suite_name,
            'status': 'success' if result.returncode == 0 else 'failed',
            'return_code': result.returncode,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'total_tests': passed + failed + errors,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'test_results': test_results
        }
        
    except Exception as e:
        logger.error(f"Error running {suite_name} tests: {e}")
        return {
            'suite_name': suite_name,
            'status': 'error',
            'error': str(e),
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'total_tests': 0,
            'test_results': []
        }

def run_all_validation_tests() -> dict:
    """
    Run all validation test suites.
    
    Returns:
        Dictionary with comprehensive test results
    """
    logger.info("Starting comprehensive validation test suite...")
    
    # Define test suites
    test_suites = [
        ('Ensemble Model Integration', '../tests/integration/test_ensemble_models.py'),
        ('Synthetic Data Generation', '../tests/validation/test_synthetic_data_generation.py'),
        ('Model Updates CI', '../tests/ci/test_model_updates.py'),
        ('False Positive Scenarios', '../tests/validation/test_false_positive_scenarios.py'),
        ('Early Detection Capability', '../tests/validation/test_early_detection_capability.py')
    ]
    
    # Run all test suites
    results = []
    overall_passed = 0
    overall_failed = 0
    overall_errors = 0
    
    for suite_name, test_module in test_suites:
        suite_result = run_test_suite(suite_name, test_module)
        results.append(suite_result)
        
        overall_passed += suite_result.get('passed', 0)
        overall_failed += suite_result.get('failed', 0)
        overall_errors += suite_result.get('errors', 0)
        
        if suite_result['status'] == 'success':
            logger.info(f"✓ {suite_name} tests completed successfully")
        else:
            logger.error(f"✗ {suite_name} tests failed")
    
    # Generate summary
    total_tests = overall_passed + overall_failed + overall_errors
    success_rate = (overall_passed / total_tests * 100) if total_tests > 0 else 0
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_suites': len(test_suites),
        'successful_suites': sum(1 for r in results if r['status'] == 'success'),
        'failed_suites': sum(1 for r in results if r['status'] != 'success'),
        'total_tests': total_tests,
        'passed': overall_passed,
        'failed': overall_failed,
        'errors': overall_errors,
        'success_rate': success_rate,
        'results': results
    }
    
    return summary

def generate_validation_report(summary: dict, output_file: str = None) -> str:
    """
    Generate comprehensive validation report.
    
    Args:
        summary: Test summary dictionary
        output_file: Optional output file path
        
    Returns:
        Report content as string
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("FLIR+SCD41 FIRE DETECTION SYSTEM VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {summary['timestamp']}")
    report_lines.append("")
    
    # Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total Test Suites: {summary['total_suites']}")
    report_lines.append(f"Successful Suites: {summary['successful_suites']}")
    report_lines.append(f"Failed Suites: {summary['failed_suites']}")
    report_lines.append(f"Total Tests: {summary['total_tests']}")
    report_lines.append(f"Passed: {summary['passed']}")
    report_lines.append(f"Failed: {summary['failed']}")
    report_lines.append(f"Errors: {summary['errors']}")
    report_lines.append(f"Success Rate: {summary['success_rate']:.1f}%")
    report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 30)
    
    for suite_result in summary['results']:
        status_symbol = "✓" if suite_result['status'] == 'success' else "✗"
        report_lines.append(f"{status_symbol} {suite_result['suite_name']}")
        report_lines.append(f"   Status: {suite_result['status'].upper()}")
        report_lines.append(f"   Tests: {suite_result.get('total_tests', 0)} "
                          f"(Passed: {suite_result.get('passed', 0)}, "
                          f"Failed: {suite_result.get('failed', 0)}, "
                          f"Errors: {suite_result.get('errors', 0)})")
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("VALIDATION COMPLETE")
    report_lines.append("=" * 80)
    
    report_content = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Validation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report to {output_file}: {e}")
    
    return report_content

def main():
    """Main function to run validation tests."""
    parser = argparse.ArgumentParser(description='Run FLIR+SCD41 Fire Detection System Validation Tests')
    parser.add_argument('--report', type=str, help='Generate report and save to file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run all validation tests
        summary = run_all_validation_tests()
        
        # Generate and display report
        report_content = generate_validation_report(summary, args.report)
        print(report_content)
        
        # Log summary
        logger.info(f"Validation completed: {summary['passed']}/{summary['total_tests']} tests passed "
                   f"({summary['success_rate']:.1f}% success rate)")
        
        # Return appropriate exit code
        return 0 if summary['success_rate'] >= 90 else 1  # Exit with error if <90% success rate
        
    except Exception as e:
        logger.error(f"Validation test execution failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
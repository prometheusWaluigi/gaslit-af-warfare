#!/usr/bin/env python3
"""
GASLIT-AF WARSTACK Test Runner

This script provides a convenient way to run tests for the GASLIT-AF WARSTACK project.
It offers various options for running specific test modules, generating reports,
and controlling test execution.
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path

# Define test modules
TEST_MODULES = {
    'biological': 'tests/test_biological_modeling.py',
    'genetic': 'tests/test_genetic_risk.py',
    'institutional': 'tests/test_institutional_feedback.py',
    'legal': 'tests/test_legal_policy.py',
    'frontend': 'tests/test_frontend.py',
    'all': 'tests'
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GASLIT-AF WARSTACK Test Runner"
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=list(TEST_MODULES.keys()),
        default='all',
        help="Test module to run (default: all)"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help="Generate coverage report"
    )
    
    parser.add_argument(
        '--html-report',
        action='store_true',
        help="Generate HTML test report"
    )
    
    parser.add_argument(
        '--json-report',
        action='store_true',
        help="Generate JSON test report"
    )
    
    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help="Skip slow tests"
    )
    
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help="Skip tests that require GPU"
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help="Stop on first failure"
    )
    
    parser.add_argument(
        '--output-dir',
        default='test_results',
        help="Directory for test reports (default: test_results)"
    )
    
    return parser.parse_args()


def run_tests(args):
    """Run the tests with the specified options."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build pytest command
    cmd = ['pytest']
    
    # Add test module
    cmd.append(TEST_MODULES[args.module])
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=src', f'--cov-report=html:{args.output_dir}/coverage'])
    
    # Add HTML report
    if args.html_report:
        try:
            import pytest_html
            cmd.extend([f'--html={args.output_dir}/report.html', '--self-contained-html'])
        except ImportError:
            print("Warning: pytest-html not installed. Skipping HTML report generation.")
    
    # Add JSON report
    if args.json_report:
        try:
            import pytest_json_report
            cmd.extend([f'--json-report', f'--json-report-file={args.output_dir}/report.json'])
        except ImportError:
            print("Warning: pytest-json-report not installed. Skipping JSON report generation.")
    
    # Skip slow tests
    if args.skip_slow:
        cmd.append('-m "not slow"')
    
    # Skip GPU tests
    if args.skip_gpu:
        cmd.append('-m "not gpu"')
    
    # Stop on first failure
    if args.failfast:
        cmd.append('--exitfirst')
    
    # Run the tests
    print(f"Running tests: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\nTest run completed in {elapsed_time:.2f} seconds")
    print(f"Exit code: {result.returncode}")
    
    if args.json_report:
        json_report_path = os.path.join(args.output_dir, 'report.json')
        if os.path.exists(json_report_path):
            try:
                with open(json_report_path, 'r') as f:
                    report = json.load(f)
                
                summary = report.get('summary', {})
                print(f"\nTest Summary:")
                print(f"  Total: {summary.get('total', 0)}")
                print(f"  Passed: {summary.get('passed', 0)}")
                print(f"  Failed: {summary.get('failed', 0)}")
                print(f"  Skipped: {summary.get('skipped', 0)}")
                print(f"  Error: {summary.get('error', 0)}")
                
                if args.coverage:
                    coverage = report.get('coverage', {})
                    if coverage:
                        print(f"\nCoverage Summary:")
                        print(f"  Overall: {coverage.get('total_percent', 0):.2f}%")
            except Exception as e:
                print(f"Error parsing JSON report: {e}")
    
    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())

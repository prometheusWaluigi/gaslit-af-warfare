#!/usr/bin/env python3
"""
Test runner script for GASLIT-AF WARSTACK.
This script runs the test suite and generates a report.
"""

import os
import sys
import argparse
import subprocess
import datetime
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run GASLIT-AF WARSTACK tests')
    parser.add_argument('--module', '-m', choices=['all', 'biological', 'genetic', 'institutional', 'legal', 'frontend'],
                        default='all', help='Module to test (default: all)')
    parser.add_argument('--output', '-o', default='test_report.json',
                        help='Output file for test report (default: test_report.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report (requires pytest-html)')
    return parser.parse_args()

def run_tests(module, verbose=False, html=False):
    """Run the tests for the specified module."""
    print(f"Running tests for module: {module}")
    
    # Base command
    cmd = ['pytest']
    
    # Add verbosity
    if verbose:
        cmd.append('-v')
    
    # Add module filter
    if module != 'all':
        cmd.append(f'-m {module}')
    
    # Add HTML report
    if html:
        cmd.append('--html=test_report.html')
    
    # Add JSON report
    cmd.append('--json-report')
    cmd.append('--json-report-file=test_report.json')
    
    # Run the command
    try:
        result = subprocess.run(' '.join(cmd), shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        print(e.stdout)
        print(e.stderr)
        return False, e.stdout + '\n' + e.stderr

def generate_report(success, output, module, output_file):
    """Generate a report from the test results."""
    # Load the JSON report if it exists
    json_report = {}
    if os.path.exists('test_report.json'):
        with open('test_report.json', 'r') as f:
            json_report = json.load(f)
    
    # Create the report
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'module': module,
        'success': success,
        'summary': {
            'total': json_report.get('summary', {}).get('total', 0),
            'passed': json_report.get('summary', {}).get('passed', 0),
            'failed': json_report.get('summary', {}).get('failed', 0),
            'skipped': json_report.get('summary', {}).get('skipped', 0),
            'error': json_report.get('summary', {}).get('error', 0),
            'xfailed': json_report.get('summary', {}).get('xfailed', 0),
            'xpassed': json_report.get('summary', {}).get('xpassed', 0),
        },
        'tests': json_report.get('tests', []),
        'output': output
    }
    
    # Write the report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report written to {output_file}")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total: {report['summary']['total']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Skipped: {report['summary']['skipped']}")
    print(f"Error: {report['summary']['error']}")
    print(f"XFailed: {report['summary']['xfailed']}")
    print(f"XPassed: {report['summary']['xpassed']}")
    
    return report

def main():
    """Main function."""
    args = parse_args()
    
    # Check if pytest-json-report is installed
    try:
        import pytest_json_report
    except ImportError:
        print("Warning: pytest-json-report is not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest-json-report'], check=True)
    
    # Check if pytest-html is installed if --html is specified
    if args.html:
        try:
            import pytest_html
        except ImportError:
            print("Warning: pytest-html is not installed. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest-html'], check=True)
    
    # Run the tests
    success, output = run_tests(args.module, args.verbose, args.html)
    
    # Generate the report
    report = generate_report(success, output, args.module, args.output)
    
    # Return success/failure
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

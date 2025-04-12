#!/usr/bin/env python3
"""
Script to analyze VCF data using the GeneticRiskScanner.
"""

import os
import sys
import json
from src.genetic_risk.genetic_scanner import GeneticRiskScanner, HAS_ALLEL, HAS_BIOPYTHON

def main():
    print("==== GASLIT-AF Genetic Risk Scanner ====")
    print(f"scikit-allel installed: {HAS_ALLEL}")
    print(f"biopython installed: {HAS_BIOPYTHON}")
    
    if not HAS_ALLEL:
        print("ERROR: scikit-allel is required for VCF analysis but not installed.")
        print("Try: pip install scikit-allel")
        return 1
    
    # Check if VCF files exist
    vcf_dir = "data/vcf"
    vcf_files = [f for f in os.listdir(vcf_dir) if f.endswith('.vcf.gz')]
    
    if not vcf_files:
        print(f"No VCF files found in {vcf_dir}")
        return 1
    
    print(f"Found {len(vcf_files)} VCF files:")
    for i, file in enumerate(vcf_files):
        print(f"{i+1}. {file}")
    
    # Process the CNV file first (it's smaller)
    cnv_file = [f for f in vcf_files if 'cnv' in f]
    if cnv_file:
        selected_file = os.path.join(vcf_dir, cnv_file[0])
        print(f"\nProcessing CNV file: {selected_file}")
    else:
        # Use the first file if no CNV file is found
        selected_file = os.path.join(vcf_dir, vcf_files[0])
        print(f"\nProcessing file: {selected_file}")
    
    # Create output directory
    output_dir = "results/genetic_risk"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the scanner
    scanner = GeneticRiskScanner({
        'output_dir': output_dir,
        'risk_threshold': 0.7,
        'collapse_threshold': 0.85
    })
    
    try:
        # Load VCF file 
        print("Loading VCF file (this may take a moment)...")
        vcf_data = scanner.load_vcf(selected_file)
        
        if 'error' in vcf_data:
            print(f"Error loading VCF: {vcf_data['error']}")
            return 1
        
        print(f"VCF loaded successfully. Found data for {len(vcf_data['samples'])} samples")
        
        # Analyze risk genes
        print("Analyzing genetic risk factors...")
        risk_results = scanner.analyze_risk_genes(vcf_data)
        
        if 'error' in risk_results:
            print(f"Error analyzing risk genes: {risk_results['error']}")
            return 1
        
        # Generate heatmap data
        heatmap_data = scanner.generate_heatmap_data(risk_results)
        
        # Export risk profile
        profile_path = os.path.join(output_dir, "risk_profile.json")
        scanner.export_risk_profile(risk_results, profile_path)
        
        # Print results summary
        print("\n==== Analysis Results ====")
        print(f"Risk category: {risk_results['risk_category']}")
        print(f"Fragility gamma: {risk_results['fragility_gamma']:.4f}")
        print(f"Allostatic lambda: {risk_results['allostatic_lambda']:.4f}")
        print(f"Allostatic omega: {risk_results['allostatic_omega']:.4f}")
        print("\nRisk scores by gene:")
        for gene, score in risk_results['risk_scores'].items():
            print(f"  {gene}: {score:.4f}")
        
        print(f"\nFull results saved to: {profile_path}")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

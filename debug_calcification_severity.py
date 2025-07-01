#!/usr/bin/env python3
"""
🔍 Standalone Calcification Severity Analysis Debug Script

This script loads pre-saved variables from the notebook session and runs
the calcification by severity analysis for debugging purposes.

Usage:
    python debug_calcification_severity.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_saved_variables(variables_dir="multi_epoch_study_analysis_results/saved_variables"):
    """Load the saved variables from pickle files."""
    print("📂 Loading saved variables...")
    
    if not os.path.exists(variables_dir):
        print(f"❌ Variables directory not found: {variables_dir}")
        print("   Run the notebook first to save the variables!")
        return None
    
    variables = {}
    required_vars = ['all_epoch_metrics', 'epoch_nums']
    
    for var_name in ['all_epoch_metrics', 'epoch_nums', 'stenosis_metrics', 
                     'calcification_metrics', 'ifr_metrics', 'OUTPUT_DIR']:
        var_path = os.path.join(variables_dir, f"{var_name}.pkl")
        
        if os.path.exists(var_path):
            try:
                with open(var_path, 'rb') as f:
                    variables[var_name] = pickle.load(f)
                print(f"   ✅ Loaded {var_name}")
                
                # Print info about loaded data
                data = variables[var_name]
                if isinstance(data, dict):
                    print(f"      📊 Dictionary with {len(data)} keys")
                elif isinstance(data, list):
                    print(f"      📊 List with {len(data)} items")
                else:
                    print(f"      📊 Type: {type(data)}")
                    
            except Exception as e:
                print(f"   ❌ Error loading {var_name}: {e}")
        else:
            print(f"   ⚠️  File not found: {var_path}")
    
    # Check if we have required variables
    missing_vars = [var for var in required_vars if var not in variables]
    if missing_vars:
        print(f"❌ Missing required variables: {missing_vars}")
        return None
    
    print(f"✅ Successfully loaded {len(variables)} variables")
    return variables

def run_severity_analysis(variables):
    """Run the calcification severity analysis using loaded variables."""
    print("\n" + "="*80)
    print("🔍 STANDALONE CALCIFICATION SEVERITY ANALYSIS")
    print("="*80)
    
    # Extract variables
    all_epoch_metrics = variables['all_epoch_metrics']
    epoch_nums = variables['epoch_nums']
    
    # Import the analysis function
    try:
        from utils.plot_metrics import debug_calcification_by_severity
        print("✅ Successfully imported debug_calcification_by_severity")
    except ImportError as e:
        print(f"❌ Failed to import analysis function: {e}")
        return None
    
    # Set matplotlib backend for saving plots
    plt.switch_backend('Agg')  # Use non-interactive backend
    
    # Run the analysis
    print(f"\n🚀 Running severity analysis on {len(epoch_nums)} epochs...")
    print(f"   📊 Epoch range: {min(epoch_nums)} - {max(epoch_nums)}")
    
    try:
        severity_results = debug_calcification_by_severity(all_epoch_metrics, epoch_nums)
        
        # Save results
        if severity_results:
            output_dir = variables.get('OUTPUT_DIR', 'debug_output')
            os.makedirs(output_dir, exist_ok=True)
            
            results_file = os.path.join(output_dir, "debug_severity_analysis_results.pkl")
            with open(results_file, 'wb') as f:
                pickle.dump(severity_results, f)
            print(f"\n💾 Debug results saved to: {results_file}")
        
        return severity_results
        
    except Exception as e:
        print(f"❌ Error during severity analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary_stats(severity_results, variables):
    """Print summary statistics about the analysis results."""
    if not severity_results:
        return
    
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    
    # Basic info
    total_matched = severity_results.get('total_matched', 0)
    all_metrics = severity_results.get('all_calcif_metrics', set())
    severity_data = severity_results.get('severity_data', {})
    
    print(f"🎯 Total calcification metrics: {len(all_metrics)}")
    print(f"🎯 Metrics matched to severity: {total_matched}")
    print(f"🎯 Match rate: {total_matched/len(all_metrics)*100:.1f}%" if all_metrics else "N/A")
    
    # Severity breakdown
    print(f"\n📈 Severity Level Breakdown:")
    for severity, metrics in severity_data.items():
        if metrics:
            print(f"   {severity.upper()}: {len(metrics)} metrics")
            for metric in metrics:
                print(f"      • {metric}")
    
    # Show sample values from latest epoch
    all_epoch_metrics = variables['all_epoch_metrics']
    epoch_nums = variables['epoch_nums']
    
    if epoch_nums and all_epoch_metrics:
        latest_epoch = max(epoch_nums)
        latest_key = f"epoch_{latest_epoch}"
        
        if latest_key in all_epoch_metrics:
            calcif_metrics = all_epoch_metrics[latest_key].get('calcification', {}).get('accuracy', {})
            
            print(f"\n📊 Sample Values (Epoch {latest_epoch}):")
            for severity, metrics in severity_data.items():
                if metrics:
                    values = []
                    for metric in metrics:
                        if metric in calcif_metrics:
                            val = calcif_metrics[metric]
                            if not np.isnan(val):
                                values.append(val)
                    
                    if values:
                        print(f"   {severity.upper()}: Mean={np.mean(values):.3f}, "
                              f"Std={np.std(values):.3f}, Range=[{np.min(values):.3f}, {np.max(values):.3f}]")

def main():
    """Main function to run the standalone analysis."""
    print("🔍 Calcification Severity Analysis - Standalone Debug Script")
    print("="*70)
    
    # Load variables
    variables = load_saved_variables()
    if not variables:
        print("\n❌ Cannot proceed without saved variables!")
        print("💡 Run the notebook first to save the required variables.")
        sys.exit(1)
    
    # Run analysis
    severity_results = run_severity_analysis(variables)
    
    # Analyze and display results
    if severity_results:
        print_summary_stats(severity_results, variables)
        
        # Import and use functions from plot_metrics
        try:
            from utils.plot_metrics import analyze_trends_over_epochs, save_plots_to_files
            
            # Call the trends analysis
            analyze_trends_over_epochs(
                severity_results, 
                variables['all_epoch_metrics'], 
                variables['epoch_nums']
            )
            
            # Save plots
            save_plots_to_files(variables.get('OUTPUT_DIR', 'debug_output'))
            
        except ImportError as e:
            print(f"⚠️ Could not import plot_metrics functions: {e}")
            print("   Trends analysis skipped")
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the plots directory for saved visualizations!")
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
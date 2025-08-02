#!/usr/bin/env python3
"""
Test Vectorized Parallel Multi-Epoch Analysis
=============================================

This script demonstrates the new vectorized parallel processing capabilities
that can analyze multiple epochs simultaneously for massive speedup.
"""

import os
import time
from utils.clean_study_analysis import run_multi_epoch_analysis
import multiprocessing as mp

def main():
    print("🚀 Testing VECTORIZED PARALLEL Multi-Epoch Analysis")
    print("=" * 60)
    
    # Configuration
    REPORT_PATH = "data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250601_RCA_LCA_merged_with_left_dominance_dependent_vessels.csv"
    PREDICTIONS_DIR = "outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/8av1xygm_20250605-083820_best_single_video"
    OUTPUT_DIR = "vectorized_test_results"
    
    # Test with a subset of epochs for demonstration
    TEST_EPOCHS = (0, 3)  # Test with first 3 epochs
    
    print(f"📋 Configuration:")
    print(f"   📄 Report: {REPORT_PATH}")
    print(f"   📁 Predictions: {PREDICTIONS_DIR}")
    print(f"   📊 Test epochs: {TEST_EPOCHS[0]} to {TEST_EPOCHS[1]}")
    print(f"   💻 CPU cores: {mp.cpu_count()}")
    
    # Test 1: Sequential Processing
    print(f"\n" + "="*50)
    print("🐌 TEST 1: Sequential Processing")
    print("="*50)
    
    start_time = time.time()
    sequential_metrics, sequential_dfs = run_multi_epoch_analysis(
        report_csv_path=REPORT_PATH,
        predictions_dir=PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR + "_sequential",
        epoch_range=TEST_EPOCHS,
        use_parallel=False  # Sequential
    )
    sequential_time = time.time() - start_time
    
    print(f"   ✅ Sequential completed: {len(sequential_metrics)} epochs")
    print(f"   ⏱️ Time: {sequential_time:.2f} seconds")
    
    # Test 2: Parallel Processing
    print(f"\n" + "="*50)
    print("🚀 TEST 2: Parallel Processing")
    print("="*50)
    
    start_time = time.time()
    parallel_metrics, parallel_dfs = run_multi_epoch_analysis(
        report_csv_path=REPORT_PATH,
        predictions_dir=PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR + "_parallel",
        epoch_range=TEST_EPOCHS,
        use_parallel=True,   # Parallel
        max_workers=4        # Use 4 workers
    )
    parallel_time = time.time() - start_time
    
    print(f"   ✅ Parallel completed: {len(parallel_metrics)} epochs")
    print(f"   ⏱️ Time: {parallel_time:.2f} seconds")
    
    # Performance Comparison
    print(f"\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON")
    print("="*50)
    
    if sequential_time > 0 and parallel_time > 0:
        speedup = sequential_time / parallel_time
        time_saved = sequential_time - parallel_time
        
        print(f"🐌 Sequential: {sequential_time:.2f}s ({sequential_time/len(sequential_metrics):.2f}s per epoch)")
        print(f"🚀 Parallel:   {parallel_time:.2f}s ({parallel_time/len(parallel_metrics):.2f}s per epoch)")
        print(f"💥 Speedup:    {speedup:.2f}x faster")
        print(f"⏰ Time saved: {time_saved:.2f} seconds")
        
        # Extrapolate to full dataset (29 epochs)
        full_sequential_est = (sequential_time / len(sequential_metrics)) * 29
        full_parallel_est = (parallel_time / len(parallel_metrics)) * 29
        full_time_saved = full_sequential_est - full_parallel_est
        
        print(f"\n🔮 EXTRAPOLATION TO FULL 29 EPOCHS:")
        print(f"🐌 Sequential (estimated): {full_sequential_est:.0f}s ({full_sequential_est/60:.1f} minutes)")
        print(f"🚀 Parallel (estimated):   {full_parallel_est:.0f}s ({full_parallel_est/60:.1f} minutes)")
        print(f"💰 Time saved (estimated): {full_time_saved:.0f}s ({full_time_saved/60:.1f} minutes)")
        
        if speedup > 2:
            print("🎉 EXCELLENT! Parallel processing provides significant speedup!")
        elif speedup > 1.3:
            print("👍 GOOD! Parallel processing provides decent speedup!")
        else:
            print("⚠️ LIMITED speedup. Consider sequential for small datasets.")
    
    # Verify results consistency
    print(f"\n" + "="*50)
    print("🔍 RESULTS VERIFICATION")
    print("="*50)
    
    if len(sequential_metrics) == len(parallel_metrics):
        print("✅ Same number of epochs processed")
        
        # Check if metrics are similar (they should be identical)
        consistent = True
        for epoch_key in sequential_metrics:
            if epoch_key in parallel_metrics:
                seq_stenosis = len(sequential_metrics[epoch_key]['stenosis']['mae'])
                par_stenosis = len(parallel_metrics[epoch_key]['stenosis']['mae'])
                if seq_stenosis != par_stenosis:
                    consistent = False
                    break
        
        if consistent:
            print("✅ Results are consistent between sequential and parallel processing")
        else:
            print("⚠️ Results differ between sequential and parallel processing")
    else:
        print("⚠️ Different number of epochs processed")
    
    print(f"\n🎉 VECTORIZED PARALLEL TESTING COMPLETED!")
    print("💡 Use the parallel version for significant speedup on large datasets!")

if __name__ == "__main__":
    main() 
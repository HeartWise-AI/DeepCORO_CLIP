#!/bin/bash

# 🚀 Run Vectorized Parallel Multi-Epoch Analysis
# ==============================================

echo "🚀 Starting Vectorized Parallel Multi-Epoch Analysis..."
echo "=================================================="

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if data files exist
REPORT_FILE="data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250601_RCA_LCA_merged_with_left_dominance_dependent_vessels.csv"
PREDICTIONS_DIR="outputs/DeepCORO_clip/dev_deep_coro_clip_single_video/8av1xygm_20250605-083820_best_single_video"

if [ ! -f "$REPORT_FILE" ]; then
    echo "❌ Report file not found: $REPORT_FILE"
    exit 1
fi

if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "❌ Predictions directory not found: $PREDICTIONS_DIR"
    exit 1
fi

echo "✅ Data files found"

# Run the vectorized analysis test
echo "🧪 Running vectorized analysis test..."
python test_vectorized_analysis.py

# Optional: Run the full analysis (uncomment to enable)
# echo "🚀 Running full vectorized analysis..."
# python -c "
# from utils.clean_study_analysis import run_multi_epoch_analysis_parallel
# import time
# 
# start_time = time.time()
# metrics, dfs = run_multi_epoch_analysis_parallel(
#     report_csv_path='$REPORT_FILE',
#     predictions_dir='$PREDICTIONS_DIR',
#     output_dir='vectorized_full_results',
#     epoch_range=(1, 29),
#     max_workers=6,
#     batch_size=10
# )
# 
# elapsed = time.time() - start_time
# print(f'🎉 Full analysis completed in {elapsed:.2f} seconds!')
# print(f'📊 Processed {len(metrics)} epochs')
# "

echo "✅ Vectorized parallel analysis completed!"
echo "💡 Check the results in the output directories" 
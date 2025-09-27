#!/usr/bin/env python3
"""
Test script for multi-prompt generation with limited rows
"""

import pandas as pd
import sys
sys.path.append('dataset_creation')

from generate_dataset_multiprompt import generate_multiprompt_dataset

# Load just 100 rows from the parquet file
print("Loading first 100 rows from parquet file...")
df = pd.read_parquet('/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL.parquet',
                      columns=None).head(100)

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.shape[1]}")

# Check what stenosis columns are available
stenosis_cols = [col for col in df.columns if col.endswith('_stenosis')]
print(f"\nFound {len(stenosis_cols)} stenosis columns:")
print(stenosis_cols[:5])  # Show first 5

# Generate multi-prompts
print("\nGenerating multi-prompt dataset...")
prompt_df = generate_multiprompt_dataset(df)

# Save to parquet
output_path = "outputs/multiprompt_test_100rows.parquet"
prompt_df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")

# Display results
print(f"\nGenerated {len(prompt_df)} total prompts from {len(df)} input rows")
print(f"Average prompts per row: {len(prompt_df) / len(df):.1f}")

print("\nPrompt type distribution:")
print(prompt_df['prompt_type'].value_counts())

print("\nPrompt weight distribution:")
print(prompt_df.groupby('prompt_type')['prompt_weight'].first())

print("\nFirst 10 prompts:")
display_df = prompt_df.head(10)[['StudyInstanceUID', 'SeriesInstanceUID', 'prompt_text', 'prompt_type', 'prompt_weight']]
for idx, row in display_df.iterrows():
    print(f"\n{idx}: {row['prompt_type']} (weight={row['prompt_weight']})")
    print(f"   Study: {row['StudyInstanceUID']}")
    print(f"   Series: {row['SeriesInstanceUID']}")
    print(f"   Text: {row['prompt_text'][:100]}...")  # First 100 chars
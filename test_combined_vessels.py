#!/usr/bin/env python3
"""Test the combined vessel mode (no vessel separation)."""

import sys
sys.path.append('dataset_creation')

import pandas as pd
from generate_dataset import (
    apply_hard_filters,
    assign_procedure_status,
    add_angiographic_view_column,
    format_main_structure_description,
    MAIN_STRUCTURE_MAP,
    DOMINANCE_MAP
)
from generate_dataset_multiprompt import (
    generate_multiprompt_dataset,
    create_default_config
)

# Load a small sample
input_path = '/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL.parquet'
df = pd.read_parquet(input_path).head(100)

# Apply mappings
if 'main_structure_class' in df.columns:
    df['main_structure_name'] = df['main_structure_class'].map(MAIN_STRUCTURE_MAP)
    df['main_structure_description'] = df['main_structure_class'].apply(format_main_structure_description)
if 'dominance_class' in df.columns:
    df['dominance_name'] = df['dominance_class'].map(DOMINANCE_MAP)

df = add_angiographic_view_column(df)

if 'Conclusion' in df.columns:
    df['bypass_graft'] = df['Conclusion'].str.contains('pontage', case=False, na=False).astype(int)

if 'StudyInstanceUID' in df.columns and 'SeriesTime' in df.columns:
    df = df.sort_values(['StudyInstanceUID', 'SeriesTime'])

# Assign procedure status
if 'stent_presence_class' in df.columns and 'StudyInstanceUID' in df.columns:
    df = assign_procedure_status(df)

# Apply filters
config = create_default_config()
df_filtered = apply_hard_filters(df, config)

print("="*60)
print("Testing WITH vessel separation (default):")
print("="*60)
prompt_df_separated = generate_multiprompt_dataset(df_filtered, vessel_separation=True)

print(f"\n✅ Generated {len(prompt_df_separated)} prompts from {len(df_filtered)} videos")

# Check for vessel mixing
for idx, row in prompt_df_separated.iterrows():
    text = row['prompt_text']
    main_structure = row.get('main_structure', '')

    # RCA vessels
    rca_found = any(vessel in text for vessel in ['RCA', 'PDA', 'PLB'])
    # LCA vessels (avoid false positives with "RCA")
    lca_found = any(vessel in text for vessel in ['Left Main', 'LAD', 'LCx', 'Lcx'])

    if rca_found and lca_found:
        print(f"⚠️ MIXED VESSELS in {main_structure} prompt: {text[:100]}...")
        break
else:
    print("✅ No vessel mixing detected in separated mode")

# Sample output
print(f"\nSample (separated):")
if len(prompt_df_separated) > 0:
    sample = prompt_df_separated.iloc[0]
    print(f"  Main structure: {sample.get('main_structure', 'N/A')}")
    print(f"  Prompt: {sample['prompt_text'][:150]}...")

print("\n" + "="*60)
print("Testing WITHOUT vessel separation (combined):")
print("="*60)
prompt_df_combined = generate_multiprompt_dataset(df_filtered, vessel_separation=False)

print(f"\n✅ Generated {len(prompt_df_combined)} prompts from {len(df_filtered)} videos")

# Check for combined vessels
combined_count = 0
for idx, row in prompt_df_combined.iterrows():
    text = row['prompt_text']

    # RCA vessels
    rca_found = any(vessel in text for vessel in ['RCA', 'PDA', 'PLB'])
    # LCA vessels
    lca_found = any(vessel in text for vessel in ['Left Main', 'LAD', 'LCx', 'Lcx'])

    if rca_found and lca_found:
        combined_count += 1
        if combined_count == 1:
            print(f"✅ COMBINED VESSELS detected (as expected):")
            print(f"   {text[:200]}...")

if combined_count > 0:
    print(f"\n✅ Found {combined_count} prompts with combined L/R vessels (expected behavior)")
else:
    print("⚠️ No combined vessels found - check if test data has both vessel types")

# Sample output
print(f"\nSample (combined):")
if len(prompt_df_combined) > 0:
    sample = prompt_df_combined.iloc[0]
    print(f"  Main structure: {sample.get('main_structure', 'N/A')}")
    print(f"  Prompt: {sample['prompt_text'][:200]}...")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Separated mode: {len(prompt_df_separated)} prompts")
print(f"Combined mode:  {len(prompt_df_combined)} prompts")
print(f"Difference:     {len(prompt_df_combined) - len(prompt_df_separated)} prompts")
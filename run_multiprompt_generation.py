#!/usr/bin/env python3
"""
Final script to generate multi-prompt dataset for SigLIP training.
Run this to process the full dataset or a sample.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import yaml

sys.path.append('dataset_creation')

from generate_dataset_multiprompt import (
    process_dataset_multiprompt,
    create_default_config
)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-prompt dataset for SigLIP training")
    parser.add_argument('--sample', type=int, help='Process only N rows for testing')
    parser.add_argument('--output-dir', default='outputs/multiprompt_final', help='Output directory')
    args = parser.parse_args()

    # Input parquet file
    input_path = '/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL.parquet'

    # Create configuration
    config = create_default_config()

    if args.sample:
        print(f"Processing sample of {args.sample} rows...")
        # For sample, load data directly and process
        df = pd.read_parquet(input_path).head(args.sample)

        # Import necessary functions
        from generate_dataset import (
            apply_hard_filters,
            assign_procedure_status,
            add_angiographic_view_column,
            format_main_structure_description,
            MAIN_STRUCTURE_MAP,
            DOMINANCE_MAP
        )
        from generate_dataset_multiprompt import generate_multiprompt_dataset

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
        df_filtered = apply_hard_filters(df, config)

        # Generate multi-prompts
        print("Generating multi-prompts...")
        prompt_df = generate_multiprompt_dataset(df_filtered)

        # Save output
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"multiprompt_sample_{args.sample}.parquet"
        prompt_df.to_parquet(output_file, index=False)

        # Display results
        print(f"\n✅ Successfully generated {len(prompt_df)} prompts from {len(df_filtered)} videos")
        print(f"📁 Saved to: {output_file}")
        print(f"\n📊 Statistics:")
        print(f"  - Average prompts per video: {len(prompt_df) / len(df_filtered):.1f}")
        print(f"\n  Prompt type distribution:")
        for ptype, count in prompt_df['prompt_type'].value_counts().items():
            weight = prompt_df[prompt_df['prompt_type'] == ptype]['prompt_weight'].iloc[0]
            print(f"    - {ptype}: {count} prompts (weight={weight})")

        print(f"\n📋 Sample output (first 5 rows):")
        print(prompt_df[['StudyInstanceUID', 'SeriesInstanceUID', 'prompt_type', 'prompt_weight', 'prompt_text']].head())

    else:
        print("Processing full dataset...")
        process_dataset_multiprompt(input_path, args.output_dir, config)


if __name__ == "__main__":
    main()
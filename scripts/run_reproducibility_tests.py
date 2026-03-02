"""
Reproducibility test: run inference on 100-study samples with different batch sizes
and compare predictions to full inference results.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/run_reproducibility_tests.py \
        --model zcb8cu0l --gpu 0
"""
import argparse
import glob
import json
import os
import sys
import time
import yaml
import subprocess
import pandas as pd
import numpy as np


def create_config(base_config_path, batch_size, sample_csv, output_dir):
    """Load base config and override batch_size, data_filename, output_dir."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config['batch_size'] = batch_size
    config['data_filename'] = sample_csv
    config['output_dir'] = output_dir
    config['shuffle_videos'] = False
    config['use_wandb'] = False

    # Write temp config
    tmp_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(tmp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return tmp_path


def run_inference(config_path, gpu, port, project_name):
    """Run inference with given config. Returns the actual output directory path."""
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': '/volume/DeepCORO_CLIP',
        'LOCAL_RANK': '0',
        'RANK': '0',
        'WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(port),
        'CUDA_VISIBLE_DEVICES': str(gpu),  # physical GPU number
    })
    # The config's device=0 maps to the physical GPU via CUDA_VISIBLE_DEVICES

    # Record dirs before run to find the newly created one
    base = f'/volume/DeepCORO_CLIP/outputs/DeepCORO_video_linear_probing/{project_name}'
    os.makedirs(base, exist_ok=True)
    before_dirs = set(glob.glob(os.path.join(base, '*_no_wandb')))

    cmd = [sys.executable, 'scripts/main.py', '--base_config', config_path]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd='/volume/DeepCORO_CLIP')

    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Inference failed with return code {result.returncode}")

    # Find the newly created output directory
    after_dirs = set(glob.glob(os.path.join(base, '*_no_wandb')))
    new_dirs = after_dirs - before_dirs
    if new_dirs:
        output_dir = new_dirs.pop()
    else:
        # Fallback: most recently modified dir
        dirs = sorted(glob.glob(os.path.join(base, '*_no_wandb')), key=os.path.getmtime)
        output_dir = dirs[-1] if dirs else None

    print(f"  Output directory: {output_dir}")

    # Give GPU time to fully release memory before next run
    time.sleep(15)

    return output_dir


def compare_predictions(pred_dir_1, pred_dir_2, label_1, label_2, filter_to_common=False):
    """Compare prediction CSVs from two runs."""
    pred_file_1 = os.path.join(pred_dir_1, 'predictions', 'inference_predictions_epoch_-1.csv')
    pred_file_2 = os.path.join(pred_dir_2, 'predictions', 'inference_predictions_epoch_-1.csv')

    df1 = pd.read_csv(pred_file_1)
    df2 = pd.read_csv(pred_file_2)

    # If filtering to common videos (for full vs sample comparison)
    if filter_to_common:
        common_videos = set(df1['video_name']) & set(df2['video_name'])
        df1 = df1[df1['video_name'].isin(common_videos)]
        df2 = df2[df2['video_name'].isin(common_videos)]

    # Sort both by video_name for consistent comparison
    df1 = df1.sort_values('video_name').reset_index(drop=True)
    df2 = df2.sort_values('video_name').reset_index(drop=True)

    # Get prediction columns
    pred_cols = [c for c in df1.columns if c.endswith('_pred')]

    results = {
        'label_1': label_1,
        'label_2': label_2,
        'n_samples': len(df1),
        'n_pred_cols': len(pred_cols),
        'exact_match': True,
        'max_abs_diff': 0.0,
        'mean_abs_diff': 0.0,
        'mismatched_cols': [],
    }

    if len(df1) != len(df2):
        results['exact_match'] = False
        results['error'] = f"Row count mismatch: {len(df1)} vs {len(df2)}"
        return results

    # Check video names match
    if not (df1['video_name'].values == df2['video_name'].values).all():
        results['exact_match'] = False
        results['error'] = "Video names don't match after sorting"
        return results

    all_diffs = []
    for col in pred_cols:
        diff = np.abs(df1[col].values - df2[col].values)
        max_diff = diff.max()
        if max_diff > 1e-6:
            results['mismatched_cols'].append({'col': col, 'max_diff': float(max_diff)})
            results['exact_match'] = False
        all_diffs.extend(diff.tolist())

    results['max_abs_diff'] = float(max(all_diffs)) if all_diffs else 0.0
    results['mean_abs_diff'] = float(np.mean(all_diffs)) if all_diffs else 0.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['zcb8cu0l', 'sarra'])
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    if args.model == 'zcb8cu0l':
        base_config = 'config/linear_probing/stenosis/zcb8cu0l_inference_config.yaml'
        sample_csv = '/volume/DeepCORO_CLIP/outputs/sample_100_zcb8cu0l.csv'
        full_pred_dir = 'outputs/DeepCORO_video_linear_probing/DeepCORO_stenosis_weighted_distal/20260223-162254_no_wandb'
        output_base = 'outputs/reproducibility/zcb8cu0l'
        project_name = 'DeepCORO_stenosis_weighted_distal'
    else:
        base_config = 'config/linear_probing/stenosis/sarra_inference_config.yaml'
        sample_csv = '/volume/DeepCORO_CLIP/outputs/sample_100_sarra.csv'
        full_pred_dir = None  # Will be set after finding Sarra's output
        output_base = 'outputs/reproducibility/sarra'
        project_name = 'DeepCORO_video_linear_probing_multiview_improved_cls_token'

        # Try to find Sarra's full inference output
        sarra_base = f'/volume/DeepCORO_CLIP/outputs/DeepCORO_video_linear_probing/{project_name}'
        sarra_dirs = sorted(glob.glob(os.path.join(sarra_base, '*_no_wandb')), key=os.path.getmtime)
        if sarra_dirs:
            full_pred_dir = sarra_dirs[-1]
            print(f"Found Sarra full inference dir: {full_pred_dir}")

    batch_sizes = [1, 2, 4] if args.model == 'sarra' else [1, 2, 4, 12]
    base_port = 29520
    run_dirs = {}

    # Run inference for each batch size
    for bs in batch_sizes:
        output_dir = os.path.join(output_base, f'bs{bs}')
        print(f"\n{'='*60}")
        print(f"Running {args.model} with batch_size={bs}")
        print(f"{'='*60}")

        config_path = create_config(base_config, bs, sample_csv, output_dir)
        actual_output_dir = run_inference(config_path, args.gpu, base_port + bs, project_name)
        run_dirs[bs] = actual_output_dir
        print(f"Completed batch_size={bs}")

    # Compare all batch sizes against each other
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    all_results = []

    # Compare each pair of batch sizes
    bs_list = sorted(run_dirs.keys())
    for i in range(len(bs_list)):
        for j in range(i+1, len(bs_list)):
            bs_a, bs_b = bs_list[i], bs_list[j]
            result = compare_predictions(
                run_dirs[bs_a], run_dirs[bs_b],
                f'bs={bs_a}', f'bs={bs_b}'
            )
            all_results.append(result)

            status = "PASS" if result['exact_match'] else "FAIL"
            print(f"\n[{status}] bs={bs_a} vs bs={bs_b}")
            print(f"  Samples: {result['n_samples']}, Pred cols: {result['n_pred_cols']}")
            print(f"  Max abs diff: {result['max_abs_diff']:.2e}")
            print(f"  Mean abs diff: {result['mean_abs_diff']:.2e}")
            if result['mismatched_cols']:
                print(f"  Mismatched columns ({len(result['mismatched_cols'])}):")
                for mc in result['mismatched_cols'][:5]:
                    print(f"    {mc['col']}: max_diff={mc['max_diff']:.2e}")

    # Compare against full inference (filter to common videos)
    if full_pred_dir and os.path.exists(os.path.join(full_pred_dir, 'predictions', 'inference_predictions_epoch_-1.csv')):
        print(f"\n{'='*60}")
        print("COMPARISON VS FULL INFERENCE")
        print(f"Full inference dir: {full_pred_dir}")
        print(f"{'='*60}")

        for bs in bs_list:
            result = compare_predictions(
                full_pred_dir, run_dirs[bs],
                'full_inference', f'sample_bs={bs}',
                filter_to_common=True
            )
            all_results.append(result)

            if 'error' in result:
                print(f"\n[INFO] full vs bs={bs}: {result['error']}")
            else:
                status = "PASS" if result['exact_match'] else "FAIL"
                print(f"\n[{status}] full vs sample_bs={bs}")
                print(f"  Common videos: {result['n_samples']}, Pred cols: {result['n_pred_cols']}")
                print(f"  Max abs diff: {result['max_abs_diff']:.2e}")
                print(f"  Mean abs diff: {result['mean_abs_diff']:.2e}")
                if result['mismatched_cols']:
                    print(f"  Mismatched columns ({len(result['mismatched_cols'])}):")
                    for mc in result['mismatched_cols'][:5]:
                        print(f"    {mc['col']}: max_diff={mc['max_diff']:.2e}")

    # Save summary
    os.makedirs(output_base, exist_ok=True)
    summary_path = os.path.join(output_base, 'reproducibility_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()

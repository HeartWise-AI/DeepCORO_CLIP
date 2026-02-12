import ast
import pydicom
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix, 
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

get_sex = False
get_age = False
compute_metrics = True

df_video = None
if get_sex:
    # Load data
    df_results = pd.read_csv('/media/data1/jdelfrate/DeepCORO_CLIP/sarra_results/preds_with_true_filled_20251020_233850.csv')
    print(f'df_results.shape: {df_results.shape}')

    df_video = pd.read_parquet('/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_STUDY_LEVEL.parquet')
    print(f'df_video.shape: {df_video.shape}')

    df_ecg = pd.read_parquet('/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet')

    # Check if all patient IDs in df_training are in df_ecg
    df_ecg_patient_id = set(df_ecg.PatientID.tolist())
    df_video_patient_id = set(df_video.Patient_ID.tolist())

    print(f'Number of patient IDs in df_ecg: {len(df_ecg_patient_id)}')
    print(f'Number of patient IDs in df_video: {len(df_video_patient_id)}')
    print(f'All patient IDs in df_video are in df_ecg: {df_video_patient_id.issubset(df_ecg_patient_id)}')

    difference = df_video_patient_id.difference(df_ecg_patient_id)
    print(f'Number of patient IDs in df_video but not in df_ecg: {len(difference)}')

    # Prepare mapping
    df_results = df_results.reset_index(drop=True)
    df_video = df_video.reset_index(drop=True)
    all_paths = {path: idx for idx, path in enumerate(df_video.FileName.tolist())}
    df_ecg_indexed = df_ecg.set_index('PatientID')
    genders = df_ecg_indexed['RestingECG_PatientDemographics_Gender'].to_dict()
    ages = df_ecg_indexed['RestingECG_PatientDemographics_PatientAge'].to_dict()

    # Fill sex and age in df_results with values from df_ecg
    df_results['sex'] = ''
    df_results['PatientAge'] = ''
    for idx, row in df_results.iterrows():
        try:
            videos = ast.literal_eval(row.video_name)
        except:
            print(f'Error parsing video_name at row {idx}: {row.video_name}')
            continue
        for path in videos:
            if path in all_paths:
                patient_id = df_video.loc[all_paths[path], 'Patient_ID']
                try:
                    df_results.loc[idx, 'sex'] = genders.get(patient_id, '')
                    break
                except:
                    print(f'Error getting sex and age for patient {patient_id} at row {idx}')
                    continue

    # Print number of rows with sex and age filled
    print(df_results[df_results.sex != ''].shape)

    # Save results
    df_results.to_csv('/media/data1/jdelfrate/DeepCORO_CLIP/sarra_results/preds_with_true_filled_20251020_233850_with_sex.csv', index=False)
else:
    df_results = pd.read_csv('/media/data1/jdelfrate/DeepCORO_CLIP/sarra_results/preds_with_true_filled_20251020_233850_with_sex.csv')


# Get age from dicoms
if get_age:
    if not df_video:
        df_video = pd.read_parquet('/media/data1/datasets/DeepCoro/2b_CathReport_HEMO_MHI_MERGED_2017-2024_VIDEO_LEVEL.parquet')

    df_results['age'] = pd.NA
    for idx, row in tqdm(df_results.iterrows(), total=len(df_results), desc='Getting age from dicoms'):
        try:
            videos = ast.literal_eval(row.video_name)
            if not videos:
                continue
            video_path = videos[0]
        except:
            print(f'Error parsing video_name at row {idx}: {row.video_name}')
            continue

        match = df_video[df_video.FileName == video_path]
        if match.empty:
            print(f'No match found for video_path: {video_path}')
            continue
        dicom_path = match.iloc[0]['DICOMPath']

        try:
            dicom = pydicom.dcmread(dicom_path)
            study_date_str = str(dicom.get((0x0008, 0x0020)).value)
            patient_birth_date_str = str(dicom.get((0x0010, 0x0030)).value)

            if study_date_str is None or patient_birth_date_str is None:
                print(f'Study date or patient birth date is missing at row {idx}: {dicom_path}')
                break

            study_date = pd.to_datetime(study_date_str, format='%Y%m%d')
            patient_birth_date = pd.to_datetime(patient_birth_date_str, format='%Y%m%d')
            age = (study_date - patient_birth_date).days / 365.25
            
            df_results.loc[idx, 'age'] = int(age)

        except Exception as e:
            print(f'Error processing dicom at row {idx}: {dicom_path}')
            print(e)
            break

    df_results.to_csv('/media/data1/jdelfrate/DeepCORO_CLIP/sarra_results/preds_with_true_filled_20251020_233850_with_sex_and_age.csv', index=False)

else:
    df_results = pd.read_csv('/media/data1/jdelfrate/DeepCORO_CLIP/sarra_results/preds_with_true_filled_20251020_233850_with_sex_and_age.csv')

if compute_metrics:
    lesion_types = ['stenosis_binary', 'stenosis', 'cto', 'thrombus', 'calcif_binary']

    arteries = (
        "left_main",
        "prox_rca",
        "mid_rca",
        "dist_rca",
        "pda",
        "posterolateral",
        "prox_lad",
        "mid_lad",
        "dist_lad",
        "D1",
        "D2",
        "prox_lcx",
        "mid_lcx",
        "dist_lcx",
        "om1",
        "om2",
        "bx",
        "lvp"    
    )

    subgroups = {
        "Male": df_results[df_results.sex == 'MALE'],
        "Female": df_results[df_results.sex == 'FEMALE'],
        "age <= 50": df_results[df_results.age <= 50],
        "age 51 - 60": df_results[(df_results.age > 50) & (df_results.age <= 60)],
        "age 61 - 70": df_results[(df_results.age > 60) & (df_results.age <= 70)],
        "age 71 - 80": df_results[(df_results.age > 70) & (df_results.age <= 80)],
        "age > 80": df_results[df_results.age > 80]
    }

    # Compute prevalence per lesion type across all artery segments
    prevalence_summary = []
    for lesion_type in lesion_types:
        for subgroup, df_subgroup in subgroups.items():
            if len(df_subgroup) == 0:
                continue

            all_y_true = []
            for artery in arteries:
                true_col = f'{artery}_{lesion_type}_true'
                all_y_true.extend(df_subgroup[true_col].tolist())

            base_dict = {
                'subgroup': subgroup,
                'lesion_type': lesion_type,
                'n_segments': len(all_y_true),
                'n_patients': len(df_subgroup)
            }

            if lesion_type == 'stenosis':
                base_dict.update({
                    'mean_stenosis': np.mean(all_y_true),
                    'std_stenosis': np.std(all_y_true)
                })
            else:
                base_dict.update({
                    'prevalence_pct': np.mean(all_y_true) * 100,
                    'n_positive': int(np.sum(all_y_true))
                })
            
            prevalence_summary.append(base_dict)

    df_prevalence = pd.DataFrame(prevalence_summary)
    print("\nPrevalence per lesion type (across all arteries):")
    print(df_prevalence)
    df_prevalence.to_csv('prevalence_per_lesion_type.csv', index=False)

    def compute_best_threshold(y_true: list[float], y_pred: list[float]) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]
        return best_threshold

    def safe_ci(arr):
        if len(arr) < 2:
            return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        return {'mean': float(np.mean(arr)), 'ci_lower': float(np.percentile(arr, 2.5)), 'ci_upper': float(np.percentile(arr, 97.5))}

    def bootstrap_classification_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_bootstrap: int = 1000,
        seed: int = 42
    ) -> dict[str, dict[str, float]]:
        """
        Bootstrap metrics and compute confidence intervals.
        """
        rng = np.random.RandomState(seed)

        n_failed = 0
        auc, auprc, sensitivity, specificity = [], [], [], []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)

            if len(np.unique(y_true[indices])) < 2:
                n_failed += 1
                continue
            
            try:
                y_true_resampled, y_pred_resampled = y_true[indices], y_pred[indices]
                auc.append(roc_auc_score(y_true_resampled, y_pred_resampled))
                auprc.append(average_precision_score(y_true_resampled, y_pred_resampled))

                best_threshold = compute_best_threshold(y_true_resampled, y_pred_resampled)
                y_pred_bin = (y_pred_resampled > best_threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_true_resampled, y_pred_bin).ravel()
                if (tp + fn) > 0:
                    sensitivity.append(tp / (tp + fn))
                if (tn + fp) > 0:
                    specificity.append(tn / (tn + fp))

            except ValueError:
                continue

        return {
            'auc': safe_ci(auc),
            'auprc': safe_ci(auprc),
            'sensitivity': safe_ci(sensitivity),
            'specificity': safe_ci(specificity)
        }

    def bootstrap_regression_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_bootstrap: int = 1000,
        seed: int = 42
    ) -> dict[str, dict[str, float]]:
        """
        Bootstrap metrics and compute confidence intervals.
        """
        rng = np.random.RandomState(seed)

        n_failed = 0
        mae, mse, rmse, r2, pearson_r = [], [], [], [], []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)

            if len(np.unique(y_true[indices])) < 2:
                n_failed += 1
                continue
            
            try:
                y_true_resampled, y_pred_resampled = y_true[indices], y_pred[indices]
                mae.append(mean_absolute_error(y_true_resampled, y_pred_resampled))
                mse.append(mean_squared_error(y_true_resampled, y_pred_resampled))
                rmse.append(np.sqrt(mean_squared_error(y_true_resampled, y_pred_resampled)))
                r2.append(r2_score(y_true_resampled, y_pred_resampled))
                pearson_r.append(np.corrcoef(y_true_resampled, y_pred_resampled)[0, 1])

            except ValueError:
                continue

        return {
            'mae': safe_ci(mae),
            'mse': safe_ci(mse),
            'rmse': safe_ci(rmse),
            'r2': safe_ci(r2),
            'pearson_r': safe_ci(pearson_r)
        }

    classification_results = []
    regression_results = []
    micro_results = []

    for lesion_type in tqdm(lesion_types, desc='Lesion types'):
        mode = 'regression' if lesion_type == 'stenosis' else 'classification'

        for artery in tqdm(arteries, desc='Arteries'):
            pred_col = f'{artery}_{lesion_type}_pred'
            true_col = f'{artery}_{lesion_type}_true'
            for subgroup, df_subgroup in subgroups.items():
                if len(df_subgroup) == 0:
                    continue
                
                y_pred = np.array(df_subgroup[pred_col].tolist())
                y_true = np.array(df_subgroup[true_col].tolist())

                if len(np.unique(y_true)) < 2:
                    print(f"Skipping {artery}_{lesion_type} because it has less than 2 unique values")
                    continue
                
                if mode == 'classification':
                    bootstrap_metrics_results = bootstrap_classification_metrics(y_true, y_pred)

                    classification_results.append({
                        'artery': artery,
                        'lesion_type': lesion_type,
                        'subgroup': subgroup,
                        'prevalence': np.mean(y_true),
                        'n_samples': len(y_true),
                        'auc': bootstrap_metrics_results['auc']['mean'],
                        'auc_ci_lower': bootstrap_metrics_results['auc']['ci_lower'],
                        'auc_ci_upper': bootstrap_metrics_results['auc']['ci_upper'],
                        'auprc': bootstrap_metrics_results['auprc']['mean'],
                        'auprc_ci_lower': bootstrap_metrics_results['auprc']['ci_lower'],
                        'auprc_ci_upper': bootstrap_metrics_results['auprc']['ci_upper'],
                        'sensitivity': bootstrap_metrics_results['sensitivity']['mean'],
                        'sensitivity_ci_lower': bootstrap_metrics_results['sensitivity']['ci_lower'],
                        'sensitivity_ci_upper': bootstrap_metrics_results['sensitivity']['ci_upper'],
                        'specificity': bootstrap_metrics_results['specificity']['mean'],
                        'specificity_ci_lower': bootstrap_metrics_results['specificity']['ci_lower'],
                        'specificity_ci_upper': bootstrap_metrics_results['specificity']['ci_upper']
                    })

                elif mode == 'regression':
                    bootstrap_metrics_results = bootstrap_regression_metrics(y_true, y_pred)

                    regression_results.append({
                        'artery': artery,
                        'lesion_type': lesion_type,
                        'subgroup': subgroup,
                        'prevalence': np.mean(y_true),
                        'n_samples': len(y_true),
                        'mae': bootstrap_metrics_results['mae']['mean'],
                        'mae_ci_lower': bootstrap_metrics_results['mae']['ci_lower'],
                        'mae_ci_upper': bootstrap_metrics_results['mae']['ci_upper'],
                        'mse': bootstrap_metrics_results['mse']['mean'],
                        'mse_ci_lower': bootstrap_metrics_results['mse']['ci_lower'],
                        'mse_ci_upper': bootstrap_metrics_results['mse']['ci_upper'],
                        'rmse': bootstrap_metrics_results['rmse']['mean'],
                        'rmse_ci_lower': bootstrap_metrics_results['rmse']['ci_lower'],
                        'rmse_ci_upper': bootstrap_metrics_results['rmse']['ci_upper'],
                    })
                else:
                    raise ValueError(f'Invalid mode: {mode}')

        # Compute micro-averaged AUC and AUPRC across all arteries (for classification lesion types)
        n_total = len(df_results)
        if mode == 'classification':
            for subgroup, df_subgroup in subgroups.items():
                if len(df_subgroup) == 0:
                    continue

                n_patients = len(df_subgroup)
                pct = 100.0 * n_patients / n_total

                all_y_true = []
                all_y_pred = []
                for artery in arteries:
                    pred_col = f'{artery}_{lesion_type}_pred'
                    true_col = f'{artery}_{lesion_type}_true'
                    all_y_true.extend(df_subgroup[true_col].tolist())
                    all_y_pred.extend(df_subgroup[pred_col].tolist())

                all_y_true = np.array(all_y_true)
                all_y_pred = np.array(all_y_pred)

                if len(np.unique(all_y_true)) < 2:
                    print(f"Skipping micro {lesion_type} for subgroup {subgroup} - less than 2 unique values")
                    continue

                bootstrap_metrics_results = bootstrap_classification_metrics(all_y_true, all_y_pred)

                micro_results.append({
                    'lesion_type': lesion_type,
                    'subgroup': subgroup,
                    'metric_type': 'micro',
                    'n_patients': n_patients,
                    'n_total': n_total,
                    'pct': pct,
                    'prevalence': np.mean(all_y_true),
                    'n_samples': len(all_y_true),
                    'micro_auc': bootstrap_metrics_results['auc']['mean'],
                    'micro_auc_ci_lower': bootstrap_metrics_results['auc']['ci_lower'],
                    'micro_auc_ci_upper': bootstrap_metrics_results['auc']['ci_upper'],
                    'micro_auprc': bootstrap_metrics_results['auprc']['mean'],
                    'micro_auprc_ci_lower': bootstrap_metrics_results['auprc']['ci_lower'],
                    'micro_auprc_ci_upper': bootstrap_metrics_results['auprc']['ci_upper'],
                    'micro_sensitivity': bootstrap_metrics_results['sensitivity']['mean'],
                    'micro_sensitivity_ci_lower': bootstrap_metrics_results['sensitivity']['ci_lower'],
                    'micro_sensitivity_ci_upper': bootstrap_metrics_results['sensitivity']['ci_upper'],
                    'micro_specificity': bootstrap_metrics_results['specificity']['mean'],
                    'micro_specificity_ci_lower': bootstrap_metrics_results['specificity']['ci_lower'],
                    'micro_specificity_ci_upper': bootstrap_metrics_results['specificity']['ci_upper']
                })

        # Compute global MAE across all arteries (for regression lesion types like stenosis)
        elif mode == 'regression':
            for subgroup, df_subgroup in subgroups.items():
                if len(df_subgroup) == 0:
                    continue

                n_patients = len(df_subgroup)
                pct = 100.0 * n_patients / n_total

                all_y_true = []
                all_y_pred = []
                for artery in arteries:
                    pred_col = f'{artery}_{lesion_type}_pred'
                    true_col = f'{artery}_{lesion_type}_true'
                    all_y_true.extend(df_subgroup[true_col].tolist())
                    all_y_pred.extend(df_subgroup[pred_col].tolist())

                all_y_true = np.array(all_y_true)
                all_y_pred = np.array(all_y_pred)

                if len(np.unique(all_y_true)) < 2:
                    print(f"Skipping global {lesion_type} for subgroup {subgroup} - less than 2 unique values")
                    continue

                bootstrap_metrics_results = bootstrap_regression_metrics(all_y_true, all_y_pred)

                micro_results.append({
                    'lesion_type': lesion_type,
                    'subgroup': subgroup,
                    'metric_type': 'global',
                    'n_patients': n_patients,
                    'n_total': n_total,
                    'pct': pct,
                    'mean_stenosis': np.mean(all_y_true),
                    'n_samples': len(all_y_true),
                    'global_mae': bootstrap_metrics_results['mae']['mean'],
                    'global_mae_ci_lower': bootstrap_metrics_results['mae']['ci_lower'],
                    'global_mae_ci_upper': bootstrap_metrics_results['mae']['ci_upper'],
                    'global_mse': bootstrap_metrics_results['mse']['mean'],
                    'global_mse_ci_lower': bootstrap_metrics_results['mse']['ci_lower'],
                    'global_mse_ci_upper': bootstrap_metrics_results['mse']['ci_upper'],
                    'global_rmse': bootstrap_metrics_results['rmse']['mean'],
                    'global_rmse_ci_lower': bootstrap_metrics_results['rmse']['ci_lower'],
                    'global_rmse_ci_upper': bootstrap_metrics_results['rmse']['ci_upper'],
                    'global_pearson_r': bootstrap_metrics_results['pearson_r']['mean'],
                    'global_pearson_r_ci_lower': bootstrap_metrics_results['pearson_r']['ci_lower'],
                    'global_pearson_r_ci_upper': bootstrap_metrics_results['pearson_r']['ci_upper'],
                })

    # df_results_classification_subgroups_ci = pd.DataFrame(classification_results)
    df_results_regression_subgroups_ci = pd.DataFrame(regression_results)
    df_results_micro_ci = pd.DataFrame(micro_results)
    # df_results_classification_subgroups_ci.to_csv('preds_with_true_filled_20251020_233850_with_sex_and_age_classification_subgroups_ci.csv', index=False)
    df_results_regression_subgroups_ci.to_csv('preds_with_true_filled_20251020_233850_with_sex_and_age_regression_subgroups_ci.csv', index=False)
    df_results_micro_ci.to_csv('preds_with_true_filled_20251020_233850_with_sex_and_age_global_metrics_ci.csv', index=False)
    print(f"\nGlobal/micro-averaged results saved. Shape: {df_results_micro_ci.shape}")
    print(df_results_micro_ci)
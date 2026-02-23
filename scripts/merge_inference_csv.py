"""
Merge inference CSVs from zcb8cu0l and Sarra runs.

Logic:
1. Read both CSVs (separator='α')
2. Filter both to Split == 'inference'
3. Find the 20 exams (StudyInstanceUID) in Sarra but NOT in zcb8cu0l
4. Merged set = all zcb8cu0l inference rows + Sarra rows for those 20 exams
5. For Sarra-only rows, columns in zcb8cu0l but not Sarra are set to NaN
6. Save merged CSV with separator='α'
"""

import pandas as pd

SEPARATOR = "α"

# --- Step 1: Read both CSVs ---
print("Reading zcb8cu0l CSV...")
df_zcb = pd.read_csv(
    "/media/data1/datasets/DeepCoro_CLIP/CTO_THROMBUS_STENOSIS_70_CALCIF_inference_with_binary.csv",
    sep=SEPARATOR,
    engine="python",
)
print(f"  zcb8cu0l total rows: {len(df_zcb)}, columns: {df_zcb.shape[1]}")

print("Reading Sarra CSV...")
df_sarra = pd.read_csv(
    "/media/data1/ravram/stenosis70/CTO_THROMBUS_STENOSIS_70_CALCIF_inference.csv",
    sep=SEPARATOR,
    engine="python",
)
print(f"  Sarra total rows: {len(df_sarra)}, columns: {df_sarra.shape[1]}")

# --- Step 2: Filter to inference split only ---
df_zcb_inf = df_zcb[df_zcb["Split"] == "inference"].copy()
df_sarra_inf = df_sarra[df_sarra["Split"] == "inference"].copy()

print(f"\nAfter filtering to Split == 'inference':")
print(f"  zcb8cu0l inference rows: {len(df_zcb_inf)}, unique exams: {df_zcb_inf['StudyInstanceUID'].nunique()}")
print(f"  Sarra inference rows:    {len(df_sarra_inf)}, unique exams: {df_sarra_inf['StudyInstanceUID'].nunique()}")

# --- Step 3: Find the exams in Sarra but NOT in zcb8cu0l ---
zcb_exams = set(df_zcb_inf["StudyInstanceUID"].unique())
sarra_exams = set(df_sarra_inf["StudyInstanceUID"].unique())

sarra_only_exams = sarra_exams - zcb_exams
print(f"\nExams in Sarra but NOT in zcb8cu0l: {len(sarra_only_exams)}")

# --- Step 4: Build the merged set ---
df_sarra_only = df_sarra_inf[df_sarra_inf["StudyInstanceUID"].isin(sarra_only_exams)].copy()
print(f"Sarra-only rows to add: {len(df_sarra_only)}")

# Step 5/6: concat — pandas will automatically set NaN for missing columns
df_merged = pd.concat([df_zcb_inf, df_sarra_only], ignore_index=True)

# --- Step 7: Save ---
output_path = "/volume/DeepCORO_CLIP/outputs/merged_inference_zcb8cu0l_sarra.csv"
df_merged.to_csv(output_path, sep=SEPARATOR, index=False)
print(f"\nSaved merged CSV to: {output_path}")

# --- Summary ---
n_unique_exams = df_merged["StudyInstanceUID"].nunique()
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Unique exams in merged set:  {n_unique_exams}  (expected 4,828)")
print(f"Total rows in merged set:    {len(df_merged)}")
print(f"Rows from zcb8cu0l:          {len(df_zcb_inf)}")
print(f"Rows from Sarra (20 exams):  {len(df_sarra_only)}")
print(f"{'='*60}")

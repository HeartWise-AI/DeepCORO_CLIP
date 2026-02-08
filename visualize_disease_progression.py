#!/usr/bin/env python3
"""
Visualize disease progression analysis - embedding change by number of vessels worsened.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the analysis data
df = pd.read_csv('outputs/pci_comparison/disease_progression_analysis.csv')

print(f"Loaded {len(df)} patients with follow-up studies")
print(f"\nColumns: {df.columns.tolist()}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ============================================================
# 1. Embedding Change by Number of Vessels with NEW Significant Disease (crossed 50%)
# ============================================================
ax1 = axes[0, 0]

# Group by vessels_new_disease
grouped_new = df.groupby('vessels_new_disease').agg({
    'embedding_change': ['mean', 'std', 'count']
}).reset_index()
grouped_new.columns = ['vessels_new_disease', 'mean', 'std', 'count']

# Cap at 3+ for visualization
grouped_new_capped = grouped_new.copy()
grouped_new_capped.loc[grouped_new_capped['vessels_new_disease'] >= 3, 'vessels_new_disease'] = 3

# Re-aggregate for 3+
grouped_new_final = grouped_new_capped.groupby('vessels_new_disease').agg({
    'mean': lambda x: np.average(x, weights=grouped_new_capped.loc[x.index, 'count']),
    'count': 'sum'
}).reset_index()

# Re-calculate std for capped groups
stds_new = []
for v in grouped_new_final['vessels_new_disease']:
    if v < 3:
        subset = df[df['vessels_new_disease'] == v]
    else:
        subset = df[df['vessels_new_disease'] >= 3]
    stds_new.append(subset['embedding_change'].std())
grouped_new_final['std'] = stds_new

colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
labels = ['0', '1', '2', '3+']

bars1 = ax1.bar(range(len(grouped_new_final)), grouped_new_final['mean'],
                yerr=grouped_new_final['std'], capsize=8,
                color=colors[:len(grouped_new_final)], alpha=0.8,
                edgecolor='black', linewidth=2)

ax1.set_xticks(range(len(grouped_new_final)))
ax1.set_xticklabels(labels[:len(grouped_new_final)], fontsize=12)
ax1.set_xlabel('Number of Vessels with NEW Significant Disease (>50%)', fontsize=12)
ax1.set_ylabel('Embedding Change (1 - cosine_sim)', fontsize=12)
ax1.set_title('Embedding Change by New Disease Onset', fontsize=14, fontweight='bold')

# Add count labels
for i, (bar, row) in enumerate(zip(bars1, grouped_new_final.itertuples())):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row.std + 0.02,
             f'n={int(row.count)}', ha='center', fontsize=10, fontweight='bold')

ax1.set_ylim(0, 0.8)

# ============================================================
# 2. Embedding Change by Number of Vessels Worsened (>20% increase)
# ============================================================
ax2 = axes[0, 1]

# Group by vessels_worsened
grouped_worse = df.groupby('vessels_worsened').agg({
    'embedding_change': ['mean', 'std', 'count']
}).reset_index()
grouped_worse.columns = ['vessels_worsened', 'mean', 'std', 'count']

# Cap at 3+ for visualization
grouped_worse_capped = grouped_worse.copy()
grouped_worse_capped.loc[grouped_worse_capped['vessels_worsened'] >= 3, 'vessels_worsened'] = 3

grouped_worse_final = grouped_worse_capped.groupby('vessels_worsened').agg({
    'mean': lambda x: np.average(x, weights=grouped_worse_capped.loc[x.index, 'count']),
    'count': 'sum'
}).reset_index()

# Re-calculate std for capped groups
stds_worse = []
for v in grouped_worse_final['vessels_worsened']:
    if v < 3:
        subset = df[df['vessels_worsened'] == v]
    else:
        subset = df[df['vessels_worsened'] >= 3]
    stds_worse.append(subset['embedding_change'].std())
grouped_worse_final['std'] = stds_worse

bars2 = ax2.bar(range(len(grouped_worse_final)), grouped_worse_final['mean'],
                yerr=grouped_worse_final['std'], capsize=8,
                color=colors[:len(grouped_worse_final)], alpha=0.8,
                edgecolor='black', linewidth=2)

ax2.set_xticks(range(len(grouped_worse_final)))
ax2.set_xticklabels(labels[:len(grouped_worse_final)], fontsize=12)
ax2.set_xlabel('Number of Vessels Worsened (>20% stenosis increase)', fontsize=12)
ax2.set_ylabel('Embedding Change (1 - cosine_sim)', fontsize=12)
ax2.set_title('Embedding Change by Disease Progression', fontsize=14, fontweight='bold')

# Add count labels
for i, (bar, row) in enumerate(zip(bars2, grouped_worse_final.itertuples())):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row.std + 0.02,
             f'n={int(row.count)}', ha='center', fontsize=10, fontweight='bold')

ax2.set_ylim(0, 0.8)

# ============================================================
# 3. Box Plot - Worsening vs Stable
# ============================================================
ax3 = axes[1, 0]

df['disease_status'] = df['vessels_new_disease'].apply(
    lambda x: 'Progressed\n(new disease)' if x > 0 else 'Stable'
)

box_data = [
    df[df['disease_status'] == 'Stable']['embedding_change'].values,
    df[df['disease_status'] == 'Progressed\n(new disease)']['embedding_change'].values
]

bp = ax3.boxplot(box_data, labels=['Stable', 'Progressed'], patch_artist=True)
bp['boxes'][0].set_facecolor('#27ae60')
bp['boxes'][1].set_facecolor('#e74c3c')

# Add significance test
stable_data = df[df['disease_status'] == 'Stable']['embedding_change']
prog_data = df[df['disease_status'] == 'Progressed\n(new disease)']['embedding_change']
t_stat, p_val = stats.ttest_ind(stable_data, prog_data)

ax3.plot([1, 1, 2, 2], [1.0, 1.03, 1.03, 1.0], 'k-', linewidth=1.5)
sig_text = f'p = {p_val:.4f}' if p_val > 0.0001 else 'p < 0.0001'
ax3.text(1.5, 1.05, f'{sig_text}\nt = {t_stat:.2f}', ha='center', fontsize=11, fontweight='bold')

ax3.set_ylabel('Embedding Change', fontsize=12)
ax3.set_title('Stable vs Progressive Disease', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1.2)

# Add means
for i, data in enumerate(box_data):
    mean_val = np.mean(data)
    ax3.scatter([i+1], [mean_val], color='white', s=80, zorder=5, edgecolor='black', linewidth=2)
    n = len(data)
    ax3.text(i+1, -0.08, f'n={n}\n$\mu$={mean_val:.3f}', ha='center', fontsize=10)

# ============================================================
# 4. Scatter plot - Net Change vs Embedding Change
# ============================================================
ax4 = axes[1, 1]

# Net change: positive = more disease, negative = improvement
sc = ax4.scatter(df['net_change'], df['embedding_change'],
                 c=df['vessels_new_disease'], cmap='RdYlGn_r',
                 alpha=0.6, s=50, edgecolor='gray', linewidth=0.5)

# Add trend line
z = np.polyfit(df['net_change'], df['embedding_change'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['net_change'].min(), df['net_change'].max(), 100)
ax4.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

# Calculate correlation
corr, corr_p = stats.pearsonr(df['net_change'], df['embedding_change'])

ax4.set_xlabel('Net Vessel Change (positive = worsening)', fontsize=12)
ax4.set_ylabel('Embedding Change', fontsize=12)
ax4.set_title(f'Disease Progression vs Embedding Change\nr = {corr:.3f}, p = {corr_p:.4f}',
              fontsize=14, fontweight='bold')
ax4.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax4.legend(loc='upper right')

cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label('Vessels with New Disease', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/pci_comparison/disease_progression_chart.png', dpi=150, bbox_inches='tight')
print(f"\nChart saved to: outputs/pci_comparison/disease_progression_chart.png")

# ============================================================
# Print summary statistics
# ============================================================
print("\n" + "="*70)
print("DISEASE PROGRESSION ANALYSIS SUMMARY")
print("="*70)

print("\n--- Vessels with NEW Significant Disease (crossed 50% threshold) ---")
for v in sorted(df['vessels_new_disease'].unique()):
    subset = df[df['vessels_new_disease'] == v]
    print(f"  {v} vessels: n={len(subset)}, "
          f"emb_change={subset['embedding_change'].mean():.3f} +/- {subset['embedding_change'].std():.3f}")

print("\n--- Vessels Worsened (>20% stenosis increase) ---")
for v in sorted(df['vessels_worsened'].unique()):
    subset = df[df['vessels_worsened'] == v]
    print(f"  {v} vessels: n={len(subset)}, "
          f"emb_change={subset['embedding_change'].mean():.3f} +/- {subset['embedding_change'].std():.3f}")

print("\n--- Stable vs Progressive ---")
stable = df[df['vessels_new_disease'] == 0]
progressed = df[df['vessels_new_disease'] > 0]
print(f"  Stable (0 new disease): n={len(stable)}, emb_change={stable['embedding_change'].mean():.3f}")
print(f"  Progressed (1+ new disease): n={len(progressed)}, emb_change={progressed['embedding_change'].mean():.3f}")
print(f"  Difference: {progressed['embedding_change'].mean() - stable['embedding_change'].mean():.3f}")
print(f"  T-test: t={t_stat:.2f}, p={p_val:.4f}")

plt.show()

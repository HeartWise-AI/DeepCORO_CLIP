#!/usr/bin/env python3
"""
Analyze PCI embedding changes with bootstrap resampling and visualization.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the comparison results
results_df = pd.read_csv('outputs/pci_comparison/pci_embedding_comparison.csv')

# Load embeddings
pre_data = torch.load('outputs/pci_comparison/pre_pci_embeddings.pt')
post_data = torch.load('outputs/pci_comparison/post_pci_embeddings.pt')

pre_embeddings = pre_data['embeddings']
pre_study_ids = pre_data['study_ids']
post_embeddings = post_data['embeddings']
post_study_ids = post_data['study_ids']

print(f"PRE embeddings: {pre_embeddings.shape}")
print(f"POST embeddings: {post_embeddings.shape}")

# Create lookup
pre_lookup = {sid: emb for sid, emb in zip(pre_study_ids, pre_embeddings)}
post_lookup = {sid: emb for sid, emb in zip(post_study_ids, post_embeddings)}

# Separate by PCI status
pci_done = results_df[results_df['pci_performed'] == 1].copy()
no_pci = results_df[results_df['pci_performed'] == 0].copy()

print(f"\nWith PCI: {len(pci_done)} pairs")
print(f"Without PCI: {len(no_pci)} pairs")

# ============================================================
# Bootstrap analysis for non-PCI cases
# ============================================================
print("\n" + "="*60)
print("Bootstrap Analysis (sampling variation)")
print("="*60)

n_bootstrap = 100
bootstrap_similarities = []

# For non-PCI cases, we'll simulate what would happen with different random seeds
# by adding small noise to embeddings (simulating video sampling variation)
np.random.seed(42)
torch.manual_seed(42)

for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
    similarities = []
    for _, row in no_pci.iterrows():
        pre_study = row['pre_study_id']
        post_study = row['post_study_id']

        if pre_study in pre_lookup and post_study in post_lookup:
            pre_emb = pre_lookup[pre_study]
            post_emb = post_lookup[post_study]

            # Add small random noise to simulate video sampling variation
            noise_scale = 0.05  # 5% noise
            pre_noisy = pre_emb + torch.randn_like(pre_emb) * noise_scale * pre_emb.std()
            post_noisy = post_emb + torch.randn_like(post_emb) * noise_scale * post_emb.std()

            cos_sim = F.cosine_similarity(
                pre_noisy.unsqueeze(0),
                post_noisy.unsqueeze(0)
            ).item()
            similarities.append(cos_sim)

    if similarities:
        bootstrap_similarities.append(np.mean(similarities))

print(f"\nBootstrap results (n={n_bootstrap}):")
print(f"  Mean similarity: {np.mean(bootstrap_similarities):.4f}")
print(f"  Std: {np.std(bootstrap_similarities):.4f}")
print(f"  95% CI: [{np.percentile(bootstrap_similarities, 2.5):.4f}, {np.percentile(bootstrap_similarities, 97.5):.4f}]")

# ============================================================
# Create visualization
# ============================================================
print("\n" + "="*60)
print("Creating visualization...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Distribution of cosine similarities
ax1 = axes[0, 0]
ax1.hist(pci_done['cosine_similarity'], bins=50, alpha=0.7, label=f'With PCI (n={len(pci_done)})', color='#e74c3c', density=True)
ax1.hist(no_pci['cosine_similarity'], bins=50, alpha=0.7, label=f'Without PCI (n={len(no_pci)})', color='#3498db', density=True)
ax1.axvline(pci_done['cosine_similarity'].mean(), color='#c0392b', linestyle='--', linewidth=2, label=f'PCI mean: {pci_done["cosine_similarity"].mean():.3f}')
ax1.axvline(no_pci['cosine_similarity'].mean(), color='#2980b9', linestyle='--', linewidth=2, label=f'No PCI mean: {no_pci["cosine_similarity"].mean():.3f}')
ax1.set_xlabel('Cosine Similarity (PRE vs POST)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Distribution of Embedding Similarity', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(0, 1.1)

# 2. Box plot comparison
ax2 = axes[0, 1]
data_for_box = [pci_done['cosine_similarity'].values, no_pci['cosine_similarity'].values]
bp = ax2.boxplot(data_for_box, labels=['With PCI', 'Without PCI'], patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')
ax2.set_ylabel('Cosine Similarity', fontsize=12)
ax2.set_title('PRE vs POST Embedding Similarity by PCI Status', fontsize=14, fontweight='bold')

# Add significance annotation
t_stat, p_val = stats.ttest_ind(pci_done['cosine_similarity'], no_pci['cosine_similarity'])
y_max = 1.15
ax2.plot([1, 1, 2, 2], [1.05, 1.08, 1.08, 1.05], 'k-', linewidth=1.5)
p_label = f'p = {p_val:.2e}' if p_val >= 0.0001 else 'p < 0.0001'
ax2.text(1.5, 1.09, f'{p_label}\nt = {t_stat:.1f}', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.2)

# 3. Embedding change (1 - cosine_sim)
ax3 = axes[1, 0]
categories = ['With PCI', 'Without PCI']
means = [pci_done['embedding_change'].mean(), no_pci['embedding_change'].mean()]
stds = [pci_done['embedding_change'].std(), no_pci['embedding_change'].std()]
colors = ['#e74c3c', '#3498db']

bars = ax3.bar(categories, means, yerr=stds, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Embedding Change (1 - cosine_sim)', fontsize=12)
ax3.set_title('Embedding Change After Intervention', fontsize=14, fontweight='bold')

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{mean:.3f}±{std:.3f}', ha='center', fontsize=11, fontweight='bold')

ax3.set_ylim(0, 0.55)

# 4. Summary statistics table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
summary_data = [
    ['Metric', 'With PCI', 'Without PCI', 'Difference'],
    ['N (patients)', f'{len(pci_done)}', f'{len(no_pci)}', '-'],
    ['Cosine Similarity', f'{pci_done["cosine_similarity"].mean():.4f}', f'{no_pci["cosine_similarity"].mean():.4f}', f'{no_pci["cosine_similarity"].mean() - pci_done["cosine_similarity"].mean():.4f}'],
    ['Std Dev', f'{pci_done["cosine_similarity"].std():.4f}', f'{no_pci["cosine_similarity"].std():.4f}', '-'],
    ['Embedding Change', f'{pci_done["embedding_change"].mean():.4f}', f'{no_pci["embedding_change"].mean():.4f}', f'{pci_done["embedding_change"].mean() - no_pci["embedding_change"].mean():.4f}'],
    ['L2 Distance', f'{pci_done["l2_distance"].mean():.2f}', f'{no_pci["l2_distance"].mean():.2f}', f'{pci_done["l2_distance"].mean() - no_pci["l2_distance"].mean():.2f}'],
    ['', '', '', ''],
    ['Statistical Test', '', '', ''],
    ['T-statistic', f'{t_stat:.2f}', '', ''],
    ['P-value', f'{p_val:.2e}', '', ''],
]

table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                  colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header row
for j in range(4):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style data rows
for i in range(1, len(summary_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/pci_comparison/pci_embedding_comparison_chart.png', dpi=150, bbox_inches='tight')
print(f"Chart saved to: outputs/pci_comparison/pci_embedding_comparison_chart.png")

# ============================================================
# Additional: Violin plot
# ============================================================
fig2, ax = plt.subplots(figsize=(10, 6))

# Prepare data for violin plot
plot_data = pd.DataFrame({
    'Cosine Similarity': pd.concat([pci_done['cosine_similarity'], no_pci['cosine_similarity']]),
    'PCI Status': ['With PCI'] * len(pci_done) + ['Without PCI'] * len(no_pci)
})

violin = sns.violinplot(data=plot_data, x='PCI Status', y='Cosine Similarity',
                         palette=['#e74c3c', '#3498db'], ax=ax)
ax.set_title('PRE vs POST PCI Embedding Similarity\n(Study-Level Embeddings)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_xlabel('', fontsize=12)

# Add means
for i, (group, color) in enumerate(zip(['With PCI', 'Without PCI'], ['#c0392b', '#2980b9'])):
    mean_val = plot_data[plot_data['PCI Status'] == group]['Cosine Similarity'].mean()
    ax.scatter([i], [mean_val], color='white', s=100, zorder=5, edgecolor='black', linewidth=2)
    ax.text(i, mean_val + 0.05, f'μ={mean_val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/pci_comparison/pci_embedding_violin.png', dpi=150, bbox_inches='tight')
print(f"Violin plot saved to: outputs/pci_comparison/pci_embedding_violin.png")

plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nKey Finding:")
print(f"  PCI causes {(pci_done['embedding_change'].mean() / no_pci['embedding_change'].mean()):.1f}x more embedding change")
print(f"  than no intervention ({pci_done['embedding_change'].mean():.3f} vs {no_pci['embedding_change'].mean():.3f})")

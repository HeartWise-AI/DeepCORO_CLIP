import torch
import pandas as pd
from tqdm import tqdm
from utils.registry import ModelRegistry
from models.text_encoder import TextEncoder, get_tokenizer

device_idx = 0
device = torch.device(f"cuda:{device_idx}")  # Define the target device

text_encoder: TextEncoder = ModelRegistry.get(
    name="text_encoder"
)(
    freeze_ratio=0.0,
    dropout=0.0,
)
text_encoder.to(device)

checkpoint_path = "outputs/dev_deep_coro_clip_single_video/mvit_pretrained_mvit_b84_f16_AdamW_lr6.1605e-05_20250225-113553_jb9rgjv9/checkpoints/best_epoch.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
text_encoder.load_state_dict(checkpoint["text_encoder"])
text_encoder.eval()

df_dataset = pd.read_csv("data/reports/reports_with_alpha_separator_with_Calcifc_Stenosis_IFR_20250108_latest_robert.csv", sep='α')
tokenizer = get_tokenizer()

# Process reports in batches for efficiency
batch_size = 64  # Adjust based on your GPU memory and performance needs
reports = df_dataset["Report"].tolist()

vessel_labels=[
    "leftmain",
    "lad",
    "mid_lad",
    "dist_lad",
    "diagonal",
    "D2",
    "D3",
    "lcx",
    "dist_lcx",
    "lvp",
    "marg_d",
    "om1",
    "om2",
    "om3",
    "prox_rca",
    "mid_rca",
    "dist_rca",
    "RVG1",
    "RVG2",
    "pda",
    "posterolateral",
    "bx",
    "lima_or_svg",
]

categories = ['stenosis', 'IFRHYPEREMIE', 'calcif']
dominance = ['coronary_dominance']

# Prepare a list to collect embeddings in the same order as the reports in df_dataset
embeddings_list = []
metadata_list = []   # Added metadata_list for corresponding f'{label}_{category}' values

for i in tqdm(range(0, len(reports), batch_size), desc="Generating text embeddings"):
    batch_texts = reports[i:i+batch_size]
    tokens = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    # Move tokens to the target device
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        text_features = text_encoder(tokens["input_ids"], tokens["attention_mask"])

    # Append the batch of embeddings as a tensor to embeddings_list
    embeddings_list.append(text_features.cpu())

    # Process each embedding for metadata
    for j in range(text_features.size(0)):
        meta_dict = {}
        for label in vessel_labels:
            for c in categories:
                col_name = f'{label}_{c}'
                meta_dict[col_name] = df_dataset[col_name].iloc[i + j]
        metadata_list.append(meta_dict)
        for c in dominance:
            col_name = f'{c}'
            meta_dict[col_name] = df_dataset[col_name].iloc[i + j]
        metadata_list.append(meta_dict)

# Save the embeddings to a pt file
embeddings_tensor = torch.cat(embeddings_list, dim=0)
print(embeddings_tensor.shape)
torch.save(embeddings_tensor, "utils/inference/reports_embeddings.pt")

# Save metadata to a parquet file
df_metadata = pd.DataFrame(metadata_list)
df_metadata.to_parquet("utils/inference/reports_metadata.parquet")

print(f"Text embeddings saved to utils/inference/reports_embeddings.pt")
print(f"Metadata saved to utils/inference/reports_metadata.parquet")
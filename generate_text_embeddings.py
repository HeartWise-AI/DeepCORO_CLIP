import torch
import pickle
import pandas as pd
from tqdm import tqdm
from utils.registry import ModelRegistry
from models.text_encoder import TextEncoder, get_tokenizer

device_idx = 1
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

# Initialize the dictionary to store embeddings: key is the report text, value is its embedding vector
report_embeddings = {}

for i in tqdm(range(0, len(reports), batch_size), desc="Generating text embeddings"):
    batch_texts = reports[i:i+batch_size]
    tokens = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    # Convert all token tensors to the target device
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():  # Disable gradients for inference
        text_features = text_encoder(tokens["input_ids"], tokens["attention_mask"])

    # Convert the tensor to a CPU numpy array for storage
    text_features_np = text_features.cpu().numpy()
    # Map each report text to its corresponding embedding vector
    for report, embedding in zip(batch_texts, text_features_np):
        report_embeddings[report] = embedding

# Save the embeddings to a pickle file
with open("utils/inference/reports_embeddings.pkl", "wb") as f:
    pickle.dump(report_embeddings, f)

print(f"Text embeddings saved to utils/inference/reports_embeddings.pkl")
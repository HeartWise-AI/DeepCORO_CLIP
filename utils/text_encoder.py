import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer

def create_global_text_pool(train_dataset, val_dataset, test_dataset=None):
    """
    Create a global pool of text reports from train, val, and optionally test sets.
    Ensures each report is included and assign them global indices.
    If duplicates need to be removed, we can use a set or dict.

    Args:
        train_dataset: Dataset instance for training set
        val_dataset: Dataset instance for validation set
        test_dataset: Dataset instance for test set (optional)

    Returns:
        all_global_reports: A list of all reports from train, val (and test if given)
    """
    train_reports = train_dataset.dataset.get_all_reports()
    val_reports = val_dataset.dataset.get_all_reports()
    test_reports = test_dataset.dataset.get_all_reports() if test_dataset is not None else []

    # If you want uniqueness:
    # unique_reports = list(set(train_reports + val_reports + test_reports))
    # But for consistent indexing, might need order:
    # Use a dict to preserve order:
    seen = {}
    for r in train_reports:
        if r not in seen:
            seen[r] = len(seen)
    for r in val_reports:
        if r not in seen:
            seen[r] = len(seen)
    for r in test_reports:
        if r not in seen:
            seen[r] = len(seen)

    # Convert keys to a list ordered by insertion
    all_global_reports = [None] * len(seen)
    for report, idx in seen.items():
        all_global_reports[idx] = report

    return all_global_reports


def precompute_global_text_embeddings(
    text_encoder: nn.Module, 
    all_global_reports: List[str], 
    tokenizer: AutoTokenizer, 
    device: torch.device, 
    batch_size: int = 64, 
    num_workers: int = 4
):
    """
    Precompute embeddings for a global set of reports.

    Args:
        text_encoder: The text encoder model
        all_global_reports: A list of all global reports
        tokenizer: The tokenizer associated with the text encoder
        device: Torch device
        batch_size: Batch size for encoding
        num_workers: Number of workers for DataLoader

    Returns:
        all_global_reports: same list as input
        all_global_text_embeddings: normalized embeddings for all reports
    """
    from torch.utils.data import DataLoader, Dataset

    class GlobalTextDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            return encoded

    text_dataset = GlobalTextDataset(all_global_reports, tokenizer)
    text_loader = DataLoader(
        text_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    all_text_embeddings = []
    text_encoder.eval()

    with torch.no_grad():
        for batch_texts in text_loader:
            input_ids = batch_texts["input_ids"].to(device)
            attention_mask = batch_texts["attention_mask"].to(device)
            text_features = text_encoder(input_ids, attention_mask)
            text_features = nn.functional.normalize(text_features, dim=1)
            all_text_embeddings.append(text_features.cpu())

    all_global_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    return all_global_reports, all_global_text_embeddings

def get_best_and_worst_retrievals(similarity_matrix, paths, reports, k=2):
    """Get the best and worst retrievals based on similarity scores, along with their top text matches.

    Args:
        similarity_matrix: Tensor of shape (num_videos, num_queries)
        paths: List of video paths
        reports: List of report texts
        k: Number of best/worst examples to return

    Returns:
        tuple: (best_indices, worst_indices, best_scores, worst_scores, best_text_indices, worst_text_indices)
    """
    # Get mean similarity score for each video-query pair
    mean_similarities = similarity_matrix.mean(dim=1)

    # Adjust k to not exceed batch size
    k = min(k, len(mean_similarities))

    # Get indices of best and worst k videos
    best_values, best_indices = torch.topk(mean_similarities, k=k)
    worst_values, worst_indices = torch.topk(mean_similarities, k=k, largest=False)

    # Get top-5 text matches for each video
    best_text_indices = []
    worst_text_indices = []

    for idx in best_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        _, top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)
        best_text_indices.append(top_n_texts)

    for idx in worst_indices:
        # Get top N text matches for this video, where N is min(5, batch_size)
        n_texts = min(5, similarity_matrix.size(1))
        _, top_n_texts = torch.topk(similarity_matrix[idx], k=n_texts)
        worst_text_indices.append(top_n_texts)

    return (
        best_indices,
        worst_indices,
        best_values,
        worst_values,
        best_text_indices,
        worst_text_indices,
    )
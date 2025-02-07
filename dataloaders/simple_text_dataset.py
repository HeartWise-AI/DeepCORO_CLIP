from torch.utils.data import Dataset


class SimpleTextDataset(Dataset):
    """Allow me to encode all reportsi n the valdiation dataset at once for validation metrics"""

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
        # Squeeze to remove batch dimension
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded
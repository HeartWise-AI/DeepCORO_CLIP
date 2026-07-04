import itertools
import os
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm


def resolve_inference_device(device_id: Any) -> torch.device:
    if isinstance(device_id, torch.device):
        return device_id
    if isinstance(device_id, str):
        try:
            device = torch.device(device_id)
        except RuntimeError:
            device = torch.device("cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def load_text_embeddings(path: str, device: torch.device) -> torch.Tensor:
    if not path:
        raise ValueError("config.text_embeddings_path is required for inference")

    loaded = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        for key in ("text_embeddings", "embeddings", "features"):
            if key in loaded:
                loaded = loaded[key]
                break

    if not isinstance(loaded, torch.Tensor):
        raise TypeError(
            "text_embeddings_path must contain a tensor or a dict with "
            "'text_embeddings', 'embeddings', or 'features'"
        )
    if loaded.ndim != 2:
        raise ValueError(f"text embeddings must be 2D [M, D], got {tuple(loaded.shape)}")
    return loaded.to(device=device, dtype=torch.float32)


def extract_video_embeddings(video_outputs: Any, video_mask: Any = None) -> torch.Tensor:
    if isinstance(video_outputs, dict):
        for key in ("video_embeds", "study_features", "video_features", "embeddings"):
            value = video_outputs.get(key)
            if isinstance(value, torch.Tensor):
                video_outputs = value
                break
        else:
            raise KeyError(
                "video encoder output dict must contain one of: "
                "video_embeds, study_features, video_features, embeddings"
            )
    elif isinstance(video_outputs, (tuple, list)):
        video_outputs = video_outputs[0]

    if not isinstance(video_outputs, torch.Tensor):
        raise TypeError(f"video encoder returned unsupported type {type(video_outputs)}")
    if video_outputs.ndim == 3:
        # [B, N, D] -> [B, D]. Average ONLY valid clips: padded (zero) clips would
        # otherwise pull sparse-view embeddings toward zero and corrupt retrieval.
        if video_mask is not None:
            m = video_mask.to(device=video_outputs.device, dtype=video_outputs.dtype)
            if m.shape == video_outputs.shape[:2]:
                m = m.unsqueeze(-1)
                video_outputs = (video_outputs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                video_outputs = video_outputs.mean(dim=1)
        else:
            video_outputs = video_outputs.mean(dim=1)
    if video_outputs.ndim != 2:
        raise ValueError(f"video embeddings must be 2D [B, D], got {tuple(video_outputs.shape)}")
    return video_outputs


def build_inference_rows(
    *,
    topk_indices: torch.Tensor,
    identifiers: list[Any],
    metadata: pd.DataFrame,
    dataset: Any,
    groupby_col_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for row_idx, top_k_meta_indices in enumerate(topk_indices.numpy()):
        identifier = identifiers[row_idx] if row_idx < len(identifiers) else row_idx
        topk_metadata = metadata.iloc[top_k_meta_indices]

        row = {}
        if getattr(dataset, "multi_video_mode", False):
            actual_video_filenames = dataset.get_video_paths(identifier)
            row[groupby_col_name] = identifier
            row["video_filenames"] = ";".join(actual_video_filenames)
        else:
            row["video_name"] = identifier

        for column in metadata.columns:
            values = topk_metadata[column]
            if pd.api.types.is_numeric_dtype(values):
                row[column] = values.mean()
            elif pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
                modes = values.dropna().mode()
                row[column] = None if modes.empty else modes.iloc[0]
            else:
                non_null = values.dropna()
                row[column] = None if non_null.empty else non_null.iloc[0]
        rows.append(row)
    return rows


def gather_inference_rows(local_rows: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return local_rows

    gathered_rows: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered_rows, local_rows)
    return list(itertools.chain.from_iterable(gathered_rows))


def save_inference_rows(
    *,
    rows: list[dict[str, Any]],
    dataset: Any,
    groupby_col_name: str,
    output_dir: str,
) -> None:
    if output_dir is None:
        raise ValueError("output_dir or config.inference_results_path is required for inference")

    averaged_metadata_df = pd.DataFrame(rows)
    if not averaged_metadata_df.empty:
        if getattr(dataset, "multi_video_mode", False):
            cols_prefix = [groupby_col_name, "video_filenames"]
        else:
            cols_prefix = ["video_name"]
        remaining_cols = [col for col in averaged_metadata_df.columns if col not in cols_prefix]
        averaged_metadata_df = averaged_metadata_df[cols_prefix + remaining_cols]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "averaged_metadata.csv")
    averaged_metadata_df.to_csv(output_path, index=False)
    print(f"Saved averaged metadata to: {output_path}")
    print("Inference completed")


def _coerce_identifiers(identifiers: Any) -> list[Any]:
    if identifiers is None:
        return []
    if isinstance(identifiers, (str, bytes)):
        return [identifiers]
    if isinstance(identifiers, torch.Tensor):
        return identifiers.detach().cpu().tolist()
    return list(identifiers)


def run_retrieval_metadata_inference(
    *,
    video_encoder: Any,
    val_loader: Any,
    config: Any,
    device_id: Any,
    world_size: int,
    output_dir: str | None,
) -> list[dict[str, Any]]:
    if val_loader is None:
        raise ValueError("Inference requires val_loader")
    if video_encoder is None:
        raise ValueError("Inference requires video_encoder")
    if not getattr(config, "metadata_path", None):
        raise ValueError("config.metadata_path is required for inference")

    device = resolve_inference_device(device_id)
    text_embeddings = load_text_embeddings(config.text_embeddings_path, device)
    metadata = pd.read_parquet(config.metadata_path)
    if metadata.empty:
        raise ValueError(f"metadata_path has no rows: {config.metadata_path}")

    candidate_count = min(text_embeddings.shape[0], len(metadata))
    if candidate_count == 0:
        raise ValueError("No text embeddings or metadata rows available for inference")
    text_embeddings = text_embeddings[:candidate_count]
    metadata = metadata.iloc[:candidate_count]

    topk = min(int(getattr(config, "topk", 1)), candidate_count)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    dataset = val_loader.dataset
    groupby_col_name = getattr(config, "groupby_column", None) or "study_id"
    inference_rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda"

    video_encoder.eval()
    for batch in tqdm(
        val_loader,
        desc=f"[GPU {device_id}] Running inference",
        disable=not getattr(config, "is_ref_device", False),
    ):
        if not isinstance(batch, dict):
            raise TypeError(f"inference batches must be dicts, got {type(batch)}")
        videos = batch.get("videos")
        if videos is None:
            continue

        with torch.no_grad():
            vids = videos.to(device).float()
            # Padded clips are exact-zero; build a per-clip validity mask for masked
            # averaging of multi-video [B, N, ...] inputs.
            video_mask = None
            if vids.dim() >= 3 and vids.shape[1] > 1:
                video_mask = vids.reshape(vids.shape[0], vids.shape[1], -1).abs().amax(dim=-1) > 0
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                video_outputs = video_encoder(vids)
            video_embeddings = extract_video_embeddings(video_outputs, video_mask).to(device=device, dtype=torch.float32)

        similarity_matrix = torch.matmul(video_embeddings, text_embeddings.t())
        _, topk_indices = torch.topk(similarity_matrix, k=topk, dim=1)
        identifiers = _coerce_identifiers(batch.get("paths", batch.get("sids")))
        inference_rows.extend(
            build_inference_rows(
                topk_indices=topk_indices.cpu(),
                identifiers=identifiers,
                metadata=metadata,
                dataset=dataset,
                groupby_col_name=groupby_col_name,
            )
        )

    inference_rows = gather_inference_rows(inference_rows, world_size)
    if getattr(config, "is_ref_device", False):
        save_inference_rows(
            rows=inference_rows,
            dataset=dataset,
            groupby_col_name=groupby_col_name,
            output_dir=output_dir or getattr(config, "inference_results_path", None),
        )
    return inference_rows

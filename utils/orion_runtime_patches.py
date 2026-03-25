from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyVasoVisionHead(nn.Module):
    """
    Head layout used by the published VasoVision checkpoint.
    """

    def __init__(self, dim_in: int, num_classes: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Conv3d(dim_in, 2048, bias=True, kernel_size=1, stride=1)
        self.regress = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = x.mean([2, 3, 4])
        x = self.regress(x)
        return x


class LegacyVasoVisionMultiOutputHead(nn.Module):
    """
    Multi-head wrapper that matches the state_dict layout in `vaso_vision.pt`.
    """

    def __init__(
        self,
        dim_in: int,
        head_structure: dict[str, int],
        head_task: dict[str, str] | None = None,
        regression_head_bias: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.heads = nn.ModuleDict(
            {
                head_name: nn.Sequential(LegacyVasoVisionHead(dim_in, num_classes))
                for head_name, num_classes in head_structure.items()
            }
        )
        self.head_structure = head_structure
        self.head_task = head_task or {}
        self.regression_head_bias = regression_head_bias or {}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.dropout(x)
        return {head_name: head_module(x) for head_name, head_module in self.heads.items()}


def _build_model_with_safe_checkpoint_loading(vte_module):
    def patched_build_model(config, device, model_path=None, for_inference=False):
        frames = config.get("frames", 16)
        resize = config.get("resize", 224)
        if config["model_name"].startswith("swin3d"):
            if frames % 2 != 0:
                raise ValueError("swin3d supports only frame counts that are multiples of 2.")
        elif config["model_name"].startswith("x3d"):
            if frames % 8 != 0:
                raise ValueError("x3d models support frame counts that are multiples of 8.")
            if config["model_name"] == "x3d_m" and resize not in [224, 256]:
                raise ValueError("x3d_m supports video values of either 224x224 or 256x256.")
        elif config["model_name"].startswith("mvit"):
            if frames != 16:
                raise ValueError("mvit supports only 16 frames.")

        if config["model_name"] not in ["x3d_m", "videopairclassifier"] and resize != 224:
            print(f"Warning: Resize value {resize} is not 224. Setting to default 224x224.")
            config["resize"] = 224

        model = vte_module.load_and_modify_model(config)
        labels_map = config.get("labels_map", None)
        print(labels_map)

        if config["task"] == "classification" and not labels_map:
            raise ValueError("labels_map is not defined in the config file.")

        if (model_path and config["resume"]) or for_inference:
            print("Device in use:", device)

            map_location = device if device.type == "cpu" else None

            if not for_inference and model_path:
                print("Loading checkpoint:", model_path)
                if device.type == "cuda":
                    map_location = {f"cuda:{0}": f"cuda:{device.index}"}
                else:
                    map_location = device
                print("Map location set to:", map_location)
            elif not model_path:
                model_path = vte_module.os.path.join(config["output_dir"], "best.pt")

            if map_location is not None:
                checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
            else:
                checkpoint = torch.load(model_path, weights_only=True)

            try:
                model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))

                if any(k.startswith("_orig_mod.module.") for k in model_state_dict.keys()):
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                        model_state_dict, "_orig_mod.module."
                    )
                    print("Removed prefix '_orig_mod.module.' from state dict")
                elif any(k.startswith("module.") for k in model_state_dict.keys()):
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                        model_state_dict, "module."
                    )
                    print("Removed prefix 'module.' from state dict")
            except RuntimeError as e:
                print(f"Error loading model state dict: {e}")
                raise

            model.load_state_dict(model_state_dict)
            print("Model loaded successfully")
            model.to(device)
            if for_inference is False:
                find_unused = config.get("task") == "masked_video_modeling"
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device.index],
                    output_device=device.index,
                    find_unused_parameters=find_unused,
                )

            optimizer_state = checkpoint.get("optimizer_state_dict")
            print("Optimizer state loaded successfully")
            scheduler_state = checkpoint.get("scheduler_state_dict")
            print("Scheduler state loaded successfully")
            epoch = checkpoint.get("epoch", 0)
            print("Epoch loaded successfully")
            best_loss = checkpoint.get("best_loss", float("inf"))
            other_metrics = {
                k: v
                for k, v in checkpoint.items()
                if k not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"]
            }
            print("Other metrics loaded successfully")

            return model, optimizer_state, scheduler_state, epoch, best_loss, other_metrics, labels_map

        model.to(device)
        find_unused = config.get("task") == "masked_video_modeling"
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused,
        )
        return model, None, None, 0, float("inf"), {}, labels_map

    return patched_build_model


def _filter_kwargs_for_callable(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    valid_names = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in valid_names}


def _build_compatible_load_dataset(vte_module):
    def patched_load_dataset(split, config, transforms, weighted_sampling):
        missing_fields = []
        if config["mean"] is None:
            missing_fields.append("mean")
        if config["std"] is None:
            missing_fields.append("std")

        if missing_fields:
            raise ValueError(f"Error: The following fields are missing: {', '.join(missing_fields)}")

        target_label = config.get("label_loc_label", None)
        head_structure = config.get("head_structure", None)

        if head_structure is not None and target_label is not None:
            if sorted(list(head_structure.keys())) != sorted(target_label) and split == "train":
                print(f"head_structure keys: {sorted(list(head_structure.keys()))}")
                print(f"target_label: {sorted(target_label)}")
                raise ValueError("Error: head_structure keys do not match target_label columns")

        kwargs = {
            "target_label": target_label,
            "mean": config["mean"],
            "std": config["std"],
            "length": config["frames"],
            "period": config["period"],
            "root": config["root"],
            "data_filename": config["data_filename"],
            "datapoint_loc_label": config.get("datapoint_loc_label", None),
            "apply_mask": config.get("apply_mask", False),
            "resize": config["resize"],
            "regression_value_scale": config.get("regression_value_scale", None),
            "head_task": config.get("head_task", None),
            "max_videos": config.get("max_videos", None),
            "use_cached_hog": config.get("use_cached_hog", False),
            "hog_cache_dir": config.get("hog_cache_dir", "hog_features_cache"),
        }

        if split == "train":
            print(f"[Dataset] apply_mask = {config.get('apply_mask', False)}")

        if split != "inference":
            if config["view_count"] is None or config["view_count"] == 1:
                dataset_cls = vte_module.orion.datasets.Video
                dataset = dataset_cls(
                    split=split,
                    video_transforms=transforms,
                    weighted_sampling=weighted_sampling,
                    debug=config["debug"],
                    **_filter_kwargs_for_callable(dataset_cls.__init__, kwargs),
                )
            else:
                dataset_cls = vte_module.orion.datasets.Video_Multi
                multi_kwargs = dict(kwargs)
                multi_kwargs["view_count"] = config["view_count"]
                multi_kwargs["labels_map"] = config.get("labels_map", None)
                dataset = dataset_cls(
                    split=split,
                    video_transforms=transforms,
                    weighted_sampling=weighted_sampling,
                    debug=False,
                    **_filter_kwargs_for_callable(dataset_cls.__init__, multi_kwargs),
                )
        else:
            if config["view_count"] is None:
                print("Loading video inference dataset")
                dataset_cls = vte_module.orion.datasets.Video_inference
                dataset = dataset_cls(
                    split=split,
                    **_filter_kwargs_for_callable(dataset_cls.__init__, kwargs),
                )
                print("Video inference dataset loaded successfully")
            else:
                dataset_cls = vte_module.orion.datasets.Video_Multi_inference
                multi_inference_kwargs = dict(kwargs)
                multi_inference_kwargs["view_count"] = config["view_count"]
                dataset = dataset_cls(
                    split=split,
                    **_filter_kwargs_for_callable(dataset_cls.__init__, multi_inference_kwargs),
                )
                print("Video multi inference dataset loaded successfully")

        return dataset

    return patched_load_dataset


def apply_orion_runtime_patches() -> None:
    import orion.models as orion_models
    import orion.utils.video_training_and_eval as vte

    if getattr(vte, "_deepcoro_runtime_patched", False):
        return

    orion_models.MultiOutputHead = LegacyVasoVisionMultiOutputHead
    vte.MultiOutputHead = LegacyVasoVisionMultiOutputHead

    original_perform_inference = vte.perform_inference
    vte.build_model = _build_model_with_safe_checkpoint_loading(vte)
    vte.load_dataset = _build_compatible_load_dataset(vte)

    def patched_perform_inference(*args, **kwargs):
        original_is_built = torch.backends.cuda.is_built
        torch.backends.cuda.is_built = torch.cuda.is_available
        try:
            return original_perform_inference(*args, **kwargs)
        finally:
            torch.backends.cuda.is_built = original_is_built

    vte.perform_inference = patched_perform_inference
    vte._deepcoro_runtime_patched = True

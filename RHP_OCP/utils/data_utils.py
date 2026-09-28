import torch
from utils.logging_utils import Logger, LogLevel, log_function

@log_function(level=LogLevel.DEBUG)
def collate_fn(batch):
    """Validate and stack fixed-size point-cloud samples."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    expected_points = batch[0]["current_pc"].shape[0]
    pointwise_fields = (
        "current_pc",
        "quality",
        "orientation",
        "push_distance",
    )

    for sample_index, item in enumerate(batch):
        point_count = item["current_pc"].shape[0]

        if point_count != expected_points:
            raise ValueError(
                "All point clouds in a batch must have the same number "
                f"of points: expected {expected_points}, got "
                f"{point_count} at sample {sample_index}"
            )

        for field in pointwise_fields[1:]:
            if item[field].shape[0] != point_count:
                raise ValueError(
                    f"{field} length does not match current_pc length "
                    f"at sample {sample_index}: "
                    f"{item[field].shape[0]} != {point_count}"
                )

    return {
        "current_pc": torch.stack(
            [item["current_pc"] for item in batch]
        ),
        "future_local_pose": torch.stack(
            [item["future_local_pose"] for item in batch]
        ),
        "quality": torch.stack(
            [item["quality"] for item in batch]
        ),
        "orientation": torch.stack(
            [item["orientation"] for item in batch]
        ),
        "push_distance": torch.stack(
            [item["push_distance"] for item in batch]
        ),
    }

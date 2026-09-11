import torch

def augment_data(
    data_inputs: torch.Tensor,   # (B, L) 或 (B, 1, L)
    data_labels: torch.Tensor,   # (B,)
) -> tuple[torch.Tensor, torch.Tensor]:
    return data_inputs, data_labels
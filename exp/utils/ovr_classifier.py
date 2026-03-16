import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_backbone_output(raw_output):
    """Extract feature tensor from backbone output (handles tuple/list or plain tensor)."""
    if isinstance(raw_output, (tuple, list)):
        # Second element is typically the feature map; fall back to first
        if len(raw_output) >= 2 and isinstance(raw_output[1], torch.Tensor):
            return raw_output[1]
        return raw_output[0]
    return raw_output


def flatten_features(features):
    """Flatten features to 2-D (batch, dim) if higher-dimensional."""
    if features.dim() > 2:
        return features.flatten(1)
    return features


class OvRBinaryClassifier(nn.Module):
    """One-vs-Rest binary classifier head on top of a shared backbone."""

    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.binary_heads = nn.ModuleList(
            [nn.Linear(feature_dim, 1) for _ in range(num_classes)]
        )

    def forward(self, x):
        features = unpack_backbone_output(self.backbone(x))
        features = flatten_features(features)
        logits = torch.cat([head(features) for head in self.binary_heads], dim=1)
        return logits, features

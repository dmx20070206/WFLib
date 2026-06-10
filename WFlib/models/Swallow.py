import torch
import torch.nn as nn
import torchvision.models as tv_models


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size=512, projection_size=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class SwallowEncoder(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(SwallowEncoder, self).__init__()
        if backbone_name == "resnet18":
            backbone = tv_models.resnet18(weights=None)
        elif backbone_name == "resnet34":
            backbone = tv_models.resnet34(weights=None)
        elif backbone_name == "resnet50":
            backbone = tv_models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported Swallow backbone: {backbone_name}")

        backbone.conv1 = nn.Conv2d(
            1,
            backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False,
        )
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.fc.in_features

    def forward(self, x):
        features = self.encoder(x)
        return features.view(features.shape[0], features.shape[1])


class Swallow(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet18"):
        super(Swallow, self).__init__()
        self.encoder = SwallowEncoder(backbone_name)
        self.fc = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.fc(features)
        return logits, features
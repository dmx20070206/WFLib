import torch.nn as nn
import math
import torch
import numpy as np

class DMX(nn.Module):
    def __init__(self, num_classes=100):
        """
        Initialize the DMX model.

        Parameters:
        num_classes (int): Number of output classes.
        """
        super(DMX, self).__init__()
        
        # Create feature extraction layers
        features = make_layers([128, 128, 'M', 256, 256, 'M', 512] + [num_classes])
        init_weights = True
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        
        # Create the initial convolutional layers
        self.first_layer = make_first_layers()
        self.features = features
        self.class_num = num_classes
        
        # Adaptive average pooling layer for classification
        self.classifier = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the network.
        """
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)

        # ``self.features`` ends with a num_classes-channel convolutional
        # block.  That block is part of the classifier, so it should not be
        # used as the metric-learning/prototype representation: its channels
        # are tied directly to the class count.  Keep the original Sequential
        # object and split it at runtime so old checkpoints remain compatible.
        embedding_map = x
        for layer in self.features[:-3]:
            embedding_map = layer(embedding_map)

        logits_map = embedding_map
        for layer in self.features[-3:]:
            logits_map = layer(logits_map)

        out = self.classifier(logits_map).flatten(1)
        # A 512-D vector is substantially better conditioned for prototypes
        # than flattening the class-logit feature map over all time positions.
        embedding = torch.nn.functional.adaptive_avg_pool1d(
            embedding_map, 1
        ).flatten(1)
        return out, embedding

    def logits_from_embedding(self, embedding):
        """Apply the original classifier path to a pooled penultimate feature.

        The normal forward path has a temporal feature map before the final
        class-dependent convolution.  The adapter receives its pooled
        512-dimensional representation, so use a singleton temporal position
        while reusing the exact original classifier layers and weights.
        """
        embedding = embedding.reshape(embedding.shape[0], 512, -1)
        logits_map = embedding
        for layer in self.features[-3:]:
            logits_map = layer(logits_map)
        return self.classifier(logits_map).flatten(1)

    def _initialize_weights(self):
        """
        Initialize weights for the network layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, in_channels=32):
    """
    Create a sequence of convolutional and pooling layers.

    Parameters:
    cfg (list): Configuration list specifying the layers.
    in_channels (int): Number of input channels.

    Returns:
    nn.Sequential: Sequential container with the layers.
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)

def make_first_layers(in_channels=1, out_channel=32):
    """
    Create the initial convolutional layers.

    Parameters:
    in_channels (int): Number of input channels.
    out_channel (int): Number of output channels.

    Returns:
    nn.Sequential: Sequential container with the initial layers.
    """
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)

if __name__ == '__main__':
    net = DMX(num_classes=100)
    x = np.random.rand(4, 1, 2, 1800)
    x = torch.tensor(x, dtype=torch.float32)
    out, feat = net(x)
    print(f"in:{x.shape} --> out:{out.shape}, {feat.shape}")

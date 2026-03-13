import torch.nn as nn
import math
import torch
import numpy as np


# ==========================================
# 1. 定义 Domain-Specific BN 模块
# ==========================================
class DSBatchNorm1d(nn.Module):
    def __init__(self, num_features, **kwargs):
        super(DSBatchNorm1d, self).__init__()
        # 实例化两套平行的 BN 层，卷积共享，但 BN 统计量完全隔离
        self.bn_source = nn.BatchNorm1d(num_features, **kwargs)
        self.bn_target = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x, domain="source"):
        if domain == "source":
            return self.bn_source(x)
        elif domain == "target":
            return self.bn_target(x)
        else:
            raise ValueError("domain parameter must be 'source' or 'target'")


class DSBatchNorm2d(nn.Module):
    def __init__(self, num_features, **kwargs):
        super(DSBatchNorm2d, self).__init__()
        self.bn_source = nn.BatchNorm2d(num_features, **kwargs)
        self.bn_target = nn.BatchNorm2d(num_features, **kwargs)

    def forward(self, x, domain="source"):
        if domain == "source":
            return self.bn_source(x)
        elif domain == "target":
            return self.bn_target(x)
        else:
            raise ValueError("domain parameter must be 'source' or 'target'")


# ==========================================
# 2. 修改层构建函数 (使用 ModuleList 替代 Sequential)
# ==========================================
def make_layers(cfg, in_channels=32):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            # 替换为 DSBatchNorm1d
            layers += [conv1d, DSBatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    # 必须使用 ModuleList，否则 forward 无法传递 domain 参数
    return nn.ModuleList(layers)


def make_first_layers(in_channels=1, out_channel=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    # 替换为 DSBatchNorm2d
    layers += [conv2d1, DSBatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, DSBatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, DSBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, DSBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.ModuleList(layers)


# ==========================================
# 3. 修改网络前向传播
# ==========================================
class RF(nn.Module):
    def __init__(self, num_classes=100):
        super(RF, self).__init__()

        init_weights = True
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32

        self.first_layer = make_first_layers()
        self.features = make_layers([128, 128, "M", 256, 256, "M", 512] + [num_classes])
        self.class_num = num_classes

        self.classifier = nn.AdaptiveAvgPool1d(1)

        if init_weights:
            self._initialize_weights()

    # 增加 domain 参数
    def forward(self, x, domain="source"):
        # 遍历 first_layer
        for layer in self.first_layer:
            if isinstance(layer, (DSBatchNorm1d, DSBatchNorm2d)):
                x = layer(x, domain=domain)
            else:
                x = layer(x)

        x = x.view(x.size(0), self.first_layer_out_channel, -1)

        # 遍历 features
        for layer in self.features:
            if isinstance(layer, (DSBatchNorm1d, DSBatchNorm2d)):
                x = layer(x, domain=domain)
            else:
                x = layer(x)

        out = self.classifier(x)
        out = out.view(out.size(0), -1)
        return out, x.view(x.size(0), -1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 这里的 self.modules() 会自动递归找到 DSBatchNorm2d 里面的 bn_source 和 bn_target
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    net = RF(num_classes=100)

    # 模拟源域数据
    x_source = torch.tensor(np.random.rand(4, 1, 2, 1800), dtype=torch.float32)
    out_s, feat_s = net(x_source, domain="source")
    print(f"Source pass -> out:{out_s.shape}, {feat_s.shape}")

    # 模拟目标域数据
    x_target = torch.tensor(np.random.rand(4, 1, 2, 1800), dtype=torch.float32)
    out_t, feat_t = net(x_target, domain="target")
    print(f"Target pass -> out:{out_t.shape}, {feat_t.shape}")

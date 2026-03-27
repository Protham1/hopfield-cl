import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class CIFARResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Per-task heads — each task gets its own 2-class linear layer
        # so Task N training cannot overwrite Task 0's output weights
        self.heads = nn.ModuleDict()
        self._init_weights()

    def add_head(self, task_id: int, num_classes: int = 2):
        """Call once per task before training that task."""
        head = nn.Linear(512, num_classes)
        nn.init.normal_(head.weight, 0, 0.01)
        nn.init.constant_(head.bias, 0)
        self.heads[str(task_id)] = head

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 512-dim embedding before the classification head."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    # Alias so both get_features and extract_features work
    extract_features = get_features

    def forward(self, x: torch.Tensor, task_id: int = None) -> torch.Tensor:
        feats = self.get_features(x)
        if task_id is not None:
            return self.heads[str(task_id)](feats)
        last_key = list(self.heads.keys())[-1]
        return self.heads[last_key](feats)


def build_model() -> CIFARResNet18:
    return CIFARResNet18()


if __name__ == "__main__":
    model = build_model()
    model.add_head(0, num_classes=2)
    model.add_head(1, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CIFARResNet18 — {n_params:,} trainable parameters")
    dummy = torch.randn(4, 3, 32, 32)
    print(f"Task 0 logits: {model(dummy, task_id=0).shape}")
    print(f"Task 1 logits: {model(dummy, task_id=1).shape}")
    print(f"Features:      {model.get_features(dummy).shape}")
    print("Model OK.")
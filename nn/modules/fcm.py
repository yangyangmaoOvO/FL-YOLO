import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(self.avg_pool(x))

class FCM(nn.Module):
    def __init__(self, dim, alpha, dim_out=None):
        super().__init__()
        print(f"[FCM __init__] 设定的 dim: {dim}")
        dim_out = dim if dim_out is None else dim_out
        self.one = int(dim * alpha)
        self.two = dim - self.one
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim_out, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)
        # 添加权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        print(f"[FCM forward] 实际输入 x.shape: {x.shape}")
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        x5 = self.conv3(x5)
        return x5
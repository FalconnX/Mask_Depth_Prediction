# Modified code of https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/3e758ab7bb0b7f8ba4e53d808b75f627cf7d8e2f/model.py


import torch
from torch import nn, optim
import torch.nn.functional as F

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=6):   
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)  # 112, 112
        x_112 = x
        x = self.layer2(x)  # 56 , 56
        x_56 = x 
        x = self.layer3(x)  # 28, 28
        x_28 = x
        x = self.layer4(x)  # 14 , 14 
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # mu = x[:, :self.z_dim]
        # logvar = x[:, self.z_dim:]
        # return mu, logvar
        return x, x_112 , x_56 , x_28

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 512

        # self.linear = nn.Linear(z_dim, 512)
        self.linear = nn.Linear(z_dim*2, 512)
        self.conv5 = ResizeConv2d(512, 512, kernel_size=3, scale_factor=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, 3, kernel_size=3, scale_factor=2)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv_out = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
            

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z,x_112 , x_56 , x_28):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.layer4(x) # 28, 28
        x = x + x_28
        x = self.layer3(x) # 56, 56
        x = x + x_56
        x = self.layer2(x)  # 112 , 112
        x = x + x_112
        x = self.layer1(x)  # 112, 112
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.conv_out(x)

        # x = torch.sigmoid(self.conv1(x))
        # x = x.view(x.size(0), 3, 112, 112)
        return x

class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim,nc=6)
        self.decoder1 = ResNet18Dec(z_dim=z_dim,nc=1)
        self.decoder2 = ResNet18Dec(z_dim=z_dim,nc=1)

    def forward(self, bg, bgfg_image):
        # bg_image = x["bg_image"]
        # bgfg_image = x["bgfg_image"] 
        x= torch.cat([bg,bgfg_image], dim=1)
        # mean, logvar = self.encoder(x)
        # z = self.reparameterize(mean, logvar)
        z, x_112 , x_56 , x_28 = self.encoder(x)
        mask_pred = self.decoder1(z, x_112 , x_56 , x_28)
        depth_pred = self.decoder2(z, x_112 , x_56 , x_28)
        return mask_pred,depth_pred
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

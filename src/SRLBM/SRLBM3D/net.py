import torch
from torch import nn
from torchview import draw_graph
import time

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,middle_channels=None,kernel_size=3):
        super(DoubleConv, self).__init__()
        if middle_channels is None:
            middle_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=kernel_size, padding=1,bias=False),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=kernel_size, padding=1,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels,middle_channels=None,kernel_size=3):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool3d(2,stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels,middle_channels,kernel_size)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels,middle_channels=None,kernel_size=3,bilinear = False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) # 通道数太多
            self.doubleconv = DoubleConv(in_channels, out_channels, middle_channels,kernel_size)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.doubleconv = DoubleConv(in_channels, out_channels, middle_channels,kernel_size)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1,x2], dim=1)
        x = self.doubleconv(x)
        return x
class GeoInput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeoInput, self).__init__()
        self.doubleconv = DoubleConv(in_channels, out_channels)
    def forward(self,x):
        x = self.doubleconv(x)
        return x
    
class PopInput19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PopInput19, self).__init__()
        self.maxpool = nn.MaxPool3d(2,stride=2)
        self.doubleconv = DoubleConv(in_channels + 19, out_channels)
    def forward(self,x_pop,x_geo):
        x_geo = self.maxpool(x_geo)
        x = torch.cat([x_geo,x_pop],dim=1)
        x = self.doubleconv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self,x):
        x = self.conv(x)
        return x

class SRUNet3D(nn.Module):
    def __init__(self,SR_depth,Unet_depth,size_input, bilinear=False,base_channels=64):
        super(SRUNet3D, self).__init__()
        self.bilinear = bilinear
        self.SR_depth = SR_depth
        self.Unet_depth = Unet_depth
        self.downLayers = []
        self.upLayers = []
        self.downLayers.append(GeoInput(1, base_channels)) # 集合信息导入
        for i in range(1,SR_depth):
            self.downLayers.append(Down(base_channels*2**(i-1),base_channels*2**i))
        self.downLayers.append(PopInput19(base_channels*2**(SR_depth-1),base_channels*2**SR_depth)) # 人口信息导入
        for i in range(SR_depth,Unet_depth):
            self.downLayers.append(Down(base_channels*2**i,base_channels*2**(i+1)))
        self.downLayers = nn.ModuleList(self.downLayers)
        for i in range(Unet_depth,0,-1):
            self.upLayers.append(Up(base_channels*2**i,base_channels*2**(i-1),bilinear=bilinear))
        self.upLayers = nn.ModuleList(self.upLayers)
        self.output_layer = OutConv(base_channels,19)
    def forward(self,x_pop,x_geo):
        down_outputs = []
        x = self.downLayers[0](x_geo)
        down_outputs.append(x)
        for i in range(1,self.Unet_depth+1):
            if i==self.SR_depth:
                x = self.downLayers[i](x_pop,x)
                down_outputs.append(x)
            else:
                x = self.downLayers[i](x)
                down_outputs.append(x)
        for i in range(0,self.Unet_depth):
            x = self.upLayers[i](x,down_outputs[self.Unet_depth-i-1])
        output = self.output_layer(x)
        return output

if __name__ == "__main__":
    model = SRUNet3D(SR_depth=1,Unet_depth=2,size_input=256)
    
    x_pop = torch.randn(1,19,128,128,128)
    x_geo = torch.randn(1,1,512,512,512)
    start_time = time.time()
    output = model(x_pop,x_geo)
    end_time = time.time()
    print(f"计算时间: {end_time - start_time:.4f} 秒")

    # 准备输入
    x_pop = torch.randn(1, 19, 32, 32, 32)
    x_geo = torch.randn(1, 1, 64, 64, 64)

    # 最简形式 - 直接打印总参数量
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 带可训练参数
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()

        dim = int(out_channels // factor)

        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

        self.deconv1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), dilation= 1)
        self.deconv2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), dilation= 1)
        self.deconv3 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), dilation= 1)
        self.deconv4 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), dilation= 1)

        self.deconv11 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 2), dilation= 2)
        self.deconv22 = nn.Conv2d(dim, dim, (3, 1), padding=(2, 0), dilation= 2)
        self.deconv33 = nn.Conv2d(dim, dim, (3, 1), padding=(2, 0), dilation= 2)
        self.deconv44 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 2), dilation= 2)

        self.deconv111 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 5), dilation= 5)
        self.deconv222 = nn.Conv2d(dim, dim, (3, 1), padding=(5, 0), dilation= 5)
        self.deconv333 = nn.Conv2d(dim, dim, (3, 1), padding=(5, 0), dilation= 5)
        self.deconv444 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 5), dilation= 5)

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, dilation=1, padding=0)

        self.spatial_attention = SpatialAttentionModule()

        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x3):
        
        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.down(x_fused)

        x_1 = self.deconv1(x_fused)
        x_11 = self.deconv11(x_1)
        x_111 = self.deconv111(x_11)
        x_1111 = self.fuse(x_1+x_11+x_111)

        x_2 = self.deconv2(x_fused)
        x_22 = self.deconv22(x_2)
        x_222 = self.deconv222(x_22)
        x_2222 = self.fuse(x_2+x_22+x_222)

        x_3 = self.inv_h_transform(self.deconv3(self.h_transform(x_fused)))
        x_33 = self.inv_h_transform(self.deconv33(self.h_transform(x_3)))
        x_333 = self.inv_h_transform(self.deconv333(self.h_transform(x_33)))
        x_3333 = self.fuse(x_3+x_33+x_333)

        x_4 = self.inv_v_transform(self.deconv4(self.v_transform(x_fused)))
        x_44 = self.inv_v_transform(self.deconv44(self.v_transform(x_4)))
        x_444 = self.inv_v_transform(self.deconv444(self.v_transform(x_44)))
        x_4444 = self.fuse(x_4+x_44+x_444)

        x_fused_s = x_1111 + x_2222 + x_3333 + x_4444
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s)

        return x_out


    def h_transform(self, x):
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x

    def inv_h_transform(self, x):
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1).contiguous()
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x

    def v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1)
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x.permute(0, 1, 3, 2)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class MDFE(nn.Module):
    def __init__(self, channels1, channels2, channels3):
        super(MDFE, self).__init__()
        all_channels = channels1 + channels2 + channels3
        self.fusion_conv1 = FusionConv(all_channels, channels1)
        self.fusion_conv2 = FusionConv(all_channels, channels2)
        self.fusion_conv3 = FusionConv(all_channels, channels3)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=True)
        x13 = F.interpolate(x3, scale_factor=4, mode="bilinear", align_corners=True)

        x21 = F.interpolate(x1, scale_factor=0.5, mode="bilinear", align_corners=True)
        x23 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=True)

        x31 = F.interpolate(x1, scale_factor=0.25, mode="bilinear", align_corners=True)
        x32 = F.interpolate(x2, scale_factor=0.5, mode="bilinear", align_corners=True)

        x_fused1 = self.fusion_conv1(x1, x12, x13)
        x_fused2 = self.fusion_conv2(x21, x2, x23)
        x_fused3 = self.fusion_conv3(x31, x32, x3)

        return x_fused1, x_fused2, x_fused3
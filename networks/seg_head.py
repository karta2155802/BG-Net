import torch
import torch.nn as nn
import torch.nn.functional as F

################
# part-seg net
################
class SegmHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, class_dim=1, upsample=False):
        super(SegmHead, self).__init__()
        if hidden_dim == None:
            hidden_dim = in_dim // 2
        if upsample == True:
            # upsample features
            self.double_conv1 = UpSampler(in_dim, hidden_dim)
        else:
            self.double_conv1 = DoubleConv(in_dim, hidden_dim)

        segm_net = DoubleConv(hidden_dim, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.double_conv2 = segm_net

    def map2labels(self, segm_hand):
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand

    def forward(self, img_feat):
        seg_mask = self.double_conv2(self.double_conv1(img_feat))
        return F.sigmoid(seg_mask)

#################
# basic modules
#################
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #self.up = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)
            #self.up = F.interpolate(scale_factor=(2, 2), mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=(2, 2), mode="bilinear", align_corners=True)
        else:
            x1 = self.up(x1)
        return self.conv(x1)

class UpSampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up1 = Up(in_dim, out_dim)

    def forward(self, x):
        x = self.up1(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def Hadamard_product(features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape

        normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)

        features = features.reshape(batch_size, -1, height*width)

        attended_features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        attended_features = attended_features.transpose(2,1)

        return attended_features
import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class obj_regHead(nn.Module):
    def __init__(self, channels=256, inter_channels=None, joint_nb=21):
        super(obj_regHead, self).__init__()
        if inter_channels is None:
            inter_channels = channels // 2

        self.conv = nn.Conv2d(channels+1, channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.residual1 = Residual(channels, channels)
        self.residual2 = Residual(channels, channels)
        self.residual3 = Residual(channels, channels)
        # out conv regression
        self.out_conv = nn.Conv2d(channels, joint_nb*3, kernel_size=1, stride=1, padding=0, bias=True)
        #nn.init.constant_(self.out_conv.bias, 0)
        # activation funcs
        #self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        # out conv regression
        out = self.out_conv(x)
        return out, x


class Pose2DLayer(nn.Module):
    def __init__(self, joint_nb=21):
        super(Pose2DLayer, self).__init__()
        self.coord_norm_factor = 10
        self.num_keypoints = joint_nb

    def forward(self, output, target=None, param=None):

        nB = output.data.size(0)
        nA = 1
        nV = self.num_keypoints
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)  # (63,B*S*S)

        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))  # (B,1,S,S,21)
        x = output[nV:2 * nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2 * nV:3 * nV].transpose(0, 1).view(nB, nA, nH, nW, nV)

        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
                   view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
                   view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()  # (B,1,S,S,21)
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()

        predx = x + grid_x
        predy = y + grid_y

        predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
        predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

        out_preds = [predx, predy, conf.view(nB, nH, nW, nV)]
        return out_preds
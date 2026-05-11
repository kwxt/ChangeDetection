import torch.nn as nn
import torch.nn.functional as F
#from models.MobileNetV2 import mobilenet_v2
import torch
#Efficient+编码双轴CCAM+解码局部



# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用平均池化和最大池化来生成空间注意力图
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入进行最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 将平均池化和最大池化的结果拼接起来
        x_out = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积层生成空间注意力图
        attention = self.conv1(x_out)

        # 应用Sigmoid激活函数
        attention = self.sigmoid(attention)

        # 将空间注意力应用到输入特征图上
        out = x * attention
        return out


# 局部注意力机制模块（卷积替代MultiheadAttention）
class LocalAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(LocalAttention, self).__init__()
        # 使用卷积层来模拟局部注意力机制
        self.local_conv = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 应用卷积增强局部特征
        x_out = self.local_conv(x)
        x_out = self.relu(x_out)
        return x_out


class LocalTransformer(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(LocalTransformer, self).__init__()

        # 替换通道注意力为空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)

        # 使用局部注意力替换原先的多头注意力
        self.local_attention = LocalAttention(in_channels, kernel_size=3)  # 这里可以调整卷积的kernel_size等参数

    def forward(self, x):
        # 应用局部注意力
        x_out = self.local_attention(x)

        # 应用空间注意力
        x_out = self.spatial_attention(x_out)

        return x_out
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeAndExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# 深度可分离卷积块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(DepthwiseSeparableConv, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.se = SqueezeAndExcitation(in_channels * expansion)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.skip:
            return self.pointwise(self.se(self.depthwise(self.expand(x)))) + x
        else:
            return self.pointwise(self.se(self.depthwise(self.expand(x))))


# EfficientNet-B0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 24, stride=1, expansion=1),  # 第一个模块
            DepthwiseSeparableConv(24, 32, stride=2, expansion=6),  # 第二个模块
            DepthwiseSeparableConv(32, 96, stride=2, expansion=6),  # 第三个模块
            DepthwiseSeparableConv(96, 320, stride=2, expansion=6),  # 第四个模块
            DepthwiseSeparableConv(80, 112, stride=1, expansion=6)  # 第五个模块
        )
        self.head = nn.Sequential(
            nn.Conv2d(112, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FFM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FFM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x2 = self.conv1(x2)
        x_x = x1 * x2
        x_x = self.relu(x_x)
        x_x = x_x + x2
        x_x = x_x * x1
        x_x = self.relu(x_x)
        return x_x


class GMM(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GMM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        return out_x1, out_x2


class SqueezeDoubleConvOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SqueezeDoubleConvOld, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.acfun = nn.GELU()
        self.gmm = GMM(out_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.gmm(x)
        block_x = self.double_conv(x)
        x = self.acfun(x + block_x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.gtc = GMM(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gtc(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


import torch
import torch.nn as nn


# 双轴特征池化模块（行轴和列轴分别池化）
class BidiagonalPooling(nn.Module):
    def __init__(self, kernel_size=3):
        super(BidiagonalPooling, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # 在行轴方向进行池化
        avg_out_h = self.avg_pool(x)
        max_out_h = self.max_pool(x)

        # 在列轴方向进行池化
        avg_out_w = self.avg_pool(x.transpose(2, 3)).transpose(2, 3)
        max_out_w = self.max_pool(x.transpose(2, 3)).transpose(2, 3)

        # 将行列轴池化结果相加
        out = avg_out_h + max_out_h + avg_out_w + max_out_w
        return out


# 上下文编码模块（用双轴池化替代全局平均池化）
class ContextEncoding(nn.Module):
    def __init__(self, in_channels, context_channels, kernel_size=3):
        super(ContextEncoding, self).__init__()
        self.context_conv = nn.Conv2d(in_channels, context_channels, kernel_size=1)
        self.context_bn = nn.BatchNorm2d(context_channels)
        self.context_relu = nn.ReLU(inplace=True)
        self.encoding_conv = nn.Conv2d(context_channels, in_channels, kernel_size=1)
        self.encoding_bn = nn.BatchNorm2d(in_channels)
        self.encoding_sigmoid = nn.Sigmoid()

        # 替换全局池化为双轴池化
        self.bidiagonal_pooling = BidiagonalPooling(kernel_size)

    def forward(self, x):
        # 使用双轴池化提取上下文信息
        context = self.bidiagonal_pooling(x)
        context = self.context_conv(context)
        context = self.context_bn(context)
        context = self.context_relu(context)

        # 将上下文信息编码回原始特征空间
        encoding = self.encoding_conv(context)
        encoding = self.encoding_bn(encoding)
        encoding = self.encoding_sigmoid(encoding)

        # 将编码后的上下文信息应用到输入特征图上
        out = x * encoding
        return out


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，压缩空间信息
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # 使用Sigmoid输出通道级别的权重
        )

    def forward(self, x):
        # 通过通道注意力模块来计算权重
        return self.channel_attention(x)


# CCAM 模块：结合双轴池化和通道注意力
class CCAM(nn.Module):
    def __init__(self, in_channels, context_channels=64, kernel_size=3, reduction=16):
        super(CCAM, self).__init__()
        # 上下文编码模块
        self.context_encoding = ContextEncoding(in_channels, context_channels, kernel_size)

        # 通道注意力模块
        self.channel_attention = ChannelAttention(in_channels, reduction)

    def forward(self, x):
        # 通过上下文编码模块处理输入
        x_context = self.context_encoding(x)

        # 通过通道注意力模块处理上下文编码后的特征图
        x_ca = self.channel_attention(x_context)

        # 将注意力权重应用到上下文编码后的特征图上
        out = x_context * x_ca

        return out


class LCD_Net(nn.Module):
    def __init__(self, ):
        super(LCD_Net, self).__init__()

        efficient_net = EfficientNetB0(num_classes=1000)

        # 提取EfficientNet-B0的前五个模块
        self.inc = efficient_net.stem  # 32
        self.down1 = efficient_net.blocks[0:1]  # 16
        self.down2 = efficient_net.blocks[1:2]  # 24
        self.down3 = efficient_net.blocks[2:3]  # 40
        self.down4 = efficient_net.blocks[3:4]  # 80
        self.cont1 = SqueezeDoubleConvOld(24, 64)
        self.cont2 = SqueezeDoubleConvOld(32, 64)
        self.cont3 = SqueezeDoubleConvOld(96, 64)
        self.cont4 = SqueezeDoubleConvOld(320, 64)

        self.decoder = nn.Sequential(SqueezeDoubleConvOld(472, 64), nn.Conv2d(64, 1, 1))

        self.decoder_4 = nn.Sequential(SqueezeDoubleConvOld(64 * 2 + 1, 64))
        #self.decoder_3 = nn.Sequential(SqueezeDoubleConvOld(64 * 3 + 1, 64))
        self.decoder_3 =LocalTransformer (64 * 3 + 1, 64)
        self.decoder_2 = nn.Sequential(SqueezeDoubleConvOld(64 * 3 + 1, 64))
        #self.decoder_1 = nn.Sequential(SqueezeDoubleConvOld(64 * 3 + 1, 64))
        self.decoder_1 = LocalTransformer(64 * 3 + 1, 64)
        self.decoder_final = nn.Sequential(SqueezeDoubleConvOld(64, 64), nn.Conv2d(64, 1, 1))

        self.ccam2 = CCAM(in_channels=32)


        self.ccam3 = CCAM(in_channels=96)
        self.ccam4 = CCAM(in_channels=320)

        self.ffm = FFM(472, 472)


    def forward(self, A, B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer2_pre = self.inc(B)
        layer1_A = self.down1(layer1_pre)
        layer1_B = self.down1(layer2_pre)

        layer2_A = self.down2(layer1_A)
        layer2_B = self.down2(layer1_B)

        layer2_A = self.ccam2(layer2_A)


        layer2_B = self.ccam2(layer2_B)

        layer3_A = self.down3(layer2_A)
        layer3_B = self.down3(layer2_B)

        layer3_A = self.ccam3(layer3_A)
        layer3_B = self.ccam3(layer3_B)

        layer4_A = self.down4(layer3_A)
        layer4_B = self.down4(layer3_B)

        layer4_A = self.ccam4(layer4_A)
        layer4_B = self.ccam4(layer4_B)

        layer4_As = F.interpolate(layer4_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)  # 320
        layer3_As = F.interpolate(layer3_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)  # 96
        layer2_As = F.interpolate(layer2_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)  # 32
        layer1_As = F.interpolate(layer1_A, layer1_A.size()[2:], mode='bilinear', align_corners=True)  # 24
        layer4_Bs = F.interpolate(layer4_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer3_Bs = F.interpolate(layer3_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer2_Bs = F.interpolate(layer2_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        layer1_Bs = F.interpolate(layer1_B, layer1_A.size()[2:], mode='bilinear', align_corners=True)

        layer_As1 = torch.cat([layer1_As, layer2_As, layer3_As, layer4_As], dim=1)
        layer_Bs1 = torch.cat([layer1_Bs, layer2_Bs, layer3_Bs, layer4_Bs], dim=1)
        layer_ss = self.ffm(layer_As1, layer_Bs1)
        layer1_A = self.cont1(layer1_A)
        layer2_A = self.cont2(layer2_A)
        layer3_A = self.cont3(layer3_A)
        layer4_A = self.cont4(layer4_A)
        layer1_B = self.cont1(layer1_B)
        layer2_B = self.cont2(layer2_B)
        layer3_B = self.cont3(layer3_B)
        layer4_B = self.cont4(layer4_B)

        layer1 = torch.cat((layer1_B, layer1_A), dim=1)
        layer2 = torch.cat((layer2_B, layer2_A), dim=1)
        layer3 = torch.cat((layer3_B, layer3_A), dim=1)
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_1 = F.interpolate(layer1, layer1.size()[2:], mode='bilinear', align_corners=True)

        feature_fuse = layer_ss
        change_map = self.decoder(feature_fuse)
        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        change_map1 = F.interpolate(change_map, layer1.size()[2:], mode='bilinear', align_corners=True)

        layer4_1 = torch.cat([layer4_1, change_map1], dim=1)
        layer4_1 = self.decoder_4(layer4_1)
        print('layer4_1', layer4_1.shape)
        layer3_1 = torch.cat([layer4_1, layer3_1, change_map1], dim=1)
        layer3_1 = self.decoder_3(layer3_1)
        print('layer3_1',layer3_1.shape)
        layer2_1 = torch.cat([layer3_1, layer2_1, change_map1], dim=1)
        layer2_1 = self.decoder_2(layer2_1)
        print('layer2_1', layer2_1.shape)
        layer1_1 = torch.cat([layer2_1, layer1_1, change_map1], dim=1)
        layer1_1 = self.decoder_1(layer1_1)
        final_map = self.decoder_final(layer1_1)
        # print(final_map.shape,'fina')
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)
        return change_map, final_map

if __name__ == '__main__':
    model = LCD_Net()
    img = torch.randn(1, 3, 256, 256)
    img1 = torch.randn(1, 3, 256, 256)
    res = model(img, img1)
    print(res[0].shape)

    from thop import profile

    # mmengine_flop_count(model, (3, 512, 512), show_table=True, show_arch=True)
    flops1, params1 = profile(model, inputs=(img, img1))
    print("flops=G", flops1)
    print("parms=M", params1)
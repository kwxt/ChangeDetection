from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F
from network.MobileNetV2 import mobilenet_v2
import  torch
from torch import Tensor
# 应用CCAM模块
class ContextEncoding(nn.Module):
    def __init__(self, in_channels, context_channels):
        super(ContextEncoding, self).__init__()
        self.context_conv = nn.Conv2d(in_channels, context_channels, kernel_size=1)
        #self.context_bn = nn.InstanceNorm2d(context_channels)
        self.context_bn = nn.LayerNorm([context_channels, 1, 1])
        self.context_relu = nn.ReLU(inplace=True)
        self.encoding_conv = nn.Conv2d(context_channels, in_channels, kernel_size=1)
        self.encoding_bn = nn.LayerNorm([in_channels, 1, 1])
        self.encoding_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取全局上下文信息
        context = torch.mean(x, dim=(2, 3), keepdim=True)
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


class CCAM(nn.Module):
    def __init__(self, in_channels, context_channels, reduction=16):
        super(CCAM, self).__init__()
        # 上下文编码模块
        self.context_encoding = ContextEncoding(in_channels, context_channels)

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通过上下文编码模块处理输入
        x_context = self.context_encoding(x)

        # 通过通道注意力模块处理上下文编码后的特征图
        x_ca = self.channel_attention(x_context)

        # 将注意力权重应用到上下文编码后的特征图上
        out = x_context * x_ca

        return out




# 多头注意力机制

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        # QK^T scaled attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / query.size(-1) ** 0.5

        # Masking
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # Attention output
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # 确保 embedding 大小可以被头数整除
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention(dropout)

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        N = query.size(0)  # batch size

        # 将 embedding 切分成多个头
        query = query.view(N, -1, self.heads, self.head_dim).transpose(1, 2)  # (N, heads, seq_len, head_dim)
        key = key.view(N, -1, self.heads, self.head_dim).transpose(1, 2)  # (N, heads, seq_len, head_dim)
        value = value.view(N, -1, self.heads, self.head_dim).transpose(1, 2)  # (N, heads, seq_len, head_dim)

        # 进行注意力计算
        out, _ = self.attention(query, key, value, mask)

        # 将各个头拼接起来并通过最后的线性层
        out = out.transpose(1, 2).contiguous().view(N, -1, self.heads * self.head_dim)  # (N, seq_len, embed_size)

        out = self.fc_out(out)
        out = self.dropout(out)

        # 调整 query 张量的形状，确保它与 out 的形状匹配
        query = query.transpose(1, 2).contiguous().view(N, -1, self.heads * self.head_dim)  # 调整 query 的形状

        out = self.norm(out + query)  # Add & Norm

        return out


class AttentionWithFusedFeatures(nn.Module):
    def __init__(self, in_channels: int, embed_size: int, heads: int, dropout: float = 0.1):
        super(AttentionWithFusedFeatures, self).__init__()

        # Multi-Head Attention
        self.mha = MultiHeadAttention(embed_size, heads, dropout)

        # 1x1 convolution for reducing dimensions
        self.conv = nn.Conv2d(in_channels, embed_size, kernel_size=1)

        # Skip connection + Layer Normalization
        self.norm = nn.BatchNorm2d(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()

        # Flatten the input for multi-head attention
        x_flat = x.view(N, C, -1).transpose(1, 2)  # (N, H*W, C)

        # Apply multi-head attention
        x_fused = self.mha(x_flat, x_flat, x_flat)

        # Reshape back to spatial dimensions
        x_fused = x_fused.transpose(1, 2).view(N, C, H, W)

        # Apply convolution to reduce dimensions
        out = self.conv(x_fused)

        # Apply residual connection and normalization
        out = self.norm(out + x_fused)

        return out

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
        embedding = (x.pow(2).sum((2, 3), keepdim=True) +self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate
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
        x=self.gmm(x)
        block_x = self.double_conv(x)
        x = self.acfun(x + block_x)
        return  x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.gtc=GMM(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x=self.gtc(x)
        #print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 应用CCAM模块+多头注意力机制
class LCD_Net(nn.Module):
    def __init__(self):
        super(LCD_Net, self).__init__()

        mob = mobilenet_v2(pretrained=True)

        self.inc = mob.features[:2]  # 16
        self.down1 = mob.features[2:4]  # 24
        self.down2 = mob.features[4:7]  # 32
        self.down3 = mob.features[7:14]  # 96
        self.down4 = mob.features[14:18]  # 320

        # 初始化CCAM模块
        self.ccam2 = CCAM(64,64)  # 64 channels after down2
        self.ccam3 = CCAM(192,96)  # 192 channels after down3
        self.ccam4 = CCAM(48,48)  # 48 channels after down4

        # 多头注意力模块
        self.mha1 = AttentionWithFusedFeatures(
            in_channels=304,embed_size = 304, heads = 8, dropout = 0.1)  # 你可以调整 embed_size 和 heads 的数量
        self.mha2 = AttentionWithFusedFeatures(in_channels=240,embed_size = 240, heads = 8, dropout = 0.1)
        self.mha3 = AttentionWithFusedFeatures(in_channels=112,embed_size = 112, heads = 8, dropout = 0.1)

        self.decoder = nn.Sequential(SqueezeDoubleConvOld(304, 64), nn.Conv2d(64, 1, 1))

        self.decoder_4 = nn.Sequential(SqueezeDoubleConvOld(640 + 1, 64))
        self.decoder_3 = nn.Sequential(SqueezeDoubleConvOld(64 +192 + 1, 64))
        self.decoder_2 = nn.Sequential(SqueezeDoubleConvOld(64 * 2 + 1, 64))
        self.decoder_1 = nn.Sequential(SqueezeDoubleConvOld(64 +48 + 1, 64))
        self.decoder_final = nn.Sequential(SqueezeDoubleConvOld(64, 64), nn.Conv2d(64, 1, 1))
    def forward(self, A, B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer2_pre = self.inc(B)
        layer1_A = self.down1(layer1_pre)
        layer1_B = self.down1(layer2_pre)

        fused_layer1 = torch.cat([layer1_A, layer1_B], dim=1)  # 拼接维度为channel维度

        # 通过CCAM模块
        fused_layer1 = self.ccam4(fused_layer1)

        layer2_A = self.down2(layer1_A)
        layer2_B = self.down2(layer1_B)

        fused_layer2 = torch.cat([layer2_A, layer2_B], dim=1)  # 拼接维度为channel维度

        # 通过CCAM模块
        fused_layer2 = self.ccam2(fused_layer2)
        #print('fused_layer2:')
        #print(fused_layer2.shape)# 64

        layer3_A = self.down3(layer2_A)
        layer3_B = self.down3(layer2_B)

        fused_layer3 = torch.cat([layer3_A, layer3_B], dim=1)  # 拼接维度为channel维度

        # 通过CCAM模块
        fused_layer3 = self.ccam3(fused_layer3)
        #print('fused_layer3:')
        #print(fused_layer3.shape)# 192

        layer4_A = self.down4(layer3_A)
        layer4_B = self.down4(layer3_B)

        fused_layer4 = torch.cat([layer4_A, layer4_B], dim=1)  # 拼接维度为channel维度

        # 通过CCAM模块
        #fused_layer4 = self.ccam4(fused_layer4)
       # print('fused_layer4:')
        #print(fused_layer4.shape)# 640


        # 将图像大小保持一致方便后续拼接
        fused_layer2 = F.interpolate(fused_layer2, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        fused_layer3 = F.interpolate(fused_layer3, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        fused_layer1 = F.interpolate(fused_layer1, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        fused_layer4 = F.interpolate(fused_layer4, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        #print('fused_layer2,3,4:')
        #print(fused_layer2.shape)
        #print(fused_layer3.shape)
        #print(fused_layer4.shape)


        # 将fused_layer2和fused_layer3拼接后送入第一个多头注意力模块
        #fused_layer1_2 = torch.cat([fused_layer1, fused_layer2], dim=1)  # 64+192+48
        #fused_layer1_3 = torch.cat([fused_layer1, fused_layer3], dim=1)
        #fused_layer2_3 = torch.cat([fused_layer2, fused_layer3], dim=1)
        fused_layer = torch.cat([fused_layer1,fused_layer2, fused_layer3], dim=1)
        #print('fused_layer:',fused_layer.shape)#(2,304,128,128)
        #print('进入多头注意力机制')
        mha_out=self.mha1(fused_layer)
        #mha_out1_2 = self.mha3(fused_layer1_2)
        #mha_out1_3 = self.mha2(fused_layer1_3)
        #mha_out2_3 = self.mha1(fused_layer2_3)
        #mha_out = torch.cat([mha_out1_2, mha_out1_3,mha_out2_3], dim=1)
        #print('多头注意力机制后：')
        #print(mha_out.shape)#(2,304,128,128)

        feature_fuse = mha_out
        change_map = self.decoder(feature_fuse)#(304,64)(64,1)
        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        change_map1 = F.interpolate(change_map, layer1_A.size()[2:], mode='bilinear', align_corners=True)

        #print('changemape1',change_map1.shape)
        layer4_1 = torch.cat([fused_layer4, change_map1], dim=1) #640+1
        layer4_1 = self.decoder_4(layer4_1)
        #print('layer4_1:',layer4_1.shape)
        layer3_1 = torch.cat([layer4_1, fused_layer3, change_map1], dim=1)#64+1+192
        layer3_1 = self.decoder_3(layer3_1)
        #print('layer3_1:', layer3_1.shape)
        layer2_1 = torch.cat([layer3_1, fused_layer2, change_map1], dim=1)#64+1+64
        layer2_1 = self.decoder_2(layer2_1)
        #print('layer2_1:', layer2_1.shape)
        layer1_1 = torch.cat([layer2_1, fused_layer1, change_map1], dim=1)#64+1+48
        layer1_1 = self.decoder_1(layer1_1)
        final_map = self.decoder_final(layer1_1)
        #print(final_map.shape,'fina')
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)
        #print('final_map:',final_map.shape)
        return change_map, final_map
'''
        # 将fused_layer3和fused_layer4拼接后送入第二个多头注意力模块
        fused_layer3 = F.interpolate(fused_layer3, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        fused_layer4 = F.interpolate(fused_layer4, layer1_A.size()[2:], mode='bilinear', align_corners=True)
        fused_layer3_4 = torch.cat([fused_layer3, fused_layer4], dim=1)
        mha_out2 = self.mha2(fused_layer3_4, fused_layer3_4, fused_layer3_4)

        # 将fused_layer2和fused_layer4拼接后送入第三个多头注意力模块
        fused_layer2_4 = torch.cat([fused_layer2, fused_layer4], dim=1)
        mha_out3 = self.mha3(fused_layer2_4, fused_layer2_4, fused_layer2_4)

        # 可以选择将这三个输出进行融合，具体取决于模型设计
        out = torch.cat([mha_out1, mha_out2, mha_out3], dim=1)  # 896/更大
'''



if __name__ == '__main__':
    model = LCD_Net()
    img = torch.randn(1, 3, 512, 512)
    img1 = torch.randn(1, 3, 512, 512)
    res = model(img, img1)
    print(res[0].shape)
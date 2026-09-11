"""
train_rgbms.py
RGB-MS双流融合网络训练主程序（添加辅分支增强 + CSoP融合版）- 完整CSoP版
"""

import json
import os
import time
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter, ModelEma
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from models.efficientformer_v2 import efficientformerv2_s0
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from datetime import datetime
import warnings
import math

# 导入数据匹配模块
import data_matching

# ===================== 全局配置 =====================
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 路径配置
DATASET_ROOT = './data-RGBMS/fold5'
CSV_OUTPUT_PATH = "rgbms_samples_list.csv"
DATA_ROOT = DATASET_ROOT
DETAILED_LOG_PATH = "binded_samples_load.log"
ERROR_LOG_PATH = "binded_samples_error.log"

# 样本定义
MS_CHANNELS = ['G', 'NIR', 'R', 'RE']
RGB_CHANNEL = 'RGB'
TARGET_SPLITS = ['train', 'val']
IGNORE_SPLITS = ['test']
SUPPORTED_EXTS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
IMAGE_SIZE = 224

# 训练参数
SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
FUSION_DIM = 640
use_amp = True
use_dp = True
CLIP_GRAD = 5.0
Best_ACC = 0.0
use_ema = True
model_ema_decay = 0.9998
start_epoch = 1
file_dir = 'checkpoints/EfficientFormer_RGBMS_Aux_CSoP5/'

# 容错配置
MATCH_TOLERANCE = True
FALLBACK_MATCH = True
MISSING_CHANNEL_ALLOW = False


# ===================== 植被指数计算模块 =====================
class RGBVegetationIndices:
    """RGB图像的植被指数计算（仅NRI）"""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.global_stats = {
            'nri': {'q1': 0.05, 'q99': 0.85, 'min': 0.0, 'max': 1.0},
        }

    def percentile_clip(self, x, vi_type):
        """分指数独立百分位裁剪"""
        if vi_type == 'nri':
            q_low = self.global_stats['nri']['q1']
            q_high = self.global_stats['nri']['q99']
        return torch.clamp(x, min=q_low, max=q_high)

    def minmax_normalize(self, x, vi_type):
        """全局Min-Max归一化到[0,1]"""
        if vi_type == 'nri':
            min_val = self.global_stats['nri']['min']
            max_val = self.global_stats['nri']['max']

        x_norm = (x - min_val) / (max_val - min_val + self.eps)
        return torch.clamp(x_norm, 0.0, 1.0)

    def calculate(self, rgb_img):
        """
        从RGB图像计算NRI指数
        输入: [B,3,H,W] 归一化后的RGB图像
        输出: [B,1,H,W] NRI特征图
        """
        # 拆分RGB通道
        R = rgb_img[:, 0:1, :, :]  # 红色通道
        G = rgb_img[:, 1:2, :, :]  # 绿色通道
        B = rgb_img[:, 2:3, :, :]  # 蓝色通道

        # 计算NRI指数: R / (R + G + B)
        denominator_nri = R + G + B + self.eps
        nri_raw = R / denominator_nri

        # 百分位裁剪
        nri_clipped = self.percentile_clip(nri_raw, 'nri')

        # 全局Min-Max归一化
        nri_norm = self.minmax_normalize(nri_clipped, 'nri')

        # 输出1通道NRI特征
        vi_features = nri_norm  # [B,1,224,224]
        return vi_features


class MSVegetationIndices:
    """多光谱图像的植被指数计算（仅CVI）"""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def calculate(self, ms_img):
        """
        从多光谱数据计算CVI植被指数
        CVI = (NIR * R) / (G * G)
        ms_data: [B,4,H,W] (G, NIR, R, RE)
        """
        # 拆分通道
        G = ms_img[:, 0:1, :, :]   # 绿波段
        NIR = ms_img[:, 1:2, :, :] # 近红外
        R = ms_img[:, 2:3, :, :]   # 红波段
        RE = ms_img[:, 3:4, :, :]  # 红边（未使用）

        # 计算 CVI = (NIR × R) / (G × G)
        numerator = NIR * R
        denominator = (G ** 2) + self.eps
        cvi_raw = numerator / denominator

        # 归一化到[0,1]
        cvi = torch.clamp(cvi_raw, torch.quantile(cvi_raw, 0.01), torch.quantile(cvi_raw, 0.99))
        cvi = (cvi - cvi.min()) / (cvi.max() - cvi.min() + self.eps)

        # 输出1通道CVI特征
        vi_maps = cvi  # [B,1,224,224]
        return vi_maps


# ===================== 植被指数特征分支 =====================
class RGBVIFeatureBranch(nn.Module):
    """RGB植被指数特征分支（仅NRI）"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # 植被指数计算模块
        self.vi_calculator = RGBVegetationIndices(eps=eps)

        # 特征提取网络
        # 1→16通道，stride=2下采样到112×112
        self.vi_1_to_16_112 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        # 16→32通道，stride=2下采样到56×56
        self.vi_16_to_32_56 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb_img):
        """
        输入：RGB图像 [B,3,224,224]
        输出：
            vi_32c_56: [B,32,56,56] (原始56特征)
            vi_16c_112: [B,16,112,112] (原始112特征)
        """
        # 1. 计算1通道NRI特征图 [B,1,224,224]
        vi_maps = self.vi_calculator.calculate(rgb_img)

        # 2. 1→16通道下采样到112尺度 [B,16,112,112]
        vi_16c_112 = self.vi_1_to_16_112(vi_maps)

        # 3. 16→32通道下采样到56尺度 [B,32,56,56]
        vi_32c_56 = self.vi_16_to_32_56(vi_16c_112)

        return vi_32c_56, vi_16c_112

    def downsample_112_to_56(self, vi_112):
        """将112×112特征下采样到56×56"""
        return self.vi_16_to_32_56(vi_112)


class MSVIFeatureBranch(nn.Module):
    """多光谱植被指数特征分支（仅CVI）"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # 植被指数计算模块
        self.vi_calculator = MSVegetationIndices(eps=eps)

        # 特征提取网络（与RGB分支结构相同）
        self.vi_1_to_16_112 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.vi_16_to_32_56 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ms_img):
        """
        输入：多光谱图像 [B,4,224,224]
        输出：
            vi_32c_56: [B,32,56,56] (原始56特征)
            vi_16c_112: [B,16,112,112] (原始112特征)
        """
        # 1. 计算1通道CVI特征图 [B,1,224,224]
        vi_maps = self.vi_calculator.calculate(ms_img)

        # 2. 1→16通道下采样到112尺度 [B,16,112,112]
        vi_16c_112 = self.vi_1_to_16_112(vi_maps)

        # 3. 16→32通道下采样到56尺度 [B,32,56,56]
        vi_32c_56 = self.vi_16_to_32_56(vi_16c_112)

        return vi_32c_56, vi_16c_112

    def downsample_112_to_56(self, vi_112):
        """将112×112特征下采样到56×56"""
        return self.vi_16_to_32_56(vi_112)


# ===================== 双层门控调制模块 =====================
class DualGateModulation(nn.Module):
    """双层门控调制模块（适用于RGB和MS流）"""

    def __init__(self):
        super().__init__()

        # 第一层门控（16通道，112×112）
        self.conv_gate_112 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 第二层门控（32通道，56×56）
        self.conv_gate_56 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_gate_112[0].weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.constant_(self.conv_gate_112[0].bias, 0)
        nn.init.kaiming_normal_(self.conv_gate_56[0].weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.constant_(self.conv_gate_56[0].bias, 0)

    def forward_112(self, main_112, aux_112):
        """
        第一层门控调制（112×112尺度）
        输入：
            main_112: 主模态112×112特征 [B,16,112,112]
            aux_112: 辅助模态112×112特征 [B,16,112,112]
        输出：
            aux_mod_112: 调制后特征 [B,16,112,112]
            gate_112: 门控值
        """
        if aux_112 is None:
            aux_112 = torch.zeros_like(main_112)

        # 基于主模态生成门控权重
        gate_112 = self.conv_gate_112(main_112)  # [B,16,112,112]

        # 调制辅助特征
        aux_mod_112 = aux_112 * gate_112

        return aux_mod_112, gate_112.mean()

    def forward_56(self, main_56, aux_56):
        """
        第二层门控调制（56×56尺度）
        输入：
            main_56: 主模态56×56特征 [B,32,56,56]
            aux_56: 辅助模态56×56特征 [B,32,56,56]
        输出：
            aux_mod_56: 调制后特征 [B,32,56,56]
            gate_56: 门控值
        """
        if aux_56 is None:
            aux_56 = torch.zeros_like(main_56)

        # 基于主模态生成门控权重
        gate_56 = self.conv_gate_56(main_56)  # [B,32,56,56]

        # 调制辅助特征
        aux_mod_56 = aux_56 * gate_56

        return aux_mod_56, gate_56.mean()


# ===================== 安全融合模块 =====================
class SecurityFusion(nn.Module):
    """安全融合模块（适用于RGB和MS流）"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.concat_channels = in_channels * 2

        # 生成融合候选特征
        self.fc_fusion = nn.Conv2d(
            in_channels=self.concat_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True
        )

        # 生成门控值（小型网络）
        self.gate_net = nn.Sequential(
            nn.Conv2d(self.concat_channels, in_channels // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 4, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, 1),
            nn.Sigmoid()
        )

        # 可学习残差缩放参数
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc_fusion.weight, mode='fan_out', nonlinearity='linear')
        if self.fc_fusion.bias is not None:
            nn.init.constant_(self.fc_fusion.bias, 0)

        for m in self.gate_net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, main_feat, aux_feat):
        """
        输入：
            main_feat：主模态特征 → [B,32,56,56]
            aux_feat：辅助模态特征 → [B,32,56,56]
        输出：
            fused_feat：融合特征 → [B,32,56,56]
            fusion_gate：融合门控值
        """
        B, C, H, W = main_feat.shape

        # 步骤1：拼接主/辅模态特征
        concat_feat = torch.cat([main_feat, aux_feat], dim=1)  # [B, 64, 56, 56]

        # 步骤2：生成融合候选特征
        fusion_candidate = self.fc_fusion(concat_feat)  # [B,32,56,56]

        # 步骤3：生成融合门控值
        fusion_gate = self.gate_net(concat_feat)  # [B,1]
        fusion_gate = fusion_gate.view(B, 1, 1, 1)  # 扩展维度适配广播

        # 步骤4：残差加权融合
        fused_feat = main_feat + fusion_gate * (self.alpha * (fusion_candidate - main_feat))

        return fused_feat, fusion_gate.mean()


# ===================== 主干网络特征提取模块（与无CSoP版保持一致） =====================
class EfficientFormerV2WithFeatures(nn.Module):
    """支持中间特征提取的EfficientFormerV2（与无CSoP版完全一致）"""

    def __init__(self, base_model, image_size=224):
        super().__init__()
        # 保留base model的所有组件
        self.patch_embed = base_model.patch_embed  # stem层
        self.network = base_model.network  # 核心网络层
        self.norm = base_model.norm  # 归一化层
        self.head = base_model.head  # 分类头

        # 提取112×112中间特征
        self.stem_112_layer = nn.Sequential(*list(base_model.patch_embed.children())[:3])

        # 计算空间尺寸
        self.spatial_size = image_size // 16  # 14×14

    def forward_tokens(self, x):
        """适配原模型的forward_tokens逻辑"""
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x, fused_feat=None):
        """
        前向传播（与无CSoP版完全一致）：
        - 如果传入fused_feat，则使用融合特征替代原始stem输出
        - 否则使用原始特征
        """
        if fused_feat is not None:
            # 使用融合特征作为输入，跳过原始patch_embed
            x = fused_feat
        else:
            # 原始前向逻辑
            x = self.patch_embed(x)

        # 执行后续network和分类头
        x = self.forward_tokens(x)
        x = self.norm(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.head(x)
        return x

    def get_stem_112_feature(self, x):
        """
        获取112×112的中间特征
        输入：[B,C,224,224]
        输出：[B,16,112,112]
        """
        return self.stem_112_layer(x)

    def get_stem_56_feature(self, x):
        """
        获取56×56的中间特征（patch_embed输出）
        输入：[B,C,224,224]
        输出：[B,32,56,56]
        """
        return self.patch_embed(x)


# ===================== CBAM差分调制权重模块（更新：更严格的权重约束） =====================
class CBAMDiffModulationWeightModule(nn.Module):
    """
    基于CBAM的差分调制权重模块 (CBAM-DMWM)
    输出：W1_norm, W2_norm = [B, 176, 1, 1]，满足 W1_norm + W2_norm = 1
    """

    def __init__(self, channels=176, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # CBAM注意力模块
        self.channel_attention = ChannelAttentionModule(channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, F1, F2):
        B, C, H, W = F1.shape

        # ========== 差分特征计算 ==========
        F_diff = F1 - F2  # [B, 176, 14, 14]
        F_diff_abs = torch.abs(F_diff)  # [B, 176, 14, 14]

        # 通道归一化
        mean = torch.mean(F_diff_abs, dim=[2, 3], keepdim=True)  # [B, 176, 1, 1]
        std = torch.std(F_diff_abs, dim=[2, 3], keepdim=True)  # [B, 176, 1, 1]
        F_diff_norm = (F_diff_abs - mean) / (std + 1e-8)  # [B, 176, 14, 14]

        # ========== CBAM注意力 ==========
        M_c = self.channel_attention(F_diff_norm)  # [B, 176, 1, 1]
        F_attn = F_diff_norm * M_c  # [B, 176, 14, 14]
        M_s = self.spatial_attention(F_attn)  # [B, 1, 14, 14]

        # 注意力融合
        M_c_expanded = M_c.expand(-1, -1, H, W)  # [B, 176, 14, 14]
        M_s_expanded = M_s.expand(-1, C, -1, -1)  # [B, 176, 14, 14]
        W_spach = M_c_expanded * M_s_expanded  # [B, 176, 14, 14]

        # ========== 模态专属权重拆分 ==========
        sigmoid_F_diff = torch.sigmoid(F_diff)  # [B, 176, 14, 14]
        W1_raw = W_spach * sigmoid_F_diff  # [B, 176, 14, 14]
        W2_raw = W_spach * torch.sigmoid(-F_diff)  # [B, 176, 14, 14]

        # ========== 权重归一化（改为单值权重） ==========
        # 方案1：全局平均后Softmax
        W1_global = torch.mean(W1_raw, dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
        W2_global = torch.mean(W2_raw, dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]

        # 拼接并进行Softmax
        weights_combined = torch.cat([W1_global, W2_global], dim=1)  # [B, 2, 1, 1]
        weights_normalized = F.softmax(weights_combined, dim=1)  # [B, 2, 1, 1]

        W1_single = weights_normalized[:, 0:1, :, :]  # [B, 1, 1, 1]
        W2_single = weights_normalized[:, 1:2, :, :]  # [B, 1, 1, 1]

        # 扩展为通道级权重
        W1_norm = W1_single.expand(-1, C, 1, 1)  # [B, 176, 1, 1]
        W2_norm = W2_single.expand(-1, C, 1, 1)  # [B, 176, 1, 1]

        # # 可选：记录权重信息用于调试
        # if self.training:
        #     print(f"CBAM单值权重: W1={W1_single.mean().item():.4f}, W2={W2_single.mean().item():.4f}")

        return W1_norm, W2_norm


class ChannelAttentionModule(nn.Module):
    """CBAM通道注意力模块 (CAM)"""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        hidden_dim = max(channels // reduction_ratio, 1)  # 确保至少为1

        # 并行池化操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP处理 (使用Conv1x1代替Linear以保持维度)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),  # 第一层: C → C/r
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, 1, bias=False)  # 第二层: C/r → C
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]

        # 特征融合与激活
        channel_att = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return channel_att


class SpatialAttentionModule(nn.Module):
    """CBAM空间注意力模块 (SAM)"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 通道维度池化 + 卷积特征提取
        self.conv = nn.Conv2d(
            in_channels=2,  # 平均池化和最大池化拼接
            out_channels=1,  # 单通道空间注意力图
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 特征拼接
        spatial_feat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # 卷积特征提取
        spatial_att = self.sigmoid(self.conv(spatial_feat))  # [B, 1, H, W]

        return spatial_att


# ===================== CSoP相似度调制模块 =====================
class CSoP_Similarity_Modulation(nn.Module):
    """
    CSoP相似度调制模块 - 完整版
    """

    def __init__(self, channels=176, height=14, width=14):
        super().__init__()
        self.C = channels  # 176
        self.H = height  # 14
        self.W = width  # 14
        self.N = height * width  # 196 (14×14)

        print(f"\n初始化CSoP-CBAM融合模块(无自适应权重版):")
        print(f"  特征维度: C={self.C}, H×W={self.H}×{self.W}, N={self.N}")
        print(f"  融合公式: F_fuse = F1×W1_comp + F2×W2_comp + F1×W1×W1_norm + F2×W2×W2_norm")
        print(f"  权重约束: W1_norm + W2_norm = 1")

        # ========== 协方差相似度提取模块 ==========
        mid_channels = min(64, self.N // 4)

        # 列协方差提取
        self.col_cov_conv = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, self.N)),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 行协方差提取
        self.row_cov_conv = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.N, 1)),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # ========== CBAM差分调制权重模块 ==========
        self.cbam_dmwm = CBAMDiffModulationWeightModule(channels=channels)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(self.weight, 1)
                nn.init.constant_(self.bias, 0)

    def _compute_covariance_similarity_matrix(self, T1, T2):
        """
        计算协方差相似度矩阵 [B,196,196]
        """
        B, N, C = T1.shape  # [B, 196, 176]

        # 中心化处理
        T1_mean = T1.mean(dim=2, keepdim=True)  # [B,196,1]
        T2_mean = T2.mean(dim=2, keepdim=True)  # [B,196,1]

        T1_centered = T1 - T1_mean  # [B,196,176]
        T2_centered = T2 - T2_mean  # [B,196,176]

        # 转置并计算协方差
        T1_T = T1_centered.transpose(1, 2)  # [B,176,196]
        T2_T = T2_centered.transpose(1, 2)  # [B,176,196]

        if C > 1:
            Cov = torch.bmm(T2_T.transpose(1, 2), T1_T) / (C - 1)  # [B,196,196]
        else:
            Cov = torch.bmm(T2_T.transpose(1, 2), T1_T)  # [B,196,196]

        # 协方差矩阵作为相似度矩阵
        M = torch.abs(Cov)  # [B,196,196]

        # 归一化到[0,1]范围
        M_min = M.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        M_max = M.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        M_normalized = (M - M_min) / (M_max - M_min + 1e-8)

        return M_normalized

    def forward(self, F1, F2):
        """
        基于CBAM差分调制的CSoP融合（无自适应权重版）
        互补信息: F1_comp = F1 × W1_comp
        一致信息: F1_consist = F1 × W1 × W1_consist_weight
        最终融合: F_fuse = 互补信息 + 一致信息
        """
        B, C, H, W = F1.shape

        # ===================== 步骤1: 特征预处理 =====================
        F1_reg = F1.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        F2_reg = F2.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]

        # ===================== 步骤2: 协方差相似度矩阵计算 =====================
        T1 = F1_reg.reshape(B, self.N, C)  # [B,N,C]
        T2 = F2_reg.reshape(B, self.N, C)  # [B,N,C]

        # 协方差相似度矩阵
        M = self._compute_covariance_similarity_matrix(T1, T2)  # [B,N,N]

        # ===================== 步骤3: 模态特异性权重图生成 =====================
        M_for_conv = M.unsqueeze(1)  # [B,1,N,N]

        # 列协方差模式提取
        V1 = self.col_cov_conv(M_for_conv)  # [B,1,1,N]

        # 行协方差模式提取
        V2 = self.row_cov_conv(M_for_conv)  # [B,1,N,1]

        # ===================== 步骤4: 权重图生成 =====================
        V1 = V1.view(B, self.N, 1)  # [B,N,1]
        V2 = V2.squeeze(1)  # [B,N,1]

        # 权重图重构
        W1 = V1.reshape(B, self.H, self.W, 1)  # [B,H,W,1] - 模态1一致信息权重
        W2 = V2.reshape(B, self.H, self.W, 1)  # [B,H,W,1] - 模态2一致信息权重

        # 互补权重计算（一致权重的互补）
        W1_comp = 1 - W1  # [B,H,W,1] - 模态1互补信息权重
        W2_comp = 1 - W2  # [B,H,W,1] - 模态2互补信息权重

        # ===================== 步骤5: CBAM差分调制权重 =====================
        # 获取一致信息的通道级权重 W1_norm + W2_norm = 1
        W1_norm, W2_norm = self.cbam_dmwm(F1, F2)  # [B,176,1,1], [B,176,1,1]

        # 扩展权重到空间维度（用于一致信息加权）
        W1_consist_weight = W1_norm.view(B, C, 1, 1).expand(B, C, H, W)  # [B,176,14,14]
        W2_consist_weight = W2_norm.view(B, C, 1, 1).expand(B, C, H, W)  # [B,176,14,14]

        # ===================== 步骤6: 双阶段信息提取与融合 =====================
        # 维度转换
        W1 = W1.permute(0, 3, 1, 2)  # [B,1,H,W]
        W2 = W2.permute(0, 3, 1, 2)  # [B,1,H,W]
        W1_comp = W1_comp.permute(0, 3, 1, 2)  # [B,1,H,W]
        W2_comp = W2_comp.permute(0, 3, 1, 2)  # [B,1,H,W]

        # 1. 互补信息提取 (差异信息) - 直接使用互补权重
        F1_comp = F1 * W1_comp  # [B,176,14,14]
        F2_comp = F2 * W2_comp  # [B,176,14,14]

        # 2. 一致信息提取 (相似信息) - 使用一致权重和CBAM差分权重
        F1_consist = F1 * W1 * W1_consist_weight  # [B,176,14,14]
        F2_consist = F2 * W2 * W2_consist_weight  # [B,176,14,14]

        # 3. 信息整合
        # 互补信息直接相加（保留各自特性）
        complementary_part = F1_comp + F2_comp  # [B,176,14,14]

        # 一致信息加权融合（CBAM权重保证W1_norm+W2_norm=1）
        consistent_part = F1_consist + F2_consist  # [B,176,14,14]

        # 4. 最终融合
        F_fuse = complementary_part + consistent_part  # [B,176,14,14]

        # ===================== 步骤7: 记录融合信息 =====================
        self.fusion_info = {
            # CBAM差分权重统计
            'W1_norm_mean': W1_norm.mean().item(),
            'W2_norm_mean': W2_norm.mean().item(),
            'W1_norm_std': W1_norm.std().item(),
            'W2_norm_std': W2_norm.std().item(),

            # CSoP权重统计
            'W1_mean': W1.mean().item(),
            'W2_mean': W2.mean().item(),
            'W1_comp_mean': W1_comp.mean().item(),
            'W2_comp_mean': W2_comp.mean().item(),

            # 信息强度统计
            'F1_comp_mean': F1_comp.mean().item(),
            'F2_comp_mean': F2_comp.mean().item(),
            'F1_consist_mean': F1_consist.mean().item(),
            'F2_consist_mean': F2_consist.mean().item(),
            'complementary_part_mean': complementary_part.mean().item(),
            'consistent_part_mean': consistent_part.mean().item(),

            # 验证权重约束
            'weight_sum': (W1_norm + W2_norm).mean().item(),
            'weight_min': torch.min(W1_norm, W2_norm).mean().item(),
            'weight_max': torch.max(W1_norm, W2_norm).mean().item(),

            # 融合比例统计
            'complementary_ratio': complementary_part.mean().item() /
                                   (complementary_part.mean().item() + consistent_part.mean().item() + 1e-8),
            'consistent_ratio': consistent_part.mean().item() /
                                (complementary_part.mean().item() + consistent_part.mean().item() + 1e-8)
        }

        return F_fuse

# ===================== 双流融合网络（辅分支增强 + CSoP融合 - 完整版） =====================
class DualStreamFusionNetWithAux_CSoP(nn.Module):
    """双流融合网络（辅分支增强 + CSoP融合）"""

    def __init__(self, num_classes, backbone_feat_dim=176, image_size=224):
        super().__init__()

        self.backbone_feat_dim = backbone_feat_dim
        self.image_size = image_size

        print(f"\n{'=' * 60}")
        print(f"初始化双流融合网络（辅分支增强 + 完整版CSoP融合）")
        print(f"{'=' * 60}")
        print(f"特征维度: {backbone_feat_dim}")
        print(f"输入尺寸: {image_size}×{image_size}")

        # RGB分支
        print("初始化RGB分支...")
        self.rgb_backbone_base = efficientformerv2_s0(
            pretrained=True,
            num_classes=num_classes,
            dist=False,
            resolution=image_size
        )
        self.rgb_backbone_base.dist = False

        # 获取特征维度
        if hasattr(self.rgb_backbone_base, 'head') and isinstance(self.rgb_backbone_base.head, nn.Linear):
            self.rgb_feat_dim = self.rgb_backbone_base.head.in_features
        else:
            self.rgb_feat_dim = backbone_feat_dim

        # 替换分类头
        self.rgb_backbone_base.head = nn.Identity()

        # 创建支持特征提取的RGB主干
        self.rgb_backbone = EfficientFormerV2WithFeatures(self.rgb_backbone_base, image_size)

        # MS分支
        print("初始化MS分支...")
        self.ms_backbone_base = efficientformerv2_s0(
            pretrained=True,
            num_classes=num_classes,
            dist=False,
            resolution=image_size
        )
        self.ms_backbone_base.dist = False

        # 适配MS输入为4通道
        print("适配MS分支输入为4通道...")
        self._adapt_ms_input_conv()

        # 获取MS特征维度
        self.ms_feat_dim = self.rgb_feat_dim

        # 替换MS分类头
        self.ms_backbone_base.head = nn.Identity()

        # 创建支持特征提取的MS主干
        self.ms_backbone = EfficientFormerV2WithFeatures(self.ms_backbone_base, image_size)

        # RGB植被指数分支
        print("初始化RGB植被指数分支...")
        self.rgb_vi_branch = RGBVIFeatureBranch()

        # MS植被指数分支
        print("初始化MS植被指数分支...")
        self.ms_vi_branch = MSVIFeatureBranch()

        # RGB流门控调制模块
        print("初始化RGB流门控调制模块...")
        self.rgb_gate_module = DualGateModulation()

        # MS流门控调制模块
        print("初始化MS流门控调制模块...")
        self.ms_gate_module = DualGateModulation()

        # RGB流安全融合模块
        print("初始化RGB流安全融合模块...")
        self.rgb_fusion_module = SecurityFusion(in_channels=32)

        # MS流安全融合模块
        print("初始化MS流安全融合模块...")
        self.ms_fusion_module = SecurityFusion(in_channels=32)

        # ===================== 完整版CSoP相似度调制模块 =====================
        print(f"\n{'=' * 40}")
        print(f"初始化完整版CSoP相似度调制模块")
        print(f"{'=' * 40}")

        # 计算预期空间尺寸
        self.spatial_size = image_size // 16  # 14×14
        print(f"空间尺寸: {self.spatial_size}×{self.spatial_size}")

        self.csop_fusion = CSoP_Similarity_Modulation(
            channels=backbone_feat_dim,  # 176
            height=self.spatial_size,  # 14
            width=self.spatial_size  # 14
        )

        # 特征重塑层（与无CSoP版保持一致）
        self.rgb_reshape = nn.Sequential(
            nn.Conv2d(self.rgb_feat_dim, self.rgb_feat_dim, kernel_size=1),
            nn.BatchNorm2d(self.rgb_feat_dim),
            nn.ReLU6(inplace=True)
        )

        self.ms_reshape = nn.Sequential(
            nn.Conv2d(self.ms_feat_dim, self.ms_feat_dim, kernel_size=1),
            nn.BatchNorm2d(self.ms_feat_dim),
            nn.ReLU6(inplace=True)
        )

        # CSoP融合后特征增强
        self.post_csop_enhance = nn.Sequential(
            nn.Conv2d(backbone_feat_dim, backbone_feat_dim, kernel_size=1),
            nn.BatchNorm2d(backbone_feat_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(backbone_feat_dim, backbone_feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_feat_dim),
            nn.ReLU6(inplace=True)
        )

        # 全局池化（与无CSoP版保持一致）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ===================== 分类头（基于CSoP融合特征） =====================
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # 添加Dropout防止过拟合
            nn.Linear(backbone_feat_dim, num_classes)  # 176 → num_classes
        )

        print(f"\n网络初始化完成!")
        print(f"RGB特征维度: {self.rgb_feat_dim}")
        print(f"MS特征维度: {self.ms_feat_dim}")
        print(f"CSoP融合维度: {backbone_feat_dim}")
        print(f"最终分类数: {num_classes}")
        print(f"{'=' * 60}")

    def _adapt_ms_input_conv(self):
        """适配MS分支输入为4通道（与无CSoP版保持一致）"""
        print("查找并替换MS分支的第一层卷积...")

        for name, module in self.ms_backbone_base.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"找到卷积层: {name}, in_channels={module.in_channels}")

                if module.in_channels == 3:
                    print(f"替换 {name} 从3通道到4通道...")

                    new_conv = nn.Conv2d(
                        in_channels=4,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=module.bias is not None
                    )

                    with torch.no_grad():
                        new_conv.weight[:, :3, :, :].copy_(module.weight.clone())
                        mean_weight = module.weight.mean(dim=1, keepdim=True)
                        new_conv.weight[:, 3:4, :, :].copy_(mean_weight)
                        if module.bias is not None:
                            new_conv.bias.copy_(module.bias.clone())

                    parent = self.ms_backbone_base
                    parts = name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], new_conv)
                    print(f"成功替换 {name}")
                    return

        print("警告：MS分支可能无法正确处理4通道输入")

    def _apply_aux_branch_flow(self, backbone, vi_branch, gate_module, fusion_module, x, stream_name="Stream"):
        """
        应用辅分支流程（与无CSoP版完全一致）：
        1. 提取主干特征
        2. 提取植被指数特征
        3. 双层门控调制
        4. 安全融合
        """
        # 1. 提取主干特征
        stem_112 = backbone.get_stem_112_feature(x)  # [B,16,112,112]
        stem_56 = backbone.get_stem_56_feature(x)  # [B,32,56,56]

        # 2. 提取植被指数特征
        vi_32c_56, vi_16c_112 = vi_branch(x)  # [B,32,56,56], [B,16,112,112]

        # 3. 第一层门控调制（112×112尺度）
        vi_16c_112_mod, gate_112 = gate_module.forward_112(stem_112, vi_16c_112)

        # 4. 下采样到56×56
        vi_32c_56_from_mod = vi_branch.downsample_112_to_56(vi_16c_112_mod)  # [B,32,56,56]

        # 5. 第二层门控调制（56×56尺度）
        final_vi_32c_56, gate_56 = gate_module.forward_56(stem_56, vi_32c_56_from_mod)

        # 6. 安全融合
        fused_feat, fusion_gate = fusion_module(stem_56, final_vi_32c_56)

        return fused_feat, gate_112, gate_56, fusion_gate

    def _process_branch_output(self, feat, branch_name="Unknown"):
        """
        处理分支输出，确保为4D特征（与无CSoP版完全一致）
        """
        if isinstance(feat, torch.Tensor):
            if feat.dim() == 4:
                return feat
            elif feat.dim() == 2:
                B, C = feat.shape
                feat = feat.view(B, C, 1, 1)
                feat = F.interpolate(feat, size=(self.spatial_size, self.spatial_size), mode='nearest')
                return feat
            elif feat.dim() == 3:
                B, dim1, dim2 = feat.shape
                if dim1 == self.rgb_feat_dim:
                    L = dim2
                    spatial_dim = int(np.sqrt(L)) if np.sqrt(L).is_integer() else 1
                    return feat.view(B, self.rgb_feat_dim, spatial_dim, spatial_dim)
                else:
                    feat = feat.permute(0, 2, 1)
                    B, C, L = feat.shape
                    spatial_dim = int(np.sqrt(L)) if np.sqrt(L).is_integer() else 1
                    return feat.view(B, C, spatial_dim, spatial_dim)

        B = 1 if feat is None else (feat.shape[0] if hasattr(feat, 'shape') else 1)
        return torch.zeros(B, self.rgb_feat_dim, self.spatial_size, self.spatial_size,
                          device=next(self.parameters()).device if self._parameters else torch.device('cpu'))

    def forward(self, rgb_x, ms_x):
        """
        前向传播（辅分支增强 + CSoP融合）
        特征提取部分与无CSoP版完全一致，只替换融合部分

        参数:
            rgb_x: RGB图像 (B, 3, H, W)
            ms_x: 多光谱图像 (B, 4, H, W)

        返回:
            output: 分类输出 (B, num_classes)
            gate_stats: 门控统计信息
        """
        B = rgb_x.shape[0]

        # ========== RGB流：应用辅分支增强（与无CSoP版完全一致） ==========
        rgb_fused, rgb_gate_112, rgb_gate_56, rgb_fusion_gate = self._apply_aux_branch_flow(
            self.rgb_backbone, self.rgb_vi_branch, self.rgb_gate_module,
            self.rgb_fusion_module, rgb_x, "RGB"
        )

        # RGB流主干前向传播（使用融合特征，与无CSoP版完全一致）
        rgb_feat = self.rgb_backbone(rgb_x, fused_feat=rgb_fused)
        rgb_feat = self._process_branch_output(rgb_feat, "RGB")
        rgb_feat = self.rgb_reshape(rgb_feat)  # [B, 176, 14, 14]

        # ========== MS流：应用辅分支增强（与无CSoP版完全一致） ==========
        ms_fused, ms_gate_112, ms_gate_56, ms_fusion_gate = self._apply_aux_branch_flow(
            self.ms_backbone, self.ms_vi_branch, self.ms_gate_module,
            self.ms_fusion_module, ms_x, "MS"
        )

        # MS流主干前向传播（使用融合特征，与无CSoP版完全一致）
        ms_feat = self.ms_backbone(ms_x, fused_feat=ms_fused)
        ms_feat = self._process_branch_output(ms_feat, "MS")
        ms_feat = self.ms_reshape(ms_feat)  # [B, 176, 14, 14]

        # ========== 完整版CSoP相似度调制融合（替换原来的拼接融合） ==========
        # 使用完整版CSoP进行特征融合
        csop_fused = self.csop_fusion(rgb_feat, ms_feat)  # [B, 176, 14, 14]

        # CSoP融合后特征增强
        csop_fused = self.post_csop_enhance(csop_fused)  # [B, 176, 14, 14]

        # ========== 全局池化与分类（与无CSoP版结构一致，但输入是CSoP融合特征） ==========
        # 全局池化
        pooled = self.global_pool(csop_fused)  # [B, 176, 1, 1]
        pooled = pooled.flatten(1)  # [B, 176]

        # 分类（直接使用176维特征）
        output = self.classifier(pooled)  # [B, num_classes]

        # 收集门控统计信息
        gate_stats = {
            'rgb_gate_112': rgb_gate_112,
            'rgb_gate_56': rgb_gate_56,
            'rgb_fusion_gate': rgb_fusion_gate,
            'ms_gate_112': ms_gate_112,
            'ms_gate_56': ms_gate_56,
            'ms_fusion_gate': ms_fusion_gate,
            'csop_used': True,
            'csop_fusion_shape': list(csop_fused.shape)
        }

        return output, gate_stats


# ===================== 多光谱预处理 =====================
class MSPreprocess:
    """可序列化的多光谱预处理"""

    def __init__(self, mean, std, train=False, image_size=224):
        mean = np.array(mean, dtype=np.float32).reshape(-1)
        std = np.array(std, dtype=np.float32).reshape(-1)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.train = train
        self.image_size = int(image_size)

    def __call__(self, x):
        # 转为torch tensor
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            t = x.float()
        else:
            t = torch.from_numpy(np.array(x)).float()

        # 处理形状
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[2] in (1, 2, 3, 4) and t.shape[0] not in (1, 2, 3, 4):
                t = t.permute(2, 0, 1)

        # 确保通道数为4
        c = t.shape[0]
        if c != 4:
            if c < 4:
                pad = torch.zeros((4 - c, t.shape[1], t.shape[2]), dtype=torch.float32)
                t = torch.cat([t, pad], dim=0)
            else:
                t = t[:4, :, :]

        # Resize到目标尺寸
        t = F.interpolate(t.unsqueeze(0), size=(self.image_size, self.image_size),
                          mode='bilinear', align_corners=False).squeeze(0)

        # 数据增强（训练集）
        if self.train:
            if torch.rand(1).item() < 0.5:
                t = torch.flip(t, dims=[-2])

        # 标准化
        t = (t - self.mean.to(t.device)) / self.std.to(t.device)

        return t


# ===================== 强绑定数据集类 =====================
class BindedSampleDataset(torch.utils.data.Dataset):
    """强绑定样本数据集"""

    def __init__(self, csv_path, split_type='train', image_size=224):
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.df = self.df[self.df['数据集划分'] == split_type].reset_index(drop=True)

        # 过滤缺失关键通道的样本
        if not MISSING_CHANNEL_ALLOW:
            self.df = self.df[
                (self.df['G路径'] != '') &
                (self.df['NIR路径'] != '') &
                (self.df['R路径'] != '') &
                (self.df['RE路径'] != '')
                ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"{split_type}集无有效强绑定样本")

        # 查找类别列
        possible_class_columns = ['类别', 'class', 'label', 'category', '物候期', 'phenology', '日期']
        class_column = None
        for col in possible_class_columns:
            if col in self.df.columns:
                class_column = col
                print(f"找到类别列: '{col}'")
                break

        if class_column is None:
            non_path_columns = [col for col in self.df.columns if '路径' not in col and col != '数据集划分']
            if non_path_columns:
                class_column = non_path_columns[0]
                print(f"使用列 '{class_column}' 作为类别列")
            else:
                raise ValueError("未找到类别列")

        # 获取类别并确保是字符串
        unique_classes = self.df[class_column].dropna().unique()
        self.classes = [str(cls).strip() for cls in unique_classes]
        self.classes = sorted(self.classes)

        # 创建类别映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.image_size = image_size
        self.class_column = class_column

        print(f"\n📦 {split_type}集强绑定数据集初始化：")
        print(f"   类别列: {class_column}")
        print(f"   类别数: {self.num_classes}")
        print(f"   完整样本数: {len(self.df)}")

        # RGB预处理变换
        self.rgb_transform = self._get_rgb_transform(split_type)

        # 多光谱预处理（延迟初始化）
        self.ms_transform = None
        self.ms_mean, self.ms_std = self._calc_ms_normalization()
        self.ms_transform = MSPreprocess(
            mean=self.ms_mean,
            std=self.ms_std,
            train=(split_type == 'train'),
            image_size=image_size
        )

    def _get_rgb_transform(self, split_type):
        """RGB图像预处理"""
        if split_type == 'train':
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _calc_ms_normalization(self):
        """计算多光谱归一化参数"""
        print(f"\n📈 计算{self.df['数据集划分'].iloc[0]}集多光谱归一化参数...")
        channel_stats = {ch: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for ch in MS_CHANNELS}
        valid_files = 0

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            ch_valid = True
            for ch in MS_CHANNELS:
                ch_path = row[f'{ch}路径']
                if ch_path == '' or not os.path.exists(ch_path):
                    ch_valid = False
                    break

            if not ch_valid:
                continue

            try:
                for ch in MS_CHANNELS:
                    img = self._read_tiff(row[f'{ch}路径'])
                    channel_stats[ch]['sum'] += img.sum()
                    channel_stats[ch]['sum_sq'] += (img ** 2).sum()
                    channel_stats[ch]['count'] += img.size
                valid_files += 1
            except Exception as e:
                continue

        print(f"   有效多光谱文件数：{valid_files}/{len(self.df)}")

        # 计算均值/标准差
        mean, std = [], []
        for ch in MS_CHANNELS:
            stats = channel_stats[ch]
            if stats['count'] == 0:
                mean.append(0.0)
                std.append(1.0)
            else:
                mean_val = stats['sum'] / stats['count']
                std_val = np.sqrt((stats['sum_sq'] / stats['count']) - (mean_val ** 2))
                std_val = std_val if std_val > 1e-6 else 1.0
                mean.append(mean_val)
                std.append(std_val)

        return np.array(mean), np.array(std)

    def _read_tiff(self, file_path):
        """读取多光谱TIFF文件"""
        try:
            with tifffile.TiffFile(file_path) as tif:
                img = tif.asarray().astype(np.float32)
        except:
            try:
                with Image.open(file_path) as img:
                    img = np.array(img, dtype=np.float32)
            except:
                return np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # 归一化到0-1
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

        # 确保2D数组
        if len(img.shape) == 3:
            img = img.mean(axis=2)

        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """加载一个强绑定样本"""
        row = self.df.iloc[idx]

        # 1. 加载RGB图像
        try:
            rgb_img = Image.open(row['RGB路径']).convert('RGB')
            rgb_img = self.rgb_transform(rgb_img)
        except Exception as e:
            rgb_img = torch.zeros(3, self.image_size, self.image_size)

        # 2. 加载4通道多光谱
        ms_channels = []
        for ch in MS_CHANNELS:
            ch_path = row[f'{ch}路径']
            try:
                ms_channels.append(self._read_tiff(ch_path))
            except:
                ms_channels.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))

        ms_img = np.stack(ms_channels, axis=0)  # (4, H, W)
        ms_img = self.ms_transform(ms_img)

        # 3. 类别标签
        class_value = str(row[self.class_column]).strip()
        if class_value not in self.class_to_idx:
            label = 0
            print(f"警告：类别 '{class_value}' 不在映射中，使用默认标签 0")
        else:
            label = self.class_to_idx[class_value]

        return rgb_img, ms_img, label


# ===================== 训练工具函数 =====================
def seed_everything(seed=42):
    """固定随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ===================== 早停机制 =====================
class EarlyStopping:
    """早停：验证精度连续patience个epoch不提升则停止训练"""
    def __init__(self, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏸️  早停计数: {self.counter}/{self.patience} (当前最佳: {self.best_score:.3f}%)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def count_model_params(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    def format_params(num):
        if num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        return f"{num}"

    print("=" * 50)
    print(f"模型参数量统计：")
    print(f"总参数：{format_params(total_params)} ({total_params:,})")
    print(f"可训练参数：{format_params(trainable_params)} ({trainable_params:,})")
    print(f"不可训练参数：{format_params(non_trainable_params)} ({non_trainable_params:,})")
    print("=" * 50)

    return total_params, trainable_params


def train(model, device, train_loader, optimizer, epoch, model_ema, writer, criterion_train, scaler):
    """训练过程（添加门控统计记录）"""
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    train_start_time = time.time()

    # 门控值统计
    rgb_gate_112_list, rgb_gate_56_list, rgb_fusion_gate_list = [], [], []
    ms_gate_112_list, ms_gate_56_list, ms_fusion_gate_list = [], [], []

    for batch_idx, (rgb_x, ms_x, target) in enumerate(train_loader):
        rgb_x = rgb_x.to(device, non_blocking=True)
        ms_x = ms_x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_forward_backward_start = time.time()

        # 前向传播（返回输出和门控统计）
        output, gate_stats = model(rgb_x, ms_x)
        optimizer.zero_grad()

        # 记录门控值
        rgb_gate_112_list.append(gate_stats['rgb_gate_112'].item())
        rgb_gate_56_list.append(gate_stats['rgb_gate_56'].item())
        rgb_fusion_gate_list.append(gate_stats['rgb_fusion_gate'].item())
        ms_gate_112_list.append(gate_stats['ms_gate_112'].item())
        ms_gate_56_list.append(gate_stats['ms_gate_56'].item())
        ms_fusion_gate_list.append(gate_stats['ms_fusion_gate'].item())

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = torch.nan_to_num(criterion_train(output, target))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion_train(output, target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            loss.backward()
            optimizer.step()

        # 更新EMA
        if model_ema is not None:
            model_to_update = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_ema.update(model_to_update)

        batch_forward_backward_time = time.time() - batch_forward_backward_start

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # 计算FPS
        batch_fps = train_loader.batch_size / batch_forward_backward_time

        # 记录TensorBoard日志
        global_step = (epoch - 1) * len(train_loader) + batch_idx + 1
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Train/Batch_Accuracy', acc1.item(), global_step)
        writer.add_scalar('Train/Learning_Rate', lr, global_step)
        writer.add_scalar('Train/Batch_FPS', batch_fps, global_step)
        writer.add_scalar('Train/Batch_Time(s)', batch_forward_backward_time, global_step)

        # 记录门控值
        writer.add_scalar('Train/RGB_Gate_112', gate_stats['rgb_gate_112'], global_step)
        writer.add_scalar('Train/RGB_Gate_56', gate_stats['rgb_gate_56'], global_step)
        writer.add_scalar('Train/RGB_Fusion_Gate', gate_stats['rgb_fusion_gate'], global_step)
        writer.add_scalar('Train/MS_Gate_112', gate_stats['ms_gate_112'], global_step)
        writer.add_scalar('Train/MS_Gate_56', gate_stats['ms_gate_56'], global_step)
        writer.add_scalar('Train/MS_Fusion_Gate', gate_stats['ms_fusion_gate'], global_step)

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}\tTime:{:.3f}s\tFPS:{:.1f}'.format(
                epoch, (batch_idx + 1) * train_loader.batch_size, total_num,
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr,
                batch_forward_backward_time, batch_fps))

    # 计算整轮训练统计
    train_total_time = time.time() - train_start_time
    train_avg_fps = total_num / train_total_time
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg

    # 计算平均门控值
    avg_rgb_gate_112 = sum(rgb_gate_112_list) / len(rgb_gate_112_list) if rgb_gate_112_list else 0
    avg_rgb_gate_56 = sum(rgb_gate_56_list) / len(rgb_gate_56_list) if rgb_gate_56_list else 0
    avg_rgb_fusion_gate = sum(rgb_fusion_gate_list) / len(rgb_fusion_gate_list) if rgb_fusion_gate_list else 0
    avg_ms_gate_112 = sum(ms_gate_112_list) / len(ms_gate_112_list) if ms_gate_112_list else 0
    avg_ms_gate_56 = sum(ms_gate_56_list) / len(ms_gate_56_list) if ms_gate_56_list else 0
    avg_ms_fusion_gate = sum(ms_fusion_gate_list) / len(ms_fusion_gate_list) if ms_fusion_gate_list else 0

    # 记录epoch日志
    writer.add_scalar('Train/Epoch_Loss', ave_loss, epoch)
    writer.add_scalar('Train/Epoch_Accuracy', acc, epoch)
    writer.add_scalar('Train/Epoch_Time(s)', train_total_time, epoch)
    writer.add_scalar('Train/Epoch_Avg_FPS', train_avg_fps, epoch)

    # 记录平均门控值
    writer.add_scalar('Train/Avg_RGB_Gate_112', avg_rgb_gate_112, epoch)
    writer.add_scalar('Train/Avg_RGB_Gate_56', avg_rgb_gate_56, epoch)
    writer.add_scalar('Train/Avg_RGB_Fusion_Gate', avg_rgb_fusion_gate, epoch)
    writer.add_scalar('Train/Avg_MS_Gate_112', avg_ms_gate_112, epoch)
    writer.add_scalar('Train/Avg_MS_Gate_56', avg_ms_gate_56, epoch)
    writer.add_scalar('Train/Avg_MS_Fusion_Gate', avg_ms_fusion_gate, epoch)

    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}\tTime:{:.3f}s\tFPS:{:.1f}'.format(
        epoch, ave_loss, acc, train_total_time, train_avg_fps))
    print(f'   RGB门控值: Gate112={avg_rgb_gate_112:.4f}, Gate56={avg_rgb_gate_56:.4f}, Fusion={avg_rgb_fusion_gate:.4f}')
    print(f'   MS门控值: Gate112={avg_ms_gate_112:.4f}, Gate56={avg_ms_gate_56:.4f}, Fusion={avg_ms_fusion_gate:.4f}')

    return ave_loss, acc, train_total_time, train_avg_fps


@torch.no_grad()
def val(model, device, test_loader, epoch, writer, criterion_val, optimizer):
    """验证过程（添加门控统计记录）"""
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    val_start_time = time.time()

    val_list = []
    pred_list = []
    batch_infer_times = []

    # 门控值统计
    rgb_gate_112_list, rgb_gate_56_list, rgb_fusion_gate_list = [], [], []
    ms_gate_112_list, ms_gate_56_list, ms_fusion_gate_list = [], [], []

    for batch_idx, (rgb_x, ms_x, target) in enumerate(test_loader):
        for t in target:
            val_list.append(t.data.item())

        rgb_x = rgb_x.to(device, non_blocking=True)
        ms_x = ms_x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        infer_start_time = time.time()
        # 验证时也获取门控统计
        output, gate_stats = model(rgb_x, ms_x)
        infer_time = time.time() - infer_start_time
        batch_infer_times.append(infer_time)

        # 记录门控值
        rgb_gate_112_list.append(gate_stats['rgb_gate_112'].item())
        rgb_gate_56_list.append(gate_stats['rgb_gate_56'].item())
        rgb_fusion_gate_list.append(gate_stats['rgb_fusion_gate'].item())
        ms_gate_112_list.append(gate_stats['ms_gate_112'].item())
        ms_gate_56_list.append(gate_stats['ms_gate_56'].item())
        ms_fusion_gate_list.append(gate_stats['ms_fusion_gate'].item())

        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        global_step = (epoch - 1) * len(test_loader) + batch_idx + 1
        writer.add_scalar('Val/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Val/Batch_Accuracy', acc1.item(), global_step)

        # 记录门控值
        writer.add_scalar('Val/RGB_Gate_112', gate_stats['rgb_gate_112'], global_step)
        writer.add_scalar('Val/RGB_Gate_56', gate_stats['rgb_gate_56'], global_step)
        writer.add_scalar('Val/RGB_Fusion_Gate', gate_stats['rgb_fusion_gate'], global_step)
        writer.add_scalar('Val/MS_Gate_112', gate_stats['ms_gate_112'], global_step)
        writer.add_scalar('Val/MS_Gate_56', gate_stats['ms_gate_56'], global_step)
        writer.add_scalar('Val/MS_Fusion_Gate', gate_stats['ms_fusion_gate'], global_step)

    # 计算验证统计
    val_total_time = time.time() - val_start_time
    val_avg_fps = total_num / val_total_time
    val_avg_batch_infer_time = sum(batch_infer_times) / len(batch_infer_times)

    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    acc5 = acc5_meter.avg

    # 计算平均门控值
    avg_rgb_gate_112 = sum(rgb_gate_112_list) / len(rgb_gate_112_list) if rgb_gate_112_list else 0
    avg_rgb_gate_56 = sum(rgb_gate_56_list) / len(rgb_gate_56_list) if rgb_gate_56_list else 0
    avg_rgb_fusion_gate = sum(rgb_fusion_gate_list) / len(rgb_fusion_gate_list) if rgb_fusion_gate_list else 0
    avg_ms_gate_112 = sum(ms_gate_112_list) / len(ms_gate_112_list) if ms_gate_112_list else 0
    avg_ms_gate_56 = sum(ms_gate_56_list) / len(ms_gate_56_list) if ms_gate_56_list else 0
    avg_ms_fusion_gate = sum(ms_fusion_gate_list) / len(ms_fusion_gate_list) if ms_fusion_gate_list else 0

    # 记录验证日志
    writer.add_scalar('Val/Epoch_Loss', ave_loss, epoch)
    writer.add_scalar('Val/Epoch_Accuracy', acc, epoch)
    writer.add_scalar('Val/Epoch_Acc5', acc5, epoch)
    writer.add_scalar('Val/Epoch_Time(s)', val_total_time, epoch)
    writer.add_scalar('Val/Epoch_Avg_FPS', val_avg_fps, epoch)
    writer.add_scalar('Val/Avg_Batch_Infer_Time(s)', val_avg_batch_infer_time, epoch)

    # 记录平均门控值
    writer.add_scalar('Val/Avg_RGB_Gate_112', avg_rgb_gate_112, epoch)
    writer.add_scalar('Val/Avg_RGB_Gate_56', avg_rgb_gate_56, epoch)
    writer.add_scalar('Val/Avg_RGB_Fusion_Gate', avg_rgb_fusion_gate, epoch)
    writer.add_scalar('Val/Avg_MS_Gate_112', avg_ms_gate_112, epoch)
    writer.add_scalar('Val/Avg_MS_Gate_56', avg_ms_gate_56, epoch)
    writer.add_scalar('Val/Avg_MS_Fusion_Gate', avg_ms_fusion_gate, epoch)

    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\tTime:{:.3f}s\tFPS:{:.1f}'.format(
        ave_loss, acc, acc5, val_total_time, val_avg_fps))
    print(f'   RGB门控值: Gate112={avg_rgb_gate_112:.4f}, Gate56={avg_rgb_gate_56:.4f}, Fusion={avg_rgb_fusion_gate:.4f}')
    print(f'   MS门控值: Gate112={avg_ms_gate_112:.4f}, Gate56={avg_ms_gate_56:.4f}, Fusion={avg_ms_fusion_gate:.4f}')

    # 保存最佳模型
    if acc > Best_ACC:
        Best_ACC = acc
        save_path = os.path.join(file_dir, 'best_model.pth')

        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'best_acc1': Best_ACC,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': test_loader.dataset.class_to_idx,
            'ms_mean': test_loader.dataset.ms_mean,
            'ms_std': test_loader.dataset.ms_std,
            'gate_stats': {
                'rgb_gate_112': avg_rgb_gate_112,
                'rgb_gate_56': avg_rgb_gate_56,
                'rgb_fusion_gate': avg_rgb_fusion_gate,
                'ms_gate_112': avg_ms_gate_112,
                'ms_gate_56': avg_ms_gate_56,
                'ms_fusion_gate': avg_ms_fusion_gate
            },
            'config': {
                'IMAGE_SIZE': IMAGE_SIZE,
                'BATCH_SIZE': BATCH_SIZE,
                'LR': LR,
                'EPOCHS': EPOCHS,
                'FUSION_METHOD': 'FullCSoP'
            }
        }, save_path)
        print(f"最佳模型已保存（精度：{acc:.3f}%）→ {save_path}")

    return val_list, pred_list, ave_loss, acc, val_total_time, val_avg_fps


# ===================== 主程序 =====================
if __name__ == '__main__':
    print("=" * 80)
    print("RGB-MS双流融合网络训练程序（辅分支增强 + 完整版CSoP融合）")
    print("=" * 80)

    # 1. 数据匹配
    print("\n步骤1: 执行数据匹配...")

    # 配置数据匹配参数
    match_config = {
        'DATASET_ROOT': DATASET_ROOT,
        'CSV_OUTPUT_PATH': CSV_OUTPUT_PATH,
        'MS_CHANNELS': MS_CHANNELS,
        'RGB_CHANNEL': RGB_CHANNEL,
        'TARGET_SPLITS': TARGET_SPLITS,
        'IGNORE_SPLITS': IGNORE_SPLITS,
        'SUPPORTED_EXTS': SUPPORTED_EXTS,
        'MATCH_TOLERANCE': MATCH_TOLERANCE,
        'FALLBACK_MATCH': FALLBACK_MATCH,
        'MISSING_CHANNEL_ALLOW': MISSING_CHANNEL_ALLOW
    }

    # 调用数据匹配模块
    samples = data_matching.scan_and_match_samples(match_config)

    if len(samples) == 0:
        print("❌ 未匹配到任何样本，训练终止")
        exit(1)

    # 导出CSV
    if not os.path.isabs(CSV_OUTPUT_PATH):
        CSV_OUTPUT_PATH = os.path.join(os.getcwd(), CSV_OUTPUT_PATH)

    export_success = data_matching.export_csv(samples, CSV_OUTPUT_PATH, match_config)
    if not export_success:
        print("❌ CSV导出失败，但继续训练...")

    # 2. 初始化训练环境
    print("\n步骤2: 初始化训练环境...")
    seed_everything(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备：{device}")
    print(f"📅 训练开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建保存目录
    os.makedirs(file_dir, exist_ok=True)

    # 初始化TensorBoard
    log_dir = os.path.join('runs', f'EfficientFormer_RGBMS_Aux_FullCSoP_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志：{log_dir}")

    # 3. 加载数据集
    try:
        train_dataset = BindedSampleDataset(CSV_OUTPUT_PATH, split_type='train', image_size=IMAGE_SIZE)
        val_dataset = BindedSampleDataset(CSV_OUTPUT_PATH, split_type='val', image_size=IMAGE_SIZE)
        classes = train_dataset.num_classes
    except ValueError as e:
        print(f"❌ 数据集加载失败：{e}")
        exit(1)

    # 4. 构建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"训练集：{len(train_dataset)} 样本")
    print(f"验证集：{len(val_dataset)} 样本")
    print(f"类别数：{classes}")

    # 5. 初始化模型（使用完整版CSoP融合模型）
    print("\n步骤5: 初始化完整版CSoP融合模型...")
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()

    # 明确指定骨干网络特征维度
    BACKBONE_FEAT_DIM = 176  # EfficientFormerV2-S0的特征维度

    model = DualStreamFusionNetWithAux_CSoP(
        num_classes=classes,
        backbone_feat_dim=BACKBONE_FEAT_DIM,
        image_size=IMAGE_SIZE
    )
    model = model.to(device)

    # 统计参数量
    total_params, trainable_params = count_model_params(model)
    writer.add_scalar('Model/Total_Params', total_params, 0)
    writer.add_scalar('Model/Trainable_Params', trainable_params, 0)

    # 6. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    cosine_schedule = CosineLRScheduler(optimizer, t_initial=EPOCHS, warmup_t=5, lr_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 多卡训练
    if torch.cuda.device_count() > 1 and use_dp:
        print(f"使用 {torch.cuda.device_count()} 块GPU进行训练！")
        model = torch.nn.DataParallel(model)

    # EMA初始化
    model_ema = ModelEma(
        model, decay=model_ema_decay, device=device, resume=None
    ) if use_ema else None

    # 7. 训练过程记录
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    train_time_list, val_time_list = [], []
    train_fps_list, val_fps_list = [], []
    epoch_list = []

    # ===================== ✅ 新增：早停初始化 =====================
    early_stopping = EarlyStopping(patience=10, verbose=True)
    # ==============================================================

    # 记录总训练开始时间
    total_training_start_time = time.time()
    print("\n" + "=" * 60)
    print(f"🚀 训练开始！总轮数：{EPOCHS}，起始轮数：{start_epoch}")
    print(f"📌 设备：{device}，批次大小：{BATCH_SIZE}，类别数：{classes}")
    print(f"📊 融合配置：双流融合 + 辅分支增强 + 完整版CSoP相似度调制")
    print(f"📊 门控策略：双层门控调制（112→56）+ 安全融合")
    print(f"🛑 早停机制：已启用（连续10轮精度不提升自动停止）")
    print(f"🌿 辅助分支：RGB=NRI | MS=CVI")
    print("=" * 60 + "\n")

    # 8. 完整训练循环
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_list.append(epoch)
        print(f"\n📅 Epoch {epoch}/{EPOCHS} 开始训练...")

        # 训练步骤
        train_loss, train_acc, train_time, train_fps = train(
            model, device, train_loader, optimizer, epoch, model_ema, writer,
            criterion_train, scaler
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_time_list.append(train_time)
        train_fps_list.append(train_fps)

        # 验证步骤
        val_list, pred_list, val_loss, val_acc, val_time, val_fps = val(
            model, device, val_loader, epoch, writer, criterion_val, optimizer
        )
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_time_list.append(val_time)
        val_fps_list.append(val_fps)

        # ===================== ✅ 新增：早停判断 =====================
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("\n" + "=" * 60)
            print(f"🛑 训练提前结束！连续 {early_stopping.patience} 个epoch精度无提升")
            print("=" * 60)
            break
        # ==============================================================

        # 学习率调度
        cosine_schedule.step(epoch)

        # 每10轮打印进度
        if epoch % 10 == 0 or epoch == EPOCHS:
            print("\n" + "-" * 50)
            print(f"📊 Epoch {epoch} 进度总结")
            print(f"  训练损失：{train_loss:.2f} | 训练精度：{train_acc:.2f}%")
            print(f"  验证损失：{val_loss:.2f} | 验证精度：{val_acc:.2f}%")
            print(f"  最佳验证精度：{Best_ACC:.2f}%")
            print(f"  训练耗时：{train_time:.1f}s | 验证耗时：{val_time:.1f}s")
            print("-" * 50 + "\n")

    # 9. 训练完成后收尾
    total_training_time = time.time() - total_training_start_time
    avg_epoch_time = total_training_time / len(epoch_list) if epoch_list else 0

    # 保存最终模型
    final_save_path = os.path.join(file_dir, 'final_model.pth')
    torch.save({
        'epoch': EPOCHS,
        'state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'best_acc1': Best_ACC,
        'optimizer': optimizer.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'ms_mean': train_dataset.ms_mean,
        'ms_std': train_dataset.ms_std,
        'train_loss_list': train_loss_list,
        'val_acc_list': val_acc_list,
        'total_training_time': total_training_time,
        'training_config': {
            'mixup_enabled': False,
            'loss_function': 'CrossEntropyLoss',
            'IMAGE_SIZE': IMAGE_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'LR': LR,
            'EPOCHS': EPOCHS,
            'FUSION_METHOD': 'FullCSoP'
        }
    }, final_save_path)
    print(f"\n💾 最终模型已保存至：{final_save_path}")

    # 生成训练总结
    print("\n" + "=" * 60)
    print("🎉 训练全部完成！")
    print("=" * 60)
    print(f"📋 训练总结：")
    print(f"  - 总训练轮数：{len(epoch_list)}（{start_epoch} ~ {EPOCHS}）")
    print(f"  - 总训练耗时：{total_training_time / 3600:.2f}h（{total_training_time:.1f}s）")
    print(f"  - 平均每轮耗时：{avg_epoch_time:.1f}s")
    print(f"  - 最佳验证精度：{Best_ACC:.3f}%")
    print(f"  - 最后一轮验证精度：{val_acc_list[-1]:.2f}%")
    print(f"  - 模型保存目录：{file_dir}")
    print(f"  - TensorBoard日志：{log_dir}")
    print(f"  - 融合配置：RGB流（NRI辅分支）+ MS流（CVI辅分支）")
    print(f"  - 双流融合：完整版CSoP相似度调制融合")
    print("=" * 60)

    # 绘制训练曲线
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 6), dpi=150)

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_loss_list, label='训练损失', color='#e74c3c', linewidth=2, marker='o', markersize=2)
    plt.plot(epoch_list, val_loss_list, label='验证损失', color='#3498db', linewidth=2, marker='s', markersize=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练/验证损失曲线（RGBMS辅分支增强+完整版CSoP）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_acc_list, label='训练精度', color='#e74c3c', linewidth=2, marker='o', markersize=2)
    plt.plot(epoch_list, val_acc_list, label='验证精度', color='#3498db', linewidth=2, marker='s', markersize=2)
    plt.axhline(y=Best_ACC, color='#2ecc71', linestyle='--', label=f'最佳精度 {Best_ACC:.1f}%', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('训练/验证精度曲线（RGBMS辅分支增强+完整版CSoP）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_save_path = os.path.join(file_dir, 'train_val_curve_rgbms_aux_full_csop.png')
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📊 训练/验证曲线已保存至：{curve_save_path}")

    # 调试信息：数据集类别检查
    print("\n" + "=" * 60)
    print("调试信息：数据集类别检查")
    print("=" * 60)

    print(f"训练集类别数: {train_dataset.num_classes}")
    print(f"训练集类别列表: {train_dataset.classes}")
    print(f"验证集类别数: {val_dataset.num_classes}")
    print(f"验证集类别列表: {val_dataset.classes}")

    # 检查验证集标签范围
    unique_val_labels = set(val_list)
    print(f"验证集唯一标签: {sorted(unique_val_labels)}")
    print(f"验证集标签范围: {min(val_list)} - {max(val_list)}")

    # 检查类别名称
    idx_to_class = {v: str(k) for k, v in train_dataset.class_to_idx.items()}
    print(f"索引到类别映射: {idx_to_class}")

    # 生成分类报告
    print("\n" + "=" * 60)
    print("📋 最后一轮验证详细分类报告")
    print("=" * 60)

    # 获取类别映射并确保所有都是字符串
    idx_to_class = {}
    for k, v in train_dataset.class_to_idx.items():
        idx_to_class[v] = str(k)

    # 创建类别名称列表
    class_names = []
    for i in range(classes):
        if i in idx_to_class:
            class_names.append(str(idx_to_class[i]))
        else:
            class_names.append(f"Class_{i}")

    print(f"类别数: {classes}")
    print(f"类别名称: {class_names}")
    print(f"验证集预测样本数: {len(pred_list)}")
    print(f"验证集真实标签数: {len(val_list)}")

    # 确保两个列表长度相同
    min_len = min(len(val_list), len(pred_list))
    if len(val_list) != len(pred_list):
        print(f"警告: 真实标签数({len(val_list)})与预测数({len(pred_list)})不一致，截取前{min_len}个")
        val_list = val_list[:min_len]
        pred_list = pred_list[:min_len]

    # 生成分类报告
    try:
        report = classification_report(
            val_list, pred_list,
            target_names=class_names,
            digits=3,
            zero_division=0
        )
        print(report)
    except Exception as e:
        print(f"分类报告生成失败: {e}")
        print("尝试生成简化报告...")

        # 生成简化的准确率统计
        accuracy = accuracy_score(val_list, pred_list)
        print(f"总体准确率: {accuracy:.3f}")

        # 生成混淆矩阵
        cm = confusion_matrix(val_list, pred_list)
        print(f"混淆矩阵形状: {cm.shape}")

        # 按类别统计准确率
        for i, class_name in enumerate(class_names):
            if i < len(cm):
                correct = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                total = sum(cm[i, :]) if i < cm.shape[0] else 0
                if total > 0:
                    print(f"{class_name}: {correct}/{total} = {correct / total:.3f}")

    # 保存训练记录
    train_record = pd.DataFrame({
        'Epoch': epoch_list,
        'Train_Loss': train_loss_list,
        'Train_Accuracy(%)': train_acc_list,
        'Val_Loss': val_loss_list,
        'Val_Accuracy(%)': val_acc_list,
        'Train_Time(s)': train_time_list,
        'Train_FPS': train_fps_list,
        'Val_Time(s)': val_time_list,
        'Val_FPS': val_fps_list
    })
    record_save_path = os.path.join(file_dir, 'train_val_metrics_rgbms_aux_full_csop.csv')
    train_record.to_csv(record_save_path, index=False, encoding='utf-8-sig')
    print(f"\n📝 训练/验证指标已保存至：{record_save_path}")

    # 保存分类报告
    report_path = os.path.join(file_dir, 'classification_report_rgbms_aux_full_csop.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"训练完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最佳验证精度：{Best_ACC:.2f}%\n")
        f.write(f"类别映射：{train_dataset.class_to_idx}\n")
        f.write(f"融合方法：完整版CSoP相似度调制融合\n")
        f.write("=" * 80 + "\n")
        try:
            report = classification_report(
                val_list, pred_list,
                target_names=class_names,
                digits=3,
                zero_division=0
            )
            f.write(report)
        except:
            f.write("无法生成完整分类报告\n")
            f.write(f"总体准确率: {accuracy_score(val_list, pred_list):.3f}\n")

    print(f"\n📋 分类报告已保存至：{report_path}")

    # 关闭TensorBoard
    writer.close()
    print(f"\n🔌 TensorBoard写入器已关闭")
    print("\n" + "=" * 60)
    print("🎉 RGBMS双流融合网络（辅分支增强 + 完整版CSoP融合）训练完成！")
    print(f"📊 最佳验证精度：{Best_ACC:.2f}%")
    print("=" * 60)
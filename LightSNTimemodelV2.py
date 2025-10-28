import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange


from torch.nn.utils import spectral_norm


# =========================================================
# 基础模块
# =========================================================
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(spectral_norm(nn.Linear(inner_dim, dim),eps=1e-8), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# =========================================================
# =========================================================


class LightSNTimeformerV2(nn.Module):
    def __init__(
        self, num_chan=32, num_time=1000, 
        temporal_kernel=11,
        num_kernel=48,
        num_classes=2, depth=3, heads=8,
        mlp_dim=256, dim_head=32, dropout=0.3
    ):
        super().__init__()
        
        # 使用固定的多尺度卷积核
        temporal_kernels = [7, 11, 15]
        channels_per_scale = num_kernel // len(temporal_kernels)
        
        # 1. 多尺度卷积编码
        self.multiscale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, channels_per_scale, (1, k), padding=(0, k//2)),
                nn.BatchNorm2d(channels_per_scale),
                nn.ELU(),
                nn.Dropout(dropout*0.5),
            ) for k in temporal_kernels
        ])
        
        # 2. 空间卷积
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_kernel, num_kernel, (num_chan, 1)),
            nn.BatchNorm2d(num_kernel),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        
        # 3. 多尺度池化
        self.pool_sizes = [max(1, int(num_time * ratio)) for ratio in [0.25, 0.125, 0.0625]]
        self.multiscale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, size)) for size in self.pool_sizes
        ])
        
        # 池化权重学习
        self.pool_weights = nn.Parameter(torch.ones(len(self.pool_sizes)))
        
        # 4. Transformer编码器
        self.to_patch = Rearrange("b k 1 t -> b t k")
        self.transformer = Transformer(num_kernel, depth, heads, dim_head, mlp_dim, dropout)
        
        # 5. 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(num_kernel, num_kernel * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(num_kernel * 2, num_kernel),
            nn.LayerNorm(num_kernel)
        )
        
        # 6. 分类头
        self.attention_pool = nn.Sequential(
            nn.Linear(num_kernel, num_kernel // 2),
            nn.Tanh(),
            nn.Linear(num_kernel // 2, 1)
        )
        
        self.mlp_head = nn.Sequential(
            nn.Linear(num_kernel, mlp_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes)
        )

    def forward(self, eeg):
        b, c, t = eeg.shape
        
        # 1. 多尺度卷积编码
        multiscale_features = []
        for conv in self.multiscale_convs:
            feat = conv(eeg.unsqueeze(1))
            multiscale_features.append(feat)
        
        # 拼接多尺度特征
        x = torch.cat(multiscale_features, dim=1)
        
        # 2. 空间卷积
        x = self.spatial_conv(x)
        
        # 3. 多尺度池化 + 加权融合
        pool_features = []
        for pool in self.multiscale_pool:
            pooled = pool(x)
            pooled = self.to_patch(pooled)
            trans_feat = self.transformer(pooled)
            
            # 注意力池化获取特征
            attn_weights = self.attention_pool(trans_feat)
            attn_weights = F.softmax(attn_weights, dim=1)
            weighted_feat = (trans_feat * attn_weights).sum(dim=1)
            
            pool_features.append(weighted_feat)
        
        # 加权融合多尺度特征
        pool_weights = F.softmax(self.pool_weights, dim=0)
        multiscale_feat = torch.zeros_like(pool_features[0])
        for i, feat in enumerate(pool_features):
            multiscale_feat += feat * pool_weights[i]
        
        # 4. 特征增强 + 残差连接
        enhanced_feat = self.feature_enhance(multiscale_feat) + multiscale_feat
        
        # 5. 分类
        return self.mlp_head(enhanced_feat)

if __name__ == "__main__":
    # 运行维度检查测试
    test_with_dimension_check()
    
    print("\n" + "=" * 60)
    print("生产版本测试")
    print("=" * 60)
    
    # 测试生产版本
    data = torch.randn(4, 32, 1000)
    model_prod = LightSNTimeformerV2()
    output_prod = model_prod(data)
    print(f"生产版本: 输入 {data.shape} -> 输出 {output_prod.shape}")
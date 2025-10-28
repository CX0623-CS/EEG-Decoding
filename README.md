### 多尺度时域卷积 + 可学习池化融合的层级时间特征建模（Temporal Multi-Scale with Learnable Fusion）
多尺度时域卷积模块，用于在不同时间感受野下并行提取短时与长时的脑电动态信息；
多尺度池化模块，在Transformer处理后，通过不同时间尺度的全局特征压缩来获得“层级时间语义”；
可学习融合权重，实现了不同时间粒度下的自适应信息整合。

以往模型（如 EEGNet、TSception）虽然考虑多尺度卷积或多时域分支，但它们的多尺度融合往往是静态的或固定结构（例如简单拼接或平均）。


“动态可学习 + 时序层级建模”不同时间窗口特征在Transformer编码后，再通过一个可学习的权重向量自动调整比例，实现了时间维度上的“自适应注意力池化”。

提出一种基于多尺度时域卷积与可学习池化融合的层级时间建模机制（Hierarchical Temporal Fusion, HTF），能够在不同时间感受野上动态聚合EEG的多层次时序信息，显著提升模型的时间泛化能力与对任务节奏的自适应性。

### 谱归一化约束的轻量Transformer解码结构（Spectrally Normalized Lightweight Timeformer）


Attention层中使用了光谱归一化（Spectral Normalization）：

self.to_out = nn.Sequential(
    spectral_norm(nn.Linear(inner_dim, dim), eps=1e-8),
    nn.Dropout(dropout)
)


能抑制特征爆炸、稳定Transformer训练过程、并强化跨被试泛化性。


传统EEG Transformer（如EEG-Conformer, EEGViT）普遍存在：参数量大；注意力不稳定；容易过拟合个体特征、跨被试性能差。

通过谱归一化 + 局部时间输入 + 浅层堆叠，使Transformer能够：
专注于局部时间-频率依赖建模，提升泛化性和训练稳定性，同时显著减小模型复杂度。
提出一种引入光谱归一化约束的轻量级Transformer解码结构（LightSNTimeformer），通过在时序特征映射阶段引入谱约束与多尺度输入机制，增强了注意力的稳定性与EEG跨被试泛化性能。



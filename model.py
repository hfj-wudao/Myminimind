#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# @Time    : 2025/3/30 00:38
# @Author  : huangfujue
# @File    : model.py
# @Date    : 2025/3/30 
"""
模块说明
"""
import math
from typing import Optional, Tuple,List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from LMConfig import LMConfig


# ----------------------------------------------------------------------------------------------------------------------

def precompute_pos_cis(dim: int, end: int, theta: float = 1e4):
    """
    预计算旋转位置编码（Rotary Position Embeddings）所需的复数值。

    旋转位置编码是一种在自注意力机制中加入位置信息的技术，不同于传统的位置编码，
    它通过对查询(Q)和键(K)向量进行复数旋转来整合位置信息，具有更好的外推性和序列长度自适应性。

    计算原理：
    1. 为每个位置生成一组频率值，频率随着特征维度索引增加而衰减
    2. 利用这些频率值通过复平面上的旋转获得位置编码
    3. 生成单位模长的复数，角度由位置和频率决定

    参数:
        dim (int): 模型隐藏维度大小，通常是注意力头维度的两倍
        end (int): 需要预计算的最大序列长度
        theta (float, 可选): 频率的基数，控制位置编码的衰减率，默认为10000

    返回值:
        torch.Tensor: 形状为[end, dim//2]的复数张量，包含每个位置的旋转编码
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """
    将旋转位置编码应用于查询(Q)和键(K)张量。

    通过将查询和键向量视为复数，并与预计算的位置复数相乘，
    实现查询和键向量的旋转变换，从而将位置信息融入自注意力计算中。

    实现步骤：
    1. 将实数查询和键张量重塑并转换为复数形式
    2. 调整位置编码形状以匹配查询和键张量
    3. 执行复数乘法以施加旋转
    4. 将结果转换回实数并恢复原始形状

    参数:
        xq (torch.Tensor): 查询张量，形状为[batch_size, seq_len, num_heads, head_dim]
        xk (torch.Tensor): 键张量，形状为[batch_size, seq_len, num_heads, head_dim]
        pos_cis (torch.Tensor): 预计算的位置复数编码，形状为[seq_len, head_dim//2]

    返回值:
        tuple(torch.Tensor, torch.Tensor):
            - 应用了旋转位置编码的查询张量，与输入xq形状相同
            - 应用了旋转位置编码的键张量，与输入xk形状相同
    """

    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键(K)和值(V)张量以适应注意力头的分组机制。

    在多头注意力机制的变体中，通常使用分组键值机制（Grouped Query Attention, GQA）
    或多查询注意力（Multi-Query Attention, MQA）来减少计算开销。
    此函数支持将较少的KV头扩展匹配更多的Q头。

    实现原理：
    1. 将KV张量在头维度上复制指定次数
    2. 保持其他维度不变
    3. 复制方式为"交错重复"(interleaved)，每个KV头对应多个Q头

    参数:
        x (torch.Tensor): 输入的键或值张量，形状为[batch_size, seq_len, n_kv_heads, head_dim]
        n_rep (int): 每个键值头需要重复的次数，通常为 n_q_heads // n_kv_heads

    返回值:
        torch.Tensor: 重复后的张量，形状为[batch_size, seq_len, n_kv_heads*n_rep, head_dim]，
                     其中n_kv_heads*n_rep通常等于n_q_heads

    注：当n_rep=1时（即KV头数与Q头数相同），函数直接返回输入张量，不进行任何操作。
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(torch.nn.Module):
    """
    RMSNorm（均方根归一化）是一种高效的归一化技术，是 LayerNorm 的变体。

    原理：
    1. 不同于 LayerNorm 同时进行中心化（减均值）和缩放（除以标准差），RMSNorm 只进行缩放操作
    2. 计算输入向量沿特定维度的均方根（Root Mean Square）值
    3. 用该均方根值对输入进行归一化
    4. 通过可学习的缩放参数调整归一化强度

    优势：
    1. 计算效率更高，省略了减均值的步骤
    2. 在深层 Transformer 网络中提供更稳定的训练过程
    3. 减轻了长序列训练中的梯度消失问题
    4. 能保持原始表示的方向，仅调整其量级

    数学公式：
    设输入向量 x，RMSNorm 的计算如下：
    RMSNorm(x) = γ · x/sqrt(mean(x²) + ε)
    其中：
    - γ 是可学习参数（维度与 x 最后一维相同）
    - ε 是防止除零的小常数
    - mean(x²) 表示对 x 的平方在指定维度上取均值

    在 Transformer 架构中，RMSNorm 通常应用在每个子层的输入和输出，
    有助于稳定深层网络的训练，是现代大型语言模型中的关键组件之一。
    """

    def __init__(self, dim: int, eps: float):
        """
        初始化 RMSNorm 层

        参数:
        - dim: 需要归一化的特征维度大小
        - eps: 用于数值稳定性的小常数，防止除零错误
        """
        super().__init__()
        self.eps = eps
        # 初始化可学习的缩放参数，全部设为1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        前向传播函数

        参数:
        - x: 输入张量，通常形状为 [batch_size, sequence_length, dim]

        返回:
        - 归一化后的张量，保持与输入相同的形状

        计算步骤:
        1. 对输入张量进行平方: x.pow(2)
        2. 在最后一个维度上计算均值: mean(-1, keepdim=True)
        3. 加上 eps 后开平方根的倒数: torch.rsqrt(... + self.eps)
        4. 将结果与原始输入相乘进行缩放
        5. 最后应用可学习的参数 weight 进一步调整
        6. 确保输出与输入具有相同的数据类型
        """
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class Attention(nn.Module):
    """
    实现带有旋转位置编码和缓存支持的多头注意力机制。

    本实现支持以下高级特性：
    1. 分组查询注意力 (Grouped Query Attention, GQA)：允许键值头少于查询头
    2. 旋转位置编码 (RoPE)：通过复数旋转集成位置信息
    3. KV缓存：用于加速自回归生成
    4. Flash Attention：使用更高效的注意力计算算法
    5. 因果注意力掩码：确保模型只能看到当前及之前的位置

    这种注意力实现是现代大型语言模型中的核心组件，支持高效训练和推理。
    """

    def __init__(self, args: LMConfig):
        """
        初始化注意力模块。

        参数:
            args (LMConfig): 包含模型配置的对象，需要包含以下属性：
                - n_heads: 注意力头数量
                - n_kv_heads: 键值头数量(可选，默认等于n_heads)
                - dim: 模型隐藏维度
                - max_seq_len: 最大序列长度
                - dropout: Dropout比率
                - flash_attn: 是否使用Flash Attention算法
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        # 每个KV头对应的查询头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 线性投影层，将输入转换为查询、键、值向量
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出投影层，将多头拼接结果映射回模型维度
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 注意力权重和残差连接的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检测是否可以使用更高效的Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # 创建上三角掩码，确保自回归模型的因果性（不能看到未来信息）
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        # 这行代码的作用是：
        # 注册模型缓冲区：将预先计算好的注意力掩码（mask）作为模型的一部分，但不是模型参数。
        # 避免梯度计算：与模型参数不同，缓冲区不参与梯度计算和参数更新，适合存储固定不变的张量。
        # 设备自动迁移：当模型移动到不同设备（如 CPU 到 GPU）时，缓冲区会自动跟随模型迁移，无需手动处理。
        # 参数解释
        # "mask"：缓冲区的名称，注册后可以通过 self.mask 访问。
        # mask：要注册的张量，这里是一个上三角矩阵，用于实现因果注意力掩码（确保模型只能看到序列中当前及之前的位置）。
        # persistent=False：表示这个缓冲区在保存模型状态字典（state_dict()）时不会被保存。设为 False 的原因是：
        # 掩码是确定性生成的，可以在加载模型时重新创建
        # 减小模型文件大小
        # 掩码不含有需要保存的学习信息

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        """
        执行多头自注意力计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]
            pos_cis (torch.Tensor): 预计算的位置编码，形状为 [seq_len, head_dim//2]
            past_key_value (Tuple[torch.Tensor, torch.Tensor], 可选):
                上一次计算的KV缓存，用于自回归生成加速
            use_cache (bool): 是否返回当前KV用于未来缓存

        返回值:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - 注意力层的输出，形状为 [batch_size, seq_len, dim]
                - 如果use_cache为True，则返回当前的KV张量作为缓存
        """
        bsz, seq_len, _ = x.shape

        # 线性投影得到查询、键、值向量
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑张量以分离头维度
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # KV缓存实现：如果有缓存，将当前KV与之前的缓存拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 准备注意力计算：调整张量维度并重复KV头以匹配Q头数量
        xq, xk, xv = (
            xq.transpose(1, 2),  # [bsz, n_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # [bsz, n_heads, kv_seq_len, head_dim]
            repeat_kv(xv, self.n_rep).transpose(1, 2)  # [bsz, n_heads, kv_seq_len, head_dim]
        )

        # 使用更高效的Flash Attention（如果可用且不是单步生成）
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,  # Flash Attention会自动处理因果掩码
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # 传统注意力计算：QK转置点积，缩放，掩码，softmax，应用于V
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]  # 应用因果掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # 加权值向量

        # 重塑输出并通过输出投影层
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))

        return output, past_kv


class FeedForward(nn.Module):
    """
    FeedForward 实现了 Transformer 架构中的前馈神经网络组件，采用 SwiGLU 激活函数变体。

    这个实现是现代大型语言模型中广泛使用的 FFN 变体，具有以下特点：
    1. SwiGLU 激活函数：结合了 SiLU(Sigmoid Linear Unit)激活和门控机制
    2. 并行门控分支：使用两个并行线性变换后进行元素级乘法
    3. 隐藏维度自动计算：通常为输入维度的4倍，但会进行一定的缩放和对齐

    与传统的 FFN(使用 ReLU 或 GELU 激活)相比，这种 SwiGLU 结构在大型模型中表现出更好的性能，
    是 LLaMA、PaLM 等现代语言模型采用的标准设计。
    """

    def __init__(self, config: LMConfig):
        """
        初始化前馈网络层。

        参数:
            config (LMConfig): 包含模型配置的对象，定义了输入维度、隐藏维度等参数
        """
        super().__init__()

        # 计算隐藏层维度，如果未指定则自动计算
        # 这种计算方式遵循了一些大型语言模型的设计规范，如 LLaMA
        if config.hidden_dim is None:
            # 初始隐藏维度为输入维度的 4 倍
            hidden_dim = 4 * config.dim
            # 缩放为原来的 2/3，这是基于经验的优化
            hidden_dim = int(2 * hidden_dim / 3)
            # 将隐藏维度调整为 multiple_of 的倍数，以便在某些硬件上获得更好的性能
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # 第一个线性变换，将输入维度映射到隐藏维度
        # 在 SwiGLU 中，这个投影的输出会经过 SiLU 激活函数
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)

        # 第二个线性变换，将隐藏维度映射回输入维度
        # 这是 FFN 的输出投影
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)

        # 第三个线性变换，并行于 w1，用于门控机制
        # 在 SwiGLU 中，这个分支的输出直接与激活后的 w1 输出相乘
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

        # Dropout 层，用于正则化，防止过拟合
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        执行前馈网络的前向传播。

        实现的是 SwiGLU 变体，计算流程为：
        1. 将输入同时送入 w1 和 w3 两个投影
        2. w1 的输出经过 SiLU 激活函数(F.silu)
        3. 将激活后的 w1 输出与 w3 输出按元素相乘，形成门控机制
        4. 将相乘结果通过 w2 投影回原始维度
        5. 应用 dropout 进行正则化

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]

        返回值:
            torch.Tensor: 前馈网络的输出，形状与输入相同
        """
        # SwiGLU 激活函数:
        # 1. F.silu(self.w1(x)): 对第一个投影应用 SiLU 激活函数
        #    SiLU(x) = x * sigmoid(x)，也称为 Swish 激活函数
        # 2. self.w3(x): 计算门控因子
        # 3. 两者相乘实现门控机制，允许网络有选择地传递信息
        # 4. 结果通过 w2 投影回原始维度，并应用 dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    """
    MoEGate 实现了混合专家模型(Mixture of Experts, MoE)的路由门控机制。

    混合专家模型是一种条件计算架构，它由多个"专家"网络和一个门控网络组成：
    1. 门控网络决定输入应该路由到哪些专家
    2. 只有被选中的专家会处理输入，从而提高计算效率
    3. 各专家的输出根据门控权重进行加权组合

    这种架构允许模型在保持推理计算量相对不变的情况下大幅增加参数量，
    是扩展大型语言模型参数规模的高效方法，被用于PaLM、Mixtral等模型。

    本实现采用了Top-K专家选择策略，同时包含辅助损失以促进专家负载均衡。
    """

    def __init__(self, config: LMConfig):
        """
        初始化MoE门控网络。

        参数:
            config (LMConfig): 包含MoE配置的对象，定义了专家数量、路由策略等
        """
        super().__init__()
        self.config = config
        # 每个token选择的专家数量
        self.top_k = config.num_experts_per_tok
        # 可路由专家的总数量
        self.n_routed_experts = config.n_routed_experts

        # 专家选择的评分函数，通常使用softmax
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数，用于平衡专家负载
        self.alpha = config.aux_loss_alpha
        # 是否在序列级别计算辅助损失
        self.seq_aux = config.seq_aux

        # 是否对top-k专家的概率进行归一化
        self.norm_topk_prob = config.norm_topk_prob
        # 门控网络的输入维度
        self.gating_dim = config.dim
        # 门控网络的参数，形状为[专家数量，输入维度]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 初始化门控网络参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化门控网络权重，使用Kaiming均匀初始化。

        Kaiming初始化有助于深层网络的训练，通过考虑非线性激活函数
        调整权重分布，使得每一层输出的方差大致相同。
        """
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        执行MoE门控网络的前向传播，确定每个token应该路由到哪些专家。

        过程包括：
        1. 计算每个token与每个专家的相关性分数
        2. 选择相关性最高的K个专家
        3. 计算辅助损失以促进专家负载均衡

        参数:
            hidden_states (torch.Tensor): 输入特征，形状为[batch_size, seq_len, dim]

        返回值:
            tuple:
                - torch.Tensor: 选中专家的索引，形状为[batch_size*seq_len, top_k]
                - torch.Tensor: 专家的权重系数，形状为[batch_size*seq_len, top_k]
                - torch.Tensor 或 float: 辅助损失，用于平衡专家负载
        """
        bsz, seq_len, h = hidden_states.shape
        # 重塑输入以便批量处理所有token
        hidden_states = hidden_states.view(-1, h)

        # 计算每个token对每个专家的路由分数
        # 本质上是计算输入与每个专家门控权重的内积
        logits = F.linear(hidden_states, self.weight, None)

        # 将分数转换为概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择概率最高的top_k个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果选择多个专家，可以选择对其权重进行归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 小常数防止除零
            topk_weight = topk_weight / denominator

        # 计算辅助损失以促进专家负载均衡
        # 这有助于防止"专家崩溃"问题，即少数专家处理大多数输入
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # 序列级别的辅助损失计算
            if self.seq_aux:
                # 重塑分数以便序列级别处理
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 计算每个batch中每个专家的使用频率
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 计算辅助损失：专家使用频率与其分配概率的乘积
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 常规辅助损失计算
                # 创建one-hot编码，表示选中的专家
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 计算专家使用频率
                ce = mask_ce.float().mean(0)
                # 计算路由概率的平均值
                Pi = scores_for_aux.mean(0)
                # 调整专家使用频率
                fi = ce * self.n_routed_experts
                # 计算最终辅助损失
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss



class MOEFeedForward(nn.Module):
    """
    MOEFeedForward 实现了混合专家模型(Mixture of Experts, MoE)版本的前馈神经网络。

    混合专家模型是一种稀疏条件计算架构，主要特点和优势包括：
    1. 大幅增加模型参数量而不显著增加推理计算量
    2. 通过"条件计算"使每个输入只激活部分网络参数
    3. 可以实现更高效的大规模模型训练和部署

    本实现的工作流程：
    - 使用门控网络(MoEGate)为每个输入token选择最合适的专家
    - 只激活被选中的专家计算输入token的表示
    - 根据门控网络分配的权重组合多个专家的输出
    - 可选的共享专家网络处理所有输入，增强模型稳定性

    MoE是现代超大规模语言模型(如Mixtral 8x7B、PaLM等)中提高参数效率的关键技术。
    """

    def __init__(self, config: LMConfig):
        """
        初始化混合专家前馈网络。

        参数:
            config (LMConfig): 包含MoE配置的对象，包括专家数量、路由策略等参数
        """
        super().__init__()
        self.config = config

        # 创建多个专家网络，每个专家是一个标准的前馈网络
        # 这些专家网络结构相同但参数不同，可以学习处理不同类型的特征
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])

        # 门控网络，负责决定每个token应路由到哪些专家
        # 门控是MoE架构的核心组件，直接影响计算效率和模型性能
        self.gate = MoEGate(config)

        # 可选的共享专家网络，会处理所有输入
        # 这是一种"混合密集-稀疏"架构，可以提高模型稳定性
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        """
        执行混合专家前馈网络的前向传播。

        过程包括：
        1. 使用门控网络为每个token选择专家
        2. 根据路由策略分发输入到相应专家
        3. 收集并组合各专家的输出
        4. 应用可选的共享专家网络

        参数:
            x (torch.Tensor): 输入张量，形状为[batch_size, seq_len, dim]

        返回值:
            torch.Tensor: 混合专家网络的输出，形状与输入相同
        """
        # 保存原始输入用于残差连接和共享专家
        identity = x
        # 记录原始形状，便于最后重塑输出
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控网络决定每个token路由到哪些专家
        # topk_idx: 选中的专家索引
        # topk_weight: 对应的专家权重
        # aux_loss: 专家负载均衡的辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 将输入展平为二维张量，便于处理
        # 形状从[batch_size, seq_len, dim]变为[batch_size*seq_len, dim]
        x = x.view(-1, x.shape[-1])

        # 将专家索引也展平为一维
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式: 使用所有选中的专家(top-k)

            # 对输入进行重复，使每个token的副本可以发送到不同的专家
            # 如果每个token选择k个专家，则每个token需要重复k次
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # 创建与输入形状相同的空张量，用于收集专家输出
            # 使用float16以节省内存，这是大模型训练的常见优化
            y = torch.empty_like(x, dtype=torch.float16)

            # 逐个专家处理相关输入
            # 这个循环处理每个专家分配到的所有token
            for i, expert in enumerate(self.experts):
                # 找出路由到当前专家的所有token
                # flat_topk_idx == i 创建一个布尔掩码，标识分配给专家i的所有token
                # 对每个专家，只处理路由到它的token，节省计算
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致

            # 重塑并加权组合专家输出
            # 1. 将输出重塑为 [batch_size*seq_len, num_experts_per_tok, dim]
            # 2. 与专家权重相乘(权重扩展一个维度以便广播)
            # 3. 沿专家维度求和，得到每个token的最终输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

            # 将输出重塑回原始形状
            y = y.view(*orig_shape)
        else:
            # 推理模式: 更高效的实现，批量处理分配给同一专家的所有token
            # 这种实现显著提高了推理速度
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 如果配置了共享专家，则将其输出添加到路由专家的输出
        # 共享专家处理所有token，不受门控机制影响
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        # 保存辅助损失，以便在模型更新时使用
        self.aux_loss = aux_loss

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的高效MoE实现，对同一专家处理的token进行批处理。

        推理时的优化策略：
        1. 将分配给同一专家的所有token分组处理
        2. 每个专家只处理一次，处理其对应的所有token
        3. 使用scatter_add_高效组合结果

        参数:
            x (torch.Tensor): 展平的输入张量，形状为[batch_size*seq_len, dim]
            flat_expert_indices (torch.Tensor): 展平的专家索引，形状为[batch_size*seq_len*num_experts_per_tok]
            flat_expert_weights (torch.Tensor): 展平的专家权重，形状为[batch_size*seq_len*num_experts_per_tok, 1]

        返回值:
            torch.Tensor: 混合专家网络的输出，形状为[batch_size*seq_len, dim]
        """
        # 创建输出缓冲区，初始化为全零
        expert_cache = torch.zeros_like(x)

        # 按专家索引对token进行排序
        # 这样可以将分配给同一专家的所有token分组在一起
        idxs = flat_expert_indices.argsort()

        # 计算每个专家处理的token数量的累积和
        # 例如，如果专家0处理6个token，专家1处理9个token，专家2处理5个token...
        # 则tokens_per_expert = [6, 15, 20, ...]
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 通过除以每个token选择的专家数，获取原始token索引
        # 这样可以将扁平化的专家-token对映射回原始token
        token_idxs = idxs // self.config.num_experts_per_tok

        # 示例解释:
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...]
        # 意味着下标在[0:6]的元素，即token_idxs[:6] -> [3, 7, 19, 21, 24, 25]位置的token都由专家0处理
        # token_idxs[6:15]位置的token都由专家1处理，以此类推

        # 逐个专家处理分配给它的所有token
        for i, end_idx in enumerate(tokens_per_expert):
            # 确定当前专家负责处理的token范围
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            # 如果当前专家没有分配到token，则跳过
            if start_idx == end_idx:
                continue

            # 获取当前专家
            expert = self.experts[i]

            # 获取分配给当前专家的所有token的原始索引
            exp_token_idx = token_idxs[start_idx:end_idx]

            # 提取这些token的输入特征
            expert_tokens = x[exp_token_idx]

            # 使用当前专家处理相应的token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # 将专家输出乘以对应的门控权重
            # 这一步实现了专家输出的加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 使用scatter_add_将结果聚合到输出缓冲区
            # 这是一种高效的原地操作，将专家输出添加到最终输出中
            # 1. exp_token_idx.view(-1, 1)将索引扩展为列向量
            # 2. repeat(1, x.shape[-1])将索引扩展到与特征维度相同的大小
            # 3. scatter_add_根据索引将expert_out加到expert_cache上
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache



class MiniMindBlock(nn.Module):
    """
    MiniMindBlock 实现了 Transformer 架构的核心构建块，包含自注意力机制和前馈网络。

    该模块采用了现代 Transformer 架构的几个关键设计：
    1. Pre-LayerNorm 结构：在每个子层（注意力和前馈网络）之前应用归一化，有助于稳定深层网络训练
    2. 残差连接：确保梯度在深层网络中能够高效传播
    3. 可选的混合专家模型(MoE)：提供计算高效的大规模参数扩展方式
    4. KV缓存支持：用于高效的自回归文本生成

    这种结构是当代大型语言模型的基本组成单元，如 GPT、LLaMA、Falcon 等。
    """

    def __init__(self, layer_id: int, config: LMConfig):
        """
        初始化 Transformer 块。

        参数:
            layer_id (int): 当前层的ID，用于某些需要识别层位置的操作
            config (LMConfig): 包含模型配置的对象，定义了模型的维度、头数、归一化参数等
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim  # 模型的隐藏维度
        self.head_dim = config.dim // config.n_heads  # 每个注意力头的维度

        # 多头自注意力层，允许模型关注输入序列的不同部分
        self.attention = Attention(config)

        self.layer_id = layer_id

        # 注意力子层的归一化，使用 RMSNorm 提高训练稳定性
        # RMSNorm 与 LayerNorm 相比省略了均值中心化步骤，仅做缩放，计算效率更高
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 前馈网络子层的归一化
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 前馈网络，根据配置决定使用标准 FFN 或混合专家模型(MoE)
        # MoE 在参数效率上有优势，可以增加模型容量而不显著增加计算量
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        """
        执行 Transformer 块的前向传播。

        过程包括：
        1. 对输入应用 RMSNorm 归一化
        2. 通过多头自注意力层处理归一化后的输入
        3. 将注意力输出通过残差连接添加到原始输入
        4. 对注意力层的输出再次应用 RMSNorm 归一化
        5. 将归一化后的输出通过前馈网络处理
        6. 将前馈网络输出通过残差连接添加到注意力层的输出

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]
            pos_cis (torch.Tensor): 预计算的位置编码，用于旋转位置编码(RoPE)
            past_key_value (tuple, 可选): 缓存的键值对，用于加速自回归生成
            use_cache (bool): 是否返回当前键值对用于未来的缓存

        返回值:
            tuple:
                - torch.Tensor: 块的输出，形状与输入相同
                - tuple 或 None: 如果use_cache为True，返回更新后的KV缓存
        """
        # 注意力子层: Pre-LayerNorm架构(先归一化后注意力)
        # 这种架构有助于深层网络的梯度流动和训练稳定性
        h_attn, past_kv = self.attention(
            self.attention_norm(x),  # 先归一化再送入注意力层
            pos_cis,
            past_key_value=past_key_value,  # 用于KV缓存的自回归生成加速
            use_cache=use_cache
        )

        # 残差连接：将注意力层输出添加到原始输入
        # 残差连接是解决深层网络退化问题的关键技术
        h = x + h_attn

        # 前馈网络子层: 同样采用Pre-LayerNorm架构
        # 前馈网络通常由两个线性变换和一个非线性激活函数组成
        # MoE版本会根据输入动态路由到不同的专家网络
        out = h + self.feed_forward(self.ffn_norm(h))

        return out, past_kv


class MiniMindLM(PreTrainedModel):
    """
    MiniMindLM 实现了完整的基于 Transformer 的语言模型，支持预训练和生成功能。

    这是一个现代自回归语言模型的完整实现，融合了当前最先进的多项技术：
    1. Transformer 解码器架构：使用自注意力机制和前馈网络构建
    2. 预计算的旋转位置编码(RoPE)：提供位置感知能力和外推性
    3. 权重绑定：输入嵌入和输出层共享权重以减少参数量
    4. KV缓存生成加速：缓存已计算的键值对以提高生成效率
    5. 多种采样策略：支持温度采样、Top-P采样等控制文本生成的随机性
    6. 参数高效扩展：可选的混合专家模型(MoE)架构
    7. 流式生成：支持增量式生成，适合交互式应用

    这种架构是现代大型语言模型(如GPT、LLaMA等)的基础设计。
    """
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        """
        初始化语言模型。

        参数:
            params (LMConfig, 可选): 模型配置对象，定义了模型架构的各项参数
        """
        self.params = params or LMConfig()
        super().__init__(self.params)

        # 基础模型参数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers

        # 词元嵌入层：将token ID转换为连续的向量表示
        # 输入维度是词表大小(vocab_size)，输出维度是模型维度(dim)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Dropout层：用于正则化，减少过拟合风险
        self.dropout = nn.Dropout(params.dropout)

        # Transformer层堆叠：模型的核心计算单元
        # 每一层都是一个完整的Transformer块，包含自注意力和前馈网络
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])

        # 最终归一化层：使输出分布更加稳定，有助于生成质量
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # 输出线性层：将模型维度映射回词表维度，用于预测下一个token
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 权重绑定：将输入嵌入层和输出层的权重绑定
        # 这种技术可以减少参数量并提高性能，被多种Transformer模型采用
        self.tok_embeddings.weight = self.output.weight

        # 预计算位置编码：创建并注册旋转位置编码(RoPE)的复数表示
        # 这些编码会在注意力计算中用于提供位置信息
        self.register_buffer("pos_cis", precompute_pos_cis(params.dim // params.n_heads, params.max_seq_len,
                                                           theta=params.rope_theta), persistent=False)

        # 输出对象：用于封装模型的各种输出
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        """
        执行模型的前向传播，完成自回归语言建模。

        过程包括：
        1. 将输入token转换为嵌入向量
        2. 提取对应的位置编码
        3. 依次通过每一个Transformer层
        4. 最终层归一化并映射到词表空间
        5. 计算混合专家模型的辅助损失(如果使用)

        参数:
            input_ids (torch.Tensor, 可选): 输入的token ID，形状为[batch_size, seq_len]
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]], 可选):
                KV缓存，加速自回归生成
            use_cache (bool): 是否返回更新后的KV缓存
            **args: 其他参数，包括start_pos用于指定生成的起始位置

        返回值:
            CausalLMOutputWithPast: 包含logits、辅助损失和KV缓存的输出对象
        """
        # 初始化KV缓存，如果未提供则为每层创建None占位符
        past_key_values = past_key_values or [None] * len(self.layers)

        # 获取生成的起始位置，默认为0
        # 这在使用KV缓存时很重要，用于确定位置编码的偏移量
        start_pos = args.get('start_pos', 0)

        # 将输入ID转换为嵌入向量并应用dropout
        h = self.dropout(self.tok_embeddings(input_ids))

        # 选取当前序列对应的位置编码
        # 从预计算的位置编码中切片获取当前需要的部分
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]

        # 用于收集每一层的KV缓存
        past_kvs = []

        # 依次通过每一个Transformer层
        for l, layer in enumerate(self.layers):
            # layer返回转换后的隐藏状态和更新的KV缓存
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

        # 最终归一化并投影到词表空间
        # 这一步生成每个位置上下一个token的概率分布
        logits = self.output(self.norm(h))

        # 计算混合专家模型(MoE)的辅助损失
        # 这个损失用于平衡专家使用率，防止"专家崩溃"问题
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        # 设置输出对象的各个字段
        self.OUT.__setitem__('logits', logits)  # 词表上的预测分布
        self.OUT.__setitem__('aux_loss', aux_loss)  # MoE辅助损失
        self.OUT.__setitem__('past_key_values', past_kvs)  # 更新的KV缓存

        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """
        生成文本序列，支持多种采样策略和生成模式。

        包含两种生成模式：
        1. 流式生成：逐个token返回生成结果，适合实时显示
        2. 批量生成：一次性返回所有生成结果，适合批处理场景

        参数:
            input_ids (torch.Tensor): 起始token序列，形状为[batch_size, seq_len]
            eos_token_id (int): 表示序列结束的token ID
            max_new_tokens (int): 最多生成的新token数量
            temperature (float): 控制采样随机性的温度参数，越小越确定性
            top_p (float): 核采样(nucleus sampling)的概率阈值，控制多样性
            stream (bool): 是否使用流式生成模式
            rp (float): 重复惩罚因子，降低已出现token的概率
            use_cache (bool): 是否使用KV缓存加速生成
            pad_token_id (int): 用于填充的token ID

        返回值:
            torch.Tensor 或 generator: 根据stream参数返回完整生成结果或生成器
        """
        # 流式生成模式
        if stream:
            return self._generate_stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache)

        # 批量生成模式
        generated = []
        for i in range(input_ids.size(0)):
            # 去除每个序列中的填充token，仅保留有效内容
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)

            # 使用流式生成器内部产生序列
            out = self._generate_stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache)

            # 收集生成器产生的所有token
            tokens_list = [tokens[:, -1:] for tokens in out]

            # 拼接所有生成的token
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad

            # 将输入序列与生成序列连接
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)

        # 找出所有生成序列中的最大长度
        max_length = max(seq.size(1) for seq in generated)

        # 将所有序列填充到相同长度，以便批量处理
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]

        # 将所有批次序列拼接返回
        return torch.cat(generated, dim=0)

    def _generate_stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """
        内部流式生成函数，逐个token生成并返回序列。

        这个函数实现了带有各种采样策略的自回归解码过程：
        1. 自回归地预测下一个token
        2. 应用温度缩放调整分布
        3. 可选地应用重复惩罚和top-p采样
        4. 从调整后的分布中采样新token
        5. 将新token添加到序列并继续生成

        参数与generate方法类似，但它是一个生成器函数。

        返回值:
            generator: 一个生成器，每次产生更新后的token序列
        """
        # 初始设置：记录起始位置，标记是否为第一次前向传播，初始化KV缓存
        start, first_seq, past_kvs = input_ids.shape[1], True, None

        # 循环生成，直到达到最大token数或生成结束标记
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首次执行或不使用缓存时，处理整个序列
            # 非首次且使用缓存时，只处理最新的token
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache), False
            else:
                # 增量处理：只将最新的token送入模型，利用缓存加速
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1)

            # 获取最后一个位置的logits和更新的KV缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values

            # 重复惩罚：降低已经出现过的token的概率
            # 这有助于减少重复生成的问题
            logits[:, list(set(input_ids.tolist()[0]))] /= rp

            # 应用温度缩放：调整logits的峰值以控制随机性
            # 温度越低，分布越尖锐，生成越确定性
            # 温度越高，分布越平坦，生成越多样
            logits /= (temperature + 1e-9)  # 添加小常数防止除零

            # 应用Top-P (nucleus) 采样
            # 只保留累积概率达到top_p阈值的最高概率token
            # 这平衡了多样性和质量，避免低概率、不合理的token被采样
            if top_p is not None and top_p < 1.0:
                # 对logits按概率从高到低排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # 计算softmax后的概率
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 创建掩码标识累积概率超过top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 掩码错位，保留第一个超过阈值的token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False  # 始终保留最高概率的token
                # 将掩码映射回原始logits顺序
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # 将被移除的位置设为负无穷，确保softmax后概率为0
                logits[indices_to_remove] = -float('Inf')

            # 从调整后的分布中采样下一个token
            # multinomial实现了按概率采样
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            # 将新token添加到输入序列
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

            # 产生当前已生成的序列(不含提示部分)
            yield input_ids[:, start:]

            # 如果生成了结束标记，提前结束生成
            if input_ids_next.item() == eos_token_id:
                break
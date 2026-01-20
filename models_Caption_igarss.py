import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_

from torch import Tensor
from typing import Optional, Any, Union, Callable
import cv2
# from cv2 import resize
import numpy as np
from torch.nn import functional as F
from imageio import imread
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.modules.container import ModuleList
import copy

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

# 不同特征层之间实现融合，且输入输出是
class Scale_aware(nn.Module):
    def __init__(self,dropout=0.1, d_model=1024, n_head=4,  activation=nn.ReLU):
        """
        :param d_model: Dimension of input features
        :param n_head: Number of attention heads
        :param dropout: Dropout rate
        :param activation: Activation function
        """
        super(Scale_aware, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # Multi-head attention layers
        self.attention_1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input_q, input_k1, input_k2):
        """
        :param input_q: Query tensor (Q)
        :param input_k1: Key tensor 1 (K1)
        :param input_k2: Key tensor 2 (K2)
        :return: Updated query tensor
        """
        # Normalize inputs
        input_q_norm = self.norm1(input_q)
        input_k1_norm = self.norm1(input_k1)
        input_k2_norm = self.norm1(input_k2)

        # Weighted concatenation of key tensors
        combined_keys = torch.cat([input_k1_norm, input_k2_norm], dim=0)

        # Multi-head attention
        attn_output, _ = self.attention_1(input_q_norm, combined_keys, combined_keys)
        attn_output = self.dropout1(attn_output)

        # Residual connection and normalization
        output = input_q + attn_output
        output_norm = self.norm2(output)

        # Feedforward network with residual connection
        ff_output = self.ffn(output_norm)
        output = output + self.dropout2(ff_output)

        return output
class DynamicMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dynamic_weights = nn.Parameter(torch.ones(num_heads, requires_grad=True))

    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        dynamic_output = attn_output * self.dynamic_weights.unsqueeze(0).unsqueeze(2)
        return dynamic_output.sum(dim=1), attn_weights
class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        super(FullyAttentionalBlock, self).__init__()
        # 定义两个全连接层，conv1和conv2
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        
        # 定义卷积层 + 归一化层 + 激活函数（ReLU）
        self.conv = nn.Sequential(
            nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),  # 卷积操作
            norm_layer(plane),  # 归一化层
            nn.ReLU()  # ReLU激活函数
        )
        
        # 定义softmax操作，用于计算关系矩阵
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化可学习的参数gamma，用于调整最终的输出
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 前向传播过程，x为输入的特征图，形状为 (batch_size, channels, height, width)
        batch_size, _, height, width = x.size()
        
        # 对输入张量进行排列和变形，获取水平和垂直方向的特征
        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)  # 水平方向特征
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)  # 垂直方向特征
        
        # 对输入张量分别在水平方向和垂直方向进行池化，并通过全连接层进行编码
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())  # 水平方向编码
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())  # 垂直方向编码
        
        # 计算水平方向和垂直方向的关系矩阵
        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))  # 计算水平方向的关系
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))  # 计算垂直方向的关系
        
        # 计算经过softmax后的关系矩阵
        full_relation_h = self.softmax(energy_h)  # 水平方向的关系
        full_relation_w = self.softmax(energy_w)  # 垂直方向的关系
        
        # 通过矩阵乘法和关系矩阵，对特征进行加权和增强
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)  # 水平方向的增强
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)  # 垂直方向的增强
        
        # 将水平和垂直方向的增强特征进行融合，并加上原始输入特征
        out = self.gamma * (full_aug_h + full_aug_w) + x
        
        # 通过卷积层进行进一步的特征处理
        out = self.conv(out)
        
        return out  # 返回处理后的特征图

class HybridAttention(nn.Module):
    def __init__(self, plane):
        super(HybridAttention, self).__init__()
        # 使用全局注意力（Cross Attention）和局部注意力（FullyAttentionalBlock）
        self.cross_attention = Cross_att(dropout=0.1, d_model=plane, n_head=4)  # 全局特征
        self.local_attention = FullyAttentionalBlock(plane)  # 局部特征
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 自适应融合权重

    def forward(self, query, key_value):
        # 计算全局特征
        global_features = self.cross_attention(query, key_value)
        
        # 计算局部特征
        local_features = self.local_attention(global_features)  # 使用局部注意力模块
        
        # 动态融合全局和局部特征
        fused_features = self.alpha * global_features + (1 - self.alpha) * local_features
        return fused_features
class CrossTransformer(nn.Module):
    """
    Cross Transformer layer with Hybrid Attention and Dynamic Multihead Attention
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        super(CrossTransformer, self).__init__()

        # 使用 HybridAttention 替代原始的注意力机制
        self.attention = HybridAttention(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.linear3 = nn.Linear(d_model, 512)

    def forward(self, input1, input2):
        # 输入图像特征
        dif = input2 - input1  # 计算输入图像之间的差异

        # 使用 HybridAttention 提取全局和局部特征
        output_1 = self.attention(input1, dif)
        output_2 = self.attention(input2, dif)
        
        # 全连接层进一步处理输出
        output_1 = self.linear3(output_1)
        output_2 = self.linear3(output_2)

        return output_1, output_2


class Cross_att(nn.Module):
    """
    Cross Transformer layer with Dynamic Multihead Attention
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        super(Cross_att, self).__init__()
        # 使用 DynamicMultiheadAttention 替代原始的 nn.MultiheadAttention
        self.attention = DynamicMultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input, dif):
        # 输入归一化
        input_norm, dif_norm = self.norm1(input), self.norm1(dif)
        # 使用动态多头注意力
        attn_output, attn_weight = self.attention(input_norm, dif_norm, dif_norm)  # (Q, K, V)
        # 残差连接和 Dropout
        output = input + self.dropout1(attn_output)

        output_norm = self.norm2(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output_norm))))
        output = output + self.dropout3(ff_output)
        return output

# 11.27日改
# 动态权重层
class DynamicWeighting(nn.Module):
    def __init__(self, n_layers):
        super(DynamicWeighting, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_layers))  # 初始化每层的权重

    def forward(self, outputs):
        """
        参数:
        outputs: List[torch.Tensor]，形状为 (batch_size, feature_dim, *) 的张量列表
        返回:
        加权融合后的输出，形状与拼接行为一致 (batch_size, feature_dim, n_layers)。
        """
        weights = torch.softmax(self.weights, dim=0)  # 计算每层的权重
        weighted_outputs = [
            w * o.unsqueeze(-1) for w, o in zip(weights, outputs)
        ]  # 对每个特征加权并添加一个新维度
        output = torch.cat(weighted_outputs, dim=-1)  # 沿最后一维拼接
        return output


class DifferenceEncoder(nn.Module):
    def __init__(self, d_model):
        super(DifferenceEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, diff):
        return self.fc(diff)

# 多尺度跨特征Transformer模块
class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model

        # n_layers = 2
        print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers

        # 添加ema模块
        # self.ema = EMA(channels=1024, factor=32)
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))

        # 加入差异识别部分，修改的部分
        self.diff_encoder = DifferenceEncoder(self.d_model)
        # self.dynamic_weighting = DynamicWeighting(self.n_layers)
        self.dynamic_weighting = DynamicWeighting(self.n_layers)
        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)
        self.transformer_cross = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        # self.transformer2 = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

        # FIXME:helpful
        encoder_self_layer = nn.TransformerEncoderLayer(2*d_model, n_head, dim_feedforward=int(4*d_model))
        self.transformer_concatlength = _get_clones(encoder_self_layer, n_layers)
        self.linear_list = nn.ModuleList([nn.Linear(2*d_model,d_model) for i in range(n_layers)])

        self.Scale_aware_list = nn.ModuleList([Scale_aware(dropout, d_model=1024, n_head=8) for i in range(n_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        if feature_dim == 1024:
            img_feat1 = self.projection(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection2(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 512:
            img_feat1 = self.projection3(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection3(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)
        if feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
            img_feat2 = self.projection4(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)

        # img_feat1 = self.ema(img_feat1)
        # img_feat2 = self.ema(img_feat2)
        
        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)  # (batch, d_model, h, w)

        img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
        img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

        encoder_output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        encoder_output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

        output1 = encoder_output1
        output2 = encoder_output2
        output1_list = list()
        output2_list = list()
        dif_list = list()
        # layer_list = list()
        for k in range(self.n_layers):
            # output1 = torch.cat([output1, output2-output1], dim=-1)
            # output1 = self.transformer_concatlength[k](output1)
            # output1 = self.linear_list[k](output1)

            # output2 = torch.cat([output2, output2 - output1], dim=-1)
            # output2 = self.transformer_concatlength[k](output2)
            # output2 = self.linear_list[k](output2)
            
            # 11.27改进
            diff = output2 - output1
            encoded_diff = self.diff_encoder(diff)
            output1 = torch.cat([output1, encoded_diff], dim=-1)
            
            # 修改output1，2
            # output1 = self.multi_scale_attn(output1, output2, output2)
            output1 = self.transformer_concatlength[k](output1)
            output1 = self.linear_list[k](output1)
            

            output2 = torch.cat([output2, encoded_diff], dim=-1)
            
            output2 = self.transformer_concatlength[k](output2)
            # output2 = self.multi_scale_attn(output2, output1, output1)
            output2 = self.linear_list[k](output2)

            output1_list.append(output1)
            output2_list.append(output2)

        output_layer1 = torch.cat([output1_list[0], output2_list[0]], dim=-1)
        output_layer2 = torch.cat([output1_list[1], output2_list[1]], dim=-1)
        output_layer3 = torch.cat([output1_list[2], output2_list[2]], dim=-1)

        # Scale_aware
        output_layer1_2 = self.Scale_aware_list[0](output_layer1,output_layer2,output_layer3)
        output_layer2_2 = self.Scale_aware_list[1](output_layer2,output_layer1,output_layer3)
        output_layer3_2 = self.Scale_aware_list[2](output_layer3,output_layer1,output_layer2)


        # output = torch.cat([output_layer1_2.unsqueeze(-1), output_layer2_2.unsqueeze(-1), output_layer3_2.unsqueeze(-1)],
        #                    dim=-1)
        output=self.dynamic_weighting([output_layer1_2, output_layer2_2, output_layer3_2])
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        # self.fc = nn.Linear(d_model, 2)
        self.fc = nn.Linear(d_model, d_model)
        # self.conv = nn.Conv2d(d_model, d_model, kernel_size=14, stride=14, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=(14, 14), stride=(14, 14))
        self.d_model = d_model
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model,num_layers=1)
        self.decode_step = nn.LSTMCell(d_model, d_model, bias=True)


        self.init_weights()


        self.init_h = nn.Linear(d_model, d_model)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(d_model, d_model)

    def init_hidden_state(self, dif):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # dif： N，d
        h = self.init_h(dif)  # (batch_size, decoder_dim)
        c = self.init_c(dif)
        return h, c

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # #
        enc_att, att_weight = self._mha_block2((self_att_tgt),
                                               torch.cat([memory[:, :, :],], dim=0), memory_mask,
                                               memory_key_padding_mask)

        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight
    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x),att_weight
    def _mha_block3(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn3(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x), att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
# 文本编码器    
class TextEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=embed_dim * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=embed_dim)  # 假设词表大小30522
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, captions, caption_lengths):
        embedded = self.embedding(captions)  # (B, L, D)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.permute(1, 0, 2)  # (L, B, D)
        encoded = self.encoder(embedded)  # (L, B, D)
        encoded = encoded.permute(1, 0, 2)  # (B, L, D)
        return encoded

class DecoderTransformer(nn.Module):
    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout=0.5, scheduled_sampling_start=0.0, scheduled_sampling_increase=0.05):
        super(DecoderTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.scheduled_sampling_prob = scheduled_sampling_start  # 初始 Teacher Forcing 概率
        
        # Embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)
        
        # Multi-Head Attention for text embedding
        self.text_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=n_head, dropout=dropout)
        
        # Transformer decoder layers
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4, dropout=dropout)
        self.transformer = _get_clones(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

        # 线性层映射到词汇表
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # 设定 Scheduled Sampling 增加步长
        self.scheduled_sampling_increase = scheduled_sampling_increase

        self.init_weights()

    def init_weights(self):
        """ 初始化 embedding 和 linear 层 """
        nn.init.uniform_(self.vocab_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.wdc.weight, -0.1, 0.1)
        nn.init.constant_(self.wdc.bias, 0)

    def caption_decoder(self, tgt_embedding, memory, tgt_mask):
        """ Transformer 解码器 """
        out = tgt_embedding
        for i, layer in enumerate(self.transformer):
            if memory.dim() == 4:
                mem_layer = memory[:, :, :, i]
            else:
                mem_layer = memory
            out = layer(out, mem_layer, tgt_mask=tgt_mask)
        return out

    def forward(self, memory, encoded_captions, caption_lengths, training=True):
        """
        memory: Transformer 编码器的输出 (batch, seq_len, feature_dim)
        encoded_captions: 编码后的目标序列 (batch, max_len)
        caption_lengths: 真实 caption 长度 (batch, 1)
        training: 是否在训练模式
        """
        device = memory.device

        # Move caption_lengths to CPU and ensure it is of type int64
        caption_lengths = caption_lengths.squeeze(1).long().cpu()

        # 对 caption 进行排序
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        memory = memory[sort_ind]

        batch_size = encoded_captions.shape[0]
        max_len = encoded_captions.shape[1]

        # 初始输入 <SOS>（假设索引 1 是 <SOS>）
        inputs = encoded_captions[:, 0].unsqueeze(1)  # (batch, 1)
        outputs = []

        for t in range(1, max_len):
            # 获取 embedding
            tgt_embedding = self.vocab_embedding(inputs)  # (batch, t, feature_dim)
            tgt_embedding = self.position_encoding(tgt_embedding)
            tgt_embedding = self.layer_norm(tgt_embedding)

            # Self-attention 处理
            attn_output, _ = self.text_attention(
                tgt_embedding.permute(1, 0, 2),  # (t, batch, feature_dim)
                tgt_embedding.permute(1, 0, 2),
                tgt_embedding.permute(1, 0, 2)
            )
            attn_output = self.layer_norm(attn_output + tgt_embedding.permute(1, 0, 2))

            # Transformer decoder
            pred = self.caption_decoder(attn_output, memory, tgt_mask=None)  # (t, batch, feature_dim)
            pred = self.wdc(self.dropout_layer(pred))  # (t, batch, vocab_size)
            pred = pred.permute(1, 0, 2)  # (batch, t, vocab_size)

            # 取出当前时间步的预测结果
            next_word_logits = pred[:, -1, :]  # (batch, vocab_size)
            next_word = next_word_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            outputs.append(next_word_logits.unsqueeze(1))  # 存储结果

            if training:
                # 计算是否使用 Teacher Forcing
                use_teacher_forcing = random.random() > self.scheduled_sampling_prob

                # 选择下一个输入（Ground Truth vs. 模型预测）
                if use_teacher_forcing:
                    next_input = encoded_captions[:, t].unsqueeze(1)  # (batch, 1) ground truth
                else:
                    next_input = next_word  # (batch, 1) 自己预测的词
            else:
                next_input = next_word  # 只在推理时使用预测结果

            # 更新输入
            inputs = torch.cat([inputs, next_input], dim=1)  # (batch, t+1)

        # 输出 (batch, max_len, vocab_size)
        outputs = torch.cat(outputs, dim=1)

        # 更新 Scheduled Sampling 概率（在训练模式下）
        if training and self.scheduled_sampling_prob < 1.0:
            self.scheduled_sampling_prob += self.scheduled_sampling_increase

        return outputs, encoded_captions, caption_lengths, sort_ind
# class DecoderTransformer(nn.Module):
#     def __init__(self, feature_dim, vocab_size, n_head, n_layers,
#                  dropout=0.5, scheduled_sampling_start=0.0, scheduled_sampling_increase=0.05):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.embed_dim = feature_dim
#         self.vocab_size = vocab_size
#         self.dropout = dropout
#         self.scheduled_sampling_prob = scheduled_sampling_start
#         self.scheduled_sampling_increase = scheduled_sampling_increase

#         # Embedding and attention
#         self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)
#         self.text_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=n_head, dropout=dropout)
#         self.position_encoding = PositionalEncoding(feature_dim)
#         self.layer_norm = nn.LayerNorm(feature_dim)

#         # Transformer decoder layers
#         decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head,
#                                                      dim_feedforward=feature_dim * 4, dropout=dropout)
#         self.transformer = _get_clones(decoder_layer, n_layers)

#         # Final linear layer
#         self.wdc = nn.Linear(feature_dim, vocab_size)
#         self.dropout_layer = nn.Dropout(p=self.dropout)

#         # 引入 TextEncoderBlock
#         self.text_encoder = TextEncoderBlock(embed_dim=feature_dim,
#                                              n_heads=n_head,
#                                              num_layers=2,
#                                              dropout=dropout)

#         self.init_weights()

#     def init_weights(self):
#         nn.init.uniform_(self.vocab_embedding.weight, -0.1, 0.1)
#         nn.init.uniform_(self.wdc.weight, -0.1, 0.1)
#         nn.init.constant_(self.wdc.bias, 0)

#     def caption_decoder(self, tgt_embedding, memory, tgt_mask=None):
#         out = tgt_embedding
#         for i, layer in enumerate(self.transformer):
#             if memory.dim() == 4:
#                 mem_layer = memory[:, :, :, i]
#             else:
#                 mem_layer = memory
#             out = layer(out, mem_layer, tgt_mask=tgt_mask)
#         return out

#     def forward(self, memory, encoded_captions, caption_lengths, training=True):
#         caption_lengths = caption_lengths.squeeze(1).long().cpu()
#         caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
#         encoded_captions = encoded_captions[sort_ind]
#         memory = memory[sort_ind]

#         # 对 GT captions 做编码（用于对比、对齐等）
#         gt_caption_embedding = self.text_encoder(encoded_captions, caption_lengths)

#         batch_size, max_len = encoded_captions.size()
#         inputs = encoded_captions[:, 0].unsqueeze(1)  # <SOS>
#         outputs = []

#         for t in range(1, max_len):
#             tgt_embedding = self.vocab_embedding(inputs)
#             tgt_embedding = self.position_encoding(tgt_embedding)
#             tgt_embedding = self.layer_norm(tgt_embedding)

#             attn_output, _ = self.text_attention(
#                 tgt_embedding.permute(1, 0, 2),
#                 tgt_embedding.permute(1, 0, 2),
#                 tgt_embedding.permute(1, 0, 2)
#             )
#             attn_output = self.layer_norm(attn_output + tgt_embedding.permute(1, 0, 2))

#             pred = self.caption_decoder(attn_output, memory)
#             pred = self.wdc(self.dropout_layer(pred))
#             pred = pred.permute(1, 0, 2)

#             next_word_logits = pred[:, -1, :]
#             next_word = next_word_logits.argmax(dim=-1, keepdim=True)
#             outputs.append(next_word_logits.unsqueeze(1))

#             if training:
#                 use_teacher_forcing = random.random() > self.scheduled_sampling_prob
#                 next_input = encoded_captions[:, t].unsqueeze(1) if use_teacher_forcing else next_word
#             else:
#                 next_input = next_word

#             inputs = torch.cat([inputs, next_input], dim=1)

#         outputs = torch.cat(outputs, dim=1)

#         if training and self.scheduled_sampling_prob < 1.0:
#             self.scheduled_sampling_prob += self.scheduled_sampling_increase

#         return outputs, encoded_captions, caption_lengths, sort_ind, gt_caption_embedding

















"""
兼容 PyTorch Geometric 2.x 的简化 GAT 层
替换 third_party/gdn/models/graph_layer.py（仅适配 PyG 2.x API）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding=None, edge_weight=None, return_attention_weights=False):
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E] (source -> target)
            embedding: [N, embed_dim] (optional)
            edge_weight: [E] learnable edge gate (optional)
        """
        N = x.size(0)
        x = self.lin(x)  # [N, heads * out_channels]
        x = x.view(N, self.heads, self.out_channels)  # [N, heads, out_channels]

        src, dst = edge_index[0], edge_index[1]  # [E]
        E = src.size(0)

        x_src = x[src]  # [E, heads, out_channels]
        x_dst = x[dst]  # [E, heads, out_channels]

        if embedding is not None:
            emb_src = embedding[src].unsqueeze(1)  # [E, 1, embed_dim]
            emb_dst = embedding[dst].unsqueeze(1)  # [E, 1, embed_dim]
            # 需要 embed_dim == out_channels，否则需要投影
            if embedding.size(-1) != self.out_channels:
                # 简单投影到 out_channels
                proj = nn.Linear(embedding.size(-1), self.out_channels, bias=False).to(embedding.device)
                emb_src = proj(emb_src.squeeze(1)).unsqueeze(1)
                emb_dst = proj(emb_dst.squeeze(1)).unsqueeze(1)
            key_src = torch.cat([x_src, emb_src.expand(-1, self.heads, -1)], dim=-1)  # [E, heads, 2*out_channels]
            key_dst = torch.cat([x_dst, emb_dst.expand(-1, self.heads, -1)], dim=-1)
            att_i = torch.cat([self.att_i, self.att_em_i], dim=-1)  # [1, heads, 2*out_channels]
            att_j = torch.cat([self.att_j, self.att_em_j], dim=-1)
        else:
            key_src = x_src
            key_dst = x_dst
            att_i = self.att_i
            att_j = self.att_j

        # 注意力分数
        alpha = (key_dst * att_i).sum(-1) + (key_src * att_j).sum(-1)  # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # softmax over source nodes for each destination node
        alpha = self._softmax(alpha, dst, N)  # [E, heads]

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 聚合: sum over neighbors
        msg = x_src * alpha.unsqueeze(-1)  # [E, heads, out_channels]
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1).unsqueeze(-1)  # [E, heads, out_channels]

        # scatter_add to destination nodes
        out = torch.zeros(N, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        dst_expanded = dst.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.out_channels)
        out.scatter_add_(0, dst_expanded, msg)

        if self.concat:
            out = out.view(N, -1)  # [N, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [N, out_channels]

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha_ret = self.__alpha__
            self.__alpha__ = None
            return out, (edge_index, alpha_ret)
        return out

    def _softmax(self, src, index, num_nodes):
        """
        按 index（目标节点）做 softmax
        src: [E, heads]
        index: [E] dst node indices
        """
        N = num_nodes
        H = src.size(1)

        # 数值稳定性: 减去每个 dst 的最大值
        src_max = torch.full((N, H), fill_value=float('-inf'), device=src.device, dtype=src.dtype)
        src_max = src_max.scatter_reduce(0, index.unsqueeze(1).expand(-1, H), src, reduce='amax', include_self=False)
        src_max = src_max[index]

        out = (src - src_max).exp()

        out_sum = torch.zeros(N, H, device=src.device, dtype=src.dtype)
        out_sum = out_sum.scatter_add(0, index.unsqueeze(1).expand(-1, H), out)
        out_sum = out_sum[index]

        return out / (out_sum + 1e-16)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

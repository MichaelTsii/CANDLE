import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GCNConv

from dataset_traffic import time_length

def get_torch_trans(heads=8, layers=1, channels=64):
    '''
    函数的作用是构建一个 Transformer 编码器，用于处理序列数据。
    参数包括头数 (heads)、层数 (layers) 和通道数 (channels)。
    '''
    #首先创建一个 Transformer 编码器层 (nn.TransformerEncoderLayer)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    
    #然后创建一个 Transformer 编码器 (nn.TransformerEncoder)，并返回该编码器的实例
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    '''
    定义并Kaiming初始化一个1D卷积层
    '''
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class Conv1dSameShape(nn.Module):
    def __init__(self, channels):
        super(Conv1dSameShape, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x shape: (B, L, C)
        # Permute to shape (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # Permute back to shape (B, L, C)
        x = x.permute(0, 2, 1)
        return x

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        '''
        num_steps:扩散步数, projection_dim:投影维度
        '''
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )#注册为模块的缓冲区（buffer）
        
        # 创建两个线性投影层，用于将嵌入映射到指定的维度
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        '''
        创建嵌入表
        '''
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, device, inputdim=1):
        super().__init__()
        self.channels = config["channels"]
        self.device = device

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1) # kernal Size 由 1 改为了 3
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    device=self.device,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, enc_outputs, cond_info, diffusion_step, observed_4G_around, latitude_4G_around, longitude_4G_around):
        B, K, L = x.shape
        x = x.unsqueeze(1)#B, 1, K, L
        x = x.reshape(B, 1, K * L)
        x1 = x
        x = self.input_projection(x)
        x2 = x
        x = F.relu(x)
        x3 = x
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, enc_outputs, cond_info, diffusion_emb, observed_4G_around, latitude_4G_around, longitude_4G_around)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, device):
        super().__init__() #调用父类的构造方法
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.device = device
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=64, nhead=8, dim_feedforward=64, activation="gelu", bias=False
            ),
            num_layers = 1,
        ).to(self.device)
        
        self.GCN_module = GCN(in_channels=time_length*64, hidden_channels=32, out_channels=time_length*64, dropout=0.3).to(self.device)

        self.gcn_projection = None 
        self.Conv1d_same_shape = Conv1dSameShape(channels)         

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def gcn_projection_init_(self, input_c, out_c):
        hidden_c = input_c * out_c
        self.gcn_projection = nn.Sequential(
            nn.Linear(input_c, hidden_c, dtype=torch.float),
            nn.Tanh(),
            nn.Linear(hidden_c, hidden_c, dtype=torch.float),
            nn.Tanh(), 
            nn.Linear(hidden_c, out_c, dtype=torch.float)
        ).to(self.device)
    
    def forward(self, x, enc_outputs, cond_info, diffusion_emb, observed_4G_around, latitude_4G_around, longitude_4G_around):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        # 对扩散嵌入进行线性投影，并加上输入数据
        diffusion_emb_after = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb_after # 可参见Figure 6

        # 在时间和特征方向上进行变换
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)=(B,C,L),下面就按K=1来做
        y = y.permute(0, 2, 1) # →(B,L,C)
        B, L, C = y.shape
        
        # 初始化GCN模型
        self.GCN_module.in_channels = L*C
        self.GCN_module.out_channels = L*C

        z = []
        
        for i in range(B):
            N = len(observed_4G_around[i]) ## 验证一下是不是传入的每个batch对应的index的4G

            tgt = y[i].unsqueeze(0).expand(N, -1, -1).permute(1, 0, 2) # (L, N, C) 
            enc_output_one = enc_outputs[i].to(torch.float) # (N, L, C)
            dec_input = enc_output_one.permute(1, 0, 2) # (L, N, C)
            
            dec_output = self.decoder(tgt, dec_input) # (L, N, C)
            dec_output = dec_output.permute(1, 0, 2) # (N, L, C)

            features = dec_output # (N, L, C) 
            
            # 构建节点图
            if N > 1:
                coords = []
                for j in range(N):
                    coords.append([float((longitude_4G_around[i])[j]), float((latitude_4G_around[i])[j])])
                coords = np.array(coords)
                # print("coords: ", coords)
                # 计算所有节点对之间的欧氏距离
                distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))

                # 为图创建边和边权重
                edge_index = [[i, j] for i in range(N) for j in range(N) if i != j]
                edge_weight = [np.exp(-5e2 * distances[i, j]) for i in range(N) for j in range(N) if i != j]                

                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(edge_weight, dtype=torch.float)

                # 构建图数据
                graph_data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight).to(self.device)
                
                # 通过GCN提取特征
                GCN_output = self.GCN_module(graph_data, N, L, C)
            else:
                GCN_output = features
            
            # 做维度变换，把N变到最后一维去：(N, L, C)->(L, C, N)
            projection_input= GCN_output.permute(1, 2, 0)           
            # 如果想跳过GCN，用下面这句
            # projection_input= dec_output.permute(1, 2, 0) #(L, C, N)
            
            # 使用MLP做维度映射,最后一维N->1——————————————————————— 
            # if self.gcn_projection ==None:
            #     self.gcn_projection_init_(N ,1)
            
            # projection_output = self.gcn_projection(projection_input) #(L, C, 1)
            # self.gcn_projection = None
            # ————————————————————————————————————————
            projection_output = torch.mean(projection_input, dim=-1).unsqueeze(-1)
            
            y_after_extraction = projection_output.permute(2, 0, 1).reshape(L, C) #(1, L, C)->(, L, C)

            # 把GCN结果加在原数据上组成新数组作为后续输入
            z.append(y[i] + y_after_extraction)
        
        z = torch.stack(z)

        z = z.permute(0, 2, 1) # →(B,C,L)
        z = self.mid_projection(z)  # (B,2*channel,K*L),1D卷积

        # 处理条件信息，并结合到处理后的数据中
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        z = z + cond_info # 可参见Figure 6,如何添加入side_info

        # 执行门控操作，使用 sigmoid 函数作为门控，tanh 函数作为过滤器
        gate, filter = torch.chunk(z, 2, dim=1)
        z = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        z = self.output_projection(z)

        # 分离残差项和用于跳跃连接的项
        residual, skip = torch.chunk(z, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        # 返回残差块的输出，包括残差项和用于跳跃连接的项
        return (x + residual) / math.sqrt(2.0), skip

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(GCN, self).__init__()       
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = GCNConv(self.in_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, self.out_channels)
        self.bn3 = nn.BatchNorm1d(self.out_channels)  # 批量归一化层1

        self.dropout_rate = dropout

    def forward(self, data, N, L, C):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index
        edge_weight = edge_weight

        # 将特征向量展平以用于GCN
        # x = x.view(N, -1)
        x = x.reshape(N, -1)

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # self.conv3 = self.conv3.to(x.device)
        x = self.conv3(x, edge_index, edge_weight)

        # 重新将特征向量变回原始维度
        x = x.view(N, L, C)
        return x
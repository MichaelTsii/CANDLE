import numpy as np
import torch
import torch.nn as nn
import math
import pickle
from diff_models import diff_CSDI

from dataset_traffic import Area_nums, data_name_4G, latitude_lowlim, latitude_uplim, longitude_lowlim, longitude_uplim, radius_5G_station, time_length, area_selected

use_4G_pattern = False

class N_for_grid_Embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(N_for_grid_Embedding, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        x = self.linear(x)  # Apply linear transformation
        x = self.relu(x)
        x = x.view(batch_size, seq_len, 1, -1)  # Reshape to [batch_size, seq_len, 1, output_size]
        return x

class CSDI_base(nn.Module):

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.mse = nn.MSELoss()

        self.noisy_rate = 1e0  # 生成噪声项的幅度

        # 从配置中获取时间嵌入维度 (timeemb)，该维度用于生成时间嵌入。
        self.emb_time_dim = config["model"]["timeemb"]
        # 从配置中获取特征嵌入维度 (featureemb)，该维度用于生成特征嵌入。
        self.emb_feature_dim = config["model"]["featureemb"]
        # 从配置中获取5G基站数量嵌入维度 (N_emb)，该维度用于生成特征嵌入。
        self.emb_N_dim = config["model"]["N_emb"]
        # 从配置中获取是否为无条件模型 (is_unconditional)，即模型是否仅基于噪声进行扩散，而不考虑条件观测。
        self.is_unconditional = config["model"]["is_unconditional"]
        # 从配置中获取是否使用4G流量的pattern (use_4G_pattern)。
        self.use_4G_pattern = config["model"]["use_4G_pattern"]
        self.channels = config["diffusion"]["channels"]

        print("self.is_unconditional: ", self.is_unconditional)
        print("self.use_4G_pattern: ", self.use_4G_pattern)

        # 计算总的嵌入维度
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + self.emb_N_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        # 配置扩散模型
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, self.device, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"]**0.5, config_diff["beta_end"]**0.5, self.num_steps)**2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)  # 累乘函数
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

        self.radius_5G_station = config["diffusion"]["radius_5G_station"]

        # 读取4G数据
        self.path_4G = ("./data/data_cache/" + "4G_" + str(longitude_lowlim) + "~" + str(longitude_uplim) + "_R" + str(radius_5G_station) + "_" + area_selected + ".pk")
        with open(self.path_4G, "rb") as f:
            self.observed_4G_around, self.latitude_4G_around, self.longitude_4G_around = pickle.load(f)
        self.observed_4G_around = [_.to(self.device) for _ in self.observed_4G_around]
        # self.clusters_4G_around = [_.to(self.device) for _ in self.clusters_4G_around]
        self.latitude_4G_around = [_.to(self.device) for _ in self.latitude_4G_around]
        self.longitude_4G_around = [_.to(self.device) for _ in self.longitude_4G_around]

        self.N_for_grid_Embedding = N_for_grid_Embedding(1, self.emb_N_dim).to(device)

    def time_embedding(self, pos, d_model=128):
        '''
        self, pos, d_model=128
        生成时间嵌入，通过对时间进行编码而得到向量表示
        返回生成的时间嵌入矩阵 pe,其中每一行代表一个时间步的嵌入向量,形状为 (B, L, d_model)
        '''
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_4G_encoded(self, observed_4G_around):

        N, L = observed_4G_around.shape  #(N, L)
        channel = self.channels
        
        encoder_4G = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64, nhead=8, dim_feedforward=64, activation="gelu", bias=False
            ),
            num_layers = 1,
        ).to(self.device)

        # 在时间维提取pattern
        src = observed_4G_around.unsqueeze(1).expand(-1, channel, -1) #用映射代替expand, (N, C, L)
        src = src.permute(2, 0, 1).to(torch.float) #(L, N, C)
        enc_output = encoder_4G(src) #输入：(L, N, C), 输出(L, N, C)
        
        enc_output = enc_output.permute(1, 0, 2) # (N, L, C)

        # 如果想跳过Encoder、Decoder
        # enc_output = src.permute(1, 0, 2).to(torch.float) 

        return enc_output #(N, L, C)
    
    def get_enc_outputs(self, B, observed_4G_around):
        enc_outputs = []
        for i in range(B):
            enc_output_one = self.get_4G_encoded(observed_4G_around[i]) # (N, L, C)
            enc_outputs.append(enc_output_one)
        
        return enc_outputs # (B, N, L, C)

    def get_side_info(self, observed_tp, observed_data, N_for_grids):
        B, K, L = observed_data.shape

        # Time Embedding
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb),生成时间嵌入
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        # Feature Embedding
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        N_for_grids = N_for_grids.unsqueeze(-1).unsqueeze(-1)
        N_for_grids = N_for_grids.expand(-1, time_length, 1)

        N_embeded = self.N_for_grid_Embedding(N_for_grids)

        # 将时间嵌入和特征嵌入在最后一维度上进行拼接，形成 side_info，其形状为 (B, L, K, emb_time_dim + emb_feature_dim + emb_N_dim)。
        side_info = torch.cat([time_embed, feature_embed, N_embeded], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L),*=emb_time_dim + emb_feature_dim + emb_N_dim

        if self.is_unconditional == False:
            side_info = torch.cat([side_info, torch.ones((B, 1, K, L)).to(self.device)], dim=1)

        return side_info

    def calc_loss_valid(self, observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, is_train):
        '''
        计算在模型验证过程中的损失, 取多个时间步并计算均值
        '''
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, is_train, set_t=t)
            loss_sum += loss.detach()

        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, is_train, set_t=-1):
        '''
        计算模型在给定时间步 t 下的损失
        '''
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (1.0 - current_alpha)**0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data)

        enc_outputs = self.get_enc_outputs(B, observed_4G_around) # (B, N, L, C)

        predicted = self.diffmodel(total_input, enc_outputs, side_info, t, 
                                   observed_4G_around, latitude_4G_around, longitude_4G_around)  # (B,K,L)

        loss = self.mse(noise, predicted)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data):
        if self.is_unconditional == True:
            total_input = noisy_data  # (B,1,K,L)
        else:
            cond_obs = observed_data.unsqueeze(1)
            noisy_target = noisy_data.unsqueeze(1)  #在没有观测到的位置引入噪声
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, n_samples):
        '''
        对观测数据进行多次插补，生成多个可能的插补样本
        返回包含生成的插补样本的张量
        '''
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                diff_input = current_sample
                
                enc_outputs = self.get_enc_outputs(B, observed_4G_around) # (B, N, L, C)

                predicted = self.diffmodel(diff_input, enc_outputs, side_info, torch.tensor([t]).to(self.device), 
                                           observed_4G_around, latitude_4G_around, longitude_4G_around)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (observed_data,
        #  clusters_5G,
        #  latitude_5G,
        #  longitude_5G,
         N_for_grids,
         observed_tp,
         idex_test
         ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, observed_data, N_for_grids)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        observed_4G_around = [self.observed_4G_around[i] for i in list(idex_test)]
        latitude_4G_around = [self.latitude_4G_around[i] for i in list(idex_test)]
        longitude_4G_around = [self.longitude_4G_around[i] for i in list(idex_test)]

        return loss_func(observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data,
        #  clusters_5G,
        #  latitude_5G,
        #  longitude_5G,
         N_for_grids,
         observed_tp,
         idex_test
         ) = self.process_data(batch)
        
        observed_4G_around = [self.observed_4G_around[i] for i in list(idex_test)]
        latitude_4G_around = [self.latitude_4G_around[i] for i in list(idex_test)]
        longitude_4G_around = [self.longitude_4G_around[i] for i in list(idex_test)]

        with torch.no_grad():
            side_info = self.get_side_info(observed_tp, observed_data, N_for_grids)

            samples = self.impute(observed_data, observed_4G_around, latitude_4G_around, longitude_4G_around, side_info, n_samples)

        return samples, observed_data, observed_tp

class CSDI_Value(CSDI_base):
    def __init__(self, config, device, target_dim=Area_nums()):
        super(CSDI_Value, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # 一个batch的5G数据
        observed_data = batch["observed_5G"].to(self.device).float()
        # clusters_5G = batch["clusters_5G"].to(self.device).int()
        # latitude_5G = batch["latitude_5G"].to(self.device).float()
        # longitude_5G = batch["longitude_5G"].to(self.device).float()
        N_for_grids = batch["N_for_grids"].to(self.device).float()
        # 时序数据
        observed_tp = batch["timepoints"].to(self.device).float()
        idex_test = batch["idex_test"].to(self.device).int()

        observed_data = observed_data.permute(0, 2, 1)

        return (
            observed_data,
            # clusters_5G,
            # latitude_5G,
            # longitude_5G,
            N_for_grids,
            observed_tp,
            idex_test
        )

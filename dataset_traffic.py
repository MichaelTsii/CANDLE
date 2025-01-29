import pickle
import torch
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import requests

# 数据集基本参数
data_name_5G = 'Traffic_5G'
data_name_4G = 'Traffic_4G'
info_name_5G = 'Cell_Info_5G'
info_name_4G = 'Cell_Info_4G'
data_filter_2022 = "./data/nanchang/2022/"
data_filter_2023 = "./data/nanchang/2023/"
file_format = '.npy'
classify_method = 'distance'
time_length = 168
radius_5G_station = 120
D = radius_5G_station * np.sqrt(2) #单位：m

# 仅研究如下城市区域中的数据
area_selected = '市区'
district_list = ['西湖区', '东湖区', '红谷滩区']
latitude_lowlim = 28.6
latitude_uplim = 28.8
longitude_lowlim = 115.83
longitude_uplim = 115.88

# ———————————————————————————————————————————————————
def Area_nums():
    area_channel = 1
    return area_channel

def d2m(l):
    out = l*8.5e4
    return out

def m2d(l):
    out = l/8.5e4
    return out

# def normalize(record):
#     flatten_record = record.reshape(1, -1)

#     mean_val = np.mean(flatten_record, axis=-1)
#     std_val = np.std(flatten_record, axis=-1)
#     flatten_normalized_data = (flatten_record.T - mean_val) / std_val
#     flatten_normalized_data = flatten_normalized_data.T

#     normalized_data = flatten_normalized_data.reshape(record.shape)

#     return normalized_data

def normalize(record):
    flatten_record = record.reshape(1, -1)

    min_val = np.min(flatten_record, axis=-1)
    max_val = np.max(flatten_record, axis=-1)
    flatten_normalized_data = (flatten_record.T - min_val) / (max_val - min_val)
    flatten_normalized_data = flatten_normalized_data.T

    normalized_data = flatten_normalized_data.reshape(record.shape)

    return normalized_data

def get_location_info(latitude, longitude):
    # 构造请求URL
    url = f"https://tools.mgtv100.com/external/v1/amap/regeo?location={longitude},{latitude}"

    try:
        # 发送GET请求
        response = requests.get(url)

        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON格式的响应数据
            data = response.json()

            # 提取地址信息
            district = data['data']['regeocode']['addressComponent']['district']
            return district
        else:
            print(f"Error: Failed to retrieve data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def data_loader_2022(data_path_5G, info_path_5G, data_path_4G, info_path_4G, latitude_lowlim, latitude_uplim, longitude_lowlim, longitude_uplim):
    
    traffic_5G = np.load(data_path_5G, allow_pickle=True).item()
    traffic_5G = {k: v for k, v in traffic_5G.items() if np.std(np.array(list(v))) != 0}

    Cell_info_5G = np.load(info_path_5G, allow_pickle=True).item()
    Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if float(v[12]) > latitude_lowlim and float(v[12]) < latitude_uplim and float(v[11]) > longitude_lowlim and float(v[11]) < longitude_uplim}
    # if area_selected == '市区':
    #     Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if get_location_info(float(v[12]), float(v[11])) in district_list}
    # else:
    #     Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if get_location_info(float(v[12]), float(v[11])) not in district_list}

    RECORD_5G = []
    latitude_5G = []
    longitude_5G = []
    num_5G = 0
    for k, v in traffic_5G.items():
        v_one = np.array(v[::2])
        if k in Cell_info_5G.keys() and np.std(v_one) != 0:
            num_5G = num_5G + 1
            RECORD_5G.append(v_one)
            latitude_5G.append(float(Cell_info_5G[k][12]))
            longitude_5G.append(float(Cell_info_5G[k][11]))

    traffic_4G = np.load(data_path_4G, allow_pickle=True).item()
    traffic_4G = {k: v for k, v in traffic_4G.items() if np.std(np.array(list(v))) != 0}

    Cell_info_4G = np.load(info_path_4G, allow_pickle=True).item()
    Cell_info_4G = {k: v for k, v in Cell_info_4G.items() if float(v[7]) > latitude_lowlim and float(v[7]) < latitude_uplim and float(v[6]) > longitude_lowlim and float(v[6]) < longitude_uplim}
    # if area_selected == '市区':
    #     Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if get_location_info(float(v[7]), float(v[6])) in district_list}
    # else:
    #     Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if get_location_info(float(v[7]), float(v[6])) not in district_list}

    RECORD_4G = []
    latitude_4G = []
    longitude_4G = []
    num_4G = 0
    for k, v in traffic_4G.items():
        v_one = np.array(v[::2])
        if k in Cell_info_4G.keys() and np.std(v_one) != 0:
            num_4G = num_4G + 1
            RECORD_4G.append(v_one)
            latitude_4G.append(float(Cell_info_4G[k][7]))
            longitude_4G.append(float(Cell_info_4G[k][6]))

    RECORD_4G = np.array(RECORD_4G)
    RECORD_5G = np.array(RECORD_5G)
    
    return RECORD_4G, latitude_4G, longitude_4G, num_4G, RECORD_5G, latitude_5G, longitude_5G, num_5G

def data_loader_2023(data_path_5G, info_path_5G, data_path_4G, info_path_4G, latitude_lowlim, latitude_uplim, longitude_lowlim, longitude_uplim):
    
    traffic_5G = np.load(data_path_5G, allow_pickle=True).item()
    traffic_5G = {k: v for k, v in traffic_5G.items() if np.std(np.array(list(v))) != 0}

    Cell_info_5G = np.load(info_path_5G, allow_pickle=True).item()
    Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if v['latitude'] > latitude_lowlim and v['latitude'] < latitude_uplim and v['longitude'] > longitude_lowlim and v['longitude'] < longitude_uplim}
    if area_selected == '市区':
        Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if v['county'] in district_list}
    else:
        Cell_info_5G = {k: v for k, v in Cell_info_5G.items() if v['county'] not in district_list}

    RECORD_5G = []
    latitude_5G = []
    longitude_5G = []
    num_5G = 0
    for k, v in traffic_5G.items():
        v_one = np.array(v[::4])
        if k in Cell_info_5G.keys() and np.std(v_one) != 0:
            num_5G = num_5G + 1
            RECORD_5G.append(v_one)
            latitude_5G.append(Cell_info_5G[k]['latitude'])
            longitude_5G.append(Cell_info_5G[k]['longitude'])


    traffic_4G = np.load(data_path_4G, allow_pickle=True).item()
    traffic_4G = {k: v for k, v in traffic_4G.items() if np.std(np.array(list(v))) != 0}

    Cell_info_4G = np.load(info_path_4G, allow_pickle=True).item()
    Cell_info_4G = {k: v for k, v in Cell_info_4G.items() if v['latitude'] > latitude_lowlim and v['latitude'] < latitude_uplim and v['longitude'] > longitude_lowlim and v['longitude'] < longitude_uplim}
    if area_selected == '市区':
        Cell_info_4G = {k: v for k, v in Cell_info_4G.items() if v['county'] in district_list}
    else:
        Cell_info_4G = {k: v for k, v in Cell_info_4G.items() if v['county'] not in district_list}

    RECORD_4G = []
    latitude_4G = []
    longitude_4G = []
    num_4G = 0
    for k, v in traffic_4G.items():
        v_one = np.array(v[::4])
        if k in Cell_info_4G.keys() and np.std(v_one) != 0:
            num_4G = num_4G + 1
            RECORD_4G.append(v_one)
            latitude_4G.append(Cell_info_4G[k]['latitude'])
            longitude_4G.append(Cell_info_4G[k]['longitude'])

    RECORD_4G = np.array(RECORD_4G)
    RECORD_5G = np.array(RECORD_5G)
    
    return RECORD_4G, latitude_4G, longitude_4G, num_4G, RECORD_5G, latitude_5G, longitude_5G, num_5G

def grid_statistic(D, num_4G, traffic_4G, latitude_4G, longitude_4G, num_5G, traffic_5G, latitude_5G, longitude_5G):
    latitude_lowlim = max(min(latitude_4G),min(latitude_5G))
    latitude_uplim = min(max(latitude_4G), max(latitude_5G))
    longitude_lowlim = max(min(longitude_4G),min(longitude_5G))
    longitude_uplim = min(max(longitude_4G), max(longitude_5G))
    
    traffic_4G_units = []
    traffic_5G_units = []
    latitude_4G_units = []
    longitude_4G_units = []

    grid_num = 0
    for i in range(int(d2m(longitude_uplim - longitude_lowlim) // D)):
        for j in range(int(d2m(latitude_uplim - latitude_lowlim) // D)):
            unit_4G = []
            unit_5G = []
            lat_unit_4G = []
            long_unit_4G = []

            for k in range(num_4G):
                if latitude_4G[k] >= latitude_lowlim + j * m2d(D) and latitude_4G[k] <= latitude_lowlim + (j+1) * m2d(D) and longitude_4G[k] >= longitude_lowlim + i * m2d(D) and longitude_4G[k] <= longitude_lowlim + (i+1) * m2d(D):
                    unit_4G.append(torch.tensor(traffic_4G[k]))
                    lat_unit_4G.append(torch.tensor(latitude_4G[k], dtype=torch.float64))
                    long_unit_4G.append(torch.tensor(longitude_4G[k], dtype=torch.float64))

            for k in range(num_5G):
                if latitude_5G[k] >= latitude_lowlim + j * m2d(D) and latitude_5G[k] <= latitude_lowlim + (j+1) * m2d(D) and longitude_5G[k] >= longitude_lowlim + i * m2d(D) and longitude_5G[k] <= longitude_lowlim + (i+1) * m2d(D):
                    unit_5G.append(traffic_5G[k])
            
            if len(unit_4G) > 0:
                unit_4G = torch.stack(unit_4G)
                lat_unit_4G = torch.stack(lat_unit_4G)
                long_unit_4G = torch.stack(long_unit_4G)
                
                traffic_4G_units.append(unit_4G)
                latitude_4G_units.append(lat_unit_4G)
                longitude_4G_units.append(long_unit_4G)
                
                unit_5G = np.array(unit_5G)
                traffic_5G_units.append(unit_5G)

                grid_num = grid_num + 1
    
    return grid_num, traffic_4G_units, traffic_5G_units, latitude_4G_units, longitude_4G_units

def parse_id(observed_data):
    return np.expand_dims(observed_data, axis=-1)
    
class Traffic_Dataset(Dataset):
    def __init__(self, eval_length=time_length, use_index_list=None,seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice
        
        path_5G = ("./data/data_cache/" + "5G_" + str(longitude_lowlim) + "~" + str(longitude_uplim) + "_R" + str(radius_5G_station) + "_" + area_selected + ".pk")
        path_4G = ("./data/data_cache/" + "4G_" + str(longitude_lowlim) + "~" + str(longitude_uplim) + "_R" + str(radius_5G_station) + "_" + area_selected + ".pk")

        # 12.16添加；当筛选、处理后的数据文件不存在时，处理和储存5G、4G数据
        if os.path.isfile(path_5G) == False or os.path.isfile(path_4G) == False:  # if datasetfile is none, create
            
            data_path_5G_2022 = data_filter_2022 + data_name_5G + file_format
            info_path_5G_2022 = data_filter_2022 + info_name_5G + file_format
            data_path_4G_2022 = data_filter_2022 + data_name_4G + file_format
            info_path_4G_2022 = data_filter_2022 + info_name_4G + file_format

            data_path_5G_2023 = data_filter_2023 + data_name_5G + file_format
            info_path_5G_2023 = data_filter_2023 + info_name_5G + file_format
            data_path_4G_2023 = data_filter_2023 + data_name_4G + file_format
            info_path_4G_2023 = data_filter_2023 + info_name_4G + file_format

            RECORD_4G_2022, latitude_4G_2022, longitude_4G_2022, num_4G_2022, RECORD_5G_2022, latitude_5G_2022, longitude_5G_2022, num_5G_2022 \
                = data_loader_2022(data_path_5G_2022, info_path_5G_2022, data_path_4G_2022, info_path_4G_2022, latitude_lowlim, latitude_uplim, longitude_lowlim, longitude_uplim)
            RECORD_4G_2023, latitude_4G_2023, longitude_4G_2023, num_4G_2023, RECORD_5G_2023, latitude_5G_2023, longitude_5G_2023, num_5G_2023 \
                = data_loader_2023(data_path_5G_2023, info_path_5G_2023, data_path_4G_2023, info_path_4G_2023, latitude_lowlim, latitude_uplim, longitude_lowlim, longitude_uplim)

            # 数据归一化
            normalized_4G_2022 = normalize(RECORD_4G_2022)
            normalized_5G_2022 = normalize(RECORD_5G_2022)

            normalized_4G_2023 = normalize(RECORD_4G_2023)
            normalized_5G_2023 = normalize(RECORD_5G_2023)

            print("num_5G_2022: ", len(RECORD_5G_2022))
            print("num_4G_2022: ", len(RECORD_4G_2022))
            print(" ")
            print("num_5G_2023: ", len(RECORD_5G_2023))
            print("num_4G_2023: ", len(RECORD_4G_2023))

            _, traffic_4G_units_2022, traffic_5G_units_2022, latitude_4G_units_2022, longitude_4G_units_2022 = grid_statistic(D, num_4G_2022, normalized_4G_2022, latitude_4G_2022, longitude_4G_2022, num_5G_2022, normalized_5G_2022, latitude_5G_2022, longitude_5G_2022)
            grid_num, traffic_4G_units_2023, traffic_5G_units_2023, _, _ = grid_statistic(D, num_4G_2023, normalized_4G_2023, latitude_4G_2023, longitude_4G_2023, num_5G_2023, normalized_5G_2023, latitude_5G_2023, longitude_5G_2023)
            # print("Grid num: ", grid_num)

            # 筛选合适的网格
            valid_grid_num = 0
            selected_traffic_4G = []
            selected_traffic_5G = []
            selected_num_5G_units = []
            selected_lat_4G = []
            selected_long_4G = []

            for i in range(grid_num):
                if traffic_5G_units_2022[i].shape[0] == 0 and traffic_5G_units_2023[i].shape[0] > 0 and traffic_4G_units_2022[i].shape[0] > 0 and traffic_4G_units_2023[i].shape[0] > 0:
                    valid_grid_num = valid_grid_num + 1
                    selected_traffic_4G.append(traffic_4G_units_2022[i])
                    selected_traffic_5G.append(traffic_5G_units_2023[i])
                    selected_num_5G_units.append(traffic_5G_units_2023[i].shape[0]) # 每个网格内的5G基站数量
                    selected_lat_4G.append(latitude_4G_units_2022[i])
                    selected_long_4G.append(longitude_4G_units_2022[i])

            N_for_grids = selected_num_5G_units
            
            observed_5G = []
            for _ in selected_traffic_5G:
                observed_5G.append(np.sum(_, axis=0))
            observed_5G = np.array(observed_5G)
        
            # 处理和储存5G数据
            self.observed_5G = np.array(parse_id(observed_5G))
            # self.clusters_5G = clusters_5G
            # self.latitude_5G = latitude_5G
            # self.longitude_5G = longitude_5G
            # self.num_4G_around = num_4G_around
            self.N_for_grids = N_for_grids
            # print("Valid_grid_num: ", len(self.observed_5G))

            with open(path_5G, "wb") as f:
                pickle.dump([self.observed_5G, self.N_for_grids], f)
        
            # 处理和储存4G数据，用于feature
            self.observed_4G_around = selected_traffic_4G
            # self.clusters_4G_around = clusters_4G_around
            self.latitude_4G_around = selected_lat_4G
            self.longitude_4G_around = selected_long_4G

            with open(path_4G, "wb") as f:
                pickle.dump([self.observed_4G_around, self.latitude_4G_around, self.longitude_4G_around], f)
        # --------------------------------------------------------

        # 读取5G数据
        with open(path_5G, "rb") as f:
            self.observed_5G, self.N_for_grids = pickle.load(f)
        print("Valid_grid_num: ", len(self.observed_5G))
        
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_5G))
        else:
            self.use_index_list = use_index_list
    
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            # 一个batch的5G数据
            "observed_5G": self.observed_5G[index],
            # "latitude_5G": np.array(self.latitude_5G[index]),
            # "longitude_5G": np.array(self.longitude_5G[index]),
            "N_for_grids": np.array(self.N_for_grids[index]),
            # 时序数据
            "timepoints": np.arange(self.eval_length),
            "idex_test": index,
        }

        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(seed=1, nfold=None, batch_size=16):

    # only to obtain total length of dataset
    dataset = Traffic_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    num_train = (int)(len(dataset) * 0.8)
    train_index = indlist[:num_train]
    test_index = indlist[num_train:]

    np.random.seed(seed)
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)

    valid_index = test_index

    dataset = Traffic_Dataset(
        use_index_list=train_index,seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Traffic_Dataset(
        use_index_list=valid_index, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=1)
    test_dataset = Traffic_Dataset(
        use_index_list=test_index, seed=seed
    )#test_index
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
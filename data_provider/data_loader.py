import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # 初始化参数
        self.seq_len = size[0] if size else 24 * 4 * 4  # 输入序列长度
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.class_names = None  # 存储类别名称
        self.label_encoder = LabelEncoder()  # 标签编码器

        # 数据集划分类型
        assert flag in ['train', 'test', 'val']
        self.set_type = flag
        self.__read_data__()

    def __read_data__(self):
        """读取数据并生成时间窗口和分类标签"""
        # 1. 读取原始数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 2. 提取特征和标签
        # --- 关键修改：明确分离特征和分类标签 ---
        # 假设标签列名为 self.target（例如 'class'）
        labels = df_raw[self.target].values  # 原始标签（可能是字符串或整数）
        
        # 标签编码为整数索引（例如 "cat"→0, "dog"→1）
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        self.class_names = self.label_encoder.classes_  

        # 提取特征列（排除标签列和日期列）
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_data = df_raw[cols]  # 特征数据

        # 3. 标准化特征（仅用训练集拟合）
        if self.scale:
            self.scaler = StandardScaler()
            if self.set_type == 'train':
                # 训练集：拟合并转换
                self.data_x = self.scaler.fit_transform(df_data.values)
            else:
                # 验证/测试集：用训练集的均值和方差转换
                train_df = pd.read_csv(os.path.join(self.root_path, self.data_path))
                train_data = train_df[cols].values
                self.scaler.fit(train_data)
                self.data_x = self.scaler.transform(df_data.values)
        else:
            self.data_x = df_data.values

        # 4. 划分数据集（训练/验证/测试）
        
        # 初始化分层抽样对象，划分训练集、验证集和测试集
        sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)  # 训练集和验证集+测试集的比例为7:3
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=1/2, random_state=42)  # 验证集和测试集的比例为1:1

        # 第一次划分，得到训练集和剩余部分（验证集+测试集）
        for train_index, val_test_index in sss_train_val.split(self.data_x, encoded_labels):
            X_train = self.data_x[train_index]
            y_train = encoded_labels[train_index]
            X_val_test = self.data_x[val_test_index]
            y_val_test = encoded_labels[val_test_index]

        # 第二次划分，将剩余部分划分为验证集和测试集
        for val_index, test_index in sss_val_test.split(X_val_test, y_val_test):
            X_val = X_val_test[val_index]
            y_val = y_val_test[val_index]
            X_test = X_val_test[test_index]
            y_test = y_val_test[test_index]

        # 根据数据集类型选择数据
        if self.set_type == 'train':
            self.data_x = X_train
            self.data_y = y_train
        elif self.set_type == 'val':
            self.data_x = X_val
            self.data_y = y_val
        else:
            self.data_x = X_test
            self.data_y = y_test

    def __getitem__(self, index):
        """返回一个时间窗口和对应的类别标签"""
        # 输入序列 X: (seq_len, feat_dim)
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]  # 形状 (seq_len, feat_dim)

        # 标签 y: 类别索引（整数）
        label = self.data_y[index]  # 形状 (1,)

        return torch.from_numpy(seq_x).float(), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """返回有效窗口数量"""
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        """反向标准化（可选）"""
        return self.scaler.inverse_transform(data) if self.scale else data


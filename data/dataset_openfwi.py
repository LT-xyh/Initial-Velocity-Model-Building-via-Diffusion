import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class OpenFWI(Dataset):
    """
    OpenFWI数据集加载类。

    该类从OpenFWI数据目录读取不同类型的数据（如速度模型、成像结果、井数据等），
    并根据配置对数据执行归一化处理。可选地，在rms_vel归一化前进行高斯平滑，
    用于数据增强或模拟测量误差。

    Args:
        root_dir (str): 数据集根目录。
        use_data (tuple[str, ...]): 需要加载的数据类型。
        datasets (tuple[str, ...]): 数据子集名称列表。
        use_normalize (str | None): 归一化方式，支持'01'、'-1_1'或None。
        rms_vel_smooth_sigma (float): rms_vel在归一化前高斯平滑的标准差，默认不处理。
    """
    def __init__(self, root_dir='',
                 use_data=('depth_vel', 'time_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel'),
                 datasets=('FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB', 'CurveFaultA'),
                 use_normalize='01', rms_vel_smooth_sigma=0.0):
        """
        OpenFWI数据集
        :param dataset_name: ['FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB']
        :param rms_vel_smooth_sigma: rms_vel在归一化前高斯平滑的标准差，默认不处理
        """
        self.use_data = use_data
        self.data_files = {data_name: [] for data_name in use_data}
        self.root_dir = root_dir
        self.use_normalize = use_normalize
        self.rms_vel_smooth_sigma = rms_vel_smooth_sigma
        self.normalize_max_min = {'depth_vel': [4500., 1500.], 'time_vel': [4500., 1500.], 'rms_vel': [4500., 1500.],
                                  'migrated_image': [1000., -700.], 'well_log': [4500., 1500.], 'horizon': [1., 0.], }
        for dataset_name in datasets:
            data_dir = os.path.join(root_dir, dataset_name)
            for data_name in self.data_files.keys():
                self.data_files[data_name].extend(sorted(
                    [os.path.join(data_dir, data_name, f) for f in os.listdir(os.path.join(data_dir, data_name)) if
                     f.endswith('.npy')]))

    def __len__(self):
        return len(self.data_files[self.use_data[0]])

    def __getitem__(self, idx):
        data_dict = {}
        for data_name in self.data_files.keys():
            data_file = self.data_files[data_name][idx]
            data = torch.from_numpy(np.load(data_file)).to(torch.float32)
            if data_name == 'rms_vel' and self.rms_vel_smooth_sigma > 0:
                data = self.gaussian_smooth_2d(data, self.rms_vel_smooth_sigma)
            if self.use_normalize == '01':
                data = self.normalize_to_zero_one(data, *self.normalize_max_min[data_name])
                normalized = True
            elif self.use_normalize == '-1_1':
                data = self.normalize_to_neg_one_to_one(data, *self.normalize_max_min[data_name])
                normalized = True
            elif self.use_normalize is None:
                data = data
            if normalized and data_name == 'rms_vel' and self.rms_vel_noise_std > 0:
                noise = torch.normal(mean=0.0, std=self.rms_vel_noise_std, size=data.shape, device=data.device)
                data = data + noise
            data_dict.update({data_name: data})

        return data_dict

    @staticmethod
    def normalize_to_zero_one(x: torch.Tensor, max_value=1, min_value=0) -> torch.Tensor:
        return (x - min_value) / (max_value - min_value)

    @staticmethod
    def normalize_to_neg_one_to_one(x: torch.Tensor, max_value=1, min_value=-1) -> torch.Tensor:
        return ((x - min_value) / (max_value - min_value)) * 2 - 1

    @staticmethod
    def gaussian_smooth_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        kernel_size = max(3, int(6 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel_2d = torch.outer(gaussian, gaussian)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        if x.dim() == 2:
            data = x.unsqueeze(0).unsqueeze(0)
            kernel = kernel_2d
            groups = 1
        elif x.dim() == 3:
            data = x.unsqueeze(0)
            channels = data.shape[1]
            kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
            groups = channels
        else:
            return x

        padding = kernel_size // 2
        smoothed = F.conv2d(data, kernel, padding=padding, groups=groups)
        return smoothed.squeeze(0) if x.dim() == 3 else smoothed.squeeze(0).squeeze(0)

    @staticmethod
    def collate_fn(batch):
        """
        将大小为batch的字典列表转换为字典的大小为batch的列表
        eg: {'a':}[batch] -> {'a':[batch]}
        在dataloader中使用，指定collate_fn=dataset.collate_fn
        :param batch:
        :return:
        """
        keys = batch[0].keys()
        result = {key: torch.stack([item[key] for item in batch]) for key in keys}
        return result


def test_max_main():
    from tqdm import tqdm
    dataset = OpenFWI(root_dir='openfwi', )
    dict_max_min = {key: {'max': 0, 'min': 1e4} for key in dataset.data_files.keys()}
    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)
    x = 0
    for batch in tqdm(dataloader):
        for key, value in batch.items():
            if value.max() > dict_max_min[key]['max']:
                dict_max_min[key]['max'] = value.max()
            if value.min() < dict_max_min[key]['min']:
                dict_max_min[key]['min'] = value.min()
        x += 1
        if x > 10:
            break
    print(dict_max_min)


def test1():
    # 创建数据集实例
    dataset = OpenFWI(root_dir='openfwi', use_data=('depth_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel'))
    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn)
    # 获取第一个批次的数据
    for batch in dataloader:
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        print(f'数据集样本数：{dataset.__len__()}')
        break


if __name__ == "__main__":
    test1()  # test_max_main()

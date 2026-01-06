import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class FieldData(Dataset):
    def __init__(self, root_dir='',
                 use_data=('depth_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel'), use_normalize='01'):
        """
        :param
        """
        self.use_data = use_data
        self.data_files = {data_name: [] for data_name in use_data}
        self.root_dir = root_dir
        self.use_normalize = use_normalize
        # self.normalize_max_min = {'depth_vel': [4500., 1500.], 'time_vel': [4500., 1500.], 'rms_vel': [4500., 1500.],
        #                           'migrated_image': [1000., -700.], 'well_log': [4500., 1500.], 'horizon': [1., 0.], }
        self.normalize_max_min = {'depth_vel': [5200., 1300.], 'time_vel': [5200., 1300.], 'rms_vel': [5200., 1300.],
                                  'migrated_image': [100., -110.], 'well_log': [5200., 1300.], 'horizon': [1., 0.], }
        data_dir = root_dir
        for data_name in self.data_files.keys():
            self.data_files[data_name].extend(
                [os.path.join(data_dir, data_name, f) for f in sorted(os.listdir(os.path.join(data_dir, data_name)),
                                                                      key=lambda p: int(
                                                                          os.path.splitext(os.path.basename(p))[0])) if
                 f.endswith('.npy')])

    def __len__(self):
        return len(self.data_files[self.use_data[0]])

    def __getitem__(self, idx):
        data_dict = {}
        for data_name in self.data_files.keys():
            data_file = self.data_files[data_name][idx]
            data = torch.from_numpy(np.load(data_file)).to(torch.float32)
            if self.use_normalize == '01':
                data = self.normalize_to_zero_one(data, *self.normalize_max_min[data_name])
            elif self.use_normalize == '-1_1':
                data = self.normalize_to_neg_one_to_one(data, *self.normalize_max_min[data_name])
            data_dict.update({data_name: data})
        return data_dict

    @staticmethod
    def normalize_to_zero_one(x: torch.Tensor, max_value=1, min_value=0) -> torch.Tensor:
        return (x - min_value) / (max_value - min_value)

    @staticmethod
    def normalize_to_neg_one_to_one(x: torch.Tensor, max_value=1, min_value=-1) -> torch.Tensor:
        return ((x - min_value) / (max_value - min_value)) * 2 - 1

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
    dataset = FieldData(root_dir='openfwi', )
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
    dataset = FieldData(root_dir='Field_data_70x70',
                        use_data=('depth_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel'))
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

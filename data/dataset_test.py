import os
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def test_shape():
    horizons_path = 'G:\\OpenFWI\\FlatVelB\\horizons_model\\model1.npy'  # (500, 70, 70)
    migration_images_path = 'G:\\OpenFWI\\FlatVelB\\migration_images\\images1.npy'  # (500, 1000, 70)
    velocity_model_path = 'G:\\OpenFWI\\FlatVelB\\model\\model1.npy'  # (500, 1, 70, 70)
    time_vel_path = 'G:\\OpenFWI\\FlatVelB\\time_velocity\\model1.npy'  # (500, 1000, 70)
    data = np.load(velocity_model_path)
    print(data.shape)


def save_single():
    # 'depth_vel', 'time_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel'
    save_path = 'openfwi/FlatVelB'
    data_dict = {
        'depth_vel': 'model\\model',
        'time_vel': 'time_velocity\\model',
        'migrated_image': 'migration_images\\images',
        'horizon': 'horizons_model\\model'
    }
    for save_name, read_name in data_dict.items():
        num = 0
        os.makedirs(f'{save_path}/{save_name}', exist_ok=True)
        for i in tqdm(range(1, 61), desc=save_name):
            datas = np.load(f'G:\\OpenFWI\\FlatVelB\\{read_name}{i}.npy').squeeze()
            datas = np.expand_dims(datas, axis=1)  # (500, 1
            for data in datas:
                np.save(f'{save_path}/{save_name}/{num}.npy', data)
                num += 1


import torch


@torch.no_grad()
def time_vel_to_vrms_equal_dt(v_t: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """
    等时间步长下的 v(t) -> V_rms(t)
    输入:
        v_t: 速度张量，等间隔采样。典型形状 (1, 1000, 70)，time_dim=1
    返回:
        Vrms: 与 v_t 同形状的 RMS 速度
    公式:
        Vrms^2(t_k) = (1/k) * sum_{i=1..k} v(t_i)^2
    """

    # 按时间维做前缀和
    v2_cum = (v_t ** 2).cumsum(dim=time_dim)

    # 构造步数 k=1..T，并在非时间维做广播
    T = v_t.size(time_dim)
    # 形如 (..., T, ...): 在 time_dim 放入长度 T，其他维为 1 以便广播
    shape = [1] * v_t.ndim
    shape[time_dim] = T
    steps = torch.arange(1, T + 1, dtype=torch.float64, device=v_t.device).view(*shape)

    vrms2 = (v2_cum / steps).clamp_min(0.0)
    vrms = torch.sqrt(vrms2)
    return vrms


def rms_vel():
    save_path = 'openfwi/FlatVelB'
    os.makedirs(f'{save_path}/rms_vel', exist_ok=True)
    for i in tqdm(range(0, 30000), desc='rms_vel'):
        time_vel = np.load(f'openfwi/FlatVelB/time_vel/{i}.npy')
        time_vel = torch.from_numpy(time_vel)
        rms_vel = time_vel_to_vrms_equal_dt(time_vel)
        rms_vel = rms_vel.numpy()
        np.save(f'{save_path}/rms_vel/{i}.npy', rms_vel)


def well_log():
    save_path = 'openfwi/FlatVelB'
    os.makedirs(f'{save_path}/well_log', exist_ok=True)
    for i in tqdm(range(0, 30000), desc='well_log'):
        depth_vel = np.load(f'openfwi/FlatVelB/depth_vel/{i}.npy')
        well_log = np.zeros_like(depth_vel)
        random_indices = random.sample(range(70), 7)
        for random_indice in random_indices:
            well_log[:, :, random_indice] = depth_vel[:, :, random_indice]
        np.save(f'{save_path}/well_log/{i}.npy', well_log)


# test_shape()
# save_single()
# rms_vel()
# well_log()


def save_single_image(img, filename='', title="Image",
                      show=False, save=True, cmap='jet',
                      extent=[0, 700, 700, 0],
                      figsize=(5, 5),
                      use_colorbar=True,
                      x_label='Length (m)',
                      y_label='Depth (m)'):
    """
    保存单个灰度图像到文件
    :param img: 要显示的图像
    :param filename: 保存的文件名(带路径)
    :param title: 图像标题
    :param show: 是否显示图像
    :param save: 是否保存图像
    :param cmap: 颜色映射，默认为'jet'
    :param extent: 图像显示范围
    :param figsize: 图像尺寸，元组(宽度, 高度)
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(img, aspect='auto', cmap=cmap, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if use_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        original_ticks = np.linspace(0, 1, 5)  # 生成5个均匀分布的刻度
        target_ticks = np.linspace(1500, 4500, 5)
        cbar.set_ticks(original_ticks)
        cbar.set_ticklabels([f'{int(t)}' for t in target_ticks])
        cbar.set_label('Velocity (m/s)')  # 设置颜色条的标签

    if save:  # 保存图像
        plt.savefig(filename)

    if show:  # 显示图像(阻塞模式)
        plt.show()

    # 关闭图形以释放内存
    plt.close(fig)

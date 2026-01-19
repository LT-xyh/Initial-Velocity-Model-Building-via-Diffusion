import os

import torch
import numpy as np

from lib.diffusers.utils.extract_tests_from_mixin import root_dir
from utils.visualize import save_visualize_image, save_multiple_curves

if __name__ == '__main__':
    root_dir = 'logs/ddpm_diffusion/test_results/test_260114/'
    dataset_name = 'CurveVelA'
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        dataset_dir = os.path.join(root_dir, dataset_name)
        data = {
            'GroundTruth': '',
            'Recon': '',
            'rms_vel': '',
            'migrated_image': '',
            'well_log': '',
        }
        for data_name in ['GroundTruth', 'Recon', 'rms_vel', 'migrated_image', 'horizon','well_log']:
            save_dir = os.path.join(dataset_dir, data_name, '0.pt')
            data[data_name] = torch.load(save_dir).numpy().squeeze()[1]

        for data_name in ['GroundTruth', 'Recon', 'rms_vel', ]:
            save_visualize_image(data[data_name], filename=f'images/DDPM/0114/{data_name}_{dataset_name}.svg', title="Image", show=True, save=True, cmap='jet', extent=None,
                              figsize=(5, 5), use_colorbar=False, x_label='Length (m)', y_label='Depth (m)')

        for data_name in ['migrated_image', 'horizon']:
            save_visualize_image(data[data_name], filename=f'images/DDPM/0114/{data_name}_{dataset_name}.svg', title="Image", show=True, save=True, cmap='grey', extent=None,
                              figsize=(5, 5), use_colorbar=False, x_label='Length (m)', y_label='Depth (m)')

        for i in range(data['well_log'].shape[1]):
            if data['well_log'][0][i] > -1:
                save_multiple_curves([data['well_log'][:, i], ], labels=None, filename=f'images/DDPM/0114/well_log_{dataset_name}_.svg', title="Well log", x_label="Depth (m)",
                                     y_label="Velocity (m/s)", show=True, save=True, figsize=(6, 6), colors=None, linestyles=None)
                break
    # dir = 'logs/ddpm_diffusion/test_results/filedata_cut_1229'
    # depth_vel_list = []
    # for i in range(3):
    #     depth_vel = torch.load(f'{dir}/{i}.pt').numpy().squeeze()  # [60, 70, 70]
    #     depth_vel_list.append(depth_vel)
    # depth_vel = np.concat(depth_vel_list, axis=0)
    #
    # print(depth_vel.shape)
    # np.save(f'{dir}/depth_vel_concat.npy', depth_vel)


    # for i in depth_vel:
    #     save_visualize_image(i, filename='', title="Image", show=True, save=False, cmap='jet', extent=None,
    #                      figsize=(5, 5), use_colorbar=False, x_label='Length (m)', y_label='Depth (m)')

    # print(os.listdir('data/data1_cut/depth_vel'))
    # print(sorted(os.listdir('data/data1_cut/depth_vel'), key=lambda p: int(os.path.splitext(os.path.basename(p))[0])))
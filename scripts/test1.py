import os

import torch
import numpy as np

from utils.visualize import save_visualize_image


if __name__ == '__main__':
    image_dir = ('logs/ddpm_diffusion/test_results/test_260103/FlatVelB/0.pt')

    depth_vel = torch.load(image_dir).numpy().squeeze()[40]
    print(depth_vel.shape)
    save_visualize_image(depth_vel, filename='images/DDPM/FB_0_40.svg', title="Image", show=True, save=True, cmap='jet', extent=None,
                          figsize=(5, 5), use_colorbar=False, x_label='Length (m)', y_label='Depth (m)')

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
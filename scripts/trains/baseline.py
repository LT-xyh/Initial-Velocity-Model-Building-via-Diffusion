import os

from scripts.trains.train_inversion_net import train_inversion_net
from scripts.trains.train_sv_inv_net import train_sv_inv_net
from scripts.trains.train_velocity_gan import train_velocity_gan

if __name__ == '__main__':

    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        print(f"------------------{dataset_name}------------------")
        train_sv_inv_net(dataset_name)
        train_inversion_net(dataset_name)
        train_velocity_gan(dataset_name)

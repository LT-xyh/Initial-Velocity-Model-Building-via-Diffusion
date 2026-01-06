from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.SVInvNetLightning import MAblationSVInvNetLightning, \
    MVAblationSVInvNetLightning, MHAblationSVInvNetLightning, MWAblationSVInvNetLightning
from scripts.trains.basetrain import base_train


def train_sv_inv_net(dataset_name):
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/sv_inv_net.yaml')
    # 获取当前日期时间
    current_date = datetime.now()
    # 格式化月份和日期为两位数，组成"MMDD"形式的字符串
    date_str = current_date.strftime("%m%d")
    conf.datasets.dataset_name[0] = dataset_name
    conf.training.logging.log_dir = 'logs/ablations/sv_inv_net'
    ablation_dict = {
        # 'M': MAblationSVInvNetLightning,
        # 'MV': MVAblationSVInvNetLightning,
        # 'MH': MHAblationSVInvNetLightning,
        'MW': MWAblationSVInvNetLightning,  # FlatVelA
    }
    for ab in ablation_dict.keys():
        print(f"\n\n\n{dataset_name}----{ab}----{ablation_dict[ab].__name__}\n")
        print(f'')
        conf.training.logging.log_version = dataset_name + "_ablation_" + ab
        model = ablation_dict[ab](conf)
        base_train(model, conf, fast_run=False, use_lr_finder=False, )


if __name__ == '__main__':
    for dataset_name in ['FlatVelA']:
        train_sv_inv_net(dataset_name)


import time
from datetime import datetime

import psutil
import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.InversionNetLightning import InversionNetLightning
from scripts.trains.basetrain import base_train


def train_inversion_net():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/inversion_net.yaml')
    # 获取当前日期时间
    current_date = datetime.now()
    # 格式化月份和日期为两位数，组成"MMDD"形式的字符串
    date_str = current_date.strftime("%m%d")

    conf.training.logging.log_version = date_str + "_"
    model = InversionNetLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False, )


if __name__ == '__main__':
    # pid = 9940
    # while psutil.pid_exists(pid):
    #     time.sleep(60)
    train_inversion_net()



import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset_field import FieldData
from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning


def field_data_test(model, conf, fast_run=False):
    test_set = FieldData(root_dir='data/data1_cut', use_data=conf.datasets.use_data,
                         use_normalize='-1_1')

    test_loader = DataLoader(test_set, batch_size=conf.training.dataloader.batch_size,
                             shuffle=False,
                             num_workers=conf.training.dataloader.num_workers, persistent_workers=True, pin_memory=True,
                             prefetch_factor=conf.training.dataloader.prefetch_factor)

    tensorboard_logger = TensorBoardLogger(save_dir=conf.training.logging.log_dir, name='tensorboard',
                                           version=conf.training.logging.log_version, )
    csv_logger = CSVLogger(save_dir=conf.training.logging.log_dir, name="csv",
                           version=conf.training.logging.log_version, )

    # 早停回调
    early_stop_callback = EarlyStopping(monitor=conf.training.callbacks.early_stopping.monitor,  # 要监控的指标
                                        min_delta=0,  # 最小变化量
                                        patience=conf.training.callbacks.early_stopping.patience,  # 连续轮数
                                        verbose=True, mode=conf.training.callbacks.early_stopping.mode)

    # 模型保存回调
    checkpoint_callback = ModelCheckpoint(  # dirpath='checkpoints',  # 指定目录
        filename=conf.training.callbacks.checkpoint.filename,  # 命名规则
        auto_insert_metric_name=False, save_top_k=conf.training.callbacks.checkpoint.save_top_k,
        monitor=conf.training.callbacks.checkpoint.monitor, mode=conf.training.callbacks.checkpoint.mode,
        save_last=True, every_n_epochs=1, )

    # 选择多 GPU 训练并指定 GPU
    if conf.training.gradient_clip_val is None:
        trainer = lightning.Trainer(precision=conf.training.precision,  # fp16混合精度训练
                                    gradient_clip_val=1.0,  # 梯度裁剪
                                    max_epochs=conf.training.max_epochs, min_epochs=conf.training.min_epochs,
                                    accelerator="gpu",  # strategy='ddp_spawn',
                                    devices=conf.training.device,  # 指定要使用的 GPU 编号
                                    logger=[tensorboard_logger, csv_logger],
                                    callbacks=[early_stop_callback, checkpoint_callback],
                                    log_every_n_steps=512 // conf.training.dataloader.batch_size, fast_dev_run=fast_run,
                                    # 只会执行一个batch 用于测试
                                    )
    else:
        trainer = lightning.Trainer(precision=conf.training.precision,  # fp16混合精度训练
                                    max_epochs=conf.training.max_epochs, min_epochs=conf.training.min_epochs,
                                    accelerator="gpu",  # strategy='ddp_spawn',
                                    devices=conf.training.device,  # 指定要使用的 GPU 编号
                                    logger=[tensorboard_logger, csv_logger],
                                    callbacks=[early_stop_callback, checkpoint_callback],
                                    log_every_n_steps=512 // conf.training.dataloader.batch_size, fast_dev_run=fast_run,
                                    # 只会执行一个batch 用于测试
                                    )

    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/filedata_cut_1229'
    conf.testing.ckpt_path = 'logs/ddpm_diffusion/tensorboard/fine_tuning_field_cut_normalize/checkpoints/epoch_29-loss0.805.ckpt'
    conf.training.logging.log_version = "test_field_1229"
    model = DDPMConditionalDiffusionLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    field_data_test(model, conf)

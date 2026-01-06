import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.tuner import Tuner
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data.dataset_field_cut import FieldCutData
from data.dataset_openfwi import OpenFWI


def base_train(model, conf, fast_run=True, use_lr_finder=False, ckpt_path=None):
    train_dataset_list = []
    val_dataset_list = []
    test_dataset_list = []
    for dataset in conf.datasets.dataset_name:
        dataset = OpenFWI(root_dir='data/openfwi', use_data=conf.datasets.use_data, datasets=(dataset,),
                          use_normalize=conf.datasets.use_normalize)
        total_size = len(dataset)
        test_size = int(0.1 * total_size)  # 取最后10%作为测试集（非随机）
        test_idx = list(range(total_size - test_size, total_size))
        remaining_idx = list(range(total_size - test_size))  # 剩下的80%用于训练和验证

        # 从剩余数据中随机划分训练集和验证集
        train_idx, val_idx = train_test_split(remaining_idx, test_size=0.25,  # 相对于剩余数据的25%，即总数据的20%
                                              random_state=42, shuffle=True  # 训练集和验证集随机划分
                                              )
        train_dataset_list.append(Subset(dataset, train_idx))
        val_dataset_list.append(Subset(dataset, val_idx))

    train_set = ConcatDataset(train_dataset_list)
    val_set = ConcatDataset(val_dataset_list)
    train_loader = DataLoader(train_set, batch_size=conf.training.dataloader.batch_size, shuffle=True,
                              num_workers=conf.training.dataloader.num_workers, persistent_workers=True,
                              pin_memory=True, prefetch_factor=conf.training.dataloader.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=conf.training.dataloader.batch_size,
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

    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(lr_finder.suggestion())
    else:
        if ckpt_path is not None:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader)


def base_train_field_cut(model, conf, fast_run=True, use_lr_finder=False, ckpt_path=None):
    train_dataset_list = []
    val_dataset_list = []
    dataset = FieldCutData(root_dir='data/data1_cut', use_data=conf.datasets.use_data,
                           use_normalize=conf.datasets.use_normalize)
    total_size = len(dataset)
    remaining_idx = list(range(total_size))  # 剩下的80%用于训练和验证

    # 从剩余数据中随机划分训练集和验证集
    train_idx, val_idx = train_test_split(remaining_idx, test_size=0.2, shuffle=True  # 训练集和验证集随机划分
                                          )
    train_dataset_list.append(Subset(dataset, train_idx))
    val_dataset_list.append(Subset(dataset, val_idx))

    train_set = ConcatDataset(train_dataset_list)
    val_set = ConcatDataset(val_dataset_list)
    train_loader = DataLoader(train_set, batch_size=conf.training.dataloader.batch_size, shuffle=True,
                              num_workers=conf.training.dataloader.num_workers, persistent_workers=True,
                              pin_memory=True, prefetch_factor=conf.training.dataloader.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=conf.training.dataloader.batch_size,
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

    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(lr_finder.suggestion())
    else:
        if ckpt_path is not None:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader)

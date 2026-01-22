import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.tuner import Tuner
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data.dataset_field_cut import FieldCutData
from data.dataset_openfwi import OpenFWI


def _get_log_every_n_steps(conf, batch_size):
    # Respect explicit config override if present; Lightning requires >= 1.
    if hasattr(conf, "training") and "log_every_n_steps" in conf.training:
        try:
            override = int(conf.training.log_every_n_steps)
            return max(1, override)
        except (TypeError, ValueError):
            return 1
    if not batch_size:
        return 1
    return max(1, 512 // int(batch_size))


def _get_persistent_workers(conf, num_workers):
    # persistent_workers only valid when num_workers > 0.
    if num_workers <= 0:
        return False
    if hasattr(conf, "training") and hasattr(conf.training, "dataloader") and "persistent_workers" in conf.training.dataloader:
        return bool(conf.training.dataloader.persistent_workers)
    return True


def base_train(model, conf, fast_run=True, use_lr_finder=False, ckpt_path=None):
    train_dataset_list = []
    val_dataset_list = []
    for dataset in conf.datasets.dataset_name:
        dataset = OpenFWI(root_dir='data/openfwi', use_data=conf.datasets.use_data, datasets=(dataset,),
                          use_normalize=conf.datasets.use_normalize)
        total_size = len(dataset)
        # Reserve last 10% for potential test split (not used here).
        test_size = int(0.1 * total_size)  # 取最后10%作为测试集（非随机）
        remaining_idx = list(range(total_size - test_size))  # 剩下的80%用于训练和验证

        # test_idx = list(range(test_size))
        # remaining_idx = list(range(test_size, total_size))  # 剩下的80%用于训练和验证

        # 从剩余数据中随机划分训练集和验证集
        train_idx, val_idx = train_test_split(remaining_idx, test_size=0.25,  # 相对于剩余数据的25%，即总数据的20%
                                              random_state=42, shuffle=True  # 训练集和验证集随机划分
                                              )
        train_dataset_list.append(Subset(dataset, train_idx))
        val_dataset_list.append(Subset(dataset, val_idx))

    train_set = ConcatDataset(train_dataset_list)
    val_set = ConcatDataset(val_dataset_list)
    batch_size = conf.training.dataloader.batch_size
    num_workers = conf.training.dataloader.num_workers
    persistent_workers = _get_persistent_workers(conf, num_workers)
    log_every_n_steps = _get_log_every_n_steps(conf, batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=persistent_workers,
                              pin_memory=True, prefetch_factor=conf.training.dataloader.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True,
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
    # Trainer config: gradient_clip_val is config-driven; None means no clipping.
    trainer_kwargs = dict(
        precision=conf.training.precision,  # fp16?????????
        max_epochs=conf.training.max_epochs, min_epochs=conf.training.min_epochs,
        accelerator="gpu",  # strategy='ddp_spawn',
        devices=conf.training.device,  # ????????? GPU ???
        logger=[tensorboard_logger, csv_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=log_every_n_steps,  # must be >= 1
        fast_dev_run=fast_run,  # ??????????atch ?????
    )
    if conf.training.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = conf.training.gradient_clip_val
    trainer = lightning.Trainer(**trainer_kwargs)

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
    batch_size = conf.training.dataloader.batch_size
    num_workers = conf.training.dataloader.num_workers
    persistent_workers = _get_persistent_workers(conf, num_workers)
    log_every_n_steps = _get_log_every_n_steps(conf, batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=persistent_workers,
                              pin_memory=True, prefetch_factor=conf.training.dataloader.prefetch_factor)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True,
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
    # Trainer config: gradient_clip_val is config-driven; None means no clipping.
    trainer_kwargs = dict(
        precision=conf.training.precision,  # fp16?????????
        max_epochs=conf.training.max_epochs, min_epochs=conf.training.min_epochs,
        accelerator="gpu",  # strategy='ddp_spawn',
        devices=conf.training.device,  # ????????? GPU ???
        logger=[tensorboard_logger, csv_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=log_every_n_steps,  # must be >= 1
        fast_dev_run=fast_run,  # ??????????atch ?????
    )
    if conf.training.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = conf.training.gradient_clip_val
    trainer = lightning.Trainer(**trainer_kwargs)

    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(lr_finder.suggestion())
    else:
        if ckpt_path is not None:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader)

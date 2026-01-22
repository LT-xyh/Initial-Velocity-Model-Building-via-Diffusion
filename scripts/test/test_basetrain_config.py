from __future__ import annotations

import inspect
from types import SimpleNamespace

from scripts.trains import basetrain
from utils import metrics


def _conf(batch_size=8, num_workers=0, log_every_n_steps=None, persistent_workers=None):
    dataloader = SimpleNamespace(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
    )
    if persistent_workers is not None:
        dataloader.persistent_workers = persistent_workers

    training = SimpleNamespace(dataloader=dataloader)
    if log_every_n_steps is not None:
        training.log_every_n_steps = log_every_n_steps

    return SimpleNamespace(training=training)


def test_log_every_n_steps():
    conf = _conf(batch_size=1024)
    assert basetrain._get_log_every_n_steps(conf, conf.training.dataloader.batch_size) == 1

    conf = _conf(batch_size=1)
    assert basetrain._get_log_every_n_steps(conf, conf.training.dataloader.batch_size) == 512

    conf = _conf(batch_size=8, log_every_n_steps=0)
    assert basetrain._get_log_every_n_steps(conf, conf.training.dataloader.batch_size) == 1

    conf = _conf(batch_size=8, log_every_n_steps="bad")
    assert basetrain._get_log_every_n_steps(conf, conf.training.dataloader.batch_size) == 1


def test_persistent_workers():
    conf = _conf(num_workers=0, persistent_workers=True)
    assert basetrain._get_persistent_workers(conf, conf.training.dataloader.num_workers) is False

    conf = _conf(num_workers=2, persistent_workers=False)
    assert basetrain._get_persistent_workers(conf, conf.training.dataloader.num_workers) is False

    conf = _conf(num_workers=2, persistent_workers=True)
    assert basetrain._get_persistent_workers(conf, conf.training.dataloader.num_workers) is True

    conf = _conf(num_workers=2)
    assert basetrain._get_persistent_workers(conf, conf.training.dataloader.num_workers) is True


def test_trainer_wiring_source_checks():
    source = inspect.getsource(basetrain.base_train)
    assert "log_every_n_steps=log_every_n_steps" in source
    assert "persistent_workers=persistent_workers" in source
    assert "if conf.training.gradient_clip_val is not None" in source
    assert "trainer_kwargs[\"gradient_clip_val\"]" in source

    source = inspect.getsource(basetrain.base_train_field_cut)
    assert "log_every_n_steps=log_every_n_steps" in source
    assert "persistent_workers=persistent_workers" in source
    assert "if conf.training.gradient_clip_val is not None" in source
    assert "trainer_kwargs[\"gradient_clip_val\"]" in source


def test_metrics_instantiation():
    _ = metrics.ValMetrics(device="cpu")


def main():
    test_log_every_n_steps()
    test_persistent_workers()
    test_trainer_wiring_source_checks()
    test_metrics_instantiation()
    print("basetrain config checks: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("basetrain config checks: FAILED")
        raise

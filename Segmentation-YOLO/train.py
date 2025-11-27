import sys
from pathlib import Path
from typing import Union, Tuple

project_root = Path(__file__).resolve().parent / 'YOLO'
sys.path.insert(0, str(project_root))

import hydra
from yolo import Config
from yolo.tools.solver import TrainModel
from lightning import Trainer
from yolo.utils.logging_utils import setup, set_seed


def parseDeviceConfig(deviceConfig: Union[int, str, list]) -> Tuple[str, Union[int, str, list]]:
    if isinstance(deviceConfig, str):
        if deviceConfig.lower() == 'cpu':
            return 'cpu', 1
        elif deviceConfig.lower() == 'auto':
            return 'auto', 'auto'
        elif deviceConfig.lower() == 'cuda':
            return 'cuda', 'auto'
        else:
            return 'gpu', int(deviceConfig)
    elif isinstance(deviceConfig, int):
        return 'gpu', [deviceConfig]
    elif isinstance(deviceConfig, list):
        return 'gpu', deviceConfig
    else:
        return 'auto', 'auto'


@hydra.main(config_path="YOLO/yolo/config", config_name="config", version_base=None)
def train(cfg: Config):
    print("=" * 60)
    print("YOLO Cone Detection Training")
    print("=" * 60)
    print("\nConfiguration Summary:")
    print(f"  Model: {cfg.name}")
    print(f"  Epochs: {cfg.task.epoch}")
    print(f"  Batch Size: {cfg.task.data.batch_size}")
    print(f"  Image Size: {cfg.image_size}")
    print(f"  Device: {cfg.device}")
    print(f"  Dataset: {cfg.dataset.path}")
    print(f"  Classes: {cfg.dataset.class_list}")
    print("=" * 60)

    set_seed(cfg.lucky_number)

    print("\nCreating model...")
    trainModel = TrainModel(cfg)

    print("Setting up trainer and logger...")
    callbacks, loggers, savePath = setup(cfg)

    from yolo.utils.logging_utils import YOLORichProgressBar
    callbacks = [cb for cb in callbacks if not isinstance(cb, YOLORichProgressBar)]

    loggers = []

    accelerator, devices = parseDeviceConfig(cfg.device)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.task.epoch,
        precision='16-mixed',
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        gradient_clip_val=10,
        default_root_dir=str(savePath),
        enable_progress_bar=True,
    )

    print("\nStarting training...")
    trainer.fit(trainModel)

    print("\nTraining complete!")
    print(f"Model saved to: {savePath}/")

if __name__ == '__main__':
    train()

from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

from utils import load_config
from dataset import SatMapDataset, custom_collate_fn
from model import MaGRoad

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for testing. "
    "If None, use the default config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--precision", default="16-mixed", help="32 or 16-mixed"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')

    # Ensure TEST_CKPT_PATH for MaGRoad.on_test_start
    if args.checkpoint is not None:
        try:
            # support attribute-style config
            setattr(config, 'TEST_CKPT_PATH', args.checkpoint)
        except Exception:
            # fallback if config is dict-like
            config['TEST_CKPT_PATH'] = args.checkpoint
    elif not getattr(config, 'TEST_CKPT_PATH', None) and not getattr(config, 'get', lambda *_: None)("TEST_CKPT_PATH", None):
        raise ValueError("Please provide --checkpoint or set TEST_CKPT_PATH in the config.")

    # Initialize model
    model = MaGRoad(config)

    # Validation dataset (align with train.py settings)
    val_dataset = SatMapDataset(
        dataset_name=config.DATASET_NAME,
        is_train=False,
        max_kp_num=config.MAX_KP_NUM,
        kp_sample_prob=config.KP_SAMPLE_PROB,
        negative_sample_ratio=config.NEGATIVE_SAMPLE_RATIO,
        negative_safe_radius=config.NEGATIVE_SAFE_RADIUS,
        graph_config=config,
        debug=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Simple callbacks and logger (mirror reference test flow)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Keep logs under the same base dir as training for easy comparison
    tb_logger = TensorBoardLogger(save_dir="wild_road_exp", name="test")

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=args.precision,
        accelerator="gpu",
        devices=[0],
        logger=tb_logger,
    )

    trainer.test(model, dataloaders=val_loader, ckpt_path=args.checkpoint)



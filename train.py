from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

from utils import load_config
from dataset import SatMapDataset, custom_collate_fn
from model import MaGRoad

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the default config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default="16-mixed", help="32 or 16-mixed"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--debug", default=False, action='store_true'
)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    debug = args.debug or args.fast_dev_run

    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(seed=2025, workers=True) # 2025

    # Initialize model
    model = MaGRoad(config)

    # Setup datasets
    # data_dir should point to wildroad directory (e.g., './wildroad')
    train_dataset = SatMapDataset(
        dataset_name=config.DATASET_NAME,
        is_train=True,  # Loads train + val tiles
        max_kp_num=config.MAX_KP_NUM,
        kp_sample_prob=config.KP_SAMPLE_PROB,
        negative_sample_ratio=config.NEGATIVE_SAMPLE_RATIO,
        negative_safe_radius=config.NEGATIVE_SAFE_RADIUS,
        graph_config=config,
        debug=debug
    )
    
    val_dataset = SatMapDataset(
        dataset_name=config.DATASET_NAME,
        is_train=False,  # Loads test tiles
        max_kp_num=config.MAX_KP_NUM,
        kp_sample_prob=config.KP_SAMPLE_PROB,
        negative_sample_ratio=config.NEGATIVE_SAMPLE_RATIO,
        negative_safe_radius=config.NEGATIVE_SAFE_RADIUS,
        graph_config=config,
        debug=debug
    )

    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Callbacks
    val_loss_checkpoint_callback = ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=False,
        save_weights_only=True
    )
    
    # add checkpoint callbacks for monitoring and saving best IoU
    keypoint_iou_checkpoint = ModelCheckpoint(
        filename="best_keypoint_iou-{epoch:02d}-{val_epoch_keypoint_iou:.3f}",
        every_n_epochs=1,
        save_top_k=1, 
        monitor='val_epoch_keypoint_iou',
        mode='max',
        save_last=False,
        save_weights_only=True
    )
    
    road_iou_checkpoint = ModelCheckpoint(
        filename="best_road_iou-{epoch:02d}-{val_epoch_road_iou:.3f}",
        every_n_epochs=1,
        save_top_k=1, 
        monitor='val_epoch_road_iou',
        mode='max',
        save_last=False,
        save_weights_only=True
    )

    topo_acc_checkpoint = ModelCheckpoint(
        filename="best_topo_acc-{epoch:02d}-{val_epoch_topo_acc:.3f}",
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_epoch_topo_acc',
        mode='max',
        save_last=False,
        save_weights_only=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    tb_logger = TensorBoardLogger(save_dir="wild_road_exp", name="wild_road_train")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2, # 2
        callbacks=[val_loss_checkpoint_callback, keypoint_iou_checkpoint, road_iou_checkpoint, topo_acc_checkpoint, lr_monitor],
        logger=tb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        log_every_n_steps=4,
        accelerator="gpu",
        devices=[0],
        strategy="ddp_find_unused_parameters_true", # ddp
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(model, train_loader, val_loader)
        
    # Test the model after training
    if config.get("RUN_TEST", False):
        trainer.test(model, val_loader)

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# import lightning.pytorch as pl

from dataset import Dataset, get_training_augmentation, get_validation_augmentation
from model import CamVidModel

# ========== 配置参数 ==========
DATA_DIR = "./data/CamVid/"          # 数据集根目录
BATCH_SIZE = 32                       # 根据 GPU 内存调整
EPOCHS = 50
NUM_WORKERS = 12                        # Windows 建议设为 0，Linux/macOS 可设为 CPU 核心数
ENCODER = "resnext50_32x4d"            # 编码器名称
ARCH = "FPN"                           # 模型架构（FPN, Unet, Unet++ 等）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径
x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "trainannot")
x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "valannot")

def main():
    # ========== 数据准备 ==========
    print(">>> 加载数据集...")
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
    )
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ========== 模型初始化 ==========
    OUT_CLASSES = len(train_dataset.CLASSES)   # 12 类
    print(f">>> 类别数: {OUT_CLASSES}")
    model = CamVidModel(
        ARCH,
        ENCODER,
        in_channels=3,
        out_classes=OUT_CLASSES,
    )

    # ========== 训练 ==========
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        accelerator=DEVICE,
        devices="auto",
        default_root_dir="./logs",          # 日志保存路径
    )

    print(">>> 开始训练...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # ========== 保存模型 ==========
    ckpt_path = "./camvid_fpn.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f">>> 模型已保存至: {ckpt_path}")

    # ========== 验证集评估 ==========
    print(">>> 在验证集上评估...")
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print("验证集指标:", valid_metrics)

if __name__ == '__main__':
    main()
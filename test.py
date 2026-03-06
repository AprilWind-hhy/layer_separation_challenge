import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import Dataset, get_validation_augmentation, visualize
from model import CamVidModel

# ========== 配置参数 ==========
DATA_DIR = "./data/CamVid/"
BATCH_SIZE = 8                          # 测试时可适当减小以便可视化
NUM_WORKERS = 0
CKPT_PATH = "./camvid_fpn.ckpt"          # 训练好的模型权重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径
x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "testannot")

def main():
    # ========== 加载数据集 ==========
    print(">>> 加载测试集...")
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ========== 加载模型 ==========
    print(">>> 加载模型...")
    # 从 checkpoint 恢复模型（需要知道类别数，可从数据集获取）
    # 注意：加载时需要传入相同的模型参数，或直接从 checkpoint 的 hyper_parameters 恢复
    # 为简化，此处重新创建模型并加载权重
    OUT_CLASSES = len(test_dataset.CLASSES)
    model = CamVidModel.load_from_checkpoint(
        CKPT_PATH,
        arch="FPN",                # 必须与训练时一致
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=OUT_CLASSES,
        map_location=torch.device(DEVICE),
    )
    model.eval()

    # ========== 测试集评估 ==========
    trainer = pl.Trainer(accelerator=DEVICE, devices="auto")
    print(">>> 在测试集上评估...")
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print("测试集指标:", test_metrics)

    # ========== 可视化部分样本 ==========
    print(">>> 可视化部分预测结果并保存...")
    images, masks = next(iter(test_loader))
    with torch.inference_mode():
        logits = model(images)
    pr_masks = logits.softmax(dim=1).argmax(dim=1)

    # 创建保存图像的目录
    os.makedirs("test_viz", exist_ok=True)

    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx >= 5:   # 只保存前 5 个样本
            break

        # 转换图像为 HWC 格式
        img_hwc = image.cpu().numpy().transpose(1, 2, 0)
        gt_mask_np = gt_mask.cpu().numpy()
        pr_mask_np = pr_mask.cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_hwc)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask_np, cmap="tab20")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask_np, cmap="tab20")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"test_viz/sample_{idx}.png", dpi=150)
        plt.close()

    print(">>> 可视化图像已保存至 test_viz/ 目录")
    
if __name__ == "__main__":
    main()
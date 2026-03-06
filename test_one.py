import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from model import CamVidModel

def get_val_transform():
    """必须与验证集预处理完全一致"""
    return A.Compose([
        A.PadIfNeeded(384, 480)   # 仅填充到固定尺寸
    ])

def preprocess_image(image_path, transform):
    """
    读取图像，应用验证增强（仅填充），返回输入张量、原始尺寸、填充后的尺寸
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    original_h, original_w = img.shape[:2]

    # 应用填充增强（结果图像尺寸为 384x480，原图位于左上角）
    augmented = transform(image=img)
    img_padded = augmented['image']   # 仍为 uint8，范围 [0, 255]

    # 转换为 CHW 格式，并转为 float32（保留 0-255 范围）
    img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)   # 添加 batch 维度

    return img_tensor, (original_h, original_w), img_padded.shape[:2]

def main():
    parser = argparse.ArgumentParser(description="单张图片推理（与验证集预处理一致）")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", default="./camvid_fpn.ckpt", help="模型 checkpoint 路径")
    parser.add_argument("--output", default="./output/prediction.png", help="输出彩色预测图路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 加载模型
    print(">>> 加载模型中...")
    model = CamVidModel.load_from_checkpoint(
        args.checkpoint,
        arch="FPN",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=12,            # CamVid 类别数（含 unlabelled）
        map_location=device,
    )
    model.eval()
    model.to(device)
    print(">>> 模型加载完成")

    # 2. 预处理（与验证集完全相同）
    val_transform = get_val_transform()
    print(">>> 预处理图片...")
    input_tensor, (orig_h, orig_w), (padded_h, padded_w) = preprocess_image(args.image, val_transform)
    input_tensor = input_tensor.to(device)

    # 3. 推理
    print(">>> 推理中...")
    with torch.inference_mode():
        logits = model(input_tensor)
        pred_padded = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (384, 480)

    # 4. 裁剪回原图尺寸（原图位于左上角，直接取前 orig_h, orig_w 区域）
    pred_original = pred_padded[:orig_h, :orig_w]

    # 可选：检查预测类别分布（调试用）
    unique, counts = np.unique(pred_original, return_counts=True)
    print("预测类别分布:", dict(zip(unique, counts)))

    # 5. 保存彩色分割图（使用 tab20 颜色映射）
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.imsave(args.output, pred_original, cmap='tab20', vmin=0, vmax=11)
    print(f">>> 预测结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
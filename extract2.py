import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from model import CamVidModel
from dataset import Dataset  # 导入类别列表

def get_val_transform():
    """与验证集预处理完全一致（仅填充）"""
    return A.Compose([A.PadIfNeeded(384, 480)])

def preprocess_image(image_path, transform):
    """
    读取图像，应用验证增强（仅填充）
    返回：输入张量 (1, C, H, W)，原始尺寸 (h,w)，填充后尺寸 (ph,pw)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]

    augmented = transform(image=img)
    img_padded = augmented['image']   # uint8, 范围 [0,255]
    padded_h, padded_w = img_padded.shape[:2]

    # 转为 CHW 并添加 batch 维度
    img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, (original_h, original_w), (padded_h, padded_w)

def save_class_elements(original_image, pred_mask, output_dir, class_names, skip_background=True):
    """
    为每个类别生成独立元素 PNG（RGBA，透明背景，仅显示原图中该类别的像素）
    Args:
        original_image: numpy array (H,W,3), RGB, uint8
        pred_mask: numpy array (H,W), 预测类别索引
        output_dir: 输出目录
        class_names: 类别名称列表，索引对应预测值
        skip_background: 是否跳过背景类（索引0，通常为 unlabelled）
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = original_image.shape[:2]
    # 转换为 BGR 以便 OpenCV 保存（OpenCV 使用 BGR 顺序）
    original_bgr = original_image[..., ::-1]

    for class_idx, class_name in enumerate(class_names):
        if skip_background and class_idx == 0:  # 假设索引0为背景
            continue

        # 创建全透明 RGBA 图像
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask = (pred_mask == class_idx)

        if np.any(mask):
            # RGB 部分填入原图对应像素（已转为 BGR）
            rgba[mask, 0:3] = original_bgr[mask]
            # Alpha 通道设为不透明
            rgba[mask, 3] = 255

        # 保存为 PNG
        out_path = os.path.join(output_dir, f"{class_name}.png")
        cv2.imwrite(out_path, rgba)
        print(f"  已保存: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="单张图片推理，支持生成整体预测图和独立类别元素图")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", default="./camvid_fpn.ckpt", help="模型 checkpoint 路径")
    parser.add_argument("--output", default="./output/prediction.png", help="输出整体预测图路径")
    parser.add_argument("--elements_dir", default=None, help="输出独立元素图的目录（若指定则生成每个类别的PNG）")
    parser.add_argument("--skip_background", action="store_true", default=True, help="生成元素图时跳过背景类（unlabelled）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 类别名称（从 dataset 导入，确保与训练一致）
    CLASSES = Dataset.CLASSES  # ['sky', 'building', ..., 'unlabelled']
    print(">>> 类别列表:", CLASSES)

    # 1. 加载模型
    print(">>> 加载模型中...")
    model = CamVidModel.load_from_checkpoint(
        args.checkpoint,
        arch="FPN",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=len(CLASSES),
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

    # 4. 裁剪回原图尺寸（原图位于左上角）
    pred_original = pred_padded[:orig_h, :orig_w]

    # 打印类别分布（调试用）
    unique, counts = np.unique(pred_original, return_counts=True)
    print("预测类别分布:", dict(zip(unique, counts)))

    # 5. 保存整体预测图（彩色掩码）
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.imsave(args.output, pred_original, cmap='tab20', vmin=0, vmax=len(CLASSES)-1)
    print(f">>> 整体预测图已保存至: {args.output}")

    # 6. 若指定元素图目录，则生成每个类别的独立元素图
    if args.elements_dir:
        print(f">>> 正在生成独立元素图到目录: {args.elements_dir}")
        # 重新读取原图（不经过任何变换，用于提取像素）
        original_rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        save_class_elements(
            original_image=original_rgb,
            pred_mask=pred_original,
            output_dir=args.elements_dir,
            class_names=CLASSES,
            skip_background=args.skip_background
        )
        print(">>> 元素图生成完毕")

if __name__ == "__main__":
    main()
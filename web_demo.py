import os
import cv2
import torch
import numpy as np
import gradio as gr
import albumentations as A
import matplotlib.pyplot as plt
from model import CamVidModel
from dataset import Dataset  # 需包含 CLASSES 列表

# -------------------- 配置 --------------------
CHECKPOINT_PATH = "./camvid_fpn.ckpt"   # 模型权重路径（根据实际情况修改）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 数据预处理 --------------------
def get_val_transform():
    """与验证集预处理一致（仅填充到 384x480）"""
    return A.Compose([A.PadIfNeeded(384, 480)])

def preprocess_image(image, transform):
    """
    输入：OpenCV 图像 (H,W,3) BGR，uint8
    返回：输入张量 (1, C, H, W)，原始尺寸 (h,w)
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    augmented = transform(image=img_rgb)
    img_padded = augmented['image']   # uint8
    padded_h, padded_w = img_padded.shape[:2]

    # 转为 CHW 并添加 batch 维度
    img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, (original_h, original_w), (padded_h, padded_w)

# -------------------- 全局加载模型 --------------------
print(">>> 正在加载模型...")
CLASSES = Dataset.CLASSES  # 类别列表
model = CamVidModel.load_from_checkpoint(
    CHECKPOINT_PATH,
    arch="FPN",
    encoder_name="resnext50_32x4d",
    in_channels=3,
    out_classes=len(CLASSES),
    map_location=DEVICE,
)
model.eval().to(DEVICE)
val_transform = get_val_transform()
print(">>> 模型加载完成，可使用 Web 界面进行推理。")

# -------------------- 推理函数（供 Gradio 调用）--------------------
def inference(input_img):
    """
    input_img: numpy array (H,W,3) RGB, uint8 (Gradio 默认以 RGB 传入)
    返回：原图（RGB）和预测彩色图（RGB）
    """
    # 转换为 BGR 以适配原有预处理函数
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # 预处理
    input_tensor, (orig_h, orig_w), _ = preprocess_image(img_bgr, val_transform)
    input_tensor = input_tensor.to(DEVICE)

    # 推理
    with torch.inference_mode():
        logits = model(input_tensor)
        pred_padded = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (384,480)

    # 裁剪回原图尺寸
    pred = pred_padded[:orig_h, :orig_w]

    # 生成彩色预测图（使用 tab20 colormap）
    cmap = plt.colormaps['tab20']
    # 归一化到 [0,1] 并映射颜色
    pred_norm = pred / (len(CLASSES) - 1)  # 类别索引从 0 开始，假设最大索引为 len-1
    pred_colored = (cmap(pred_norm)[:, :, :3] * 255).astype(np.uint8)

    # 返回原图（保持 RGB）和预测彩色图
    return input_img, pred_colored

# -------------------- 构建 Gradio 界面 --------------------
with gr.Blocks(title="语义分割演示", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 语义分割演示（CamVid 模型）")
    gr.Markdown("上传一张图片，模型将输出每个像素的类别预测（使用 FPN+ResNeXt50 架构）。")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="上传图片")
            submit_btn = gr.Button("开始分割", variant="primary")
        with gr.Column():
            image_original = gr.Image(label="原图", interactive=False)
            image_pred = gr.Image(label="预测结果", interactive=False)

    submit_btn.click(
        fn=inference,
        inputs=image_input,
        outputs=[image_original, image_pred]
    )

    gr.Markdown("---")
    gr.Markdown("类别列表：\n" + ", ".join(CLASSES))

if __name__ == "__main__":
    # 启动服务（默认地址 http://127.0.0.1:7860）
    demo.launch(share=False, server_name="0.0.0.0")
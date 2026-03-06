#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像独立元素分离脚本

功能：
    - 从 PNG 图片中自动分离独立视觉元素
    - 每个元素输出为带透明通道的 PNG 文件，保存在 output/{图片名}/ 下
    - 优先使用图片自带的透明通道；若无则基于颜色背景估计 + 连通分量生成前景

用法：
    # 处理 test_img 目录下所有 PNG
    python extract.py

    # 处理单张图片
    python extract.py --image test_img/xxx.png
"""

import argparse
import os
import sys
import warnings

# 可选：如需屏蔽 OpenCV 或 NumPy 的警告可取消下一行注释
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 配置：目录与默认参数
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "test_img")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
# 连通分量最小面积（像素），过小的斑点视为噪点
MIN_ELEMENT_AREA = 100
# 颜色距离阈值（无 alpha 时的背景分割）
DEFAULT_COLOR_THRESHOLD = 30


def _ensure_rgba(img_bgr, alpha=None):
    """将 BGR 转为 BGRA；若提供 alpha 则使用，否则不透明。"""
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[2] == 4:
        return img_bgr  # 已是 BGRA
    if alpha is not None:
        return cv2.merge((img_bgr, alpha))
    return cv2.merge((img_bgr, np.full(img_bgr.shape[:2], 255, dtype=np.uint8)))


def _read_png(path):
    """
    读取 PNG，兼容带/不带透明通道。
    返回 (BGR 或 BGRA 的 ndarray, 是否有 alpha)。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, False
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    has_alpha = img.shape[2] == 4
    return img, has_alpha


def _get_mask_from_alpha(img):
    """从带透明通道的图中取二值前景掩码（非完全透明为前景）。"""
    if img.shape[2] != 4:
        return None
    alpha = img[:, :, 3]
    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    return mask


def _detect_background_color(img_bgr):
    """从图像四角区域估计背景颜色（BGR）。"""
    h, w = img_bgr.shape[:2]
    s = min(15, h // 2, w // 2)
    if s <= 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    regions = [
        img_bgr[0:s, 0:s],
        img_bgr[0:s, w - s : w],
        img_bgr[h - s : h, 0:s],
        img_bgr[h - s : h, w - s : w],
    ]
    return np.median(np.vstack([r.reshape(-1, 3) for r in regions]), axis=0).astype(np.uint8)


def _color_distance(p, bg):
    return np.sqrt(np.sum((p.astype(np.float32) - bg.astype(np.float32)) ** 2))


def _get_mask_from_color(img_bgr, bg_color, threshold):
    """根据与背景色的距离得到二值前景掩码。"""
    h, w = img_bgr.shape[:2]
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    dist = np.sqrt(np.sum((img_bgr.astype(np.float32) - bg) ** 2, axis=2))
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _connected_components(mask, min_area=MIN_ELEMENT_AREA):
    """
    对二值掩码做连通分量分析，返回每个前景成分的 (标签图, 成分数量)。
    标签 0 为背景，1..N 为各独立元素。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 过滤面积过小的成分（可选：将小成分并入背景）
    h, w = mask.shape[:2]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            labels[labels == i] = 0
    # 重新编号，使标签连续 0,1,2,...
    unique = np.unique(labels)
    unique = unique[unique > 0]
    new_labels = np.zeros_like(labels)
    for idx, old in enumerate(unique, start=1):
        new_labels[labels == old] = idx
    return new_labels, len(unique)


def _save_elements(rgba, labels, num_elements, output_dir):
    """
    将每个连通分量保存为独立 PNG（带透明通道）。
    输出文件：output_dir/element_1.png, element_2.png, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i in range(1, num_elements + 1):
        mask_i = (labels == i).astype(np.uint8)
        if mask_i.sum() == 0:
            continue
        # 仅保留当前成分的像素，其余透明
        out = np.zeros((rgba.shape[0], rgba.shape[1], 4), dtype=np.uint8)
        out[:, :, :3] = rgba[:, :, :3]
        out[:, :, 3] = rgba[:, :, 3] * mask_i
        # 可选：裁剪到外接矩形以减小文件（此处保存整图以保持坐标一致）
        out_path = os.path.join(output_dir, f"element_{i}.png")
        cv2.imwrite(out_path, out)
        saved.append(out_path)
    return saved


def process_one_image(image_path, output_base_dir, min_area=MIN_ELEMENT_AREA, color_threshold=DEFAULT_COLOR_THRESHOLD):
    """
    处理单张 PNG：得到前景掩码 → 连通分量 → 写出 element_*.png。
    """
    img, has_alpha = _read_png(image_path)
    if img is None:
        return False, f"无法读取图片: {image_path}"

    # 统一为 BGRA
    if has_alpha:
        rgba = img
        mask = _get_mask_from_alpha(img)
    else:
        rgb_only = img[:, :, :3].copy()
        # 使用颜色法生成前景掩码
        bg_color = _detect_background_color(rgb_only)
        mask = _get_mask_from_color(rgb_only, bg_color, color_threshold)
        rgba = _ensure_rgba(rgb_only, mask)

    labels, num_elements = _connected_components(mask, min_area=min_area)
    if num_elements == 0:
        return False, "未检测到任何独立元素"

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_base_dir, base_name)
    saved = _save_elements(rgba, labels, num_elements, out_dir)
    return True, out_dir


def main():
    parser = argparse.ArgumentParser(description="从 PNG 图片中分离独立元素（输出至 output/{图片名}/element_*.png）")
    parser.add_argument("--image", default=None, help="输入 PNG 路径；不指定则处理 test_img 下全部 PNG")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="默认输入目录（未指定 --image 时）")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出根目录")
    parser.add_argument("--min-area", type=int, default=MIN_ELEMENT_AREA, help="最小元素面积（像素）")
    parser.add_argument("--threshold", type=int, default=DEFAULT_COLOR_THRESHOLD, help="颜色法背景阈值（0-255）")
    args = parser.parse_args()

    if args.image:
        paths = [os.path.abspath(args.image)]
        if not os.path.isfile(paths[0]):
            print(f"错误：文件不存在 - {paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.isdir(args.input_dir):
            print(f"错误：输入目录不存在 - {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        paths = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(".png")
        ]
        if not paths:
            print(f"未在 {args.input_dir} 中找到 PNG 文件。", file=sys.stderr)
            sys.exit(1)

    # 执行处理
    success_count = 0
    fail_count = 0
    for path in paths:
        ok, result = process_one_image(
            path,
            args.output_dir,
            min_area=args.min_area,
            color_threshold=args.threshold,
        )
        if ok:
            print(f"✅ 已处理: {path} -> 输出目录: {result}")
            success_count += 1
        else:
            print(f"❌ 处理失败: {path} -> {result}", file=sys.stderr)
            fail_count += 1

    print(f"\n处理完成：成功 {success_count} 张，失败 {fail_count} 张")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import rp

from motion_aware_modulation import apply_motion_scale_to_flow, compute_motion_scaling
from video_save_compat import install_video_save_compat


def import_raft_module():
    # 动态导入 RAFT 光流模块，优先尝试 CommonSource 路径
    rp.git_import("CommonSource")
    candidates = (
        "rp.git.CommonSource.raft",
        "raft",
    )
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError("Unable to import a RAFT module from CommonSource")


# 导入 CommonSource 中的噪声形变模块
rp.git_import("CommonSource")
import rp.git.CommonSource.noise_warp as nw


def parse_arguments():
    # 命令行参数定义
    parser = argparse.ArgumentParser(
        description="Generate Go-with-the-Flow-compatible warped noise from videos with motion-aware flow scaling."
    )
    # 输入模式互斥：单视频 或 视频目录
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Single local video path or URL.")
    input_group.add_argument("--video_dir", help="Directory of local video files to process.")

    # 输出与处理参数
    parser.add_argument("--output_root", default="motion_aware_noise_outputs")
    parser.add_argument("--target_num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--noise_channels", type=int, default=16)
    parser.add_argument("--resize_flow", type=int, default=8)
    parser.add_argument("--downscale_factor", type=int, default=8)
    parser.add_argument("--min_scale", type=float, default=0.5)
    parser.add_argument("--max_scale", type=float, default=2.0)
    parser.add_argument("--low_percentile", type=float, default=20.0)
    parser.add_argument("--high_percentile", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_video_sources(args):
    # 如果指定了单视频，直接返回
    if args.video:
        return [args.video]

    # 否则扫描目录中的视频文件
    video_extensions = {".avi", ".mov", ".mp4", ".mkv", ".webm", ".gif"}
    video_dir = Path(args.video_dir)
    return sorted(
        str(path)
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in video_extensions
    )


def sanitize_video_name(video_source):
    # URL 和本地路径分别提取 stem，并将空格替换为下划线
    if "://" in video_source:
        stem = Path(video_source.split("?")[0]).stem or "remote_video"
    else:
        stem = Path(video_source).stem
    return stem.replace(" ", "_")


def load_and_preprocess_video(video_source, target_num_frames, height, width):
    # 加载视频并统一帧数与分辨率
    video = rp.load_video(video_source)
    video = rp.resize_list(video, length=target_num_frames)
    video = rp.resize_images_to_hold(video, height=height, width=width)
    video = rp.crop_images(video, height=height, width=width, origin="center")
    video = rp.as_rgb_images(video)
    video = np.asarray(video, dtype=np.float32)
    # 若是 0~255 像素值，归一化到 0~1
    if video.max() > 1.5:
        video = video / 255.0
    return video


def compute_flows_and_motions(video, raft_model):
    # 逐帧计算光流与运动强度
    flows_dx = []
    flows_dy = []
    motions = []

    previous_frame = video[0]
    for frame in video[1:]:
        dx, dy = raft_model(previous_frame, frame)
        flows_dx.append(dx)
        flows_dy.append(dy)
        motions.append(float((dx.abs() + dy.abs()).sum().item()))
        previous_frame = frame

    return flows_dx, flows_dy, motions


def downscale_noise(noise, downscale_factor):
    # 先面积插值下采样，再按比例放大振幅
    down_noise = rp.torch_resize_image(noise, 1 / downscale_factor, interp="area")
    return down_noise * downscale_factor


def tensor_to_nested_list(tensor):
    # Tensor -> Python 嵌套列表（用于外部缩放函数）
    return tensor.detach().cpu().tolist()


def nested_list_to_tensor(nested, reference_tensor):
    # Python 嵌套列表 -> Tensor（保持设备与 dtype 一致）
    return reference_tensor.new_tensor(nested)


def save_compat_outputs(output_dir, video, latent_noises):
    # 保存兼容 Go-with-the-Flow 的输出文件
    video_uint8 = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    np.save(output_dir / "noises.npy", latent_noises.astype(np.float16))
    np.save(output_dir / "latent.npy", latent_noises.astype(np.float16))
    rp.save_video_mp4(video_uint8, str(output_dir / "input.mp4"), framerate=12, video_bitrate="max")
    rp.save_image(video_uint8[0], str(output_dir / "first_frame.png"))


def save_metadata(output_dir, motions, scales):
    # 保存运动值、缩放值及摘要信息
    np.save(output_dir / "motion.npy", np.asarray(motions, dtype=np.float32))
    np.save(output_dir / "scales.npy", np.asarray(scales, dtype=np.float32))
    (output_dir / "modulation.json").write_text(
        json.dumps(
            {
                "mode": "motion_aware",
                "motion_count": len(motions),
                "min_scale": min(scales) if scales else 1.0,
                "max_scale": max(scales) if scales else 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def process_video(video_source, device, raft_model, args):
    # 处理单个视频：光流 -> 运动缩放 -> 噪声扭曲 -> 保存
    video_name = sanitize_video_name(video_source)
    output_dir = Path(args.output_root) / video_name
    if output_dir.exists() and not args.overwrite:
        print(f"[skip] {output_dir} exists")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    video = load_and_preprocess_video(
        video_source=video_source,
        target_num_frames=args.target_num_frames,
        height=args.height,
        width=args.width,
    )
    _, height, width, _ = video.shape
    flows_dx, flows_dy, motions = compute_flows_and_motions(video, raft_model)
    scales = compute_motion_scaling(
        motions,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )

    # 初始化噪声 warper
    warper = nw.NoiseWarper(
        c=args.noise_channels,
        h=args.resize_flow * height,
        w=args.resize_flow * width,
        device=device,
        post_noise_alpha=0.0,
        progressive_noise_alpha=0.0,
        warp_kwargs=dict(),
    )

    noises = []
    # 第一帧使用初始噪声
    initial_noise = downscale_noise(warper.noise, args.downscale_factor)
    noises.append(rp.as_numpy_image(initial_noise).astype(np.float16))

    # 后续帧按 motion scale 调制光流后进行 warping
    for dx, dy, frame_scale in zip(flows_dx, flows_dy, scales):
        modulated_dx_list, modulated_dy_list = apply_motion_scale_to_flow(
            dx_frame=tensor_to_nested_list(dx),
            dy_frame=tensor_to_nested_list(dy),
            frame_scale=frame_scale,
        )
        modulated_dx = nested_list_to_tensor(modulated_dx_list, dx)
        modulated_dy = nested_list_to_tensor(modulated_dy_list, dy)
        warped_noise = warper(modulated_dx, modulated_dy).noise
        downscaled_noise = downscale_noise(warped_noise, args.downscale_factor)
        noises.append(rp.as_numpy_image(downscaled_noise).astype(np.float16))

    # 堆叠并保存输出
    latent_noises = np.stack(noises, axis=0)
    save_compat_outputs(output_dir=output_dir, video=video, latent_noises=latent_noises)
    save_metadata(output_dir=output_dir, motions=motions, scales=scales)
    print(f"[ok] saved motion-aware warped noise for {video_source} -> {output_dir}")


def main():
    install_video_save_compat()
    args = parse_arguments()

    # macOS 默认使用 CPU，其他平台自动选 torch 设备
    if rp.currently_running_mac():
        device = "cpu"
    else:
        device = rp.select_torch_device(prefer_used=True)
    print(f"Using device: {device}")

    # 加载 RAFT 光流模型
    raft_module = import_raft_module()
    raft_model = raft_module.RaftOpticalFlow(device=device, version="large")

    # 收集输入视频列表
    video_sources = discover_video_sources(args)
    if not video_sources:
        raise FileNotFoundError("No video sources found for the selected input mode")

    # 批量处理每个视频
    for video_source in video_sources:
        process_video(
            video_source=video_source,
            device=device,
            raft_model=raft_model,
            args=args,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

NOISE_ROOT="output/noise/motion_aware_noise_outputs"
NOISE_DIR="${NOISE_ROOT}/demo"
OUT_DIR="output/mag_videos/demo"
OUT_MP4="${OUT_DIR}/demo_motion_aware.mp4"

mkdir -p "${OUT_DIR}"

python make_motion_aware_warped_noise.py \
  --video "input_videos/demo/demo.mp4" \
  --output_root "${NOISE_ROOT}" \
  --resize_flow 8 \
  --downscale_factor 64 \
  --overwrite

python cut_and_drag_inference.py "${NOISE_DIR}" \
  --prompt "Preserve the original video content, subject identity, facial features, clothing, background, composition, camera viewpoint, and lighting. Only subtly amplify the existing natural human motion with a very small increase in motion amplitude. Keep the original action timing, direction, and trajectory unchanged. Maintain sharp details, clear boundaries, stable structure, and strong temporal consistency. No motion blur, no ghosting, no deformation, no camera shake, no new objects, no scene change." \
  --output_mp4_path "${OUT_MP4}" \
  --device "cuda" \
  --num_inference_steps 24 \
  --guidance_scale 3 \
  --degradation 0.1 \
  --low_vram False

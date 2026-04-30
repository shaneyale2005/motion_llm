import os
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


Box = List[float]


def clip_box(box: Sequence[float], frame_width: int, frame_height: int) -> Optional[Box]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, float(frame_width)))
    y1 = max(0.0, min(y1, float(frame_height)))
    x2 = max(0.0, min(x2, float(frame_width)))
    y2 = max(0.0, min(y2, float(frame_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def union_boxes(boxes: Iterable[Sequence[float]], frame_width: int, frame_height: int) -> Optional[Box]:
    clipped = [box for box in (clip_box(box, frame_width, frame_height) for box in boxes) if box]
    if not clipped:
        return None
    return [
        min(box[0] for box in clipped),
        min(box[1] for box in clipped),
        max(box[2] for box in clipped),
        max(box[3] for box in clipped),
    ]


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def box_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _read_frame(path: str) -> Optional[np.ndarray]:
    frame = cv2.imread(path)
    if frame is None:
        return None
    return frame


def _boxes_from_magnitude(
    magnitude: np.ndarray,
    percentile: float,
    min_area_ratio: float,
    max_boxes: int,
) -> List[Box]:
    if magnitude.size == 0 or float(np.max(magnitude)) <= 1e-6:
        return []

    threshold = float(np.percentile(magnitude, percentile))
    if threshold <= 1e-6:
        return []

    mask = (magnitude >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    height, width = mask.shape[:2]
    min_area = max(16, int(width * height * min_area_ratio))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w <= 1 or h <= 1:
            continue
        components.append((int(area), [float(x), float(y), float(x + w), float(y + h)]))

    components.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in components[:max_boxes]]


def compute_frame_motion_boxes(
    frames_path: str,
    len_frames: int,
    percentile: float = 90.0,
    min_area_ratio: float = 0.002,
    max_boxes: int = 5,
) -> Dict[int, List[Box]]:
    """Compute high-motion boxes for each saved frame using Farneback optical flow."""
    boxes_by_frame: Dict[int, List[Box]] = {idx: [] for idx in range(len_frames)}
    if len_frames <= 1:
        return boxes_by_frame

    prev_frame = _read_frame(os.path.join(frames_path, "0000.jpg"))
    if prev_frame is None:
        return boxes_by_frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    last_boxes: List[Box] = []

    for frame_idx in range(1, len_frames):
        frame = _read_frame(os.path.join(frames_path, f"{frame_idx:04d}.jpg"))
        if frame is None:
            boxes_by_frame[frame_idx] = last_boxes
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        last_boxes = _boxes_from_magnitude(
            magnitude=magnitude,
            percentile=percentile,
            min_area_ratio=min_area_ratio,
            max_boxes=max_boxes,
        )
        boxes_by_frame[frame_idx] = last_boxes
        prev_gray = gray

    if len_frames > 1:
        boxes_by_frame[0] = boxes_by_frame.get(1, [])
    return boxes_by_frame


def merge_box_with_motion(
    base_box: Sequence[float],
    motion_boxes: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
    near_ratio: float = 0.12,
) -> Box:
    clipped_base = clip_box(base_box, frame_width, frame_height)
    if clipped_base is None:
        clipped_base = [0.0, 0.0, float(frame_width), float(frame_height)]

    near_distance = max(frame_width, frame_height) * near_ratio
    merge_candidates = [clipped_base]
    for motion_box in motion_boxes:
        clipped_motion = clip_box(motion_box, frame_width, frame_height)
        if clipped_motion is None:
            continue
        if box_iou(clipped_base, clipped_motion) > 0 or box_distance(clipped_base, clipped_motion) <= near_distance:
            merge_candidates.append(clipped_motion)

    merged = union_boxes(merge_candidates, frame_width, frame_height)
    return merged if merged is not None else clipped_base


def build_motion_fallback_clusters(
    motion_boxes_by_frame: Dict[int, List[Box]],
    len_frames: int,
    frame_width: int,
    frame_height: int,
) -> List[dict]:
    """Create frame-covering clusters from flow boxes for tracking-failure fallback."""
    clusters = []
    last_box: Optional[Box] = None
    for frame_idx in range(len_frames):
        motion_box = union_boxes(motion_boxes_by_frame.get(frame_idx, []), frame_width, frame_height)
        if motion_box is None:
            motion_box = last_box
        if motion_box is None:
            motion_box = [0.0, 0.0, float(frame_width), float(frame_height)]
        last_box = motion_box
        clusters.append({"start_frame": frame_idx, "end_frame": frame_idx, "union_box": motion_box})
    return clusters

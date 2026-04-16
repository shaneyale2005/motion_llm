import math


# 计算给定列表在指定百分位上的值（线性插值）
def _percentile(values, percentile):
    # 空列表时返回 0.0，避免后续计算报错
    if not values:
        return 0.0
    # 统一转为 float 并排序
    ordered = sorted(float(value) for value in values)
    # 仅一个元素时直接返回该元素
    if len(ordered) == 1:
        return ordered[0]
    # 计算百分位对应的浮点下标
    rank = (len(ordered) - 1) * (percentile / 100.0)
    # 下界与上界索引
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    # 若刚好命中整数索引，直接返回
    if lower == upper:
        return ordered[lower]
    # 否则做线性插值
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# 根据整段 motion 序列计算每帧缩放系数
def compute_motion_scaling(
    motions,
    min_scale=0.5,
    max_scale=2.0,
    low_percentile=20.0,
    high_percentile=80.0,
):
    # 空输入返回空列表
    if not motions:
        return []

    # 全部转 float 便于统一计算
    motions = [float(motion) for motion in motions]
    # 所有 motion 几乎相等时，统一返回 1.0（不缩放）
    if all(abs(motion - motions[0]) < 1e-9 for motion in motions):
        return [1.0 for _ in motions]

    # 计算低/高百分位阈值
    lo = _percentile(motions, low_percentile)
    hi = _percentile(motions, high_percentile)
    # 阈值异常时回退为不缩放
    if hi <= lo or hi <= 0:
        return [1.0 for _ in motions]

    scales = []
    for motion in motions:
        # 低于低阈值：给最大缩放
        if motion <= lo:
            scale = max_scale
        # 高于高阈值：给最小缩放
        elif motion >= hi:
            scale = min_scale
        else:
            # 中间区间：线性插值（motion 越大，scale 越小）
            alpha = (motion - lo) / (hi - lo)
            scale = max_scale - alpha * (max_scale - min_scale)
        # 双重保护，确保落在[min_scale, max_scale]
        scale = max(min_scale, min(max_scale, scale))
        # 保留 4 位小数
        scales.append(round(scale, 4))
    return scales


# 根据光流 dx/dy 计算空间权重图（每像素）
def spatial_motion_weights(dx_frame, dy_frame):
    rows = len(dx_frame)
    cols = len(dx_frame[0]) if rows else 0
    magnitudes = []
    # 逐像素计算运动幅值 sqrt(dx^2 + dy^2)
    for row_index in range(rows):
        magnitude_row = []
        for col_index in range(cols):
            magnitude = math.sqrt(
                float(dx_frame[row_index][col_index]) ** 2
                + float(dy_frame[row_index][col_index]) ** 2
            )
            magnitude_row.append(magnitude)
        magnitudes.append(magnitude_row)

    # 拉平成一维，便于求全局 min/max
    flat = [value for row in magnitudes for value in row]
    # 空输入时直接返回原结果
    if not flat:
        return magnitudes
    minimum = min(flat)
    maximum = max(flat)
    # 若全图幅值几乎一致，返回全 1.0 权重
    if abs(maximum - minimum) < 1e-9:
        return [[1.0 for _ in range(cols)] for _ in range(rows)]

    # 将幅值归一化到 [0.5, 1.0] 作为空间权重
    output = []
    for row in magnitudes:
        output.append([0.5 + 0.5 * ((value - minimum) / (maximum - minimum)) for value in row])
    return output


# 将帧级缩放系数 frame_scale 结合空间权重应用到光流上
def apply_motion_scale_to_flow(dx_frame, dy_frame, frame_scale):
    # 先计算每像素空间权重
    weights = spatial_motion_weights(dx_frame=dx_frame, dy_frame=dy_frame)
    modulated_dx = []
    modulated_dy = []
    # 逐行处理 dx/dy 与对应权重
    for dx_row, dy_row, weight_row in zip(dx_frame, dy_frame, weights):
        dx_out = []
        dy_out = []
        # 逐像素应用增益
        for dx_value, dy_value, pixel_weight in zip(dx_row, dy_row, weight_row):
            # gain 在 pixel_weight=0 时为 1，在 1 时接近 frame_scale
            gain = 1.0 + (frame_scale - 1.0) * pixel_weight
            dx_out.append(float(dx_value) * gain)
            dy_out.append(float(dy_value) * gain)
        modulated_dx.append(dx_out)
        modulated_dy.append(dy_out)
    return modulated_dx, modulated_dy

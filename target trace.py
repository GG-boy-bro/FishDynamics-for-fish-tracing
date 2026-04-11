# -*- coding: utf-8 -*-
"""
YOLO-Seg + 稳定追踪(B+ReID) + 椭圆拟合 + 摄食强度指标

前端：基本沿用你当前这版 B（ReID + ID-lock + 每 detect_interval 帧检测）
后端：在“检测帧”上，用轨迹位移 + 椭圆参数计算：
- 单鱼：速度、体长速度、方向、动能 proxy、阻力功率
- 群体：K_bl_mean, P_drag_mean, D_density, H_angle, F

需要你手动标定的参数（必改）：
----------------------------------------------------------------
FISH_LENGTH_CM = 18       # 单鱼体长 (cm)
FISH_WEIGHT_G = 25        # 单鱼质量 (g)
POOL_RADIUS_CM = 50       # 水池半径 (cm)
PIXEL_TO_CM = 0.12        # 每像素对应长度 (cm/px)，标定得到

使用说明：
- 仍然是跳帧检测：detect_interval 决定检测频率
- 速度/能量在“相邻两次检测之间”计算
"""

import cv2
import numpy as np
import math
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import os

# ===================== 标定与物理参数 =====================

FISH_LENGTH_CM = 18.0   # ⭐ 必填：鱼体长(cm)
FISH_WEIGHT_G  = 25.0   # ⭐ 必填：鱼体质量(g)
POOL_RADIUS_CM = 50.0   # ⭐ 必填：池半径(cm)
PIXEL_TO_CM    = 0.12   # ⭐ 必填：像素 -> cm 比例

RHO_WATER = 1000.0      # 水密度 kg/m^3
DRAG_COEFF = 0.47       # 阻力系数，球/椭球量级，近似

ANGLE_BINS = 12         # 方向离散化 bin 数，用于熵

# ===================== 路径 & 算法参数 =====================

P = {
    "model_path": "D:/Desk/feed_data/best.pt",     # YOLO-seg 模型
    "video_path": "D:/Desk/feed_data/00.avi",      # 输入视频
    "out_video": "D:/Desk/feed_data/fish_final_with_metrics.mp4",  # 输出视频
    "csv_base":  "D:/Desk/feed_data/fish_metrics",  # 输出 CSV 基名（会生成 *_tracks.csv 和 *_group.csv）

    "conf_thresh": 0.5,        # 检测置信度
    "detect_interval": 3,      # 每 N 帧检测一次

    "max_lost": 30,            # 允许丢失的检测帧数量

    # 匹配代价主参数（不使用椭圆参与匹配）
    "max_match_cost": 0.7,     # 总匹配代价阈值，越小越严格
    "w_center": 0.65,          # 中心距离权重
    "w_iou":    0.25,          # IoU 权重
    "w_hist":   0.10,          # 颜色相似度权重

    # ID-locking：防止一次检测中，轨迹中心跳太远
    # 单位：相对于图像对角线的比例
    "max_center_jump": 0.25,   # 每次更新时允许的最大归一化位移（0.25 ≈ 1/4 对角线）

    # ReID 参数：尝试用颜色+位置找回刚刚丢失的轨迹
    "reid_max_lost": 60,       # 最多允许丢失多少个检测帧还能被 ReID 找回
    "reid_max_dist": 0.30,     # ReID 时中心归一化距离上限（0~1，越小越严格）
    "reid_min_hist": 0.65,     # ReID 时颜色相似度下限（0~1，越大越严格）

    # 绘制相关
    "alpha_draw": 0.35,
    "ellipse_color": (0, 0, 255),
    "font_scale": 0.8,
    "font_thick": 2,
}

# ================== 工具函数 ==================

def iou_xyxy(a, b):
    """IoU 计算, a/b: [x1,y1,x2,y2]"""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, (a[2] - a[0]) * (a[3] - a[1]))
    areaB = max(0, (b[2] - b[0]) * (b[3] - b[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def ellipse_from_mask(mask):
    """从二值 mask 拟合椭圆（只用于显示/分析，不参与匹配）"""
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    try:
        e = cv2.fitEllipse(cnt)
        vals = [e[0][0], e[0][1], e[1][0], e[1][1], e[2]]
        if any(np.isnan(v) for v in vals):
            return None
        return e
    except:
        return None

def compute_hist(img, mask):
    """HSV 2D 直方图，H+S，用于颜色匹配/ReID"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def hist_similarity(h1, h2):
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - d)

def match_cost(det, trk, diag_len):
    """
    cost = w_center * center_cost + w_iou * iou_cost + w_hist * hist_cost
    越小越好
    """
    cx_d, cy_d = det["center"]
    cx_t, cy_t = trk.center

    center_dist = math.hypot(cx_d - cx_t, cy_d - cy_t)
    center_cost = center_dist / (diag_len + 1e-6)   # 归一化

    iou = iou_xyxy(det["bbox"], trk.bbox)
    iou_cost = 1.0 - iou

    hsim = hist_similarity(det["hist"], trk.hist)
    hist_cost = 1.0 - hsim

    total = (
        P["w_center"] * center_cost +
        P["w_iou"]    * iou_cost +
        P["w_hist"]   * hist_cost
    )
    return total, center_cost, iou, hsim

def angle_entropy(angle_list, bins=ANGLE_BINS):
    """
    速度方向的熵，angle_list: 0~π
    """
    if len(angle_list) == 0:
        return 0.0
    hist, _ = np.histogram(angle_list, bins=bins, range=(0, math.pi), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))

# ================== Track 结构 ==================

class Track:
    def __init__(self, tid, det, frame_idx):
        self.id = tid
        self.center = det["center"]          # (cx, cy) 当前检测帧中心
        self.bbox   = det["bbox"]            # [x1,y1,x2,y2]
        self.hist   = det["hist"]            # 颜色特征
        self.ellipse = det["ellipse"]        # ((cx,cy),(MA,ma),angle)
        self.lost   = 0                      # 未匹配的检测帧数
        self.last_seen = frame_idx           # 最后一次检测所在帧

        # 用于速度/能量计算：
        self.prev_center = None              # 上一次检测位置
        self.prev_frame  = None              # 上一次检测帧号

        # 最近一次速度/方向/能量（方便导出）
        self.last_speed_cm_s = 0.0
        self.last_speed_BL_s = 0.0
        self.last_angle_v    = 0.0  # 0~π
        self.last_drag_power = 0.0
        self.last_ke         = 0.0  # 动能 proxy (J)

    def update(self, det, frame_idx):
        # 将当前 center 变为 prev_center
        self.prev_center = self.center
        self.prev_frame  = self.last_seen

        self.center = det["center"]
        self.bbox   = det["bbox"]
        self.hist   = det["hist"]
        self.ellipse = det["ellipse"]
        self.lost   = 0
        self.last_seen = frame_idx

    def compute_metrics(self, fps):
        """
        在两次检测之间，用 prev_center -> center 的位移 + 椭圆，计算速度/能量
        """
        if self.prev_center is None or self.prev_frame is None:
            # 第一次出现，没有前一帧，返回0
            self.last_speed_cm_s = 0.0
            self.last_speed_BL_s = 0.0
            self.last_angle_v    = 0.0
            self.last_drag_power = 0.0
            self.last_ke         = 0.0
            return

        dt_frames = (self.last_seen - self.prev_frame)
        if dt_frames <= 0:
            return
        dt = dt_frames / fps  # 时间(s)

        x0, y0 = self.prev_center
        x1, y1 = self.center
        dx = x1 - x0
        dy = y1 - y0
        dist_px = math.hypot(dx, dy)
        dist_cm = dist_px * PIXEL_TO_CM
        dist_m  = dist_cm / 100.0

        # 速度
        v_m_s = dist_m / dt if dt > 0 else 0.0
        v_cm_s = v_m_s * 100.0
        v_BL_s = v_cm_s / FISH_LENGTH_CM if FISH_LENGTH_CM > 0 else 0.0

        # 运动方向（0~π）
        ang_v = math.atan2(dy, dx)  # [-π, π]
        if ang_v < 0:
            ang_v += math.pi
        elif ang_v > math.pi:
            ang_v -= math.pi

        # 椭圆主轴角 & 有效投影面积
        (cx, cy), (MA, ma), ang_e_deg = self.ellipse
        # 像素 -> m
        a_m = (MA * PIXEL_TO_CM / 100.0) / 2.0
        b_m = (ma * PIXEL_TO_CM / 100.0) / 2.0
        if a_m <= 0 or b_m <= 0:
            A_eff = 0.0
        else:
            ang_e = math.radians(ang_e_deg)
            rel = ang_v - ang_e
            # 椭圆在速度方向上的有效迎风面积近似：πab * |sin(rel)|
            A_eff = math.pi * a_m * b_m * abs(math.sin(rel))

        # 阻力功率 proxy：P = 1/2 ρ C_d A_eff v^3
        P_drag = 0.5 * RHO_WATER * DRAG_COEFF * A_eff * (v_m_s ** 3)

        # 动能 proxy：K = 1/2 m v^2
        mass_kg = FISH_WEIGHT_G / 1000.0
        K = 0.5 * mass_kg * (v_m_s ** 2)

        self.last_speed_cm_s = float(v_cm_s)
        self.last_speed_BL_s = float(v_BL_s)
        self.last_angle_v    = float(ang_v)
        self.last_drag_power = float(P_drag)
        self.last_ke         = float(K)

# ================== 主逻辑 ==================

model = YOLO(P["model_path"])
cap = cv2.VideoCapture(P["video_path"])

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
diag_len = math.hypot(W, H)

# 视频输出（MP4）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(P["out_video"], fourcc, fps, (W, H))

tracks = {}
next_id = 1
frame_idx = 0

# 群体指标记录
track_records = []   # 单鱼逐“检测帧”记录
group_records = []   # 每个检测帧群体指标

# 池面积 (m^2)
POOL_AREA_M2 = math.pi * (POOL_RADIUS_CM / 100.0) ** 2

print("== Tracking with ReID + ID-locking + metrics ==")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect = (frame_idx % P["detect_interval"] == 0)
    detections = []

    # -------- 检测帧：跑 YOLO-Seg --------
    if detect:
        results = model(frame)[0]
        if results.masks is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            masks  = results.masks.data.cpu().numpy()

            for box, score, m in zip(boxes, scores, masks):
                if score < P["conf_thresh"]:
                    continue

                x1, y1, x2, y2 = box
                x1 = max(0, min(W-1, x1))
                x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(0, min(H-1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                # mask resize 到整帧
                m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bin = (m_resized > 0.5).astype(np.uint8)*255

                e = ellipse_from_mask(mask_bin)
                if e is None:
                    continue
                (cx, cy), (MA, ma), ang = e
                if not np.isfinite(cx) or not np.isfinite(cy):
                    continue

                hist = compute_hist(frame, mask_bin)

                detections.append({
                    "bbox":   [float(x1), float(y1), float(x2), float(y2)],
                    "center": (float(cx), float(cy)),
                    "hist":   hist,
                    "ellipse": e,
                })

    # -------- 匹配（只在检测帧进行） --------
    if detect:
        tr_ids = list(tracks.keys())
        used_tracks = set()
        used_dets   = set()

        # 1) Hungarian 匹配 + ID-locking
        if tr_ids and detections:
            cost_mat = np.zeros((len(tr_ids), len(detections)), dtype=float)
            aux_center = np.zeros_like(cost_mat)

            for i, tid in enumerate(tr_ids):
                t = tracks[tid]
                for j, det in enumerate(detections):
                    c, center_c, _, _ = match_cost(det, t, diag_len)
                    cost_mat[i, j]   = c
                    aux_center[i, j] = center_c

            row_ind, col_ind = linear_sum_assignment(cost_mat)

            for r, c in zip(row_ind, col_ind):
                total_cost = cost_mat[r, c]
                center_c   = aux_center[r, c]
                if total_cost > P["max_match_cost"]:
                    continue
                if center_c > P["max_center_jump"]:
                    continue

                tid = tr_ids[r]
                tracks[tid].update(detections[c], frame_idx)
                used_tracks.add(tid)
                used_dets.add(c)

            # 2) ReID：对未分配的 detection，尝试从“最近丢失的轨迹”中找回旧ID
            for j, det in enumerate(detections):
                if j in used_dets:
                    continue

                best_tid = None
                best_score = -1e9

                for tid, t in tracks.items():
                    if tid in used_tracks:
                        continue
                    if t.lost == 0:
                        continue
                    if t.lost > P["reid_max_lost"]:
                        continue

                    # 位置约束
                    cx_d, cy_d = det["center"]
                    cx_t, cy_t = t.center
                    dist_norm = math.hypot(cx_d - cx_t, cy_d - cy_t) / (diag_len + 1e-6)
                    if dist_norm > P["reid_max_dist"]:
                        continue

                    # 颜色相似度
                    hsim = hist_similarity(det["hist"], t.hist)
                    if hsim < P["reid_min_hist"]:
                        continue

                    # 综合评分：颜色优先，其次距离
                    score = hsim - 0.3 * dist_norm
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None:
                    tracks[best_tid].update(detections[j], frame_idx)
                    tracks[best_tid].lost = 0
                    used_tracks.add(best_tid)
                    used_dets.add(j)

            # 3) 未被匹配、也未被 ReID 利用的 detection -> 新建 track
            for j, det in enumerate(detections):
                if j not in used_dets:
                    tracks[next_id] = Track(next_id, det, frame_idx)
                    used_tracks.add(next_id)
                    next_id += 1

            # 4) 未成功匹配/恢复的老 track，lost++
            for tid in tr_ids:
                if tid not in used_tracks:
                    tracks[tid].lost += 1

        else:
            # 没有已有 track 或没有 detection
            if detections:
                for det in detections:
                    tracks[next_id] = Track(next_id, det, frame_idx)
                    next_id += 1
            else:
                for t in tracks.values():
                    t.lost += 1

        # 清理长期丢失的轨迹
        remove_ids = [tid for tid, t in tracks.items() if t.lost > P["max_lost"]]
        for tid in remove_ids:
            del tracks[tid]

        # ====== 在“检测帧”上计算单鱼和群体指标 ======
        v_BL_list = []
        drag_list = []
        angle_list = []

        for tid, t in tracks.items():
            # 只对本帧刚刚被更新过的轨迹计算（last_seen == frame_idx）
            if t.last_seen != frame_idx:
                continue
            # 更新速度/能量
            t.compute_metrics(fps)

            v_BL_list.append(t.last_speed_BL_s)
            drag_list.append(t.last_drag_power)
            angle_list.append(t.last_angle_v)

            # 记录单鱼指标
            track_records.append({
                "frame": frame_idx,
                "id": tid,
                "cx_px": t.center[0],
                "cy_px": t.center[1],
                "speed_cm_s": t.last_speed_cm_s,
                "speed_BL_s": t.last_speed_BL_s,
                "angle_v_rad": t.last_angle_v,
                "drag_power_W": t.last_drag_power,
                "kinetic_J": t.last_ke,
            })

        # 群体指标
        if len(v_BL_list) > 0:
            K_bl_mean = float(np.mean(v_BL_list))
            P_drag_mean = float(np.mean(drag_list))
            H_angle = angle_entropy(angle_list, bins=ANGLE_BINS)
            N_active = len(v_BL_list)
            D_density = N_active / (POOL_AREA_M2 + 1e-9)

            group_records.append({
                "frame": frame_idx,
                "N_active": N_active,
                "K_bl_mean": K_bl_mean,
                "P_drag_mean": P_drag_mean,
                "D_density": D_density,
                "H_angle": H_angle,
            })

    # -------- 绘制视频（所有帧都画，椭圆沿用最近检测结果） --------
    overlay = frame.copy()
    for tid, t in tracks.items():
        cx, cy = t.center

        if t.ellipse is not None:
            (ex, ey), (MA, ma), ang = t.ellipse
            if np.isfinite(ex) and np.isfinite(ey):
                ell_int = ((int(ex), int(ey)), (int(MA), int(ma)), int(ang))
                try:
                    cv2.ellipse(overlay, ell_int, P["ellipse_color"], -1)
                except:
                    pass

        cv2.putText(
            overlay, f"ID:{tid}",
            (int(cx)+5, int(cy)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            P["font_scale"],
            (255, 255, 255),
            P["font_thick"]
        )

    out_frame = cv2.addWeighted(overlay, P["alpha_draw"], frame, 1 - P["alpha_draw"], 0)
    out.write(out_frame)

    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"[B+ReID+metrics] frame={frame_idx}, tracks={len(tracks)}")

cap.release()
out.release()
print("✅ 视频输出：", P["out_video"])

# ================== 写 CSV & 计算 F 指标 ==================

base = P["csv_base"]
tracks_csv = base + "_tracks.csv"
group_csv  = base + "_group.csv"

os.makedirs(os.path.dirname(base), exist_ok=True)

# 单鱼逐帧
df_tracks = pd.DataFrame(track_records)
df_tracks.to_csv(tracks_csv, index=False, encoding="utf-8-sig")
print("✅ 单鱼指标 CSV:", tracks_csv)

# 群体逐帧 + F
if len(group_records) > 0:
    df_group = pd.DataFrame(group_records)

    # Z-score 标准化（每个视频内部）
    for col in ["K_bl_mean", "P_drag_mean", "D_density", "H_angle"]:
        mu = df_group[col].mean()
        sigma = df_group[col].std()
        if sigma < 1e-9:
            df_group[col + "_z"] = 0.0
        else:
            df_group[col + "_z"] = (df_group[col] - mu) / sigma

    # 这里给一个基础权重，可以以后根据标定/回归调整
    wK, wP, wD, wH = 0.3, 0.3, 0.2, 0.2
    df_group["F"] = (
        wK * df_group["K_bl_mean_z"] +
        wP * df_group["P_drag_mean_z"] +
        wD * df_group["D_density_z"] +
        wH * df_group["H_angle_z"]
    )

    df_group.to_csv(group_csv, index=False, encoding="utf-8-sig")
    print("✅ 群体指标 CSV:", group_csv)
else:
    print("⚠ 没有群体记录（可能视频中没有有效检测）")

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ------------------------ 参数结构 ------------------------

@dataclass
class CalibParams:
    fish_length_cm: float = 18.0
    fish_weight_g: float = 25.0
    pool_radius_cm: float = 50.0
    pixel_to_cm: float = 0.12

@dataclass
class AlgoParams:
    conf_thresh: float = 0.5
    detect_interval: int = 3

    max_lost: int = 30

    max_match_cost: float = 0.7
    w_center: float = 0.65
    w_iou: float = 0.25
    w_hist: float = 0.10

    max_center_jump: float = 0.25

    reid_max_lost: int = 60
    reid_max_dist: float = 0.30
    reid_min_hist: float = 0.65

    alpha_draw: float = 0.35


# ------------------------ 工具函数 ------------------------

def iou_xyxy(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0

def ellipse_from_mask(mask: np.ndarray):
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
    except Exception:
        return None

def compute_hist(img: np.ndarray, mask: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask.shape[:2] != hsv.shape[:2]:
        mask = cv2.resize(mask, (hsv.shape[1], hsv.shape[0]), interpolation=cv2.INTER_NEAREST)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def hist_similarity(h1, h2) -> float:
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - d))

def match_cost(det: Dict[str, Any], trk: "Track", diag_len: float, ap: AlgoParams):
    cx_d, cy_d = det["center"]
    cx_t, cy_t = trk.center
    center_dist = math.hypot(cx_d - cx_t, cy_d - cy_t)
    center_cost = center_dist / (diag_len + 1e-6)

    iou = iou_xyxy(det["bbox"], trk.bbox)
    iou_cost = 1.0 - iou

    hsim = hist_similarity(det["hist"], trk.hist)
    hist_cost = 1.0 - hsim

    total = ap.w_center * center_cost + ap.w_iou * iou_cost + ap.w_hist * hist_cost
    return float(total), float(center_cost), float(iou), float(hsim)

def angle_entropy(angle_list, bins=12):
    if len(angle_list) == 0:
        return 0.0
    hist, _ = np.histogram(angle_list, bins=bins, range=(0, math.pi), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-np.sum(hist * np.log(hist)))

# ------------------------ Track ------------------------

RHO_WATER = 1000.0
DRAG_COEFF = 0.47
ANGLE_BINS = 12

class Track:
    def __init__(self, tid: int, det: Dict[str, Any], frame_idx: int):
        self.id = tid
        self.center = det["center"]
        self.bbox = det["bbox"]
        self.hist = det["hist"]
        self.ellipse = det["ellipse"]
        self.lost = 0
        self.last_seen = frame_idx

        self.prev_center = None
        self.prev_frame = None

        # last metrics
        self.last_speed_cm_s = 0.0
        self.last_speed_BL_s = 0.0
        self.last_angle_v = 0.0
        self.last_drag_power = 0.0
        self.last_ke = 0.0

    def update(self, det: Dict[str, Any], frame_idx: int):
        self.prev_center = self.center
        self.prev_frame = self.last_seen

        self.center = det["center"]
        self.bbox = det["bbox"]
        self.hist = det["hist"]
        self.ellipse = det["ellipse"]
        self.lost = 0
        self.last_seen = frame_idx

    def compute_metrics(self, fps: float, calib: CalibParams):
        if self.prev_center is None or self.prev_frame is None:
            self.last_speed_cm_s = 0.0
            self.last_speed_BL_s = 0.0
            self.last_angle_v = 0.0
            self.last_drag_power = 0.0
            self.last_ke = 0.0
            return

        dt_frames = (self.last_seen - self.prev_frame)
        if dt_frames <= 0:
            return
        dt = dt_frames / float(fps)

        x0, y0 = self.prev_center
        x1, y1 = self.center
        dx, dy = (x1 - x0), (y1 - y0)

        dist_px = math.hypot(dx, dy)
        dist_cm = dist_px * calib.pixel_to_cm
        dist_m = dist_cm / 100.0

        v_m_s = dist_m / dt if dt > 1e-9 else 0.0
        v_cm_s = v_m_s * 100.0
        v_BL_s = v_cm_s / calib.fish_length_cm if calib.fish_length_cm > 1e-9 else 0.0

        ang_v = math.atan2(dy, dx)  # [-pi, pi]
        # fold to [0, pi]
        ang_v = ang_v % math.pi

        (cx, cy), (MA, ma), ang_e_deg = self.ellipse
        a_m = (MA * calib.pixel_to_cm / 100.0) / 2.0
        b_m = (ma * calib.pixel_to_cm / 100.0) / 2.0
        if a_m <= 0 or b_m <= 0:
            A_eff = 0.0
        else:
            ang_e = math.radians(ang_e_deg) % math.pi
            rel = abs(ang_v - ang_e)
            if rel > math.pi/2:
                rel = math.pi - rel
            A_eff = math.pi * a_m * b_m * abs(math.sin(rel))

        P_drag = 0.5 * RHO_WATER * DRAG_COEFF * A_eff * (v_m_s ** 3)

        mass_kg = calib.fish_weight_g / 1000.0
        K = 0.5 * mass_kg * (v_m_s ** 2)

        self.last_speed_cm_s = float(v_cm_s)
        self.last_speed_BL_s = float(v_BL_s)
        self.last_angle_v = float(ang_v)
        self.last_drag_power = float(P_drag)
        self.last_ke = float(K)


# ------------------------ 主处理函数 ------------------------

def process_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    calib: Optional[CalibParams] = None,
    algo: Optional[AlgoParams] = None,
    progress_cb=None,
) -> Dict[str, str]:
    """
    运行整套流程并输出结果文件路径字典：
      - out_video
      - tracks_csv
      - group_csv
      - plot_png
      - f_curve_png
    progress_cb: callable(percent:int, message:str) or None
    """
    calib = calib or CalibParams()
    algo = algo or AlgoParams()

    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{stem}_processed.mp4")
    tracks_csv = os.path.join(output_dir, f"{stem}_tracks.csv")
    group_csv = os.path.join(output_dir, f"{stem}_group.csv")
    plot_png = os.path.join(output_dir, f"{stem}_metrics.png")
    f_curve_png = os.path.join(output_dir, f"{stem}_F_curve.png")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag_len = math.hypot(W, H)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("无法创建输出视频文件（可能是编码器/路径问题）")

    tracks: Dict[int, Track] = {}
    next_id = 1
    frame_idx = 0

    track_records = []
    group_records = []

    pool_area_m2 = math.pi * (calib.pool_radius_cm / 100.0) ** 2

    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect = (frame_idx % max(1, algo.detect_interval) == 0)
        detections = []

        if detect:
            results = model(frame)[0]
            if results.masks is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                masks = results.masks.data.cpu().numpy()

                for box, score, m in zip(boxes, scores, masks):
                    if float(score) < float(algo.conf_thresh):
                        continue

                    x1, y1, x2, y2 = map(float, box)
                    x1 = max(0.0, min(W - 1.0, x1))
                    x2 = max(0.0, min(W - 1.0, x2))
                    y1 = max(0.0, min(H - 1.0, y1))
                    y2 = max(0.0, min(H - 1.0, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (m_resized > 0.5).astype(np.uint8) * 255

                    e = ellipse_from_mask(mask_bin)
                    if e is None:
                        continue
                    (cx, cy), (MA, ma), ang = e
                    if not np.isfinite(cx) or not np.isfinite(cy):
                        continue

                    hist = compute_hist(frame, mask_bin)

                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": (float(cx), float(cy)),
                        "hist": hist,
                        "ellipse": e,
                    })

        # 只在检测帧做匹配/指标
        if detect:
            tr_ids = list(tracks.keys())
            used_tracks = set()
            used_dets = set()

            if tr_ids and detections:
                cost_mat = np.zeros((len(tr_ids), len(detections)), dtype=float)
                aux_center = np.zeros_like(cost_mat)

                for i, tid in enumerate(tr_ids):
                    t = tracks[tid]
                    for j, det in enumerate(detections):
                        c, center_c, _, _ = match_cost(det, t, diag_len, algo)
                        cost_mat[i, j] = c
                        aux_center[i, j] = center_c

                row_ind, col_ind = linear_sum_assignment(cost_mat)

                for r, c in zip(row_ind, col_ind):
                    total_cost = float(cost_mat[r, c])
                    center_c = float(aux_center[r, c])
                    if total_cost > algo.max_match_cost:
                        continue
                    if center_c > algo.max_center_jump:
                        continue
                    tid = tr_ids[r]
                    tracks[tid].update(detections[c], frame_idx)
                    used_tracks.add(tid)
                    used_dets.add(c)

                # ReID
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
                        if t.lost > algo.reid_max_lost:
                            continue

                        cx_d, cy_d = det["center"]
                        cx_t, cy_t = t.center
                        dist_norm = math.hypot(cx_d - cx_t, cy_d - cy_t) / (diag_len + 1e-6)
                        if dist_norm > algo.reid_max_dist:
                            continue
                        hsim = hist_similarity(det["hist"], t.hist)
                        if hsim < algo.reid_min_hist:
                            continue

                        score = hsim - 0.3 * dist_norm
                        if score > best_score:
                            best_score = score
                            best_tid = tid

                    if best_tid is not None:
                        tracks[best_tid].update(detections[j], frame_idx)
                        tracks[best_tid].lost = 0
                        used_tracks.add(best_tid)
                        used_dets.add(j)

                # 新建
                for j, det in enumerate(detections):
                    if j not in used_dets:
                        tracks[next_id] = Track(next_id, det, frame_idx)
                        used_tracks.add(next_id)
                        next_id += 1

                # lost++
                for tid in tr_ids:
                    if tid not in used_tracks:
                        tracks[tid].lost += 1
            else:
                if detections:
                    for det in detections:
                        tracks[next_id] = Track(next_id, det, frame_idx)
                        next_id += 1
                else:
                    for t in tracks.values():
                        t.lost += 1

            # 清理
            remove_ids = [tid for tid, t in tracks.items() if t.lost > algo.max_lost]
            for tid in remove_ids:
                del tracks[tid]

            # 指标
            v_BL_list, drag_list, angle_list = [], [], []
            for tid, t in tracks.items():
                if t.last_seen != frame_idx:
                    continue
                t.compute_metrics(fps, calib)

                v_BL_list.append(t.last_speed_BL_s)
                drag_list.append(t.last_drag_power)
                angle_list.append(t.last_angle_v)

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

            if len(v_BL_list) > 0:
                K_bl_mean = float(np.mean(v_BL_list))
                P_drag_mean = float(np.mean(drag_list))
                H_angle = angle_entropy(angle_list, bins=ANGLE_BINS)
                N_active = int(len(v_BL_list))
                D_density = float(N_active / (pool_area_m2 + 1e-9))

                group_records.append({
                    "frame": frame_idx,
                    "N_active": N_active,
                    "K_bl_mean": K_bl_mean,
                    "P_drag_mean": P_drag_mean,
                    "D_density": D_density,
                    "H_angle": H_angle,
                })

        # 绘制（所有帧）
        overlay = frame.copy()
        for tid, t in tracks.items():
            if t.ellipse is not None:
                (ex, ey), (MA, ma), ang = t.ellipse
                if np.isfinite(ex) and np.isfinite(ey):
                    ell_int = ((int(ex), int(ey)), (int(MA), int(ma)), int(ang))
                    try:
                        cv2.ellipse(overlay, ell_int, (0, 0, 255), -1)
                    except Exception:
                        pass

            cx, cy = t.center
            cv2.putText(
                overlay, f"ID:{tid}",
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        out_frame = cv2.addWeighted(overlay, algo.alpha_draw, frame, 1 - algo.alpha_draw, 0)
        out.write(out_frame)

        frame_idx += 1
        if progress_cb is not None and total_frames > 0 and frame_idx % 5 == 0:
            pct = int(100 * frame_idx / max(1, total_frames))
            progress_cb(pct, f"处理帧 {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    # 写 CSV
    df_tracks = pd.DataFrame(track_records)
    df_tracks.to_csv(tracks_csv, index=False, encoding="utf-8-sig")

    if len(group_records) > 0:
        df_group = pd.DataFrame(group_records)

        # Z-score
        for col in ["K_bl_mean", "P_drag_mean", "D_density", "H_angle"]:
            mu = float(df_group[col].mean())
            sigma = float(df_group[col].std())
            if sigma < 1e-9:
                df_group[col + "_z"] = 0.0
            else:
                df_group[col + "_z"] = (df_group[col] - mu) / sigma

        # 权重可按需要调整
        wK, wP, wD, wH = 0.3, 0.3, 0.2, 0.2
        df_group["F"] = (
            wK * df_group["K_bl_mean_z"] +
            wP * df_group["P_drag_mean_z"] +
            wD * df_group["D_density_z"] +
            wH * df_group["H_angle_z"]
        )
        df_group.to_csv(group_csv, index=False, encoding="utf-8-sig")
    else:
        df_group = pd.DataFrame(columns=["frame","N_active","K_bl_mean","P_drag_mean","D_density","H_angle","F"])
        df_group.to_csv(group_csv, index=False, encoding="utf-8-sig")

    # 绘图
    from .plotting import plot_group_metrics
    plot_group_metrics(group_csv, plot_png, f_curve_png)

    if progress_cb is not None:
        progress_cb(100, "完成")

    return {
        "out_video": out_video,
        "tracks_csv": tracks_csv,
        "group_csv": group_csv,
        "plot_png": plot_png,
        "f_curve_png": f_curve_png,
}

# 写 CSV
    df_tracks = pd.DataFrame(track_records)
    df_tracks.to_csv(tracks_csv, index=False, encoding="utf-8-sig")

    if len(group_records) > 0:
        df_group = pd.DataFrame(group_records)

        # Z-score 标准化处理
        for col in ["K_bl_mean", "P_drag_mean", "D_density", "H_angle"]:
            mu = float(df_group[col].mean())
            sigma = float(df_group[col].std())
            if sigma < 1e-9:
                df_group[col + "_z"] = 0.0
            else:
                df_group[col + "_z"] = (df_group[col] - mu) / sigma

        # 摄食强度指数 F 计算（加权综合指标）
        wK, wP, wD, wH = 0.3, 0.3, 0.2, 0.2
        df_group["F"] = (
            wK * df_group["K_bl_mean_z"] +
            wP * df_group["P_drag_mean_z"] +
            wD * df_group["D_density_z"] +
            wH * df_group["H_angle_z"]
        )
        df_group.to_csv(group_csv, index=False, encoding="utf-8-sig")
    else:
        df_group = pd.DataFrame(columns=["frame","N_active","K_bl_mean","P_drag_mean","D_density","H_angle","F"])
        df_group.to_csv(group_csv, index=False, encoding="utf-8-sig")

    # 调用绘图模块生成可视化结果
    from .plotting import plot_group_metrics
    plot_group_metrics(group_csv, plot_png, f_curve_png)	

    # 进度回调通知完成
    if progress_cb is not None:
        progress_cb(100, "完成")

    # 返回所有输出文件路径
    return {
        "out_video": out_video,
        "tracks_csv": tracks_csv,
        "group_csv": group_csv,
        "plot_png": plot_png,
        "f_curve_png": f_curve_png,
    }


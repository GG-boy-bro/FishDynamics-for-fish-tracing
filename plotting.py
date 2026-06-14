# ------------------------ plotting.py ------------------------

from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 让打包/无界面环境也能画
import matplotlib.pyplot as plt

def plot_group_metrics(group_csv: str, out_png: str, f_curve_png: str) -> None:
    df = pd.read_csv(group_csv)
    # 允许空
    if len(df) == 0:
        # 生成一张提示图，避免 GUI 读不到文件
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No valid group metrics (no detections).", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No F curve.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f_curve_png, dpi=200)
        plt.close()
        return

    t = df["frame"].values

    # 指标
    K = df.get("K_bl_mean", pd.Series([0]*len(df))).values
    P = df.get("P_drag_mean", pd.Series([0]*len(df))).values
    D = df.get("D_density", pd.Series([0]*len(df))).values
    H = df.get("H_angle", pd.Series([0]*len(df))).values
    F = df.get("F", pd.Series([0]*len(df))).values

    # 总览
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t, K)
    plt.title("K_bl_mean (mean speed in BL/s)")
    plt.xlabel("frame")

    plt.subplot(3, 2, 2)
    plt.plot(t, P)
    plt.title("P_drag_mean (W)")
    plt.xlabel("frame")

    plt.subplot(3, 2, 3)
    plt.plot(t, D)
    plt.title("D_density (N/m²)")
    plt.xlabel("frame")

    plt.subplot(3, 2, 4)
    plt.plot(t, H)
    plt.title("H_angle (entropy)")
    plt.xlabel("frame")

    plt.subplot(3, 2, 5)
    plt.plot(t, F, linewidth=2.2)
    plt.title("F (feeding intensity)")
    plt.xlabel("frame")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    # F 曲线
    plt.figure(figsize=(12, 4))
    plt.plot(t, F, linewidth=2.5)
    plt.title("F(t) - feeding intensity")
    plt.xlabel("frame")
    plt.ylabel("F")
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(f_curve_png, dpi=260)
    plt.close()


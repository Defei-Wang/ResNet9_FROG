import os



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_20k", "Ere_20k.csv")
SAVE_DIR = os.path.join(BASE_DIR, "dataset_20k", "Pulse_Plots")
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_PLOTS = 50   # 先跑 50 张验证，确认无误后改为 20000
N_GRID = 256
TIME_WINDOW_FS = 800.0
time_axis = np.linspace(-TIME_WINDOW_FS / 2, TIME_WINDOW_FS / 2, N_GRID)


# ==========================================
# 锚定函数：与 DataLoader.py __getitem__ 完全对齐
# ==========================================
def anchor_pulse(real, imag, mask_threshold=0.05):
    """
    执行与 DataLoader 完全一致的锚定：
      1. 时间中心对齐：最高峰滚动到索引 128
      2. 全局相位归零：最高峰处 Im=0, Re>0
      3. 振幅归一化
      4. 低能量区相位掩膜 -> NaN
    返回: complex_E, abs_E, phase (已掩膜)
    """
    complex_E = real + 1j * imag

    abs_E = np.abs(complex_E)
    peak_idx = np.argmax(abs_E)

    # 步骤1：时间中心对齐（与 DataLoader 一致，先 roll）
    shift = 128 - peak_idx
    complex_E = np.roll(complex_E, shift)

    # 步骤2：全局相位锚定（roll 之后再取峰值相位）
    global_phase = np.angle(complex_E[128])
    complex_E = complex_E * np.exp(-1j * global_phase)

    # 步骤3：归一化
    abs_E = np.abs(complex_E)
    if abs_E.max() > 0:
        complex_E /= abs_E.max()
        abs_E /= abs_E.max()

    # 步骤4：相位提取 + 掩膜
    phase = np.unwrap(np.angle(complex_E))
    phase -= phase[128]  # 锚定后峰值处相位为 0
    phase_masked = phase.copy()
    phase_masked[abs_E < mask_threshold] = np.nan

    return complex_E, abs_E, phase_masked


# ==========================================
# 主程序
# ==========================================
def main():
    print(f"读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"共 {total} 组数据，本次处理 {min(total, MAX_PLOTS)} 组")

    plt.switch_backend('agg')

    for idx in tqdm(range(min(total, MAX_PLOTS)), desc="生成图像"):
        row = df.iloc[idx]
        sid = str(row['id']).zfill(5) if 'id' in df.columns else f"{idx:05d}"

        real_vals = row[[f'real_{i}' for i in range(N_GRID)]].values.astype(float)
        imag_vals = row[[f'imag_{i}' for i in range(N_GRID)]].values.astype(float)

        complex_E, abs_E, phase = anchor_pulse(real_vals, imag_vals)

        # ---- 绘图：3 行子图 ----
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=120)
        fig.suptitle(f"Sample {sid}", fontsize=14, fontweight='bold')

        # 子图1：|E(t)|
        ax = axes[0]
        ax.plot(time_axis, abs_E, color='black', linewidth=2)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)   # 中心线
        ax.set_ylabel("|E(t)| (a.u.)")
        ax.set_xlim([-300, 300])
        ax.set_ylim([-0.05, 1.15])
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(["|E(t)|"], loc='upper right')

        # 子图2：Re(E) 和 Im(E)
        ax = axes[1]
        ax.plot(time_axis, np.real(complex_E), color='steelblue', linewidth=1.5, label='Re(E)')
        ax.plot(time_axis, np.imag(complex_E), color='tomato',    linewidth=1.5, label='Im(E)', alpha=0.8)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_ylabel("Electric Field (a.u.)")
        ax.set_xlim([-300, 300])
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        # 子图3：相位 φ(t)（掩膜后）
        ax = axes[2]
        ax.plot(time_axis, phase, color='green', linewidth=2, label=r'$\phi(t)$')
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Phase (rad)")
        ax.set_xlim([-300, 300])
        valid = phase[~np.isnan(phase)]
        if len(valid) > 0:
            margin = max(0.5, (valid.max() - valid.min()) * 0.15)
            ax.set_ylim([valid.min() - margin, valid.max() + margin])
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"pulse_{sid}.png"))
        plt.close(fig)

    print(f"\n完成！图像保存至: {SAVE_DIR}")


if __name__ == '__main__':
    main()

# ======================================================================
# 泛化测试专属生成器：高斯相干叠加版 (无零点穿透，无 pi 跳变)
# 专为测试 ResNet 在常规物理脉冲下的平滑反演能力设计
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from tqdm import tqdm
import os
import pandas as pd

# ----------------------------- 配置参数 -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ✅ 修改点 1：输出到全新的专属测试文件夹
DATASET_DIR = os.path.join(BASE_DIR, "dataset_clean_test") 
SHG_DIR = os.path.join(DATASET_DIR, "SHG")
THG_DIR = os.path.join(DATASET_DIR, "THG")
PREVIEW_DIR = os.path.join(DATASET_DIR, "Previews") 
os.makedirs(SHG_DIR, exist_ok=True)
os.makedirs(THG_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

N_GRID = 256                    
IMG_SIZE = (256, 256)           
T_WINDOW_FS = 800.0             
LAMBDA0_NM = 1064.0             

WL_SH_MIN_NM = 490.0   
WL_SH_MAX_NM = 570.0   
WL_TH_MIN_NM = 320.0   
WL_TH_MAX_NM = 390.0   

# ✅ 修改点 2：生成 500 组用于高级测试
NUM_SAMPLES = 500

c = 3e8                         
lambda0 = LAMBDA0_NM * 1e-9      
nu0 = c / lambda0                

t_fs = np.linspace(-T_WINDOW_FS / 2, T_WINDOW_FS / 2, N_GRID)
t_sec = t_fs * 1e-15
dt = t_sec[1] - t_sec[0]
delays_fs = (np.arange(N_GRID) - N_GRID // 2) * (T_WINDOW_FS / N_GRID)

new_wl_sh = np.linspace(WL_SH_MIN_NM * 1e-9, WL_SH_MAX_NM * 1e-9, N_GRID)
new_wl_th = np.linspace(WL_TH_MIN_NM * 1e-9, WL_TH_MAX_NM * 1e-9, N_GRID)
new_delays = np.linspace(delays_fs.min(), delays_fs.max(), N_GRID)

def compute_frog_trace(E, harmonic=2):
    trace = np.zeros((N_GRID, N_GRID))
    for i in range(N_GRID):
        shift = i - N_GRID // 2
        E_delayed = np.roll(E, shift)
        if harmonic == 2:
            gated = E * E_delayed
        else:
            gated = (E**2) * E_delayed  
            
        spec = np.abs(np.fft.fft(gated))**2
        trace[i, :] = np.fft.fftshift(spec)
    return trace

def get_wavelength_grid(harmonic):
    freq_delta = np.fft.fftshift(np.fft.fftfreq(N_GRID, dt))
    freq_full = (2 if harmonic == 2 else 3) * nu0 + freq_delta
    return c / freq_full  

pulse_data = []  
t_norm = t_fs / 150.0
phase_scales = np.array([0.01, 0.05, 0.5, 1.5, 0.0, 0.0]) * np.pi

# ======================================================================
# ✅ 修改点 3：核心脉冲生成逻辑全面替换为“纯物理相干叠加”
# ======================================================================
for idx in tqdm(range(NUM_SAMPLES), desc="Generating 500 Clean Generalization Pulses"):
    
    while True:
        is_complex = (np.random.rand() < 0.7)
        n_peaks_target = np.random.randint(2, 4) if is_complex else 1
        
        E_temp = np.zeros(N_GRID, dtype=complex)
        
        # 叠加多个独立的高斯子脉冲
        for _ in range(n_peaks_target):
            tau = np.random.uniform(30, 70)
            t0 = np.random.uniform(-100, 100) if is_complex else np.random.uniform(-20, 20)
            amp = np.random.uniform(0.4, 1.0)
            phi0 = np.random.uniform(-np.pi, np.pi) # 载波相位
            
            # 每个子脉冲附带平滑色散
            chirp_coeffs = (np.random.rand(6) - 0.5) * phase_scales
            chirp_phase = np.polyval(chirp_coeffs, t_norm)
            
            sub_pulse = amp * np.exp(-((t_fs - t0)/tau)**2) * np.exp(1j * (phi0 + chirp_phase))
            E_temp += sub_pulse
        
        # 加软时间窗以防止边缘突变
        window = np.exp(-0.5 * (t_fs / (T_WINDOW_FS * 0.45))**10) 
        E = E_temp * window
        abs_E = np.abs(E) 
        
        if np.max(abs_E) == 0: continue
        E /= np.max(abs_E) 
        abs_E /= np.max(abs_E)    
        
        # 智能筛选器
        peaks, _ = find_peaks(abs_E, height=0.15, distance=15)
        
        if is_complex:
            if not (2 <= len(peaks) <= 4): continue
        else:
            if len(peaks) != 1: continue
            
        # ✅ 因为物理上没有跳变，可以直接进行标准 unwrap 提取平滑相位用于画预览图
        smooth_phase = np.unwrap(np.angle(E))
        peak_idx = np.argmax(abs_E)
        smooth_phase -= smooth_phase[peak_idx]
        
        break

# ----------------------------- 计算迹线与保存 -----------------------------
    pulse_data.append([f"{idx:05d}"] + np.real(E).tolist() + np.imag(E).tolist())
    
    trace_sh = compute_frog_trace(E, harmonic=2)
    trace_th = compute_frog_trace(E, harmonic=3)
    
    wl_sh_raw = get_wavelength_grid(2)
    wl_th_raw = get_wavelength_grid(3)
    
    interp_sh = RegularGridInterpolator((delays_fs, wl_sh_raw), trace_sh, bounds_error=False, fill_value=0.0)
    interp_th = RegularGridInterpolator((delays_fs, wl_th_raw), trace_th, bounds_error=False, fill_value=0.0)
    
    delay_mesh, wl_mesh = np.meshgrid(new_delays, new_wl_sh, indexing='ij')
    trace_sh_interp = interp_sh(np.column_stack((delay_mesh.ravel(), wl_mesh.ravel()))).reshape(IMG_SIZE)
    
    delay_mesh, wl_mesh = np.meshgrid(new_delays, new_wl_th, indexing='ij')
    trace_th_interp = interp_th(np.column_stack((delay_mesh.ravel(), wl_mesh.ravel()))).reshape(IMG_SIZE)
    
    if trace_sh_interp.max() > 0: trace_sh_interp = trace_sh_interp / trace_sh_interp.max()
    if trace_th_interp.max() > 0: trace_th_interp = trace_th_interp / trace_th_interp.max()
    
    trace_sh_interp = np.sqrt(trace_sh_interp)  
    trace_th_interp = np.sqrt(trace_th_interp)
    
    sh_img = (trace_sh_interp * 255).astype(np.uint8)
    th_img = (trace_th_interp * 255).astype(np.uint8)
    
    plt.imsave(os.path.join(SHG_DIR, f"shg_{idx:05d}.png"), sh_img, cmap='hot')
    plt.imsave(os.path.join(THG_DIR, f"thg_{idx:05d}.png"), th_img, cmap='hot')

    # 生成前 20张预览图供你检查
    if idx < 20:
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
        
        ax1 = axes[0]
        ax1.plot(t_fs, abs_E, color='#222222', linewidth=3, label=r'$|E(t)|$')
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Absolute Electric Field (a.u.)')
        ax1.set_xlim([-250, 250])
        ax1.set_ylim([-0.05, 1.1])
        
        ax1_twin = ax1.twinx()
        mask = (t_fs >= -150) & (t_fs <= 150)
        ax1_twin.plot(t_fs[mask], smooth_phase[mask], color='#2CA02C', linestyle='--', linewidth=2.5, label=r'$\phi(t)$')
        ax1_twin.set_ylabel('Phase (rad)', color='#2CA02C')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.set_title(f'Clean Sample {idx:05d}')
        
        axes[1].imshow(trace_sh_interp.T, origin='lower', aspect='auto', cmap='hot', extent=[new_delays[0], new_delays[-1], new_wl_sh[0]*1e9, new_wl_sh[-1]*1e9])
        axes[1].set_title('SHG FROG Trace')
        
        axes[2].imshow(trace_th_interp.T, origin='lower', aspect='auto', cmap='hot', extent=[new_delays[0], new_delays[-1], new_wl_th[0]*1e9, new_wl_th[-1]*1e9])
        axes[2].set_title('THG FROG Trace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PREVIEW_DIR, f"preview_trace_{idx:05d}.png"))
        plt.close(fig)

# ----------------------------- 保存 CSV -----------------------------
columns = ['id'] + [f'real_{i}' for i in range(N_GRID)] + [f'imag_{i}' for i in range(N_GRID)]
df = pd.DataFrame(pulse_data, columns=columns)
# ✅ 修改点 4：CSV 名字换成独立的
csv_path = os.path.join(DATASET_DIR, "Ere_clean.csv")
df.to_csv(csv_path, index=False)

print(f"\n🎉 测试泛化集生成完毕！去 Previews 文件夹看看这完美的无跳变曲线吧。")
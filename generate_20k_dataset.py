# 20,000组 (|E|多峰+平滑相位)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import pandas as pd

# ----------------------------- 配置参数 -----------------------------
# 强制指定项目根目录，不再依赖终端当前位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset_20k")
SHG_DIR = os.path.join(DATASET_DIR, "SHG")
THG_DIR = os.path.join(DATASET_DIR, "THG")
PREVIEW_DIR = os.path.join(DATASET_DIR, "Previews")

# 确保路径存在
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

NUM_SAMPLES = 20000 #全量测试

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

def get_base_shape(t, tau, shape_type):
    if shape_type == "gaussian": return np.exp(-(t / tau)**2)
    elif shape_type == "lorentzian": return 1.0 / (1.0 + (t / tau)**2)
    elif shape_type == "sech2": return 1.0 / (np.cosh(t / tau)**2)  

def compute_frog_trace(E, harmonic=2):
    trace = np.zeros((N_GRID, N_GRID))
    for i in range(N_GRID):
        shift = i - N_GRID // 2
        E_delayed = np.roll(E, shift)
        if harmonic == 2:
            gated = E * E_delayed
        else:
            gated = (E**2) * E_delayed  # 物理正确的 THG 积分
            
        spec = np.abs(np.fft.fft(gated))**2
        trace[i, :] = np.fft.fftshift(spec)
    return trace

def get_wavelength_grid(harmonic):
    freq_delta = np.fft.fftshift(np.fft.fftfreq(N_GRID, dt))
    freq_full = (2 if harmonic == 2 else 3) * nu0 + freq_delta
    return c / freq_full  

pulse_data = []  

# 归一化时间基准，保证多项式不过度发散
t_norm = t_fs / 150.0

# ... 前面的代码保持不变 ...

for idx in tqdm(range(NUM_SAMPLES), desc="Generating 20k Balanced Pulses"):
    
    while True:
        # 决定当前脉冲的命运：70% 是复杂多峰，30% 是简单单峰
        is_complex = (np.random.rand() < 0.7)
        
        if is_complex:
            # 【复杂模式】：强制过零点，产生多峰
            c0 = np.random.uniform(-0.2, 0.2)  
            c1_c3 = np.random.uniform(-0.8, 0.8, 3) 
            c4_c5 = np.random.uniform(-0.1, 0.1, 2)
        else:
            # 【简单模式】：常数项很大，保证多项式不会在中心穿过零点，维持单峰
            c0 = np.random.uniform(0.5, 1.2)  
            c1_c3 = np.random.uniform(-0.3, 0.3, 3) # 减弱低阶扰动
            c4_c5 = np.random.uniform(-0.05, 0.05, 2)
            
        amp_coeffs = np.concatenate((c4_c5, c1_c3, [c0]))
        amp_poly = np.polyval(amp_coeffs, t_norm)
        
        shape_type = np.random.choice(["gaussian", "lorentzian", "sech2"])

        if shape_type == "lorentzian":
            tau = np.random.uniform(30, 45)  # 比 Gaussian/sech2 更窄
        else:
            tau = np.random.uniform(40, 60)

        base = get_base_shape(t_fs, tau, shape_type)
        
        envelope = base * amp_poly
        abs_E = np.abs(envelope) 
        
        if np.max(abs_E) == 0: continue
        envelope /= np.max(abs_E) 
        abs_E /= np.max(abs_E)    
        
        # 智能筛选器
        peaks, _ = find_peaks(abs_E, height=0.15, distance=15)
        
        if is_complex:
            # 复杂模式下，只放行 2 或 3 个峰
            if not (2 <= len(peaks) <= 3): continue
        else:
            # 简单模式下，只放行 1 个峰（纯正的单峰脉冲）
            if len(peaks) != 1: continue
            
        # --- 相位调制部分保持不变，对简单和复杂脉冲一视同仁 ---
        phase_scales = np.array([0.01, 0.05, 0.5, 1.5, 0.0, 0.0]) * np.pi
        chirp_coeffs = (np.random.rand(6) - 0.5) * phase_scales
        applied_phase = np.polyval(chirp_coeffs, t_norm)
        
        # 1. 严苛的高斯窗 (0.22 改为 0.18，物理隔绝边界混叠)
        window = np.exp(-0.5 * (t_fs / (T_WINDOW_FS * 0.18))**10) 
        E = envelope * np.exp(1j * applied_phase) * window
        
        # 2. 提取相位
        peak_idx = np.argmax(np.abs(E))
        smooth_phase = applied_phase - applied_phase[peak_idx]

        # ========================================================
        # 提前模拟 Dataloader 的 roll 操作
        # ========================================================
        E_sim_rolled = np.roll(E, 128 - peak_idx) 
        abs_rolled = np.abs(E_sim_rolled) / np.abs(E_sim_rolled).max()
        
        # 检查首尾各 5 个像素的最大值
        boundary_max = max(abs_rolled[:5].max(), abs_rolled[-5:].max())
        

        
        if boundary_max > 0.03: 
            continue

        break

    # 计算迹线和保存代码
        
    pulse_data.append([f"{idx:05d}"] + np.real(E).tolist() + np.imag(E).tolist())
    
    # 计算双谐波迹线
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
    
    plt.imsave(os.path.join(SHG_DIR, f"shg_{idx:05d}.png"), sh_img, cmap='gray')
    plt.imsave(os.path.join(THG_DIR, f"thg_{idx:05d}.png"), th_img, cmap='gray')

    # 生成前 20 张物理预览图
    if idx < 20:
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
        
        # 1. 绝对电场 |E| 与平滑相位
        ax1 = axes[0]
        ax1.plot(t_fs, np.abs(E), color='#222222', linewidth=3, label=r'$|E(t)|$')
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Absolute Electric Field (a.u.)')
        ax1.set_xlim([-400, 400])
        ax1.set_ylim([-0.05, 1.1])
        
        ax1_twin = ax1.twinx()
        mask = (t_fs >= -150) & (t_fs <= 150)
        ax1_twin.plot(t_fs[mask], smooth_phase[mask], color='#2CA02C', linestyle='--', linewidth=2.5, label=r'$\phi(t)$')
        p_min, p_max = np.min(smooth_phase[mask]), np.max(smooth_phase[mask])
        ax1_twin.set_ylim([p_min - 1, p_max + 1])
        ax1_twin.set_ylabel('Phase (rad)', color='#2CA02C')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.set_title(f'Sample {idx:05d}: |E| & Smooth Phase')
        
        # 2. SHG FROG
        im1 = axes[1].imshow(trace_sh_interp.T, origin='lower', aspect='auto', cmap='gray',
                             extent=[new_delays[0], new_delays[-1], new_wl_sh[0]*1e9, new_wl_sh[-1]*1e9])
        axes[1].set_title('SHG FROG Trace')
        axes[1].set_xlabel('Delay (fs)')
        axes[1].set_ylabel('Wavelength (nm)')
        
        # 3. THG FROG
        im2 = axes[2].imshow(trace_th_interp.T, origin='lower', aspect='auto', cmap='gray',
                             extent=[new_delays[0], new_delays[-1], new_wl_th[0]*1e9, new_wl_th[-1]*1e9])
        axes[2].set_title('THG FROG Trace')
        axes[2].set_xlabel('Delay (fs)')
        axes[2].set_ylabel('Wavelength (nm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PREVIEW_DIR, f"preview_trace_{idx:05d}.png"))
        plt.close(fig)

# ----------------------------- 保存 CSV -----------------------------
columns = ['id'] + [f'real_{i}' for i in range(N_GRID)] + [f'imag_{i}' for i in range(N_GRID)]
df = pd.DataFrame(pulse_data, columns=columns)
csv_path = os.path.join(DATASET_DIR, "Ere_20k.csv")
df.to_csv(csv_path, index=False)

print(f"\n✅包含多峰与相位的 20,000 组数据已就绪。")

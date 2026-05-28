import numpy as np

def get_phys_props(real, imag, mask_threshold=0.08):
    """
    终极物理特征提取（放弃伪装，直面物理真实）
    """
    complex_E = real + 1j * imag
    abs_E = np.abs(complex_E)
    if np.max(abs_E) > 0: 
        abs_E /= np.max(abs_E)
    
    # 放弃倍角法，直接进行标准的相位解包
    # 我们承认并保留因包络翻转带来的真实 pi 跳变
    phase = np.unwrap(np.angle(complex_E))
    
    # 相位锚定：依然对齐最高峰为 0，保证对比的绝对基准
    peak_idx = np.argmax(abs_E)
    phase -= phase[peak_idx]
    
    # 核心策略：强力掩膜
    # 在电场绝对值极低（<8%）的区域，相位在物理上失去意义（变成纯噪声）
    # 直接将其置为 NaN，在图表上形成自然的“断点”，避免画出垂直的断崖线
    phase[abs_E < mask_threshold] = np.nan
    
    return abs_E, phase

def get_time_axis(n_grid=256, window_fs=800.0):
    return np.linspace(-window_fs/2, window_fs/2, n_grid)
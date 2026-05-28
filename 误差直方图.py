import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取你本地真实的 Dual 测试集 MSE 分布数据
csv_path = 'test_mse_distribution_Dual.csv'
try:
    df = pd.read_csv(csv_path)
    # 假设你的 CSV 列名为 'MSE'，若没有表头则取第一列
    mse_values = df['MSE'].values if 'MSE' in df.columns else df.iloc[:, 0].values
except FileNotFoundError:
    print(f"Error: 找不到文件 {csv_path}")
    exit()

# 清洗可能存在的极小负值或 0（防止 log10 报错）
mse_values = np.clip(mse_values, 1e-8, None)

plt.figure(figsize=(8, 5))
# 生成对数刻度的 bins
bins = np.logspace(np.log10(mse_values.min()), np.log10(mse_values.max()), 50)
plt.hist(mse_values, bins=bins, color='#01579b', edgecolor='black', alpha=0.75)

plt.xscale('log')
plt.xlabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency (Number of Samples)', fontsize=12, fontweight='bold')
plt.title('Log-Scale MSE Distribution of Dual-Channel Model (N=4000)', fontsize=14)

# 标注平均值与 95% 分位数 (量化长尾)
mean_mse = np.mean(mse_values)
p95_mse = np.percentile(mse_values, 95)
plt.axvline(mean_mse, color='red', linestyle='dashed', linewidth=2, label=f'Mean MSE: {mean_mse:.2e}')
plt.axvline(p95_mse, color='orange', linestyle='dotted', linewidth=2, label=f'95th Percentile: {p95_mse:.2e}')

plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figure_MSE_Histogram.png', dpi=300)
print("误差直方图已生成: Figure_MSE_Histogram.png")
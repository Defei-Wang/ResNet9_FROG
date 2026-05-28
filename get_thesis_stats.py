import pandas as pd
import numpy as np
import os

BASE_DIR = r"E:\下载\FROG_DL_Project"
# 假设你的 4000 个 MSE 数据保存在这个文件里（请确认文件名）
csv_path = os.path.join(BASE_DIR, "logs_Dual_20k", "test_mse_distribution_Dual.csv") 

try:
    df = pd.read_csv(csv_path)
    # 假设 MSE 数据在第一列，或者列名叫 'MSE'
    # 如果列名不一样，请改为 df['你的列名']
    # 方案 A：直接读取第二列（索引为 1）
    mses = df.iloc[:, 1].values 

    # 方案 B（更稳妥）：直接通过你在表头写的列名来读，比如你的列名叫 'MSE' 或 'mse_loss'
    # mses = df['MSE'].values

    print("========= 核心统计指标 =========")
    print(f"均值 (Mean):          {np.mean(mses):.4e}")
    print(f"中位数 (Median):      {np.median(mses):.4e}")
    print(f"标准差 (Std Dev):     {np.std(mses):.4e}")
    print(f"95% 分位数 (95th P):  {np.percentile(mses, 95):.4e}")
    print(f"99% 分位数 (99th P):  {np.percentile(mses, 99):.4e}")
    print(f"最大值 (Max):         {np.max(mses):.4e}")
    print(f"最小值 (Min):         {np.min(mses):.4e}")
    print("================================")
except Exception as e:
    print(f"读取失败，请检查路径或列名: {e}")
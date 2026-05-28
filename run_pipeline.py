import subprocess
import sys
import os

# ==========================================
# 1. 定义你要顺序执行的脚本清单
# ==========================================
# 请确保这里的名字和你实际的评估脚本名字完全对应！
pipeline_scripts = [
    "trainDual_20k.py",               # 1. 训练双通道
    "evaluate_Dual_20k.py",
    "diagnose_dual_errors.py",        # 2. 评估双通道 (或者是你的 evaluate_dual.py)
    "train_SHG_only_20k.py",          # 3. 训练单通道
    "evaluate_SHG_only.py",
    "diagnose_single_errors.py",     # 4. 评估单通道 (或者是你的 evaluate_SHG_only.py)
    "plot_final_ablation.py"
]

# 获取当前 Python 解释器的绝对路径 (极其重要，防止 VS Code 环境串台)
python_exe = sys.executable
base_dir = os.path.dirname(os.path.abspath(__file__))

print("=====================================================")
print("🤖 自动化超快光学反演流水线已启动")
print("=====================================================\n")

# ==========================================
# 2. 顺序执行循环
# ==========================================
for script in pipeline_scripts:
    script_path = os.path.join(base_dir, script)
    
    if not os.path.exists(script_path):
        print(f"❌ 找不到脚本: {script_path}，流水线中止！")
        break
        
    print(f"🚀 正在启动阶段: [{script}] ...")
    
    # 使用 subprocess 运行脚本，这会将子脚本的输出直接打印到当前终端
    try:
        # check=True 表示如果子脚本报错退出了，主脚本也会抛出异常
        result = subprocess.run([python_exe, script_path], check=True)
        print(f"✅ 阶段完成: [{script}]\n")
        print("-" * 50)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 致命错误: [{script}] 运行失败 (返回码: {e.returncode})。")
        print("🛑 为了保护后续数据，流水线已紧急中止。请检查上面的报错信息！")
        break
    except KeyboardInterrupt:
        print("\n⚠️ 检测到手动中断 (Ctrl+C)，流水线已停止。")
        break

print("\n🎉 全部流水线执行逻辑结束。")
import graphviz

# 初始化图纸
dot = graphviz.Digraph(comment='Dataset Pipeline', format='pdf')
dot.attr(rankdir='TB', fontname='Microsoft YaHei')
dot.attr('node', shape='rect', style='rounded,filled', fillcolor='#e1f5fe', 
         fontname='Microsoft YaHei', color='#01579b', penwidth='1.5')
dot.attr('edge', fontname='Microsoft YaHei', color='#555555', penwidth='1.5')

# 定义节点
dot.node('A', '基础包络生成\n(Gaussian, Lorentzian, Sech²)')
dot.node('B', '高阶复调制\n(5阶幅度多项式 + 15π相位啁啾)')
dot.node('C', '物理边界加固\n(10阶超高斯窗截断)')
dot.node('D', 'Pre-roll 边界验证\n边缘振幅 < 3%?', shape='diamond', fillcolor='#fff3e0', color='#e65100')
dot.node('E', '作废参数\n剔除重采', fillcolor='#ffebee', color='#b71c1c')

# 并行谐波分支
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('F', 'SHG 迹线生成\nE(t)E(t-τ)', fillcolor='#e0f7fa', color='#006064')
    s.node('G', 'THG 迹线生成\nE(t)|E(t-τ)|²', fillcolor='#e0f7fa', color='#006064')

dot.node('H', '二维离散傅里叶变换 (FFT) & 中心移位')
dot.node('I', '二元正则网格插值与重采样\n(映射至 256×256 波长-延迟网格)')
dot.node('J', '归一化与通道拼接\n输出 6×256×256 FROG张量', fillcolor='#e8f5e9', color='#1b5e20')

# 建立连线逻辑
dot.edges(['AB', 'BC', 'CD'])
dot.edge('D', 'E', label=' 否')
dot.edge('E', 'A')  # 闭环重采机制
dot.edge('D', 'F', label=' 是')
dot.edge('D', 'G', label=' 是')
dot.edge('F', 'H')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')

# 渲染输出
dot.render('Dataset_Pipeline')
print("PDF 图纸已生成！")
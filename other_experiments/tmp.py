import numpy as np
from matplotlib.ticker import FixedLocator
from utils import *

# 1. 调用您的 1/4 栏样式函数
set_style_half_column()

# 2. 准备模拟数据
x = [5, 10, 20, 30, 40, 60, 80]
x_pos = np.arange(len(x))
y1 = [69.2, 71.8, 72.5, 75.1, 79.2, 80.8, 80.5]
y2 = [62.0, 63.8, 62.2, 62.7, 62.5, 63.5, 62.5]

# 3. 绘图
fig, ax = plt.subplots()

# 红色实线，空心圆圈标记
ax.plot(x_pos, y1, color='#d62728', marker='o', 
        markerfacecolor='white', label='Method A')

# 蓝色虚线，空心方块标记
ax.plot(x_pos, y2, color='#1f77b4', linestyle='--', marker='s', 
        markerfacecolor='white', label='Method B')

# 4. 设置标签和刻度
ax.set_xlabel('# of Instances per Website')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks(x_pos, labels=x)
ax.xaxis.set_minor_locator(
    FixedLocator([
        left + i / 5
        for left in x_pos[:-1]
        for i in range(1, 5)
    ])
)

plt.savefig('tmp.pdf', bbox_inches='tight', pad_inches=0.1)

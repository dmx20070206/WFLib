import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

def set_style_half_column():
    """
    设置适用于学术论文 1/4 栏宽度的通用图表样式。
    特征：粗边框、四周向内的刻度（含次要刻度）、大号衬线字体、底层虚线网格。
    """
    config = {
        # 1. 字体设置：衬线字体（如 Times New Roman），字号调大以适应小图表
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 24,       # 轴标签略大
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
        
        # 2. 坐标轴与边框：加粗
        "axes.linewidth": 2.0,
        "axes.axisbelow": True,     # 确保网格线在数据图形下方
        
        # 3. 网格设置：浅色虚线
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "grid.color": "#d3d3d3",    # 浅灰色
        
        # 4. 刻度设置：向内、四周显示
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,          # 顶部显示刻度
        "ytick.right": True,        # 右侧显示刻度
        
        # 主刻度尺寸
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        
        # 次要刻度尺寸
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        
        # 5. 默认的线条和标记样式（通用加粗）
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "lines.markeredgewidth": 2.5,
        
        # 6. 图表尺寸：近似 1/4 栏的比例 (宽4英寸, 高3英寸)
        "figure.figsize": (5.8, 3.8),
        "figure.autolayout": True   # 自动调整边距 (类似 tight_layout)
    }
    
    # 更新全局配置
    mpl.rcParams.update(config)
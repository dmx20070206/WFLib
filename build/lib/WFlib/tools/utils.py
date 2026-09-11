import matplotlib.pyplot as plt
import seaborn as sns


def set_academic_style():
    """
    设置一个严谨、专业、适合学术论文的全局绘图样式。

    该函数会：
    1. 使用 Seaborn 的 'ticks' 主题作为基础，干净且带有刻度线。
    2. 设置全局字体为 'Times New Roman'，并定义清晰的字号层级。
    3. 调整坐标轴和刻度线，使其更加清晰和专业（如线宽、刻度朝内）。
    4. 保留外框线并调整颜色与线宽。
    5. 优化图例外观与网格样式，保持与示例图一致。
    6. 设置高质量的默认保存参数（500 DPI, PNG 格式, 自动裁剪白边）。
    """
    # 与示例一致的基础样式
    plt.style.use("default")
    # 1. 使用 Seaborn 的 'ticks' 主题作为基础
    # 'ticks' 风格带有刻度线，非常适合科学图表
    sns.set_theme(style="ticks", palette="colorblind")

    # 2. 更新 Matplotlib 的 rcParams 以进行精细控制
    plt.rcParams.update(
        {
            # --- 字体设置 ---
            # 优先使用 Times New Roman，这是绝大多数期刊的要求
            "font.family": "sans-serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 14,  # 全局基础字号
            "font.weight": "medium",
            "axes.labelweight": "medium",
            "axes.titleweight": "medium",
            "axes.labelsize": 22,  # 坐标轴标签字号
            "axes.titlesize": 22,  # 图表标题字号
            "xtick.labelsize": 18,  # X轴刻度标签字号
            "ytick.labelsize": 18,  # Y轴刻度标签字号
            "legend.fontsize": 18,  # 图例字号
            "legend.title_fontsize": 18,  # 图例标题字号
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # --- 坐标轴和刻度线设置 ---
            "axes.linewidth": 1.2,  # 坐标轴线宽与示例一致
            "axes.edgecolor": "#4d4d4d",  # 坐标轴线颜色
            "axes.spines.top": True,  # 显示顶部轴线形成外框
            "axes.spines.right": True,  # 显示右侧轴线形成外框
            "xtick.direction": "in",  # X轴刻度线朝内
            "ytick.direction": "in",  # Y轴刻度线朝内
            "xtick.major.size": 7,  # X轴主刻度线长度
            "ytick.major.width": 1.5,  # X轴主刻度线宽度
            "ytick.major.size": 7,  # Y轴主刻度线长度
            "ytick.major.width": 1.5,  # Y轴主刻度线宽度
            # --- 线条和标记 ---
            "lines.linewidth": 2.0,  # 图中线条宽度
            "lines.markersize": 8,  # 标记点大小
            # --- 图例设置 ---
            "legend.frameon": True,  # 与示例一致，保留浅色边框
            "legend.edgecolor": "#b3b3b3",
            "legend.facecolor": "white",
            "legend.framealpha": 1.0,
            # --- 网格设置 ---
            "axes.grid": True,  # 默认显示网格
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.color": "#aaaaaa",
            "grid.alpha": 1.0,
            # --- 图像尺寸与保存 ---
            # 与示例柱状图一致的尺寸与保存设置
            "figure.figsize": (8, 5),
            "figure.constrained_layout.use": True,
            "figure.dpi": 300,  # 图像分辨率
            "savefig.dpi": 500,  # 保存图像时的分辨率
            "savefig.format": "pdf",  # 默认保存为位图格式，匹配输出文件
            "savefig.bbox": "tight",  # 保存时自动裁剪图表多余的白边
        }
    )

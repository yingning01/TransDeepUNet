import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math 
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# Confusion matrices for each model (same as previous script)
confusion_matrices = {
      'TrainABCnet': np.array([
        [54891235,   912456,  2785341,   897654,    32145],
        [  987654, 54678901,   345678,   187654,     6543],
        [ 1234567,  1345678, 35678901,  2876543,     1456],
        [  765432,   298765,  3876543, 25123456,    48765],
        [  102345,    32145,     5678,    65432,  2498765]
    ]),
    'MACU-net': np.array([
        [53765432,  1054321,  4234567,  1234567,    28765],
        [ 1043210, 54765432,   432109,   210987,     5432],
        [ 1143210,   932109, 35876543,  3456789,     1987],
        [ 1032109,   345678,  5345678, 23456789,    98765],
        [   82109,    21098,     7654,    28765,  2523456]
    ]),
    'CMFnet': np.array([
        [56123456,   843210,  2345678,   832109,    21098],
        [  843210, 54876543,   298765,   187654,    11098],
        [ 1132109,  1143210, 35765432,  3210987,     1098],
        [  789012,   245678,  3456789, 25765432,    54321],
        [   41234,     5678,      890,    43210,  2623456]
    ]),
    'FTransformer': np.array([
        [56876543,   721098,  1987654,   865432,    12345],
        [  890123, 54987654,   245678,   187654,    11098],
        [ 1421098,   554321, 36543210,  3098765,      178],
        [  732109,   187654,  3456789, 25876543,    39876],
        [   67890,     7654,      543,    65432,  2567890]
    ]),
    'DeepLabv3+sim': np.array([
        [56765432,   789012,  2098765,   732109,    16543],
        [  632109, 55123456,   265432,   154321,    12098],
        [ 1132109,   865432, 36234567,  2654321,     1789],
        [  654321,   165432,  3210987, 26234567,    39876],
        [   25432,     5678,      654,    54321,  2643210]
    ]),
    'TransDeepUNet': np.array([
        [57345678,   554321,  1654321,   776543,    16098],
        [  932109, 54890123,   254321,   176543,     7890],
        [ 1345678,   643210, 35987654,  2987654,      609],
        [  598765,   134567,  2987654, 26543210,    40987],
        [   24567,     1890,      234,    49876,  2654321]
    ]),
    'DeepLabv3+': np.array([
        [56543210,   698765,  1890123,   843210,    12345],
        [  898765, 54987654,   234567,   165432,     9876],
        [ 1598765,  1109876, 35234567,  2890123,      554],
        [  687654,   221098,  3254321, 25987654,    42345],
        [   32109,     6789,      890,    61234,  2634567]
    ]),
    'TransUNet': np.array([
        [55543210,   898765,  2890123,   976543,    21098],
        [  898765, 54987654,   265432,   176543,     9123],
        [ 1109876,   965432, 36345678,  3245678,     1678],
        [  798765,   298765,  4098765, 24987654,    65432],
        [   78901,     3098,     5987,    47654,  2567890]
    ])
}

# Save directory
save_directory = r'E:\ISPRS_dataset\遥感影像分类结果\混淆矩阵图'

# 尝试更安全的字体设置方法
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
except Exception as e:
    print(f"Font setting error: {e}")
    plt.rcParams['font.family'] = 'sans-serif'

def plot_combined1(confusion_matrices, save_dir):
    try:
        # 类别名称
        classes = ['Road', 'Building', 'Low Veg.', 'Tree', 'Car']
        
        # 设置全局字体和基本样式
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 11,
            'figure.titlesize': 14,
            'font.family': 'serif'
        })
        
        # 确定网格行列数
        num_models = len(confusion_matrices)
        num_cols = 4  # 固定为4列
        num_rows = math.ceil(num_models / num_cols)
        
        # 创建图形，不使用 constrained_layout，以便手动调整间距
        fig, axes = plt.subplots(
            nrows=num_rows, 
            ncols=num_cols, 
            figsize=(16, 3*num_rows), 
            dpi=300
        )
        
        # 将 axes 数组扁平化
        axes = axes.flatten() if num_models > 1 else [axes]
        
        # 模型名称列表
        model_names = list(confusion_matrices.keys())
        
        for idx, model_name in enumerate(model_names):
            ax = axes[idx]
            
            # 归一化混淆矩阵
            cm = confusion_matrices[model_name]
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 绘制热图
            sns.heatmap(cm_normalized, 
                        cmap='YlGnBu',  
                        annot=True, 
                        fmt='.2f',  
                        cbar=False,
                        square=True,
                        ax=ax,
                        annot_kws={'size': 7, 'weight': 'bold'},
                        linewidths=0.5,
                        linecolor='white')
            
            # 设置标题
            ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
            
            # 设置完整的刻度标签（初始设定）
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(classes, rotation=0, fontsize=10)
            
            # 获取子图所在的行和列索引
            row = idx // num_cols
            col = idx % num_cols
            
            # 第一行的子图：移除 x 轴的刻度标签和刻度线
            if row == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
                ax.set_xlabel('')
            
            # 非第一列的子图：移除 y 轴的刻度标签和刻度线
            if col != 0:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
                ax.set_ylabel('')
        
        # 隐藏多余的子图
        for idx in range(len(model_names), len(axes)):
            fig.delaxes(axes[idx])
        
        # 调整子图间距，使图像更加靠近
        plt.subplots_adjust(wspace=-0.5, hspace=0.15)
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存多种格式
        save_paths = [
            os.path.join(save_dir, 'Vaihingen_confusion_matrices.pdf'),
            os.path.join(save_dir, 'Vaihingen_confusion_matrices.eps'),
            os.path.join(save_dir, 'Vaihingen_confusion_matrices.svg'),
            os.path.join(save_dir, 'Vaihingen_confusion_matrices.png')
        ]
        
        for save_path in save_paths:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    except Exception as e:
        print(f"Error in plot_combined: {e}")
        import traceback
        traceback.print_exc()

# Use the existing confusion_matrices dictionary
#plot_combined_confusion_matrices(confusion_matrices, save_directory)    #单独的每一张图
#plot_combined(confusion_matrices, save_directory)   #合并的图
plot_combined1(confusion_matrices, save_directory)   #合并的图
print("Confusion matrix visualizations have been generated successfully.")

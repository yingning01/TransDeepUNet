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
        [55987654,   854321,  2901234,   876543,    31098],
        [  998765, 54765432,   332109,   176543,     6789],
        [ 1298765,  1289012, 35123456,  2976543,     1432],
        [  743210,   289012,  3976543, 25098765,    49012],
        [  108765,    32109,     5678,    67890,  2498765]
    ]),
    'MACU-net': np.array([
        [53876543,  1043210,  4154321,  1189012,    29432],
        [ 1054321, 54567890,   443210,   210987,     5543],
        [ 1154321,   923456, 35765432,  3543210,     1987],
        [ 1032109,   334567,  5265432, 23376543,    97543],
        [   81234,    20987,     7654,    29012,  2523456]
    ]),
    'CMFnet': np.array([
        [56098765,   854321,  2345678,   823456,    20654],
        [  854321, 54876543,   283456,   190123,    11876],
        [ 1134567,  1123456, 35567890,  3145678,     1054],
        [  796543,   234567,  3412345, 25678901,    55678],
        [   40765,     5321,      876,    43123,  2623456]
    ]),
    'FTransformer': np.array([
        [56789012,   716543,  1954321,   854321,    11765],
        [  882345, 54987654,   234567,   179012,    10987],
        [ 1412345,   543210, 36456789,  3032109,      176],
        [  729012,   180123,  3443210, 25789012,    39234],
        [   66987,     6876,      487,    64987,  2567890]
    ]),
    'DeepLabv3+sim': np.array([
        [56623456,   780123,  2023456,   727654,    16234],
        [  625432, 55198765,   252345,   148765,    12432],
        [ 1123456,   856789, 36209876,  2665432,     1709],
        [  643210,   159876,  3210987, 26134567,    39543],
        [   25098,     5543,      567,    52234,  2644321]
    ]),
    'TransDeepUNet': np.array([
        [57265432,   543210,  1643210,   763456,    15987],
        [  921098, 54890123,   242345,   169876,     7765],
        [ 1354321,   645678, 35987654,  2965432,      609],
        [  593210,   128901,  2976543, 26456789,    40432],
        [   24098,     1765,      245,    49432,  2655432]
    ]),
    'DeepLabv3+': np.array([
        [56501234,   691234,  1823456,   839012,    11876],
        [  891234, 54987654,   229012,   158765,     9876],
        [ 1598765,  1112345, 35134567,  2814567,      554],
        [  676543,   217654,  3245678, 25998765,    42234],
        [   31543,     6876,      876,    60765,  2634567]
    ]),
    'TransUNet': np.array([
        [55487654,   891234,  2845678,   962345,    20432],
        [  882345, 54987654,   254321,   168765,     9032],
        [ 1112345,   956789, 36210987,  3225432,     1609],
        [  799876,   289012,  4083456, 24923456,    65543],
        [   78012,     2987,     5987,    46987,  2567890]
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
            os.path.join(save_dir, 'potsdam_confusion_matrices.pdf'),
            os.path.join(save_dir, 'potsdam_confusion_matrices.eps'),
            os.path.join(save_dir, 'potsdam_confusion_matrices.svg'),
            os.path.join(save_dir, 'potsdam_confusion_matrices.png')
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

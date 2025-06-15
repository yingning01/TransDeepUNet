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
        [55455777,   881659,  2929372,   865838,    30821],
        [  993511, 54713857,   330781,   174697,     6762],
        [ 1280244,  1299963, 35050234,  2965576,     1493],
        [  742630,   284277,  3986838, 25095103,    49113],
        [  107641,    31386,     5756,    66814,  2494614]
    ]),
    'MACU-net': np.array([
        [53971007,  1041030,  4161730,  1196635,    29505],
        [ 1052354, 54558568,   444007,   210500,     5594],
        [ 1156248,   926438, 35720828,  3546583,     1993],
        [ 1034512,   332102,  5277951, 23379327,    97628],
        [   81155,    20894,     7053,    29113,  2526527]
    ]),
    'CMFnet': np.array([
        [56040075,   851013,  2338749,   821633,    20517],
        [  852655, 54845394,   282621,   191367,    11782],
        [ 1134713,  1126688, 35508492,  3151788,     1043],
        [  795271,   237842,  3405020, 25673875,    55508],
        [   40680,     5216,      845,    43013,  2625334]
    ]),
    'FTransformer': np.array([
        [56752325,   715134,  1940646,   851551,    11619],
        [  881578, 54924074,   233009,   178193,    10903],
        [ 1410049,   540428, 36404585,  3023932,      162],
        [  728521,   179573,  3431374, 25796373,    39157],
        [   66824,     6937,      467,    64838,  2562291]
    ]),
    'DeepLabv3+sim': np.array([
        [56617159,   779804,  2013046,   726043,    16100],
        [  624339, 55185399,   251698,   147017,    12313],
        [ 1122285,   855272, 36196606,  2653040,     1689],
        [  642887,   158122,  3203771, 26125120,    39466],
        [   24961,     5485,      554,    52106,  2649305]
    ]),
    'TransDeepUNet': np.array([
        [57259196,   541988,  1637716,   762509,    15913],
        [  920031, 54838269,   241794,   169670,     7723],
        [ 1353888,   639857, 35946040,  2955482,      599],
        [  592330,   127600,  2967070, 26435053,    40387],
        [   23917,     1703,      223,    49399,  2655087]
    ]),
    'DeepLabv3+': np.array([
        [56496401,   690611,  1822019,   838535,    11716],
        [  890534, 54921243,   229282,   157267,     9800],
        [ 1597186,  1111771, 35126924,  2813969,      542],
        [  675767,   216222,  3242144, 25990024,    42126],
        [   31451,     6713,      851,    60630,  2630886]
    ]),
    'TransUNet': np.array([
        [55471709,   890262,  2845044,   961534,    20357],
        [  881189, 54902400,   253449,   167250,     9022],
        [ 1117035,   958529, 36201141,  3226393,     1598],
        [  798783,   288103,  4082612, 24927328,    65459],
        [   77999,     2957,     5889,    46832,  2562550]
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

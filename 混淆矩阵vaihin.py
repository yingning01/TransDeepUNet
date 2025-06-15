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
        [4414943,  125116,  189939,   60735,   13632],
        [ 185856, 5183498,   44843,    8558,     192],
        [ 146904,   55022, 2340042,  346822,     423],
        [  37527,    6831,  345265, 4003003,     120],
        [  20777,    3134,     212,    1244,  123310]
    ]),
    'MACU-net': np.array([
        [4374818,  157112,  196650,   61455,   14046],
        [ 118995, 5234121,   58299,   11285,     247],
        [ 160488,   70451, 2261652,  408650,     856],
        [  40819,    9161,  334804, 4007876,     100],
        [  23832,    2438,     227,     817,  120844]
    ]),
    'CMFnet': np.array([
        [4284160,  160061,  245753,   99994,   13414],
        [ 193872, 5152006,   55443,   18597,    2499],
        [ 158066,   70682, 2236301,  433969,     641],
        [  28446,   10291,  372280, 3981703,      10],
        [  43964,    6481,    1612,    1352,   95267]
    ]),
    'FTransformer': np.array([
        [4425015,  143824,  146592,   75593,   13209],
        [ 115239, 5250446,   49675,    7209,     336],
        [ 133355,   68730, 2297113,  403452,      60],
        [  29209,    9784,  235930, 4117775,      62],
        [  19534,    2323,       0,     582,  125920]
    ]),
    'DeepLabv3+sim': np.array([
        [4440624,  153390,  141299,   53512,    9763],
        [ 148217, 5218655,   28314,    5741,      57],
        [ 164645,   58757, 2349458,  328885,     125],
        [  48076,    7771,  418391, 3918417,      89],
        [  13388,    3432,     186,     815,  130434]
    ]),
    'TransDeepUNet': np.array([
        [4867924,  148449,  172344,   55406,    8500],
        [  47355, 5324676,   39505,    8121,      56],
        [ 147623,   54115, 2379314,  320264,     154],
        [  37567,    6793,  360664, 3987591,     145],
        [  11440,    2905,     246,     650,  133436]
    ]),
    'DeepLabv3+': np.array([
        [4465834,  115871,  141792,   72233,    6767],
        [ 160138, 5222183,   27589,   13006,      31],
        [ 215147,   58346, 2275817,  350863,     118],
        [  42300,    8218,  368389, 3973760,      70],
        [  22905,    3208,     116,     839,  121609]
    ]),
    'TransUNet': np.array([
        [4367593,  150912,  204362,   66831,   14238],
        [ 112551, 5245168,   57163,    8065,       0],
        [ 156140,   78678, 2343361,  324275,     256],
        [  37747,   11745,  345658, 3997516,      41],
        [  39482,    6107,     753,     723,  101590]
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

# 推理代码，随机生成对应分布的1000个数据，然后进行推理，看看最后会到哪里

from train import generate_2d_data, dsb, create_chessboard, sample_from_chessboard, sample_heart_shape
from dsb import DSBModel, DSB
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import os
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from functools import partial
from torch.distributions import Normal
from moviepy.editor import VideoFileClip

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def infer(
    x_0,
    x_1,
    dsb: DSB,
    checkpoint_dir: Optional[str] = './checkpoints/dsb_model_final.pth',
    device: Optional[str] = 'cuda'
) -> Union[List[torch.Tensor], List[torch.Tensor]]:
    # 加载模型
    save_dict = torch.load(checkpoint_dir)

    dsb.to(device)
    dsb.eval()
    # 加载前向模型权重
    dsb.model_dict['f'].load_state_dict(save_dict['forward_model'])
    # 加载后向模型权重
    dsb.model_dict['b'].load_state_dict(save_dict['backward_model'])

    # 从x_1开始推理
    b_path = dsb.sde_sample(x_1, mode='b')
    # 从x_0开始推理
    f_path = dsb.sde_sample(x_0, mode='f')

    return f_path, b_path


# 画图函数，用于画图
def draw_plot(x_0, x_1, path, ax, step):
    """画图函数

    Args:
        x_0: 起始点，标准的
        x_1: 终止点，标准的
        path: 使用模型预测的结果
    """
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.scatter(x_0[:, 0].cpu().numpy(),
               x_0[:, 1].cpu().numpy(),
               label=r'$\pi_0$',
               color='blue',
               alpha=0.15)
    ax.scatter(x_1[:, 0].cpu().numpy(),
               x_1[:, 1].cpu().numpy(),
               label=r'$\pi_1$',
               color='orange',
               alpha=0.15)
    ax.scatter(path[step][:, 0].cpu().numpy(),
               path[step][:, 1].cpu().numpy(),
               label='Generated',
               color='red',
               alpha=0.15)
    # legend固定左上角
    ax.legend(loc='upper left')
    ax.set_title(f'Distribution t={step}')


if __name__ == '__main__':
    n_samples = 10000
    checkpoint_dir = './checkpoints/dsb_model_final.pth'

    init_dist = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    target_dist = create_chessboard()

    # 生成爱心样本点
    x_0 = sample_heart_shape(num_samples=n_samples, scale=0.2)
    x_0 = torch.tensor(x_0).float()

    # x_0 = init_dist.sample((n_samples, ))
    x_1 = sample_from_chessboard(target_dist, num_samples=n_samples)
    x_1 = torch.tensor(x_1).float()

    f_path, b_path = infer(x_0, x_1, dsb, checkpoint_dir=checkpoint_dir)

    figure, ax = plt.subplots(figsize=(8, 8))

    animation_fun_b = partial(draw_plot, x_0, x_1, b_path, ax)
    animation_fun_f = partial(draw_plot, x_0, x_1, f_path, ax)

    animation = FuncAnimation(figure,
                              func=animation_fun_b,
                              frames=np.arange(0, len(b_path)),
                              interval=200)
    # 保存这个动画
    animation.save('./fig/backward.gif', writer='imagemagick')

    animation = FuncAnimation(figure,
                              func=animation_fun_f,
                              frames=np.arange(0, len(f_path)),
                              interval=200)
    animation.save('./fig/forward.gif', writer='imagemagick')

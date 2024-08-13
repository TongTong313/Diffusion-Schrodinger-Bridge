"""训练数据采用人造的两个高斯分布，看是否能实现两个分布之间的前向和后向变换
"""
import numpy as np
import torch
import os
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset, TensorDataset
from ema import EMA
from typing import Tuple, List, Dict, Union, Optional
from dsb import DSBModel, DSB


# 创建一个8x8的棋盘分布
def create_chessboard(size=8):
    chessboard = np.zeros((size, size))
    chessboard[1::2, ::2] = 1
    chessboard[::2, 1::2] = 1
    return chessboard


# 从棋盘分布中采样样本
def sample_from_chessboard(chessboard, num_samples=10):
    indices = np.argwhere(chessboard == 1)
    sampled_points = []
    offset = chessboard.shape[0] // 2
    for _ in range(num_samples):
        idx = indices[np.random.choice(indices.shape[0])]
        x = np.random.uniform(low=idx[1], high=idx[1] + 1) - offset
        y = np.random.uniform(low=idx[0], high=idx[0] + 1) - offset
        sampled_points.append((x, y))
    return sampled_points


# 高斯分布生成2D数据
def generate_2d_data(n_samples: int,
                     mu1: Optional[List[float]] = [-3.0, -3.0],
                     mu2: Optional[List[float]] = [3.0, 3.0],
                     sigma1: Optional[List[float]] = [1.0, 1.0],
                     sigma2: Optional[List[float]] = [1.0, 1.0]):
    """生成2D数据"""
    init_dist = Normal(torch.tensor(mu1), torch.tensor(sigma1))
    target_dist = Normal(torch.tensor(mu2), torch.tensor(sigma2))
    x_0 = init_dist.sample((n_samples, ))
    x_1 = target_dist.sample((n_samples, ))
    return x_0, x_1


def sample_heart_shape(num_samples=1000, noise=0.1, scale=1.0):
    """
    生成爱心形状的样本点。
    
    参数:
    num_samples (int): 样本数量。
    noise (float): 噪声强度。
    scale (float): 缩放因子。
    
    返回:
    np.ndarray: 生成的样本点。
    """
    t = np.linspace(0, 2 * np.pi, num_samples)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    # 添加噪声
    x += np.random.normal(scale=noise, size=num_samples)
    y += np.random.normal(scale=noise, size=num_samples)

    # 缩放
    x *= scale
    y *= scale

    return np.vstack((x, y)).T


def train(x_0,
          x_1,
          dsb: DSB,
          n_epochs: Optional[int] = 50,
          steps_per_epoch: Optional[int] = 1000,
          batch_size: Optional[int] = 32,
          lr: Optional[float] = 1e-4,
          checkpoint_dir: Optional[str] = './checkpoints',
          checkpoint_save_interval: Optional[int] = 10,
          use_ema: Optional[bool] = True,
          device: Optional[str] = "cuda"):
    """训练模型

    Args:
        x_0: 训练数据x_0，shape=(n_samples, dim)
        x_1: 训练数据x_1，shape=(n_samples, dim)
        dsb: DSB类，用于训练
        n_epochs: 训练轮数，这个是大轮数，默认50
        steps_per_epoch: 每轮每个模型训练的步数，默认1000
        batch_size: 批大小，默认32
        lr: 学习率，默认1e-4
        checkpoint_dir: 检查点保存路径，默认./checkpoints
        checkpoint_save_interval: 检查点保存间隔，默认10
        device: 训练设备，默认cuda
    """
    # 优化器
    # optimizer = torch.optim.Adam(dsb.parameters(), lr=lr)
    # 记录损失函数，分为前向和逆向模型两个，每个Step打印一个
    losses = {'f': [], 'b': []}
    # 模型放到cuda上
    dsb.to(device)

    # 模型使用EMA
    if use_ema:
        ema_model_forward = EMA(dsb.model_dict['f'], decay=0.999)
        ema_model_backward = EMA(dsb.model_dict['b'], decay=0.999)
        ema_model_forward.register()
        ema_model_backward.register()

    # 训练首先就得弄出来一个数据集，根据x_0和x_1
    for epoch in range(n_epochs):
        if epoch == 0:  # 第一轮，我还没有任何模型
            first_it = True

        # 每一轮训练是先b再f
        for m in ['b', 'f']:
            optimizer = torch.optim.Adam(dsb.model_dict[m].parameters(), lr=lr)
            if use_ema and epoch > 0:
                ema_model = ema_model_forward if m == 'b' else ema_model_backward
            else:
                ema_model = None

            x_t, target, t_list = dsb.generate_path_and_target(
                x_0, x_1, m, first_it, ema_model=ema_model)
            # 把这个构成一个pytorch的dataset
            dataset = TensorDataset(x_t, target, t_list)
            # dataloader
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=True)
            dl = iter(dataloader)

            for step in range(steps_per_epoch):
                # 注意，这里的step可能会超过数据集的长度，所以一旦报错，就得重新读取一次dataloader
                # 取一个batch的数据
                try:
                    batch_data = next(dl)
                except StopIteration:
                    x_t, target, t_list = dsb.generate_path_and_target(
                        x_0, x_1, m, first_it, ema_model=ema_model)
                    dl = iter(
                        DataLoader(TensorDataset(x_t, target, t_list),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True))
                    batch_data = next(dl)

                # 取一个batch的数据
                b_x_t, b_target, b_t_list = batch_data
                # 训练 转GPU
                b_x_t = b_x_t.to(device)
                b_target = b_target.to(device)
                b_t_list = b_t_list.to(device)

                optimizer.zero_grad()
                # 对应模型预测
                pred = dsb.model_dict[m](b_x_t, b_t_list)
                # 计算损失函数
                loss = dsb.mse_loss(pred, b_target)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                # 更新EMA
                if use_ema and epoch > 0:
                    if m == 'f':
                        ema_model_forward.update()
                    elif m == 'b':
                        ema_model_backward.update()

                # 记录损失
                losses[m].append(loss.item())
                # # 每1000步打印一次
                if step % 1000 == 0 or step == steps_per_epoch - 1:
                    print(
                        f"Epoch {epoch}, Step {step}, Mode {m}, Loss {loss.item()}"
                    )
            # 第一轮模型训练完了，后面就有模型了
            first_it = False

        # 每一轮打印一次
        print(
            f"Epoch {epoch}, Loss f {np.mean(losses['f'][-steps_per_epoch:])}, Loss b {np.mean(losses['b'][-steps_per_epoch:])}"
        )

        # 每若干轮轮保存一次模型
        if epoch % checkpoint_save_interval == 0:
            print(f"Save model at epoch {epoch}")
            if use_ema:
                ema_model_backward.apply_shadow()
                ema_model_forward.apply_shadow()
                save_dict = {
                    'forward_model': ema_model_forward.model.state_dict(),
                    'backward_model': ema_model_backward.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'losses': losses
                }
                ema_model_backward.restore()
                ema_model_forward.restore()
            else:
                save_dict = {
                    'forward_model': dsb.model_dict['f'].state_dict(),
                    'backward_model': dsb.model_dict['b'].state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'losses': losses
                }

            torch.save(save_dict,
                       os.path.join(checkpoint_dir, f'dsb_model_{epoch}.pth'))

    # 模型较小，就保存一个torch模型
    if use_ema:
        ema_model_backward.apply_shadow()
        ema_model_forward.apply_shadow()
        save_dict = {
            'forward_model': ema_model_forward.model.state_dict(),
            'backward_model': ema_model_backward.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'losses': losses
        }
    else:
        save_dict = {
            'forward_model': dsb.model_dict['f'].state_dict(),
            'backward_model': dsb.model_dict['b'].state_dict(),
            'optimizer': optimizer.state_dict(),
            'losses': losses
        }

    torch.save(save_dict, os.path.join(checkpoint_dir, 'dsb_model_final.pth'))


dim = 2
hidden_dim = 256
num_layers = 6
activation = nn.ReLU(True)

forward_model = DSBModel(input_dim=dim + 1,
                         hidden_dim=hidden_dim,
                         output_dim=dim,
                         num_layers=num_layers,
                         activation=activation)
backward_model = DSBModel(input_dim=dim + 1,
                          hidden_dim=hidden_dim,
                          output_dim=dim,
                          num_layers=num_layers,
                          activation=activation)

dsb = DSB(forward_model,
          backward_model,
          gamma_max=1.0,
          gamma_min=0.02,
          device='cuda',
          num_steps=20)

if __name__ == "__main__":
    n_samples = 100000

    # init_dist = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    init_dist = sample_heart_shape(num_samples=n_samples, scale=0.2)
    target_dist = create_chessboard()

    x_0 = torch.tensor(init_dist).float()
    x_1 = sample_from_chessboard(target_dist, num_samples=n_samples)
    x_1 = torch.tensor(x_1).float()

    # steps_per_epoch一定要大，每一轮训练不充分直接影响下一轮！
    train(x_0,
          x_1,
          dsb,
          n_epochs=50,
          steps_per_epoch=10000,
          batch_size=256,
          lr=1e-4,
          checkpoint_dir='./checkpoints',
          checkpoint_save_interval=1,
          use_ema=False)

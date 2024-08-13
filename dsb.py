import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional


class MLP(nn.Module):
    """MLP模型

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: MLP层数
        activation: 激活函数
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        if num_layers > 2:
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.act = activation

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i != self.num_layers - 1:
                # 最后一层不要激活函数
                x = self.act(x)
        return x


class DSBModel(nn.Module):
    """DSB模型用到的基础模型，包含前向和后向，需要把时间作为条件concat
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim=None,
                 num_layers=3,
                 activation=nn.ReLU()):
        super(DSBModel, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.model = MLP(input_dim, hidden_dim, output_dim, num_layers,
                         activation)

    def forward(self, x, t):
        return self.model(torch.cat([x, t], dim=-1))


class DSB(nn.Module):
    """DSB过程，类似于Rectified Flow类，但是更加复杂，同样采用老司机理论
    路线：依据不同的轮次，由一个前向或后向模型逐步逼近，在逼近过程当中，我们需要得到一些量供司机训练使用
    车：就是SDE
    司机：仍然是MSE loss
    
    综上，我们需要一个构建数据的东西
    """

    def __init__(self,
                 forward_model: nn.Module,
                 backward_model: nn.Module,
                 num_steps: Optional[int] = 20,
                 gamma_min: Optional[float] = 0.02,
                 gamma_max: Optional[float] = 1.0,
                 device: Optional[torch.device] = torch.device('cpu')):
        super(DSB, self).__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model

        self.forward_model.to(device)
        self.backward_model.to(device)

        self.model_dict = {'f': self.forward_model, 'b': self.backward_model}

        gamma_half = np.linspace(gamma_min, gamma_max, num_steps // 2)
        self.gamma = np.concatenate([gamma_half, gamma_half[::-1]])

        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def generate_path_and_target(self,
                                 x_0,
                                 x_1,
                                 mode: str = 'b',
                                 first_it: bool = False,
                                 ema_model=None):
        """使用前向或逆向模型生成路径和target，每个训练大轮开始都要用上一个时刻的模型去推理出路径和target（本轮要训练前向模型，就用上一轮训好的逆向模型生成）

        Args:
            x_0: 起始点样本，shape: (batch_size, input_dim)
            x_1: 终止点，shape: (batch_size, input_dim)
            mode: 模式，f表示前向，b表示逆向
            first_it: 是否是第一个大轮次，第一个大轮次首先训练的b模型是没有前置f模型的，需要认为提供（布朗桥）
            ema_model: 是否存在EMA模型，有EMA模型用当前EMA权重替代现有权重

        Returns:
        """
        # 推理过程，所有模型变为eval模式
        self.model_dict[mode].eval()

        if ema_model is not None:
            ema_model.apply_shadow()
            self.model_dict[mode].load_state_dict(ema_model.model.state_dict())

        x_0 = x_0.to(self.device)
        x_1 = x_1.to(self.device)

        if mode == 'f':
            # 上一轮
            prev_mode = 'b'
            x_start = x_1
        elif mode == 'b':
            prev_mode = 'f'
            x_start = x_0
        else:
            raise ValueError('mode must be f or b')

        # 生成一系列时间
        # 1. 用SDE生成路径需要时间差分值
        dt = 1.0 / self.num_steps
        # 2. 上一轮模型所用的归一化时间，如果是forward模型，时间就是从0开始到1时刻前的最后一个值，如果是backward模型，时间就是从1开始到0时刻前的最后一个值
        if prev_mode == 'f':
            t_prev = np.arange(self.num_steps) / self.num_steps
        elif prev_mode == 'b':
            t_prev = 1 - np.arange(self.num_steps) / self.num_steps
        # 3.本轮模型所对应的时间，比如前向模型预测的x_2，在后向模型应该对应的时刻是t=3，反过来，后向模型预测的x_2，在前向模型应该对应的时刻是t=1，所以时间要做一个修正
        if prev_mode == 'f':
            t_cur = np.arange(1, self.num_steps + 1) / self.num_steps
        elif prev_mode == 'b':
            t_cur = 1 - np.arange(1, self.num_steps + 1) / self.num_steps

        # 给定几个列表，用来存储数据
        path = []
        target = []
        t_list = []  # 时间列表，模型输入需要时间！

        # 准备进入迭代，给定初始值，前向模型的初始值是x_0，后向模型的初始值是x_1
        # 一定不要乱，这里的0和1指的就是正常的时间
        x = x_start

        # 生成路径，路径要考虑是否为第一轮
        if first_it:
            # 难点来了，第一轮的时候，我们是不知道前一个时刻的F是什么的，只能靠自己定义，比如布朗桥
            assert mode == 'b'  # 必须是backward先开始
            for k in range(self.num_steps):
                t = t_prev[k]  # shape: (1)
                t = torch.ones((x.shape[0], 1), device=self.device) * t
                # 按照SDE的形式生成路径，前向和逆向现在都统一了
                # 1. 维纳过程，就认为F是一个恒等变换，也即F(x) = x
                dw = torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
                dw = dw.to(self.device)

                # 线性插值
                vec = (x_1 - x) / (1 - t)
                pred_x = x + vec * dt

                # pred = x  # F(x)预测的就是x
                pred_x = pred_x.to(self.device)
                # 维纳过程计算x_{k+1}
                x = pred_x + torch.sqrt(torch.tensor(2 * self.gamma[k])) * dw
                # x = x + pred + torch.sqrt(torch.tensor(2 * self.gamma)) * dw
                # 2. 当前的x已经是下一个时刻的值，可以来算损失函数里面F(X[k+1])了
                # 同样计算pred_x_next，只要vec是上一个时刻的就好了
                pred_x_next = x + vec * dt

                # pred_x_next = x
                # 3. 计算这一轮的B的回归目标是什么
                target_cur = x + pred_x - pred_x_next  # 原论文目标
                # target_cur = pred - pred_next  # 作者代码实现
                # target_cur = -torch.sqrt(torch.tensor(
                #     2 * self.gamma)) * dw  # 其他实现
                # 现在都有都有了，我们来整理整理
                path.append(x.detach().clone())
                target.append(target_cur)
                t_list.append(torch.ones((x.shape[0], 1)) * t_cur[k])

        else:
            # 每一个时刻，都计算一次，一个batch的每一个样本获得一个路径的位置、时间和目标
            for k in range(self.num_steps):
                # 这个t是上一轮模型对应的时间
                t = t_prev[k]  # shape: (1)
                # 为了计算，要把t的维度变为和x的batch_size一样，也即是(batch_size, 1)
                t = torch.ones((x.shape[0], 1), device=self.device) * t
                # 按照SDE的形式生成路径，前向和逆向现在都统一了
                # 1. 用上一轮的B或者F生成上一个时刻或下一个时刻的x，也即x[k]或者x[k+1]}，同时要保留B(x[k+1])或者F(x[k])的值(bf_x)，这两个值也是损失函数的一部分
                # 维纳过程dz
                dw = torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)

                x = x.to(self.device)
                dw = dw.to(self.device)

                pred_x = self.model_dict[prev_mode](x, t)
                # x -> x_next 也即 x[k] -> x[k+1] 或 x[k+1] -> x[k]
                x = pred_x + torch.sqrt(torch.tensor(2 * self.gamma[k])) * dw
                # x = x + pred + torch.sqrt(torch.tensor(2 * self.gamma)) * dw

                # 2. 当前的x已经是上一个或下一个时刻的值，可以来算损失函数里面B(X[k])和F(x[k+1])这两项了
                pred_next_x = self.model_dict[prev_mode](x, t)
                # 3. 计算这一轮的F或者B的回归目标是什么
                target_cur = x + pred_x - pred_next_x  # 原论文目标
                # target_cur = pred - pred_next  # 作者代码实现，在训练的时候的pred再减去x
                # target_cur = -pred_next - torch.sqrt(
                #     torch.tensor(2 * self.gamma)) * dw  # 其他实现
                # 现在都有都有了，我们来整理整理
                path.append(x.detach().clone())
                target.append(target_cur)
                t_list.append(torch.ones((x.shape[0], 1)) * t_cur[k])

        # 最后就是把我们得到的所有信息stack一下，构建新的数据
        path = torch.stack(path).to(
            self.device)  # shape: (num_steps, batch_size, input_dim)
        target = torch.stack(target).to(
            self.device)  # shape: (num_steps, batch_size, input_dim)
        t_list = torch.stack(t_list).to(
            self.device)  # shape: (num_steps, batch_size, 1)

        # 每个batch的每一个样本都采样一个时刻
        t_sample = torch.randint(
            0, self.num_steps, (1, x_0.shape[0], 1),
            device=self.device)  # shape: (1, batch_size, 1)
        # 从路径中取出对应时刻的路径和目标，用gather函数
        t_list = torch.gather(t_list, 0, t_sample).squeeze(0)
        x_t = torch.gather(path, 0,
                           t_sample.expand(1, x_0.shape[0],
                                           x_0.shape[1])).squeeze(0)

        target = torch.gather(target, 0,
                              t_sample.expand(1, x_0.shape[0],
                                              x_0.shape[1])).squeeze(0)

        # 恢复训练模式
        self.model_dict[mode].train()
        # 恢复模型权重
        if ema_model is not None:
            ema_model.restore()
            self.model_dict[mode].load_state_dict(ema_model.model.state_dict())

        return x_t, target, t_list

    @torch.no_grad()
    def sde_sample(self,
                   x_start,
                   mode: str = 'b',
                   num_steps: Optional[int] = None,
                   return_path: Optional[bool] = True):
        # 欧拉丸山法根据模型预测的值，生成结果，用于推理
        # 这个函数可以选择输出每一个时刻的值，便于画图
        if num_steps is None:
            num_steps = self.num_steps

        dt = 1.0 / num_steps
        path = []
        x = x_start
        x = x.to(self.device)
        path.append(x.detach().clone())

        # 输入模型的时间序列
        t = np.arange(num_steps) / num_steps
        if mode == 'b':
            t = 1 - t

        for k in range(num_steps):
            t_cur = torch.ones((x.shape[0], 1), device=self.device) * t[k]
            pred = self.model_dict[mode](x, t_cur)
            dw = torch.sqrt(torch.tensor(dt)) * torch.randn_like(x)
            dw = dw.to(self.device)
            if k == num_steps - 1:
                # 采样过程最后一步就不加噪声了
                x = pred
            else:
                x = pred + torch.sqrt(torch.tensor(2 * self.gamma[k])) * dw
            # x = x + pred + torch.sqrt(torch.tensor(2 * self.gamma)) * dw

            path.append(x.detach().clone())

        if return_path:
            return path
        else:  # 仅返回最终结果
            return x

    # 最基本的MSE loss
    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)


if __name__ == '__main__':
    input_dim = 11
    hidden_dim = 10
    output_dim = 10
    num_layers = 3

    forward_model = DSBModel(input_dim, hidden_dim, output_dim, num_layers)
    backward_model = DSBModel(input_dim, hidden_dim, output_dim, num_layers)

    dsb = DSB(forward_model, backward_model)
    x_0 = torch.randn(10, 10)
    x_1 = torch.randn(10, 10)
    path, target, t_list = dsb.generate_path_and_target(x_0,
                                                        x_1,
                                                        mode='b',
                                                        first_it=False)

    print(path.shape, target.shape, t_list.shape)

    path = dsb.sde_sample(x_1, mode='b')

    print(path)

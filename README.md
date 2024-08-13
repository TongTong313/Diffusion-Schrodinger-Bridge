# 从零手搓扩散薛定谔桥（Diffusion Schrödinger Bridge）

作者：Tong Tong 

B站主页：[Tong发发](https://space.bilibili.com/323109608)

本套代码**未来会出讲解视频**，为了让大家更好的理解，强烈建议观看本人下面的几个B站视频：
- [白话薛定谔桥](https://www.bilibili.com/video/BV1dsYieMEvj/)
- [扩散模型随机微分方程（SDE）公式保姆级手推](https://www.bilibili.com/video/BV1y1YpejEB4/)
- [你一定能听懂的扩散模型Flow Matching基本原理深度解析](https://www.bilibili.com/video/BV1Wv3xeNEds/)
- [你一定能听懂的Recitified Flow基本原理深度解析](https://www.bilibili.com/video/BV19m421G7W8/)
- [零门槛掌握DDPM](https://www.bilibili.com/video/BV1zz421i7UM/)

**特别推荐看一下本人的[扩散模型之老司机开车理论视频](https://www.bilibili.com/video/BV1qW42197dv/)，对你理解扩散模型有很大帮助~**

**TODO**：
- [ ] 加班加点准备代码讲解视频…… 
- [ ] 模型权重文件上传
- [ ] 计划实现一些DSB的变种

**一些bug修复说明**:
- 暂无


## V1.0：Diffusion Schrödinger Bridge

### 说明

* 代码基于人为生成的二维分布数据，一个为棋盘分布一个为爱心曲线分布。
* 本项目完全**从零手搓**，尽可能不参考其他任何代码，从论文原理出发逐步实现，因此算是**极简实现**的一种，并**不能保证最优性能**，各位大佬可以逐步修改完善，欢迎交流。
* 为了让大家都能上手，本代码只基于深度学习框架Pytorch和一些其他必要的库。该数据集随着训练代码生成，数据集维度与规模较小，也方便展示效果，最重要的是**即使是使用CPU都能训练**！！！
* 模型结构自己手搓了一个MLP模型，大家可以根据自己的需求修改，也可以使用其他更复杂的模型。
* 代码中有很多注释，希望能帮助大家理解代码，如果有问题欢迎留言交流。
* 代码环境要求很低，甚至不需要GPU都可以
    * Python 3.8+
    * Pytorch 2.0+ 
    * Numpy
    * Matplotlib
    * 其他的就缺啥装啥
* 代码运行方式
    * 训练：`python train.py`
    * 推理：`python infer.py`
* 代码实现原理参考论文
    * Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
* 代码结果展示
    * 前向过程：爱心分布 -> 棋盘分布
    ![result_forward](/fig/forward.gif)


    * 逆向过程：棋盘分布 -> 爱心分布
    ![result_backwrad](/fig/backward.gif)

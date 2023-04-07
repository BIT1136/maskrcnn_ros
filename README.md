# maskrcnn_ros

[Maskrcnn](https://github.com/BIT1136/Maskrcnn) 的 ROS 包装器.

## 安装

    git clone --recurse-submodules git@github.com:BIT1136/maskrcnn_ros.git

## 安装依赖

使用 conda/mamba 创建虚拟环境并安装依赖:

    conda create -n maskrcnn_ros python=3.10 pytorch=1.13 torchvision numpy

可能兼容更旧的环境.

将 `nodes/node.py` 第一行中的 Python 解释器路径修改为安装了依赖的解释器.

## 添加模型

将模型文件置于 `models` 文件夹中,修改 `nodes/node.py` 中的 `model_path` 变量.

## 运行节点

    roslaunch maskrcnn_ros node.launch
    

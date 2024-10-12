# DRL_crypto

**A deep reinforcement learning research platform for crypto trading.**

`DRL_crypto` 是一个基于深度强化学习的加密货币交易研究平台，提供了用于构建、训练和验证算法的完整工具链，集成了多模态输入、数据处理、回测和强化学习引擎。

## Features

- **RL Engine**:
  - 封装了 Stable Baselines3（SB3）和 Gymnasium，用于强化学习算法的训练。
  - 提供了简化的接口，使得在自定义环境中应用多种 RL 算法更加便捷。

- **Backtest Engine**:
  - 集成回测模块，允许用户通过历史数据验证策略的表现。
  - 支持账户信息模拟，包括初始资金、仓位、交易记录等，可以作为观测空间的一部分。

- **Algo Environment**:
  - 支持多模态输入（multi-modal observation），例如 `candle`（蜡烛图）、`trade`（交易数据）、`LOB`（订单簿）等。
  - **Data Preprocessing**:
    - 提供内置的数据处理模块，支持常用的市场数据，如 K线、交易记录、订单簿等。
    - 支持因子嵌入，可以使用 `dollar bar`、`price percent change` 等多种采样方法。
    - 包含默认的归一化方法，采用滚动窗口的 scaled sigmoid 归一化，确保不同类型数据都能有效参与强化学习过程。

- **Customizable Observations**:
  - 可以将账户信息嵌入到环境的观测空间中，作为状态的一部分，帮助模型更好地理解账户状态和市场动态。

## Data Normalization

目前内置的归一化方法使用了 **scaled sigmoid rolling window normalization**，这是一个滚动窗口的归一化技术，确保数据平滑、标准化后输入神经网络，提高训练效果。

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone 

import torch

def prepare_data(train_X, test_X, train_Y, test_Y, device,length):
    """
    预处理数据函数，将 numpy 数组转换为 PyTorch 张量并进行相应的处理。

    参数:
    - train_X (np.ndarray): 训练数据特征
    - test_X (np.ndarray): 测试数据特征
    - train_Y (np.ndarray): 训练数据标签
    - test_Y (np.ndarray): 测试数据标签
    - device (torch.device): 设备 (CPU 或 GPU)

    返回:
    - train_X (torch.Tensor): 处理后的训练数据特征
    - test_X (torch.Tensor): 处理后的测试数据特征
    - train_Y (torch.Tensor): 处理后的训练数据标签
    - test_Y (torch.Tensor): 处理后的测试数据标签
    """

    # 将 numpy 数组转换为 PyTorch 张量
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_Y = torch.from_numpy(train_Y)
    test_Y = torch.from_numpy(test_Y)

    # 将标签从 one-hot 编码转换为整型标签
    train_Y = torch.argmax(train_Y, dim=1).long()  # 转换为整型
    test_Y = torch.argmax(test_Y, dim=1).long()  # 转换为整型

    # 重新调整训练和测试数据的形状
    train_X = train_X.view(-1, 1, length)  # 使用 -1 代替硬编码的批量大小
    test_X = test_X.view(-1, 1, length)  # 使用 -1 代替硬编码的批量大小

    # 转换数据类型
    train_X = train_X.to(torch.float32)
    test_X = test_X.to(torch.float32)

    # 将数据转移到指定设备 (CPU 或 GPU)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    return train_X, test_X, train_Y, test_Y

def val_prepare_data(val_X, val_Y, device,length):
    
    # 将 numpy 数组转换为 PyTorch 张量
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)
    # 将标签从 one-hot 编码转换为整型标签
    val_Y = torch.argmax(val_Y, dim=1).long()  # 转换为整型

    # 重新调整训练和测试数据的形状
    val_X = val_X.view(-1, 1, length)  # 使用 -1 代替硬编码的批量大小

    # 转换数据类型
    val_X = val_X.to(torch.float32)

    # 将数据转移到指定设备 (CPU 或 GPU)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)

    return val_X, val_Y
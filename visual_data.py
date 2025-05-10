import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_model_confusion_matrix(model, dataloader, device, class_names=None, figsize=(10, 7)):
    """
    自动计算并绘制模型在测试集上的混淆矩阵。

    参数:
        model (torch.nn.Module): 已训练好的模型
        dataloader (DataLoader): 测试数据集加载器
        device (torch.device): 模型所在设备
        class_names (list or None): 类别名称列表；如果为 None，则自动生成 ['0', '1', ..., 'N-1']
        figsize (tuple): 图像大小
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 自动生成类别名称
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.show()

def visualize_tsne(model, dataloader, device, min_val=-80, max_val=80, title='t-SNE Visualization'):
    """
    使用 t-SNE 对模型的特征输出进行可视化。

    参数:
        model (torch.nn.Module): 已训练好的模型
        dataloader (DataLoader): 用于测试的 dataloader
        device (torch.device): 当前计算设备
        min_val (float): t-SNE 映射的最小值（用于归一化）
        max_val (float): t-SNE 映射的最大值（用于归一化）
        title (str): 图像标题
    """

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(label.cpu().numpy())

    # 合并所有批次的特征与标签
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 归一化到 [0, 1]
    def scale_to_01(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val)

    features_tsne_scaled = np.zeros_like(features_tsne)
    features_tsne_scaled[:, 0] = scale_to_01(features_tsne[:, 0], min_val, max_val)
    features_tsne_scaled[:, 1] = scale_to_01(features_tsne[:, 1], min_val, max_val)

    # 可视化绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne_scaled[:, 0], features_tsne_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()
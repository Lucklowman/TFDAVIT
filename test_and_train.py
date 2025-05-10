import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import time

def evaluate_model(model, dataloader, device):
    model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集上的模型准确率: {accuracy:.2f}%')
    return accuracy

def train_one_epoch(model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x, batch_y
        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_time = time.time() - start_time
    accuracy = correct / total
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss, accuracy, epoch_time

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x, batch_y
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss, accuracy

def train_and_evaluate(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    num_epochs, 
    train_one_epoch, 
    evaluate
):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_time = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        epoch_times.append(train_time)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Time: {train_time:.2f}s")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Learning Rate: {current_lr:.8f}")

    best_epoch = test_accuracies.index(max(test_accuracies))
    best_test_accuracy = test_accuracies[best_epoch]
    best_test_loss = test_losses[best_epoch]

    total_time = sum(epoch_times)

    print(f"\n最高测试精度: {best_test_accuracy:.4f} 在第 {best_epoch + 1} 次迭代时")
    print(f"对应的测试损失: {best_test_loss:.4f}")
    print(f"所有迭代的总用时: {total_time:.2f} 秒")

    plot_results(num_epochs, train_losses, test_losses, train_accuracies, test_accuracies)

def plot_results(num_epochs, train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

